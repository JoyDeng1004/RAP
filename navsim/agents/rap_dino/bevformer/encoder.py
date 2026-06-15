from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
from datetime import datetime
import os
import warnings
from mmengine.registry import MODELS
from mmcv.cnn.bricks.transformer import TransformerLayerSequence

import numpy as np
import torch
from mmengine.utils.dl_utils.parrots_wrapper import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
import torch.nn as nn

_RAP_REF2D_DUMP_DONE = False
_RAP_REF2D_DUMP_STAGE_COUNT = 0


def _rap_dump_ref2d_enabled():
    return os.environ.get('RAP_DUMP_REF2D', '').lower() in ('1', 'true', 'yes', 'on')


def _rap_numpy_batch_item(value, batch_index):
    if isinstance(value, torch.Tensor):
        return value[batch_index].detach().cpu().numpy()
    return np.asarray(value[batch_index])


def _rap_uint8_rgb_images_from_debug(camera_image_debug, batch_index, num_cam):
    camera_images = _rap_numpy_batch_item(camera_image_debug, batch_index)
    assert camera_images.ndim == 4 and camera_images.shape[0] == num_cam
    assert camera_images.shape[-1] == 3
    return np.clip(camera_images, 0, 255).astype(np.uint8)


def _rap_uint8_rgb_images_from_camera_feature(camera_feature, batch_index, num_cam):
    assert camera_feature.ndim == 5
    camera_images = camera_feature[batch_index].detach().to(torch.float32).cpu().numpy()
    assert camera_images.shape[0] == num_cam and camera_images.shape[1] == 3
    camera_images = camera_images.transpose(0, 2, 3, 1)
    image_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    image_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    camera_images = camera_images * image_std + image_mean
    return np.clip(camera_images, 0, 255).astype(np.uint8)

@MODELS.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args,bev_h,bev_w, pc_range=None, num_points_in_pillar=4,lidar_height=0,half_width=0,half_length=0,rear_axle_to_center=0, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):
        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

        bev_h=bev_h
        bev_w=bev_w

        ref_3d = self.get_reference_points( bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d')
        #1,4,10000,3       height=4 0.125,0.375,0.625,0.875
        ref_2d = self.get_reference_points(bev_h, bev_w, dim='2d')#1,64,1,2

        bs, len_bev, num_bev_level, _ = ref_2d.shape#bev_level 1
        hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, len_bev, num_bev_level, 2)#2,64,1,2

        reference_points = ref_3d.to(torch.float32).clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1).permute(1, 0, 2, 3)

        self.reference_points =nn.Parameter(reference_points,requires_grad=False)#num_points_in_pillar,bs,w*h,xy

        self.ref_3d=nn.Parameter(ref_3d,requires_grad=False)
        self.hybird_ref_2d=nn.Parameter(hybird_ref_2d,requires_grad=False)
        self.ref_2d=nn.Parameter(ref_2d,requires_grad=False)


        self.half_length = half_length
        self.half_width = half_width
        self.rear_axle_to_center = rear_axle_to_center
        self.lidar_height=lidar_height

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cpu', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z#4,8,8
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W#4,8,8
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H#4,8,8
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 2, 1).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)#1,4,64,3
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_x,ref_y = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    #@force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points,   img_metas):

        #lidar2img = []
        # for img_meta in img_metas:
        #     lidar2img.append(img_meta['lidar2img'])
        # lidar2img=img_metas['lidar2img']
        # lidar2img = np.asarray(lidar2img)
        # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        lidar2img=img_metas['lidar2img']
        num_cam = lidar2img.size(1)
        D, B, num_query = reference_points.size()[:3]

        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_metas['img_shape'][0][0][1]
        reference_points_cam[..., 1] /= img_metas['img_shape'][0][0][0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask


    def compute_corners(self,boxes):
        # Calculate half dimensions
        x = boxes[:, 0]        # x-coordinate of the center
        y = boxes[:, 1]        # y-coordinate of the center
        headings= boxes[:, 2]

        half_width =torch.zeros_like(x)+self.half_width
        half_length = torch.zeros_like(x)+self.half_length

        cos_yaw = torch.cos(headings)[...,None]
        sin_yaw = torch.sin(headings)[...,None]

        x=x[...,None]+self.rear_axle_to_center * cos_yaw
        y=y[...,None]+self.rear_axle_to_center * sin_yaw

        # Compute the four corners
        corners_x = torch.stack([half_length, half_length, -half_length, -half_length],dim=-1)
        corners_y = torch.stack([half_width, -half_width, -half_width, half_width],dim=-1)

        # Rotate corners by yaw
        rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
        rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

        # Translate corners to the center of the bounding box
        corners = torch.stack((rot_corners_x + x, rot_corners_y + y), dim=-1)

        return corners


    #@auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                ref_2d=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        output = bev_query
        intermediate = []

        bs=bev_query.shape[0]
        len_bev=bev_query.shape[1]

        with torch.autocast(device_type='cuda', enabled=False):  # Disable autocasting
            if ref_2d is not None:
                ref_pos =( ref_2d[:, :, None, :2]+32)/64

                hybird_ref_2d = torch.cat([ref_pos, ref_pos])

                zs = torch.linspace(self.pc_range[2]-self.lidar_height, self.pc_range[5]-self.lidar_height, self.num_points_in_pillar, dtype=torch.float32,
                                    device=ref_2d.device)

                zs = zs[None, None, :, None].repeat(bs, len_bev, 1, 1)

                # ref_pos = self.compute_corners(ref_2d.reshape(-1, 3)).reshape(-1, len_bev, 4, 2)

                # P = self.num_points_in_pillar
                # B = ref_pos.shape[0]
                # zs_rep = zs.unsqueeze(2).repeat(1, 1, 4, 1, 1).reshape(B, len_bev, 4 * P, 1)  # (B, len_bev, 4*P, 1)

                # ref_pos_rep = ref_pos.unsqueeze(3).repeat(1, 1, 1, P, 1).reshape(B, len_bev, 4 * P, 2)  # (B, len_bev, 4*P, 2)

                # ref_3d = torch.cat([ref_pos_rep, zs_rep], dim=-1).permute(0, 2, 1, 3)
                ref_pos =self.compute_corners(ref_2d.reshape(-1,3)).reshape(-1,len_bev,4,2)

                zs=zs.repeat(1,1,4,1)

                ref_3d = torch.cat([ref_pos.repeat(1, 1, self.num_points_in_pillar, 1), zs], dim=-1).permute(0, 2, 1, 3)
                
                reference_points = ref_3d.to(torch.float32).clone()

                reference_points = torch.cat(
                    (reference_points, torch.ones_like(reference_points[..., :1])), -1).permute(1, 0, 2, 3)

            else:
                ref_3d = self.ref_3d.repeat(bs, 1, 1, 1)  # 2,4,64,3

                reference_points = self.reference_points.repeat(1, bs, 1, 1)  # 4,2,64,4

                if prev_bev is not None:
                    # prev_bev = prev_bev.permute(1, 0, 2)
                    prev_bev = torch.stack(
                        [prev_bev, bev_query], 1).reshape(bs * 2, len_bev, -1)

                    ref_2d = self.ref_2d.repeat(bs, 1, 1, 1)
                    shift_ref_2d = ref_2d.clone()
                    shift_ref_2d += shift[:, None, None, :]
                    hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                        bs * 2, len_bev, -1, 2)
                else:
                    hybird_ref_2d = self.hybird_ref_2d.repeat(bs, 1, 1, 1)  # 4,64,1,2

            reference_points_cam, bev_mask = self.point_sampling(
                reference_points,  kwargs['img_metas'])
            if ref_2d is not None and _rap_dump_ref2d_enabled():
                global _RAP_REF2D_DUMP_DONE, _RAP_REF2D_DUMP_STAGE_COUNT
                if not _RAP_REF2D_DUMP_DONE:
                    dump_stage = int(os.environ.get('RAP_DUMP_STAGE', '0'))
                    current_stage = _RAP_REF2D_DUMP_STAGE_COUNT
                    _RAP_REF2D_DUMP_STAGE_COUNT += 1
                    if current_stage == dump_stage:
                        with torch.no_grad():
                            ref_pos_t = (ref_2d[:, :, None, :2] + 32) / 64
                            corners = self.compute_corners(ref_2d.reshape(-1, 3)).reshape(
                                ref_2d.shape[0], ref_2d.shape[1], 4, 2)

                            batch_index = 0
                            num_cam = reference_points_cam.shape[0]
                            D = reference_points_cam.shape[3]
                            Q = ref_2d.shape[1]
                            npp = self.num_points_in_pillar

                            assert ref_2d.ndim == 3 and ref_2d.shape[2] == 3
                            assert ref_pos_t.shape == (ref_2d.shape[0], Q, 1, 2)
                            assert corners.shape == (ref_2d.shape[0], Q, 4, 2)
                            assert ref_3d.shape == (ref_2d.shape[0], D, Q, 3)
                            assert reference_points.shape == (D, ref_2d.shape[0], Q, 4)
                            assert reference_points_cam.shape == (num_cam, ref_2d.shape[0], Q, D, 2)
                            assert bev_mask.shape == (num_cam, ref_2d.shape[0], Q, D)
                            assert D == 4 * npp

                            dump_dir = os.environ.get('RAP_DUMP_DIR', './ref2d_debug')
                            os.makedirs(dump_dir, exist_ok=True)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            dump_path = os.path.join(
                                dump_dir, f'ref2d_stage{current_stage}_{timestamp}.npz')

                            dump_data = dict(
                                ref_2d=ref_2d[batch_index].detach().cpu().numpy(),
                                ref_pos=ref_pos_t[batch_index, :, 0].detach().cpu().numpy(),
                                corners=corners[batch_index].detach().cpu().numpy(),
                                ref_3d=ref_3d[batch_index].detach().cpu().numpy(),
                                reference_points_cam=reference_points_cam[:, batch_index].detach().cpu().numpy(),
                                bev_mask=bev_mask[:, batch_index].detach().cpu().numpy(),
                                bev_h=np.array(bev_h),
                                bev_w=np.array(bev_w),
                                npp=np.array(npp),
                                D=np.array(D),
                                proposal_num=np.array(bev_h),
                                num_poses=np.array(bev_w),
                                Q=np.array(Q),
                                num_cam=np.array(num_cam),
                                pc_range=np.array(self.pc_range),
                                half_width=np.array(self.half_width),
                                half_length=np.array(self.half_length),
                                rear_axle_to_center=np.array(self.rear_axle_to_center),
                                lidar_height=np.array(self.lidar_height),
                                query_order=np.array('q = p*T + t'),
                            )

                            img_metas = kwargs['img_metas']
                            if 'img_shape' in img_metas:
                                dump_data['img_shape'] = img_metas['img_shape'][batch_index].detach().cpu().numpy()

                            camera_image_debug = img_metas.get('camera_image_debug')
                            camera_feature = img_metas.get('camera_feature')
                            if camera_image_debug is not None:
                                camera_images = _rap_uint8_rgb_images_from_debug(
                                    camera_image_debug, batch_index, num_cam)
                                dump_data.update(
                                    camera_images=camera_images,
                                    camera_order=np.array(['cam_b0', 'cam_f0', 'cam_l0', 'cam_r0']),
                                    camera_image_source=np.array(
                                        'features.camera_image_debug resized/padded unnormalized real image'),
                                    camera_image_raw_color_order=np.array('RGB'),
                                    camera_image_color_order=np.array('RGB'),
                                    camera_image_to_rgb=np.array(False),
                                )
                            elif camera_feature is not None:
                                image_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
                                image_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
                                camera_images = _rap_uint8_rgb_images_from_camera_feature(
                                    camera_feature, batch_index, num_cam)
                                dump_data.update(
                                    camera_images=camera_images,
                                    camera_order=np.array(['cam_b0', 'cam_f0', 'cam_l0', 'cam_r0']),
                                    camera_image_source=np.array(
                                        'features.camera_feature denormalized fallback'),
                                    camera_image_raw_color_order=np.array('RGB'),
                                    camera_image_color_order=np.array('RGB'),
                                    camera_image_mean=image_mean,
                                    camera_image_std=image_std,
                                    camera_image_to_rgb=np.array(False),
                                )

                            np.savez_compressed(dump_path, **dump_data)
                        _RAP_REF2D_DUMP_DONE = True
            # reference_points_cam.shape: [4,2,640,16,2]
            # bev_mask.shape:[4,2,640,16]
            # bev_mask.sum(): 12656 (valid)
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@MODELS.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
