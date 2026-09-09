import numpy as np
from .transform3d import PhotoMetricDistortionMultiViewImage ,NormalizeMultiviewImage \
    ,RandomScaleImageMultiViewImage ,PadMultiViewImage
import torch
import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)
_RENDERED_PLACEHOLDER_LOGGED_PATHS = set()

PhotoMetricDistortionMultiViewImage = PhotoMetricDistortionMultiViewImage(
    brightness_delta=32,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=18)
NormalizeMultiviewImage = NormalizeMultiviewImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)
RandomScaleImageMultiViewImage = RandomScaleImageMultiViewImage(scales=[0.4])
PadMultiViewImage = PadMultiViewImage(size=None, size_divisor=32, pad_val=0)


def LoadMultiViewImageFromFiles(agent_input,synthetic=False):
    image_result = {}

    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
   # image_result["camera2ego"] = []
    image_result["camera_intrinsics"] = []
    image_result["img"] = []
    validity_list = []

    for cameras in agent_input.cameras[-1:]:
        for cam in [cameras.cam_b0, cameras.cam_f0, cameras.cam_l0,cameras.cam_r0]:#, cameras.cam_l1, cameras.cam_r1 cameras.cam_l1, cameras.cam_l2, , cameras.cam_r1, cameras.cam_r2
            selected_image = cam.rendered_image if synthetic else cam.image
            if selected_image is None:
                continue
            if synthetic:
                img = selected_image.astype(np.float32)
                validity = torch.tensor(bool(cam.rendered_valid))
            else:
                img = selected_image.astype(np.float32)
                validity = torch.tensor(bool(cam.real_valid))

            image_result["img"].append(img)
            validity_list.append(validity)

            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam.sensor2lidar_rotation)
            lidar2cam_t = cam.sensor2lidar_translation @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam.intrinsics
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = cam.intrinsics
            image_result["camera_intrinsics"].append(camera_intrinsics)


    image_result.update(
        dict(
            lidar2img=lidar2img_rts,
            cam_intrinsic=cam_intrinsics,
            lidar2cam=lidar2cam_rts,
        ))
    if len(validity_list) != 4:
        raise ValueError(f"Expected four camera views, got {len(validity_list)}")
    image_result["validity"] = torch.all(torch.stack(validity_list))
    return image_result



def _rendered_view_count(agent_input) -> int:
    cameras = agent_input.cameras[-1]
    return sum(
        cam.rendered_image is not None
        for cam in (cameras.cam_b0, cameras.cam_f0, cameras.cam_l0, cameras.cam_r0)
    )


def _record_rendered_placeholder_once(log_path: str) -> None:
    if not log_path or log_path in _RENDERED_PLACEHOLDER_LOGGED_PATHS:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "event": "all_rendered_views_missing_placeholder_used",
        "pid": os.getpid(),
        "rendered_camera_feature": "zeros_like(real_camera_feature)",
    }, indent=2, sort_keys=True) + "\n")
    logger.warning("All four rendered views are missing; using the pdm-scoring-only zero placeholder. Record: %s", path)
    _RENDERED_PLACEHOLDER_LOGGED_PATHS.add(log_path)


def _get_bev_feature(
    agent_input,
    training: bool = False,
    allow_missing_rendered_placeholder: bool = False,
    rendered_placeholder_log_path: str = "",
):
    rendered_view_count = _rendered_view_count(agent_input)
    use_placeholder = rendered_view_count == 0 and allow_missing_rendered_placeholder
    synthetic_image_result = None if use_placeholder else LoadMultiViewImageFromFiles(agent_input,synthetic=True)
    real_image_result=LoadMultiViewImageFromFiles(agent_input,synthetic=False)
    # if training:
    #     image_result = PhotoMetricDistortionMultiViewImage(image_result)
    if synthetic_image_result is not None:
        image_result = synthetic_image_result
        image_result = NormalizeMultiviewImage(image_result)
        image_result = RandomScaleImageMultiViewImage(image_result)  # 432,768
        image_result = PadMultiViewImage(image_result)  # 448,768
        imgs = [img.transpose(2, 0, 1) for img in image_result['img']]
        synthetic_camera_feature = torch.tensor(np.ascontiguousarray(np.stack(imgs, axis=0)))

    image_result = real_image_result
    image_result = NormalizeMultiviewImage(image_result)
    image_result = RandomScaleImageMultiViewImage(image_result)  # 432,768
    image_result = PadMultiViewImage(image_result)  # 448,768
    imgs = [img.transpose(2, 0, 1) for img in image_result['img']]
    real_camera_feature = torch.tensor(np.ascontiguousarray(np.stack(imgs, axis=0)))

    if use_placeholder:
        synthetic_camera_feature = torch.zeros_like(real_camera_feature)
        _record_rendered_placeholder_once(rendered_placeholder_log_path)

    features = {"camera_feature": real_camera_feature,
                "camera_valid":real_image_result["validity"],
                "rendered_camera_feature":synthetic_camera_feature,
                "img_shape": torch.FloatTensor(np.stack(real_image_result["img_shape"])),#8,3
                "lidar2img": torch.FloatTensor(np.stack(real_image_result["lidar2img"]))#8,4,4
                }

    return features
