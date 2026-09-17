import numpy as np
from .transform3d import PhotoMetricDistortionMultiViewImage ,NormalizeMultiviewImage \
    ,RandomScaleImageMultiViewImage ,PadMultiViewImage
import torch

PhotoMetricDistortionMultiViewImage = PhotoMetricDistortionMultiViewImage(
    brightness_delta=32,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=18)
NormalizeMultiviewImage = NormalizeMultiviewImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
RandomScaleImageMultiViewImage = RandomScaleImageMultiViewImage(scales=[0.4])
PadMultiViewImage = PadMultiViewImage(size=None, size_divisor=32, pad_val=0)


def LoadMultiViewImageFromFiles(agent_input,synthetic=False):
    image_result = {}

    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
    image_result["camera_intrinsics"] = []
    image_result["img"] = []
    validity_list = []

    for cameras in agent_input.cameras[-1:]:
        for cam in [cameras.cam_b0, cameras.cam_f0, cameras.cam_l0, cameras.cam_r0]:
            if cam.image is None:
                continue
            # Select calibration that matches the image source.
            if synthetic:
                img = cam.rendered_image.astype(np.float32)
                validity = torch.tensor(True)
                sensor2lidar_rotation = cam.render_sensor2lidar_rotation
                sensor2lidar_translation = cam.render_sensor2lidar_translation
                intrinsic = cam.render_intrinsics
            else:
                img = cam.image.astype(np.float32)
                validity = torch.tensor(cam.real_valid)
                sensor2lidar_rotation = cam.sensor2lidar_rotation
                sensor2lidar_translation = cam.sensor2lidar_translation
                intrinsic = cam.intrinsics

            image_result["img"].append(img)
            validity_list.append(validity)

            lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
            lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = intrinsic
            image_result["camera_intrinsics"].append(camera_intrinsics)


    image_result.update(
        dict(
            lidar2img=lidar2img_rts,
            cam_intrinsic=cam_intrinsics,
            lidar2cam=lidar2cam_rts,
        ))
    image_result["validity"] = torch.all(torch.stack(validity_list))
    return image_result



def _get_bev_feature( agent_input, training: bool=False):
    synthetic_image_result=LoadMultiViewImageFromFiles(agent_input,synthetic=True)
    real_image_result=LoadMultiViewImageFromFiles(agent_input,synthetic=False)
    image_result = synthetic_image_result
    image_result = NormalizeMultiviewImage(image_result)
    image_result = RandomScaleImageMultiViewImage(image_result)
    image_result = PadMultiViewImage(image_result)
    imgs = [img.transpose(2, 0, 1) for img in image_result['img']]
    synthetic_camera_feature = torch.tensor(np.ascontiguousarray(np.stack(imgs, axis=0)))

    image_result = real_image_result
    image_result = NormalizeMultiviewImage(image_result)
    image_result = RandomScaleImageMultiViewImage(image_result)
    image_result = PadMultiViewImage(image_result)
    imgs = [img.transpose(2, 0, 1) for img in image_result['img']]
    real_camera_feature = torch.tensor(np.ascontiguousarray(np.stack(imgs, axis=0)))


    # Preserve rendered calibration for the rendered-image forward pass.
    features = {"camera_feature": real_camera_feature,
                "camera_valid":real_image_result["validity"],
                "rendered_camera_feature":synthetic_camera_feature,
                "img_shape": torch.FloatTensor(np.stack(real_image_result["img_shape"])),
                "lidar2img": torch.FloatTensor(np.stack(real_image_result["lidar2img"])),
                "rendered_lidar2img": torch.FloatTensor(np.stack(synthetic_image_result["lidar2img"]))
                }

    return features
