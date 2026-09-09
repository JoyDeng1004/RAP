#!/usr/bin/env python3
import json
import pickle
import shutil
import tarfile
from pathlib import Path, PurePosixPath

NUM_HISTORY = 4
NUM_FUTURE = 10
NUM_FRAMES = NUM_HISTORY + NUM_FUTURE

root = Path("/Users/joy/pythonProject/RAP/navsim_one_scene")
meta_archive = root / "downloads/openscene_metadata_mini.tgz"
camera_archive = root / "downloads/openscene_sensor_mini_camera_5.tgz"
output_root = root / "dataset"

if not meta_archive.is_file() or not camera_archive.is_file():
    raise FileNotFoundError("下载分片不存在，请先完成 wget 命令。")


def normalized_camera_path(data_path, log_name):
    parts = list(PurePosixPath(str(data_path)).parts)
    if log_name not in parts:
        return None
    return "/".join(parts[parts.index(log_name):])


print("正在索引 CAM_F0 分片……")
with tarfile.open(camera_archive, "r:gz") as camera_tar:
    camera_index = {}

    for member in camera_tar:
        if not member.isfile():
            continue

        parts = list(PurePosixPath(member.name).parts)
        if "sensor_blobs" not in parts:
            continue

        # 官方归档可能是 sensor_blobs/<log>/CAM_F0，也可能在中间
        # 多一层 split：sensor_blobs/mini/<log>/CAM_F0。以 CAM_F0
        # 的前一层识别 log，可同时兼容两种结构。
        if "CAM_F0" in parts:
            cam_idx = parts.index("CAM_F0")
            if cam_idx >= 1 and len(parts) > cam_idx + 1:
                relative = "/".join(parts[cam_idx - 1:])
                camera_index[relative] = member.name

    log_names = sorted({path.split("/", 1)[0] for path in camera_index})
    if not log_names:
        raise RuntimeError("相机分片中没有找到 CAM_F0 数据。")

    print(f"分片中找到 {len(log_names)} 个候选 log。")

    selected = None

    with tarfile.open(meta_archive, "r:gz") as meta_tar:
        metadata_members = {
            PurePosixPath(member.name).stem: member
            for member in meta_tar.getmembers()
            if member.isfile() and member.name.endswith(".pkl")
        }

        for log_name in log_names:
            member = metadata_members.get(log_name)
            if member is None:
                continue

            stream = meta_tar.extractfile(member)
            frames = pickle.load(stream)

            for start in range(0, len(frames) - NUM_FRAMES + 1):
                window = frames[start:start + NUM_FRAMES]

                # NAVSIM 默认要求当前帧拥有有效 route。
                current = window[NUM_HISTORY - 1]
                if not current.get("roadblock_ids"):
                    continue

                image_paths = []
                valid = True

                for frame in window:
                    cameras = frame.get("cams", {})
                    camera = cameras.get("CAM_F0") or cameras.get("cam_f0")

                    if not camera or not camera.get("data_path"):
                        valid = False
                        break

                    relative = normalized_camera_path(
                        camera["data_path"], log_name
                    )
                    if relative not in camera_index:
                        valid = False
                        break

                    image_paths.append(relative)

                if valid:
                    selected = {
                        "log_name": log_name,
                        "frames": window,
                        "image_paths": image_paths,
                        "current_token": current["token"],
                        "scene_token": current.get("scene_token"),
                        "map_location": current.get("map_location"),
                    }
                    break

            if selected:
                break

    if selected is None:
        raise RuntimeError("该分片中没有找到可组成完整 Scene 的数据。")

    log_name = selected["log_name"]

    log_output = output_root / "navsim_logs" / "mini"
    sensor_output = output_root / "sensor_blobs" / "mini"
    log_output.mkdir(parents=True, exist_ok=True)
    sensor_output.mkdir(parents=True, exist_ok=True)

    # 保存恰好 14 帧，使 SceneLoader 只产生一个 Scene。
    with open(log_output / f"{log_name}.pkl", "wb") as file:
        pickle.dump(selected["frames"], file, protocol=pickle.HIGHEST_PROTOCOL)

    # 仅提取这 14 帧的前视图。
    for relative in selected["image_paths"]:
        destination = sensor_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        source = camera_tar.extractfile(camera_index[relative])
        with open(destination, "wb") as target:
            shutil.copyfileobj(source, target)

manifest = {
    "split": "mini",
    "log_name": selected["log_name"],
    "token": selected["current_token"],
    "scene_token": selected["scene_token"],
    "map_location": selected["map_location"],
    "num_history_frames": NUM_HISTORY,
    "num_future_frames": NUM_FUTURE,
    "num_front_images": len(selected["image_paths"]),
}

with open(output_root / "scene_manifest.json", "w") as file:
    json.dump(manifest, file, indent=2)

print("\n提取成功：")
print(json.dumps(manifest, indent=2))
print(f"\n数据根目录：{output_root.resolve()}")
