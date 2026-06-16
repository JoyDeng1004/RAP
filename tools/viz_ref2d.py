#!/usr/bin/env python3
import argparse
import math

import numpy as np


HIGHLIGHT = "#d62728"
SELECTED = "#ffcc00"
CORNER_NAMES = ("FL", "FR", "RR", "RL")


def _scalar(data, key, default=None):
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _default_query(Q, P, T):
    return min(Q - 1, (P // 2) * T + (T // 2))


def _text_array(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return [str(v) for v in value.tolist()]
    return str(value)


def _closed_polygon(points):
    return np.concatenate([points, points[:1]], axis=0)


def _points_in_unit_image(points):
    return (
        np.isfinite(points[..., 0])
        & np.isfinite(points[..., 1])
        & (points[..., 0] >= 0.0)
        & (points[..., 0] <= 1.0)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] <= 1.0)
    )


def _points_in_pixel_image(points, width, height):
    return (
        np.isfinite(points[..., 0])
        & np.isfinite(points[..., 1])
        & (points[..., 0] >= 0.0)
        & (points[..., 0] < width)
        & (points[..., 1] >= 0.0)
        & (points[..., 1] < height)
    )


def _oriented_box_corners(x, y, heading, half_length, half_width, rear_axle_to_center):
    center_x = x + rear_axle_to_center * np.cos(heading)
    center_y = y + rear_axle_to_center * np.sin(heading)
    local = np.array([
        [half_length, half_width],
        [half_length, -half_width],
        [-half_length, -half_width],
        [-half_length, half_width],
    ], dtype=np.float32)
    rot = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)],
    ], dtype=np.float32)
    return local @ rot.T + np.array([center_x, center_y], dtype=np.float32)


def _plot_ego(plt, ax, ref_2d, P, T, q_star):
    traj = ref_2d.reshape(P, T, 3)
    time = np.arange(T)
    cmap = plt.get_cmap("viridis", T)
    p_star, t_star = divmod(q_star, T)

    for p in range(P):
        xy = traj[p, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="0.75", linewidth=0.7, alpha=0.7)
        ax.scatter(xy[:, 0], xy[:, 1], c=time, cmap=cmap, vmin=0, vmax=max(T - 1, 1),
                   s=12, alpha=0.75)

    xy_star = traj[p_star, :, :2]
    ax.plot(xy_star[:, 0], xy_star[:, 1], color=HIGHLIGHT, linewidth=2.0)
    ax.scatter([traj[p_star, t_star, 0]], [traj[p_star, t_star, 1]],
               marker="*", s=160, color=HIGHLIGHT, edgecolor="black", linewidth=0.6, zorder=5)

    step = max(1, T // 5)
    for p in range(P):
        pts = traj[p, ::step]
        ax.quiver(pts[:, 0], pts[:, 1], np.cos(pts[:, 2]), np.sin(pts[:, 2]),
                  angles="xy", scale_units="xy", scale=4.0, width=0.002,
                  color="0.25", alpha=0.35)

    ax.set_title(f"ego-frame ref_2d (q*={q_star}, p={p_star}, t={t_star})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4)


def _plot_metric_bev(plt, ax, ref_2d, corners, P, T, q_star, data):
    traj = ref_2d.reshape(P, T, 3)
    p_star, t_star = divmod(q_star, T)

    for p in range(P):
        xy = traj[p, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="0.68", linewidth=0.7, alpha=0.45)

    xy_star = traj[p_star, :, :2]
    ax.plot(xy_star[:, 0], xy_star[:, 1], color=HIGHLIGHT, linewidth=2.2, zorder=4,
            label="red line: q* proposal trajectory")
    ax.scatter(xy_star[:, 0], xy_star[:, 1], s=24, color=HIGHLIGHT, zorder=5)
    ax.scatter([traj[p_star, t_star, 0]], [traj[p_star, t_star, 1]],
               marker="*", s=180, color=HIGHLIGHT, edgecolor="black", linewidth=0.6, zorder=7,
               label="red star: q* pose")

    q_box = _closed_polygon(corners[q_star])
    ax.plot(q_box[:, 0], q_box[:, 1], color=HIGHLIGHT, linewidth=2.0, zorder=6,
            label="red box: q* footprint")
    ax.fill(q_box[:, 0], q_box[:, 1], color=HIGHLIGHT, alpha=0.10, zorder=3)
    for name, xy in zip(CORNER_NAMES, corners[q_star]):
        ax.text(xy[0], xy[1], name, color=HIGHLIGHT, fontsize=8)

    half_length = float(_scalar(data, "half_length", 2.8))
    half_width = float(_scalar(data, "half_width", 1.2))
    rear_axle_to_center = float(_scalar(data, "rear_axle_to_center", 1.46))
    ego_box = _closed_polygon(_oriented_box_corners(
        0.0, 0.0, 0.0, half_length, half_width, rear_axle_to_center))
    ax.plot(ego_box[:, 0], ego_box[:, 1], color="black", linewidth=1.4)
    ax.fill(ego_box[:, 0], ego_box[:, 1], color="black", alpha=0.08)
    ax.arrow(0.0, 0.0, 2.8, 0.0, width=0.08, head_width=0.55, head_length=0.8,
             color="black", length_includes_head=True)

    pc_range = np.asarray(_scalar(data, "pc_range", np.array([-32, -32, -2, 32, 32, 6])))
    if pc_range.shape[0] >= 6:
        ax.set_xlim(float(pc_range[0]), float(pc_range[3]))
        ax.set_ylim(float(pc_range[1]), float(pc_range[4]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"metric BEV overlay (q*={q_star}, p={p_star}, t={t_star})")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.grid(True, linewidth=0.35, alpha=0.45)
    ax.legend(loc="best", fontsize=8)


def _plot_temporal(fig, outer_spec, ref_pos, P, T, q_star):
    inner = outer_spec.subgridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.28)
    ax = fig.add_subplot(inner[0, 0])
    ax_hm = fig.add_subplot(inner[0, 1])

    in_bounds = np.all((ref_pos >= 0.0) & (ref_pos <= 1.0), axis=1)
    oob_ratio = 1.0 - float(np.mean(in_bounds))

    ax.scatter(ref_pos[in_bounds, 0], ref_pos[in_bounds, 1],
               s=12, color="#1f77b4", alpha=0.65, label="in [0,1]")
    if np.any(~in_bounds):
        ax.scatter(ref_pos[~in_bounds, 0], ref_pos[~in_bounds, 1],
                   s=22, marker="x", color="0.45", alpha=0.75, label="out")
    ax.scatter([ref_pos[q_star, 0]], [ref_pos[q_star, 1]], marker="*", s=160,
               color=HIGHLIGHT, edgecolor="black", linewidth=0.6, zorder=5)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linewidth=1.0)
    ax.set_title(f"temporal ref_pos, out-of-bounds {oob_ratio:.1%}")
    ax.set_xlabel("(x + 32) / 64")
    ax.set_ylabel("(y + 32) / 64")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    oob_grid = (~in_bounds).reshape(P, T).astype(float)
    ax_hm.imshow(oob_grid, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    p_star, t_star = divmod(q_star, T)
    ax_hm.scatter([t_star], [p_star], marker="*", s=120, color=HIGHLIGHT,
                  edgecolor="black", linewidth=0.6)
    ax_hm.set_title("P x T query order")
    ax_hm.set_xlabel("t")
    ax_hm.set_ylabel("proposal p")


def _plot_cameras(fig, outer_spec, reference_points_cam, bev_mask, data, q_star):
    num_cam, _, D, _ = reference_points_cam.shape
    cols = int(math.ceil(math.sqrt(num_cam)))
    rows = int(math.ceil(num_cam / cols))
    inner = outer_spec.subgridspec(rows, cols, wspace=0.25, hspace=0.35)

    npp = int(_scalar(data, "npp", 4))
    pc_range = np.asarray(_scalar(data, "pc_range"))
    lidar_height = float(_scalar(data, "lidar_height", 0.0))
    if pc_range.shape[0] >= 6:
        heights = np.linspace(pc_range[2] - lidar_height, pc_range[5] - lidar_height, npp)
    else:
        heights = np.arange(npp)

    axes = []
    for cam in range(num_cam):
        ax = fig.add_subplot(inner[cam // cols, cam % cols])
        axes.append(ax)
        pts = reference_points_cam[cam]
        mask = bev_mask[cam].astype(bool)
        valid = mask & _points_in_unit_image(pts)
        valid_pts = pts[valid]
        if valid_pts.size:
            ax.scatter(valid_pts[:, 0], valid_pts[:, 1], s=3, color="#1f77b4", alpha=0.18)

        q_pts = pts[q_star]
        q_mask = mask[q_star]
        unique_count = min(4, D)
        for d in range(unique_count):
            x, y = q_pts[d]
            z = heights[d % len(heights)]
            in_image = bool(_points_in_unit_image(q_pts[d]))
            if q_mask[d] and in_image:
                ax.scatter([x], [y], marker="*", s=120, color=HIGHLIGHT,
                           edgecolor="black", linewidth=0.5, alpha=1.0, zorder=5)
                ax.text(x, y, f"{CORNER_NAMES[d]} z={z:.1f}", fontsize=7,
                        color=HIGHLIGHT, clip_on=True)
            elif in_image:
                ax.scatter([x], [y], marker="x", s=40, color="0.55",
                           linewidth=0.6, alpha=0.35, zorder=4)

        ax.set_title(f"camera {cam}")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.grid(True, linewidth=0.3, alpha=0.35)

    for idx in range(num_cam, rows * cols):
        ax = fig.add_subplot(inner[idx // cols, idx % cols])
        ax.axis("off")

    if axes:
        axes[0].text(0.0, -0.18, "q* shows 4 corner-height pairs; D points repeat this staircase.",
                     transform=axes[0].transAxes, fontsize=8)


def _plot_camera_image_overlays(fig, outer_spec, camera_images, reference_points_cam,
                                bev_mask, data, q_star, P, T, show_all_camera_points=False):
    num_cam, H, W, _ = camera_images.shape
    cols = int(math.ceil(math.sqrt(num_cam)))
    rows = int(math.ceil(num_cam / cols))
    inner = outer_spec.subgridspec(rows, cols, wspace=0.12, hspace=0.22)

    camera_order = _text_array(data["camera_order"]) if "camera_order" in data else [
        f"camera {idx}" for idx in range(num_cam)]
    npp = int(_scalar(data, "npp", 4))
    pc_range = np.asarray(_scalar(data, "pc_range"))
    lidar_height = float(_scalar(data, "lidar_height", 0.0))
    if pc_range.shape[0] >= 6:
        heights = np.linspace(pc_range[2] - lidar_height, pc_range[5] - lidar_height, npp)
    else:
        heights = np.arange(npp)

    p_star, t_star = divmod(q_star, T)
    proposal_slice = slice(p_star * T, (p_star + 1) * T)

    for cam in range(num_cam):
        ax = fig.add_subplot(inner[cam // cols, cam % cols])
        ax.imshow(camera_images[cam])

        pts = reference_points_cam[cam]
        mask = bev_mask[cam].astype(bool)
        pixel_pts = pts * np.array([W, H], dtype=np.float32)
        if show_all_camera_points:
            all_valid_mask = mask & _points_in_unit_image(pts)
            all_valid = pts[all_valid_mask]
            if all_valid.size:
                ax.scatter(all_valid[:, 0] * W, all_valid[:, 1] * H,
                           s=2, color="#1f77b4", alpha=0.18, linewidths=0)

        selected_pts = pts[proposal_slice]
        selected_mask = mask[proposal_slice]
        selected_valid_mask = selected_mask & _points_in_unit_image(selected_pts)
        selected_valid = selected_pts[selected_valid_mask]
        if selected_valid.size:
            ax.scatter(selected_valid[:, 0] * W, selected_valid[:, 1] * H,
                       s=14, color=SELECTED, alpha=0.75, linewidths=0)

        q_pts = pts[q_star]
        q_pixel_pts = pixel_pts[q_star]
        q_mask = mask[q_star]
        for d in range(min(4, q_pts.shape[0])):
            x, y = q_pixel_pts[d]
            in_image = bool(_points_in_unit_image(q_pts[d])) and bool(
                _points_in_pixel_image(q_pixel_pts[d], W, H))
            if q_mask[d] and in_image:
                ax.scatter([x], [y], marker="*", s=150, color=HIGHLIGHT,
                           edgecolor="black", linewidth=0.5, alpha=1.0, zorder=6)
                ax.text(x, y, f"{CORNER_NAMES[d]} z={heights[d % len(heights)]:.1f}",
                        fontsize=7, color=HIGHLIGHT, zorder=7, clip_on=True)
            elif in_image:
                ax.scatter([x], [y], marker="x", s=45, color="0.55",
                           linewidth=0.6, alpha=0.35, zorder=5)

        title = camera_order[cam] if cam < len(camera_order) else f"camera {cam}"
        ax.set_title(f"{title}: selected proposal p={p_star}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(num_cam, rows * cols):
        ax = fig.add_subplot(inner[idx // cols, idx % cols])
        ax.axis("off")


def render(dump_path, query=None, out_path=None, show_all_camera_points=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(dump_path, allow_pickle=False)
    ref_2d = data["ref_2d"]
    ref_pos = data["ref_pos"]
    corners = data["corners"]
    reference_points_cam = data["reference_points_cam"]
    bev_mask = data["bev_mask"]

    Q = int(_scalar(data, "Q", ref_2d.shape[0]))
    P = int(_scalar(data, "proposal_num"))
    T = int(_scalar(data, "num_poses"))
    D = int(_scalar(data, "D", reference_points_cam.shape[2]))
    num_cam = int(_scalar(data, "num_cam", reference_points_cam.shape[0]))
    if P * T != Q:
        raise ValueError(f"proposal_num * num_poses != Q: {P} * {T} != {Q}")

    if query is None:
        query = _default_query(Q, P, T)
    if query < 0 or query >= Q:
        raise ValueError(f"--query must be in [0, {Q - 1}], got {query}")

    assert ref_2d.shape == (Q, 3)
    assert ref_pos.shape == (Q, 2)
    assert corners.shape == (Q, 4, 2)
    assert reference_points_cam.shape == (num_cam, Q, D, 2)
    assert bev_mask.shape == (num_cam, Q, D)

    if out_path is None:
        out_path = dump_path.rsplit(".", 1)[0] + "_viz.png"

    if "camera_images" in data:
        camera_images = data["camera_images"]
        assert camera_images.shape[0] == num_cam and camera_images.shape[-1] == 3
        color_order = _text_array(data["camera_image_color_order"]) if "camera_image_color_order" in data else "RGB"
        if color_order == "BGR":
            camera_images = camera_images[..., ::-1]
        fig = plt.figure(figsize=(22, 13), constrained_layout=True)
        outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
        _plot_camera_image_overlays(
            fig, outer[0, 0], camera_images, reference_points_cam, bev_mask,
            data, query, P, T, show_all_camera_points=show_all_camera_points)
        bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.22)
        ax_bev = fig.add_subplot(bottom[0, 0])
        _plot_metric_bev(plt, ax_bev, ref_2d, corners, P, T, query, data)
        _plot_temporal(fig, bottom[0, 1], ref_pos, P, T, query)
    else:
        fig = plt.figure(figsize=(21, 7), constrained_layout=True)
        outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.3])
        ax_bev = fig.add_subplot(outer[0, 0])
        _plot_metric_bev(plt, ax_bev, ref_2d, corners, P, T, query, data)
        _plot_temporal(fig, outer[0, 1], ref_pos, P, T, query)
        _plot_cameras(fig, outer[0, 2], reference_points_cam, bev_mask, data, query)

    fig.suptitle(str(dump_path), fontsize=10)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Render RAP ref_2d projection-chain debug dump.")
    parser.add_argument("dump", help="Path to ref2d debug .npz")
    parser.add_argument("--query", type=int, default=None, help="Query index to highlight")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--show-all-camera-points", action="store_true",
                        help="Also draw all valid camera projection points on RGB images")
    args = parser.parse_args()

    out_path = render(
        args.dump,
        query=args.query,
        out_path=args.out,
        show_all_camera_points=args.show_all_camera_points,
    )
    print(out_path)


if __name__ == "__main__":
    main()
