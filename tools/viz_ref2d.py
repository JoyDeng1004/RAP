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


def _first_present(data, keys):
    for key in keys:
        if key in data:
            return key, data[key]
    return None, None


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


def _derive_ref_2d_sca(data, ref_2d):
    key, value = _first_present(data, ("ref_2d_sca", "ref_2d_shifted", "ref_2d_post"))
    if value is not None:
        ref_2d_sca = np.asarray(value)
        if ref_2d_sca.ndim == 3 and ref_2d_sca.shape[0] == 1:
            ref_2d_sca = ref_2d_sca[0]
        return ref_2d_sca, key

    if "shift_y" not in data:
        return None, None

    ref_2d_sca = np.array(ref_2d, copy=True)
    ref_2d_sca[:, 1] += float(_scalar(data, "shift_y", 0.0))
    return ref_2d_sca, "derived from shift_y"


def _style_metric_bev_axes(ax, data, title, show_labels=True):
    pc_range = np.asarray(_scalar(data, "pc_range", np.array([-32, -32, -2, 32, 32, 6])))
    if pc_range.shape[0] >= 6:
        ax.set_xlim(float(pc_range[0]), float(pc_range[3]))
        ax.set_ylim(float(pc_range[1]), float(pc_range[4]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    if show_labels:
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y left (m)")
    ax.grid(True, linewidth=0.35, alpha=0.45)


def _coverage_per_query(reference_points_cam, bev_mask):
    valid = bev_mask.astype(bool) & _points_in_unit_image(reference_points_cam)
    return valid.sum(axis=(0, 2))


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

    _style_metric_bev_axes(ax, data, f"metric BEV overlay (q*={q_star}, p={p_star}, t={t_star})")
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


def _c6_present(data):
    return "sampling_locations_baseline" in data or "sampling_locations_shifted" in data


def _camera_display_order(data, num_cam):
    """Return (order, names): camera column order front-centric (left, front, right, back)."""
    names = list(_text_array(data["camera_order"])) if "camera_order" in data else [
        f"camera {i}" for i in range(num_cam)]
    preferred = ["cam_l0", "cam_f0", "cam_r0", "cam_b0"]
    order = []
    for pref in preferred:
        for i in range(num_cam):
            if i < len(names) and names[i] == pref and i not in order:
                order.append(i)
    for i in range(num_cam):
        if i not in order:
            order.append(i)
    return order, names


def _gather_sampling(data, prefix, cam, q_star):
    """Return (pixel-free normalized points (N, 2), alpha weights (N,)) for one cam/query."""
    key = f"sampling_locations_{prefix}"
    if key not in data:
        return None, None
    valid = data.get(f"sampling_valid_{prefix}")
    if valid is not None and not bool(np.asarray(valid)[cam, q_star]):
        return None, None
    locs = np.asarray(data[key])[cam, q_star].reshape(-1, 2)  # (heads*all_pts, 2)
    weights_key = f"attention_weights_{prefix}"
    if weights_key in data:
        weights = np.asarray(data[weights_key])[cam, q_star].reshape(-1)
    else:
        weights = np.ones(locs.shape[0], dtype=np.float32)
    finite = np.isfinite(locs).all(axis=1) & np.isfinite(weights)
    return locs[finite], weights[finite]


def _plot_sampling_points_overlay(fig, outer_spec, camera_images, reference_points_cam,
                                  bev_mask, data, q_star):
    """C6 — deformable sampling points on real images, alpha-weighted by attention.

    Baseline points are blue, shifted points red. The reference_points_cam footprint
    for q* is drawn as an anchor star in each color.
    """
    num_cam, H, W, _ = camera_images.shape
    order, names = _camera_display_order(data, num_cam)
    inner = outer_spec.subgridspec(1, num_cam, wspace=0.06)
    camera_order = names
    scale = np.array([W, H], dtype=np.float32)

    ref_cam_base = data.get("reference_points_cam_baseline", reference_points_cam)
    ref_cam_shift = data.get("reference_points_cam_shifted")

    for col, cam in enumerate(order):
        ax = fig.add_subplot(inner[0, col])
        ax.imshow(camera_images[cam])

        n_total = 0
        for prefix, color in (("baseline", "#1f77b4"), ("shifted", HIGHLIGHT)):
            locs, weights = _gather_sampling(data, prefix, cam, q_star)
            if locs is None or locs.size == 0:
                continue
            pix = locs * scale
            in_img = _points_in_pixel_image_arr(pix, W, H)
            pix, w = pix[in_img], weights[in_img]
            if pix.size == 0:
                continue
            alpha = _normalize01(w)
            # Per-point alpha encoded via RGBA (array alpha= isn't supported on old mpl).
            from matplotlib.colors import to_rgb
            rgb = to_rgb(color)
            rgba = np.empty((pix.shape[0], 4), dtype=np.float32)
            rgba[:, :3] = rgb
            rgba[:, 3] = np.clip(0.15 + 0.8 * alpha, 0.0, 1.0)
            ax.scatter(pix[:, 0], pix[:, 1], s=10 + 40 * alpha, c=rgba,
                       linewidths=0, zorder=4)
            n_total += pix.shape[0]

        # Anchor footprint (mean over D height samples that are valid) for q*.
        for ref_cam, mask_key, color in (
            (ref_cam_base, "bev_mask_baseline", "#1f77b4"),
            (ref_cam_shift, "bev_mask_shifted", HIGHLIGHT),
        ):
            if ref_cam is None:
                continue
            pts = np.asarray(ref_cam)[cam, q_star]  # (D, 2)
            mask = data.get(mask_key)
            keep = _points_in_unit_image(pts)
            if mask is not None:
                keep = keep & np.asarray(mask)[cam, q_star].astype(bool)
            pts = pts[keep]
            if pts.size == 0:
                continue
            anchor = pts.mean(axis=0) * scale
            ax.scatter([anchor[0]], [anchor[1]], marker="*", s=180, color=color,
                       edgecolor="black", linewidth=0.6, zorder=6)

        title = camera_order[cam] if cam < len(camera_order) else f"camera {cam}"
        ax.set_title(f"C6 {title}: q*={q_star} pts={n_total}", fontsize=9)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])


def _points_in_pixel_image_arr(pixel_points, width, height):
    x = pixel_points[:, 0]
    y = pixel_points[:, 1]
    return (x >= 0) & (x < width) & (y >= 0) & (y < height)


def _plot_ref2d_shift_overlay(plt, ax, ref_2d, ref_2d_sca, ref_2d_sca_source, P, T, q_star, data):
    traj = ref_2d.reshape(P, T, 3)
    traj_sca = ref_2d_sca.reshape(P, T, 3)
    p_star, t_star = divmod(q_star, T)

    for p in range(P):
        xy = traj[p, :, :2]
        xy_sca = traj_sca[p, :, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="0.68", linewidth=0.7, alpha=0.35)
        ax.plot(xy_sca[:, 0], xy_sca[:, 1], color="#1f77b4", linewidth=0.7, alpha=0.45)

    step = max(1, ref_2d.shape[0] // 80)
    delta = ref_2d_sca[:, :2] - ref_2d[:, :2]
    anchors = ref_2d[::step, :2]
    arrows = delta[::step]
    nonzero = np.linalg.norm(arrows, axis=1) > 1e-8
    if np.any(nonzero):
        ax.quiver(
            anchors[nonzero, 0],
            anchors[nonzero, 1],
            arrows[nonzero, 0],
            arrows[nonzero, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="#1f77b4",
            alpha=0.75,
        )

    ax.scatter(ref_2d[:, 0], ref_2d[:, 1], s=10, color="0.25", alpha=0.35, label="ref_2d")
    ax.scatter(ref_2d_sca[:, 0], ref_2d_sca[:, 1], s=10, color="#1f77b4", alpha=0.45,
               label=f"ref_2d_sca ({ref_2d_sca_source})")

    ax.plot(traj[p_star, :, 0], traj[p_star, :, 1], color=HIGHLIGHT, linewidth=2.0)
    ax.plot(traj_sca[p_star, :, 0], traj_sca[p_star, :, 1], color=SELECTED, linewidth=2.0)
    ax.scatter([ref_2d[q_star, 0]], [ref_2d[q_star, 1]], marker="*", s=160,
               color=HIGHLIGHT, edgecolor="black", linewidth=0.6, zorder=5)
    ax.scatter([ref_2d_sca[q_star, 0]], [ref_2d_sca[q_star, 1]], marker="*", s=150,
               color=SELECTED, edgecolor="black", linewidth=0.6, zorder=5)

    shift_y = float(np.median(delta[:, 1])) if delta.size else 0.0
    _style_metric_bev_axes(ax, data, f"B3 ref_2d vs SCA shift, dy={shift_y:.3g}")
    ax.legend(loc="best", fontsize=8)


def _camera_debug_pair(data, reference_points_cam, bev_mask):
    base_key, base_cam = _first_present(
        data,
        (
            "reference_points_cam_baseline",
            "reference_points_cam_pre",
            "baseline_reference_points_cam",
        ),
    )
    mask_key, base_mask = _first_present(
        data,
        (
            "bev_mask_baseline",
            "bev_mask_pre",
            "baseline_bev_mask",
        ),
    )
    shift_key, shift_cam = _first_present(
        data,
        (
            "reference_points_cam_shifted",
            "reference_points_cam_post",
            "shifted_reference_points_cam",
        ),
    )
    shifted_mask_key, shifted_mask = _first_present(
        data,
        (
            "bev_mask_shifted",
            "bev_mask_post",
            "shifted_bev_mask",
        ),
    )
    if shift_cam is None:
        shift_cam = reference_points_cam
        shift_key = "reference_points_cam"
    if shifted_mask is None:
        shifted_mask = bev_mask
        shifted_mask_key = "bev_mask"
    if base_cam is None or base_mask is None:
        return None, None, shift_cam, shifted_mask, base_key, mask_key, shift_key, shifted_mask_key
    return base_cam, base_mask, shift_cam, shifted_mask, base_key, mask_key, shift_key, shifted_mask_key


def _plot_camera_shift_overlay(fig, outer_spec, reference_points_cam, bev_mask, data, q_star):
    (
        base_cam,
        base_mask,
        shift_cam,
        shift_mask,
        base_key,
        mask_key,
        shift_key,
        shifted_mask_key,
    ) = _camera_debug_pair(data, reference_points_cam, bev_mask)

    if base_cam is None:
        ax = fig.add_subplot(outer_spec)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "B4 camera baseline-vs-shift overlay\nmissing baseline camera fields",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title("B4 camera footprint shift")
        return

    num_cam, _, D, _ = shift_cam.shape
    order, names = _camera_display_order(data, num_cam)
    inner = outer_spec.subgridspec(1, num_cam, wspace=0.06)

    camera_images = data["camera_images"] if "camera_images" in data else None
    if camera_images is not None:
        color_order = _text_array(data["camera_image_color_order"]) if "camera_image_color_order" in data else "RGB"
        if color_order == "BGR":
            camera_images = camera_images[..., ::-1]
        _, height, width, _ = camera_images.shape
    else:
        height, width = 1.0, 1.0

    for col, cam in enumerate(order):
        ax = fig.add_subplot(inner[0, col])
        if camera_images is not None:
            ax.imshow(camera_images[cam])

        base_pts = base_cam[cam, q_star]
        shift_pts = shift_cam[cam, q_star]
        base_valid = base_mask[cam, q_star].astype(bool) & _points_in_unit_image(base_pts)
        shift_valid = shift_mask[cam, q_star].astype(bool) & _points_in_unit_image(shift_pts)

        base_draw = base_pts * np.array([width, height], dtype=np.float32)
        shift_draw = shift_pts * np.array([width, height], dtype=np.float32)
        for d in range(D):
            if base_valid[d] and shift_valid[d]:
                ax.plot(
                    [base_draw[d, 0], shift_draw[d, 0]],
                    [base_draw[d, 1], shift_draw[d, 1]],
                    color=SELECTED,
                    linewidth=0.8,
                    alpha=0.8,
                )

        if np.any(base_valid):
            ax.scatter(base_draw[base_valid, 0], base_draw[base_valid, 1], s=30,
                       color="#1f77b4", alpha=0.85, label="baseline")
        if np.any(shift_valid):
            ax.scatter(shift_draw[shift_valid, 0], shift_draw[shift_valid, 1], s=30,
                       color=HIGHLIGHT, alpha=0.9, label="shifted")

        name = names[cam] if cam < len(names) else f"camera {cam}"
        ax.set_title(f"B4 {name}", fontsize=9)
        if camera_images is not None:
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.set_xlabel("u")
            ax.set_ylabel("v")
            ax.grid(True, linewidth=0.3, alpha=0.35)
        if col == 0 and (np.any(base_valid) or np.any(shift_valid)):
            ax.legend(loc="upper right", fontsize=7)


def _plot_bev_mask_coverage(fig, outer_spec, reference_points_cam, bev_mask, data, P, T, grid=True):
    base_cam, base_mask, shift_cam, shift_mask, *_ = _camera_debug_pair(data, reference_points_cam, bev_mask)
    if grid:
        inner = outer_spec.subgridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.28)
        ax_bev = fig.add_subplot(inner[0, 0])
        ax_grid = fig.add_subplot(inner[0, 1])
    else:
        ax_bev = fig.add_subplot(outer_spec)
        ax_grid = None

    ref_2d = data["ref_2d"]
    current_count = _coverage_per_query(shift_cam, shift_mask)
    if base_cam is not None:
        baseline_count = _coverage_per_query(base_cam, base_mask)
        color_values = current_count - baseline_count
        title = "B5 BEV coverage delta"
        cmap = "coolwarm"
        vmax = max(float(np.max(np.abs(color_values))), 1.0)
        vmin = -vmax
        grid_values = color_values.reshape(P, T)
    else:
        color_values = current_count
        title = "B5 BEV coverage count"
        cmap = "viridis"
        vmin = 0.0
        vmax = max(float(np.max(color_values)), 1.0)
        grid_values = current_count.reshape(P, T)

    scatter = ax_bev.scatter(ref_2d[:, 0], ref_2d[:, 1], c=color_values, cmap=cmap,
                             vmin=vmin, vmax=vmax, s=16, linewidths=0)
    _style_metric_bev_axes(ax_bev, data, title)
    fig.colorbar(scatter, ax=ax_bev, fraction=0.046, pad=0.03)

    if ax_grid is not None:
        image = ax_grid.imshow(grid_values, aspect="auto", interpolation="nearest",
                               cmap=cmap, vmin=vmin, vmax=vmax)
        ax_grid.set_title("P x T coverage")
        ax_grid.set_xlabel("t")
        ax_grid.set_ylabel("proposal p")
        fig.colorbar(image, ax=ax_grid, fraction=0.046, pad=0.03)


def _plot_shift_geometry_checks(fig, outer_spec, plt, ref_2d, reference_points_cam,
                                bev_mask, data, P, T, q_star):
    ref_2d_sca, ref_2d_sca_source = _derive_ref_2d_sca(data, ref_2d)
    grid = outer_spec.subgridspec(1, 3, width_ratios=[1.15, 1.2, 1.2], wspace=0.25)

    ax_shift = fig.add_subplot(grid[0, 0])
    if ref_2d_sca is None:
        ax_shift.axis("off")
        ax_shift.text(0.5, 0.5, "B3 requires ref_2d_sca or shift_y", ha="center", va="center")
        ax_shift.set_title("B3 ref_2d shift")
    else:
        _plot_ref2d_shift_overlay(plt, ax_shift, ref_2d, ref_2d_sca, ref_2d_sca_source, P, T, q_star, data)

    _plot_camera_shift_overlay(fig, grid[0, 1], reference_points_cam, bev_mask, data, q_star)
    _plot_bev_mask_coverage(fig, grid[0, 2], reference_points_cam, bev_mask, data, P, T)


def _has_shift_geometry_checks(data):
    return any(
        key in data
        for key in (
            "ref_2d_sca",
            "ref_2d_shifted",
            "ref_2d_post",
            "shift_y",
            "reference_points_cam_baseline",
            "reference_points_cam_pre",
            "baseline_reference_points_cam",
            "bev_mask_baseline",
            "bev_mask_pre",
            "baseline_bev_mask",
        )
    )


def _normalize01(values):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _nearest_shift_indices(ref_xy, shift_y):
    try:
        from sklearn.neighbors import NearestNeighbors

        target_xy = ref_xy + np.array([0.0, shift_y], dtype=np.float32)
        return NearestNeighbors(n_neighbors=1).fit(ref_xy).kneighbors(target_xy, return_distance=False)[:, 0]
    except Exception:
        target_xy = ref_xy + np.array([0.0, shift_y], dtype=np.float32)
        distances = np.linalg.norm(target_xy[:, None, :] - ref_xy[None, :, :], axis=-1)
        return np.argmin(distances, axis=1)


def _feature_pca_rgb(feature_pre, feature_post):
    features = np.concatenate([feature_pre, feature_post], axis=0)
    try:
        from sklearn.decomposition import PCA

        components = PCA(n_components=3).fit_transform(features)
    except Exception:
        centered = features - features.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        n_components = min(3, vt.shape[0])
        components = centered @ vt[:n_components].T
        if n_components < 3:
            pad = np.zeros((components.shape[0], 3 - n_components), dtype=components.dtype)
            components = np.concatenate([components, pad], axis=1)
    rgb = np.stack([_normalize01(components[:, idx]) for idx in range(3)], axis=-1)
    return rgb[: feature_pre.shape[0]], rgb[feature_pre.shape[0] :]


def _plot_bev_feature_delta(fig, outer_spec, ref_2d, feature_pre, feature_post, shift_y, data):
    from matplotlib.colors import TwoSlopeNorm

    ref_xy = ref_2d[:, :2]
    delta = feature_post - feature_pre
    delta_norm = np.linalg.norm(delta, axis=1)
    delta_color = delta_norm / max(float(np.max(delta_norm)), 1e-12)

    pre_norm = np.linalg.norm(feature_pre, axis=1)
    post_norm = np.linalg.norm(feature_post, axis=1)
    cosine = np.sum(feature_pre * feature_post, axis=1) / np.maximum(pre_norm * post_norm, 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)

    pre_rgb, post_rgb = _feature_pca_rgb(feature_pre, feature_post)

    shifted_indices = _nearest_shift_indices(ref_xy, shift_y)
    shift_residual = np.linalg.norm(feature_post - feature_pre[shifted_indices], axis=1)
    shift_residual_color = shift_residual / max(float(np.max(shift_residual)), 1e-12)

    grid = outer_spec.subgridspec(1, 4, wspace=0.22)
    ax_delta = fig.add_subplot(grid[0, 0])
    ax_cos = fig.add_subplot(grid[0, 1])
    pca_grid = grid[0, 2].subgridspec(1, 2, wspace=0.08)
    ax_pca_pre = fig.add_subplot(pca_grid[0, 0])
    ax_pca_post = fig.add_subplot(pca_grid[0, 1])
    ax_shift = fig.add_subplot(grid[0, 3])

    scatter = ax_delta.scatter(ref_xy[:, 0], ref_xy[:, 1], c=delta_color, cmap="magma",
                               vmin=0.0, vmax=1.0, s=14, linewidths=0)
    _style_metric_bev_axes(ax_delta, data, "V1 delta-norm")
    fig.colorbar(scatter, ax=ax_delta, fraction=0.046, pad=0.03)

    cos_norm = TwoSlopeNorm(vmin=-1.0, vcenter=1.0, vmax=1.000001)
    scatter = ax_cos.scatter(ref_xy[:, 0], ref_xy[:, 1], c=cosine, cmap="coolwarm",
                             norm=cos_norm, s=14, linewidths=0)
    _style_metric_bev_axes(ax_cos, data, "V2 cosine")
    fig.colorbar(scatter, ax=ax_cos, fraction=0.046, pad=0.03)

    ax_pca_pre.scatter(ref_xy[:, 0], ref_xy[:, 1], c=pre_rgb, s=14, linewidths=0)
    _style_metric_bev_axes(ax_pca_pre, data, "V3 PCA pre", show_labels=False)
    ax_pca_post.scatter(ref_xy[:, 0], ref_xy[:, 1], c=post_rgb, s=14, linewidths=0)
    _style_metric_bev_axes(ax_pca_post, data, "V3 PCA post", show_labels=False)
    ax_pca_post.set_yticklabels([])

    scatter = ax_shift.scatter(ref_xy[:, 0], ref_xy[:, 1], c=shift_residual_color, cmap="viridis",
                               vmin=0.0, vmax=1.0, s=14, linewidths=0)
    _style_metric_bev_axes(ax_shift, data, f"V4 shift residual y={shift_y:.3g}")
    fig.colorbar(scatter, ax=ax_shift, fraction=0.046, pad=0.03)


def render(dump_path, query=None, out_path=None, show_all_camera_points=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(dump_path, allow_pickle=False)
    ref_2d = data["ref_2d"]
    ref_pos = data["ref_pos"]
    corners = data["corners"]

    if "bev_feature_pre" in data and "bev_feature_post" in data:
        feature_pre = data["bev_feature_pre"]
        feature_post = data["bev_feature_post"]
        shift_y = float(_scalar(data, "shift_y", 0.0))
        Q = int(_scalar(data, "Q", ref_2d.shape[0]))
        P = int(_scalar(data, "proposal_num"))
        T = int(_scalar(data, "num_poses"))
        if P * T != Q:
            raise ValueError(f"proposal_num * num_poses != Q: {P} * {T} != {Q}")
        assert ref_2d.shape == (Q, 3)
        assert ref_pos.shape == (Q, 2)
        assert corners.shape == (Q, 4, 2)
        assert feature_pre.shape == feature_post.shape
        assert feature_pre.shape[0] == Q

        if out_path is None:
            out_path = dump_path.rsplit(".", 1)[0] + "_bev_feature_viz.png"
        fig = plt.figure(figsize=(24, 6), constrained_layout=True)
        outer = fig.add_gridspec(1, 1)
        _plot_bev_feature_delta(fig, outer[0, 0], ref_2d, feature_pre, feature_post, shift_y, data)
        fig.suptitle(str(dump_path), fontsize=10)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return out_path

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

    has_shift_checks = _has_shift_geometry_checks(data)
    if "camera_images" in data and has_shift_checks:
        # Clean stacked layout for real-sample shift dumps (tools/dump_ref2d_real.py):
        #   Row 1: B4 - baseline vs shifted reference_points_cam footprints on real images (1xN)
        #   Row 2: C6 - deformable sampling points on real images (1xN, only if present)
        #   Row 3: BEV diagnostics - metric BEV | B3 ref_2d shift | B5 coverage
        camera_images = data["camera_images"]
        assert camera_images.shape[0] == num_cam and camera_images.shape[-1] == 3
        has_c6 = _c6_present(data)
        # Camera rows are wide-and-short (1xN); the diagnostics row is taller.
        cam_aspect = float(camera_images.shape[1]) / float(camera_images.shape[2])  # H/W per cam
        cam_row_h = max(0.45, cam_aspect * (24.0 / max(num_cam, 1)) / 6.4)
        height_ratios = [cam_row_h]
        if has_c6:
            height_ratios.append(cam_row_h)
        height_ratios.append(1.0)
        fig = plt.figure(figsize=(24, 6.4 * sum(height_ratios)), constrained_layout=True)
        outer = fig.add_gridspec(len(height_ratios), 1, height_ratios=height_ratios)

        row = 0
        _plot_camera_shift_overlay(fig, outer[row, 0], reference_points_cam, bev_mask, data, query)
        row += 1
        if has_c6:
            _plot_sampling_points_overlay(
                fig, outer[row, 0], camera_images, reference_points_cam, bev_mask, data, query)
            row += 1

        diag = outer[row, 0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.26)
        ax_bev = fig.add_subplot(diag[0, 0])
        _plot_metric_bev(plt, ax_bev, ref_2d, corners, P, T, query, data)
        ax_shift = fig.add_subplot(diag[0, 1])
        ref_2d_sca, ref_2d_sca_source = _derive_ref_2d_sca(data, ref_2d)
        if ref_2d_sca is not None:
            _plot_ref2d_shift_overlay(plt, ax_shift, ref_2d, ref_2d_sca, ref_2d_sca_source, P, T, query, data)
        else:
            ax_shift.axis("off")
            ax_shift.set_title("B3 ref_2d shift (needs ref_2d_sca/shift_y)")
        _plot_bev_mask_coverage(fig, diag[0, 2], reference_points_cam, bev_mask, data, P, T, grid=False)
    elif "camera_images" in data:
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
        if has_shift_checks:
            fig = plt.figure(figsize=(24, 14), constrained_layout=True)
            outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05])
            top = outer[0, 0].subgridspec(1, 3, width_ratios=[1.0, 1.35, 1.3])
        else:
            fig = plt.figure(figsize=(21, 7), constrained_layout=True)
            outer = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.3])
            top = outer
        ax_bev = fig.add_subplot(top[0, 0])
        _plot_metric_bev(plt, ax_bev, ref_2d, corners, P, T, query, data)
        _plot_temporal(fig, top[0, 1], ref_pos, P, T, query)
        _plot_cameras(fig, top[0, 2], reference_points_cam, bev_mask, data, query)
        if has_shift_checks:
            _plot_shift_geometry_checks(
                fig, outer[1, 0], plt, ref_2d, reference_points_cam, bev_mask, data, P, T, query)

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
