import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import Optional
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

@dataclass
class CostmapParams:
    resolution: float          # [m / pixel]
    map_size_m: float          # 전체 맵 한 변 [m]
    robot_radius_m: float      # 로봇 반경 [m]
    inflation_radius_m: float  # 인플레이션 반경 [m]
    max_range_m: float         # 라이다 최대 거리 [m]

def get_costmap_gray_cmap():
    """
    MATLAB imshow-style grayscale for int8 costmap:
    -1 (unknown) -> mid gray
    0 (free)     -> black
    1~99         -> increasing gray
    100          -> white
    """
    # 0..255 colormap
    colors = np.zeros((101, 3), dtype=float)
    # unknown (-1) -> ~128 (중간 회색)
    colors[0] = [0.5, 0.5, 0.5]
    # free (0) -> black
    colors[1] = [0.0, 0.0, 0.0]
    # decay(1~99) -> dark gray → bright gray
    for i in range(2, 101):
        # 2..100
        t = (i - 2) / 100.0         # normalize 0..1
        g = t                      # keep linear straight mapping
        colors[i] = [g, g, g]
    colors[102:] = 1.0
    return ListedColormap(colors)

def get_depth_colormap():
    """
    Red (close) → Yellow (mid) → Blue (far)
    """
    colors = [
        (1.0, 0.0, 0.0),   # red
        (1.0, 1.0, 0.0),   # yellow
        #(1.0, 0.5, 0.0),   # orange
        (0.6, 1.0, 0.6),   # light-green
        (0.0, 0.7, 0.0),   # green
        (0.0, 1.0, 1.0),   # cyan
        (0.0, 0.2, 1.0),   # blue
    ]
    return LinearSegmentedColormap.from_list("depth_map_r_y_b", colors, N=256)

class Visualizer:
    """
    - build_costmap: range/angle → costmap (-1,0~100)
    - draw_pose_seq / draw_poses_2d: 포즈 시퀀스 시각화
    - draw: RGB/Depth/costmap 디버그 figure 생성
    """

    def __init__(self, cm_params: CostmapParams):
        self.cm_params = cm_params
        self.resolution = cm_params.resolution
        self.max_depth = 3.6   # max reliable depth
        self.min_depth = 0.05
    # ============================================================
    # 1) COSTMAP 관련 내부 메서드들
    # ============================================================



    def bresenham(self, y0: int, x0: int, y1: int, x1: int):
        """
        Bresenham line algorithm (0-based index).
        Returns (rows, cols) as numpy arrays.
        """
        points = []

        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while True:
            points.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

        rr = np.array([p[0] for p in points], dtype=int)
        cc = np.array([p[1] for p in points], dtype=int)
        return rr, cc

    def remap_costmap(self, costmap_i8: np.ndarray) -> np.ndarray:
        """
        MATLAB remap_costmap(final_costmap)에 대응.
        필요하면 이 로직을 MATLAB 버전에 맞게 수정해서 사용.
        """
        cm = costmap_i8.astype(np.int16)

        # shift so that unknown(-1) -> 0
        cm_shifted = cm + 1  # -1 -> 0, 0 -> 1, 100 -> 101
        cm_shifted = np.clip(cm_shifted, 0, 101)

        # scale 0..101 → 0..255
        cm_u8 = (cm_shifted.astype(np.float32) * (255.0 / 101.0)).astype(np.uint8)
        return cm_u8

    def build_costmap(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
    ):
        """
        Python version of MATLAB build_costmap.m

        Parameters
        ----------
        ranges : np.ndarray
            1D array of lidar ranges [m]
        angles : np.ndarray
            1D array of angles [rad]

        Returns
        -------
        costmap_i8 : np.ndarray (int8)
            raw costmap: -1(unknown), 0~100 (free~obstacle)
        costmap_u8 : np.ndarray (uint8)
            remapped costmap (0~255), for visualization or further use
        """
        p = self.cm_params
        resolution = p.resolution
        map_size_m = p.map_size_m
        robot_radius_m = p.robot_radius_m
        inflation_radius_m = p.inflation_radius_m
        max_range_m = p.max_range_m

        map_size_px = int(round(map_size_m / resolution))

        U = -1   # unknown
        O = 100  # obstacle cost

        roi_front_m = 1.0
        assert roi_front_m < map_size_m
        roi_back_m = 0.25
        assert roi_back_m <= roi_front_m

        robot_radius_px = int(round(robot_radius_m / resolution))
        inflation_radius_px = int(round(inflation_radius_m / resolution))

        half_map = map_size_px / 2.0  # 중앙 기준 (float)

        # === Convert scan to local frame ===
        angles = -angles  # MATLAB: CCW (right -> front -> left)
        x_local = ranges * np.cos(angles)
        y_local = ranges * np.sin(angles)

        # --- NaN / inf 제거 (여기서 먼저 필터링) ---
        finite_mask = np.isfinite(x_local) & np.isfinite(y_local)
        x_local_f = x_local[finite_mask]
        y_local_f = y_local[finite_mask]

        # === Map scan to grid indices (MATLAB 1-based) → Python 0-based ===
        ix = np.round(x_local_f / resolution + half_map).astype(int)
        iy = np.round(y_local_f / resolution + half_map).astype(int)

        # Initialize costmap with -1 (unknown)
        final_costmap = np.full((map_size_px, map_size_px), U, dtype=np.int8)

        # Valid indices in [1, map_size_px] for MATLAB → [0, map_size_px-1] in Python
        valid = (ix >= 1) & (ix <= map_size_px) & (iy >= 1) & (iy <= map_size_px)
        ix = ix[valid] - 1
        iy = iy[valid] - 1

        # Mark obstacles with 100
        final_costmap[iy, ix] = O

        # === Raycasting for all beams ===
        # 중심 좌표 (0-based)
        x0 = int(round(half_map)) - 1  # MATLAB의 round(half_map)를 1 줄임
        y0 = int(round(half_map)) - 1

        for k in range(len(angles)):
            theta = angles[k]

            # End of beam: actual hit or max range
            if (not np.isnan(ranges[k])) and (ranges[k] > 0) and (ranges[k] < max_range_m):
                r = ranges[k]
            else:
                r = max_range_m

            # local → grid
            x_local_k = r * np.cos(theta)
            y_local_k = r * np.sin(theta)

            x1 = int(round(x_local_k / resolution + half_map)) - 1
            y1 = int(round(y_local_k / resolution + half_map)) - 1

            # clamp to grid
            x1 = np.clip(x1, 0, map_size_px - 1)
            y1 = np.clip(y1, 0, map_size_px - 1)

            # Raytrace to mark free space
            rr, cc = self.bresenham(y0, x0, y1, x1)
            rr = np.clip(rr, 0, map_size_px - 1)
            cc = np.clip(cc, 0, map_size_px - 1)

            # exclude last if obstacle (MATLAB: 1:length(rr)-1)
            for n in range(len(rr) - 1):
                if final_costmap[rr[n], cc[n]] == U:
                    final_costmap[rr[n], cc[n]] = 0

            # Mark obstacle at the end (if range is valid)
            if (ranges[k] > 0) and (ranges[k] < max_range_m):
                final_costmap[y1, x1] = O

        # === Obstacle Inflation with Decay ===
        obstacle_mask = (final_costmap == O)
        decayed_costmap = np.zeros_like(final_costmap, dtype=np.uint8)

        for r in range(1, inflation_radius_px + 1):
            # disk structuring element
            y, x = np.ogrid[-r:r + 1, -r:r + 1]
            se_ring = (x ** 2 + y ** 2) <= r ** 2

            ring = ndimage.binary_dilation(obstacle_mask, structure=se_ring)

            # Decay value 100 -> 0
            decay_val = np.uint8(100 - round((r / float(inflation_radius_px)) * 100))

            mask = ring & (final_costmap != O)
            # take max decay if overlapping
            decayed_costmap[mask] = np.maximum(decayed_costmap[mask], decay_val)

        # === Merge decay onto free space ===
        free_mask = (final_costmap == 0)
        merged = final_costmap.astype(np.int16)
        merged[free_mask] = np.maximum(
            merged[free_mask],
            decayed_costmap[free_mask].astype(np.int16),
        )
        final_costmap = merged.astype(np.int8)

        costmap_i8 = final_costmap
        costmap_u8 = self.remap_costmap(final_costmap)

        return costmap_i8, costmap_u8

    # ============================================================
    # 2) 포즈 시퀀스 시각화 메서드들
    # ============================================================

    def draw_pose_seq(
        self,
        pts_xyzq_corrected_px: np.ndarray,   # [N, 7] (x,y,z,qw,qx,qy,qz)
        rx: float,
        ry: float,
        vstep: int,
        pose_marker: str = "o",
        fcolor: str = "b",
        scale: float = 8.0,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Python equivalent of MATLAB drawPoseSeq()

        pts_xyzq_corrected_px : [N,7] in pixel coordinates
        rx, ry : robot origin (px)
        vstep  : sampling interval
        """
        if ax is None:
            ax = plt.gca()

        pts = pts_xyzq_corrected_px
        if pts.ndim != 2 or pts.shape[1] not in (4, 7):
            raise ValueError(
                f"Unexpected pts_xyzq_corrected_px shape {pts.shape}, "
                "expected (N,7) or (N,4)."
            )

        if pts.shape[1] == 7:
            # x, y, z, qw, qx, qy, qz
            x = pts[:, 0]
            y = pts[:, 1]
            qw = pts[:, 3]
            qx = pts[:, 4]
            qy = pts[:, 5]
            qz = pts[:, 6]
        else:  # (N, 4): x, y, qw, qz
            x = pts[:, 0]
            y = pts[:, 1]
            qw = pts[:, 2]
            qz = pts[:, 3]
            # 2D 회전이므로 x, y 축 회전은 0으로 가정
            qx = np.zeros_like(qw)
            qy = np.zeros_like(qw)

        # shift to map frame
        pts_px = np.stack([
            x + rx,
            y + ry
        ], axis=1)

        # yaw extraction
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        # orientation vectors
        dir_x = np.cos(yaw)
        dir_y = np.sin(yaw)

        # decimate (vstep)
        pts_px_s = pts_px[0::vstep]
        dir_x_s = dir_x[0::vstep]
        dir_y_s = dir_y[0::vstep]

        # arrow vectors
        dvx = scale * dir_x_s
        dvy = scale * dir_y_s

        # markers
        ax.plot(
            pts_px_s[:, 0],
            pts_px_s[:, 1],
            pose_marker,
            markersize=8,
            markerfacecolor=fcolor,
            markeredgecolor=fcolor,
        )

        # arrows
        ax.quiver(
            pts_px_s[:, 0],
            pts_px_s[:, 1],
            dvx,
            dvy,
            angles='xy',
            scale_units='xy',
            scale=1,
            linewidth=1,
            color='r',
            headwidth=8,
            headlength=10,
        )

    def draw_poses_2d(
        self,
        poses: np.ndarray,              # Nx7 [x y z qw qx qy qz] OR Nx3 [x y theta]
        arrow_len: float = 1.0,
        color: str = "m",
        linewidth: float = 1.0,
        ax: Optional[plt.Axes] = None,
    ):
        """
        Python version of MATLAB drawPoses2D

        poses: Nx7 [x y z qw qx qy qz]  OR  Nx3 [x y theta]
        """
        if ax is None:
            ax = plt.gca()

        if poses.shape[1] == 7:
            # quaternion -> yaw (ZYX)
            qw = poses[:, 3]
            qx = poses[:, 4]
            qy = poses[:, 5]
            qz = poses[:, 6]
            yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
            x = poses[:, 0]
            y = poses[:, 1]
        elif poses.shape[1] == 3:
            x = poses[:, 0]
            y = poses[:, 1]
            yaw = poses[:, 2]
        else:
            raise ValueError("poses must be Nx7 or Nx3")

        u = arrow_len * np.cos(yaw)
        v = arrow_len * np.sin(yaw)

        ax.quiver(
            x,
            y,
            u,
            v,
            angles='xy',
            scale_units='xy',
            scale=1,
            color=color,
            linewidth=linewidth,
            headwidth=8,
            headlength=10,
        )
        ax.plot(x, y, ".", color=color)

    # ============================================================
    # 3) 디버그 전체 figure 그리기
    # ============================================================

    def draw(
        self,
        str_data_info: str,
        rgb_img: np.ndarray,
        rgb_sg: np.ndarray,
        depth_img: np.ndarray,
        depth_sg: np.ndarray,
        costmap_i8: np.ndarray,
        sgs_xyzq_corrected_m: np.ndarray,
        pose_context_m: np.ndarray,
        waypt_label_m: np.ndarray,
        rx: float,
        ry: float,
        old_sg_m: np.ndarray,  #
        gt_sg_m:  np.ndarray,  #  GT target sg (to avoid obstacle)
        pred_waypoints_m: np.ndarray,
        pred_pose_diff_m: np.ndarray,
        pred_collprob: float,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Draws the debug figure (RGB/Depth grid + costmap overlay).
        """
        #tmp_data_id_list = str_data_info.split(' ')[1].split('/')[-2:]
        #data_id = "_".join(tmp_data_id_list)
        fig = plt.figure(figsize=(16, 7.5))
        fig.suptitle(f"DATA ID: {str_data_info}", fontsize=14, fontweight="bold")
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.2])

        ax_rgb      = fig.add_subplot(gs[0, 0])
        ax_rgb_sg   = fig.add_subplot(gs[0, 1])
        ax_depth    = fig.add_subplot(gs[1, 0])
        ax_depth_sg = fig.add_subplot(gs[1, 1])
        ax_costmap  = fig.add_subplot(gs[:, 2])

        # =========================
        # LEFT: 2×2 image grid
        # =========================
        ax_rgb.imshow(rgb_img)
        ax_rgb.set_title("Observed RGB [▶]", color="green", fontweight="bold")
        ax_rgb.axis("off")

        ax_rgb_sg.imshow(rgb_sg)
        ax_rgb_sg.set_title("SG (GT) RGB [⬟]", color="limegreen", fontweight="bold")
        ax_rgb_sg.axis("off")

        cmap = get_depth_colormap()
        im_depth = ax_depth.imshow(depth_img, cmap=cmap, vmin=self.min_depth, vmax=self.max_depth)
        ax_depth.set_title("Observed Depth")
        ax_depth.axis("off")
        fig.colorbar(im_depth, ax=ax_depth, fraction=0.046, pad=0.04)

        im_depth_sg = ax_depth_sg.imshow(depth_sg, cmap=cmap, vmin=self.min_depth, vmax=self.max_depth)
        ax_depth_sg.set_title("SG Depth")
        ax_depth_sg.axis("off")
        fig.colorbar(im_depth_sg, ax=ax_depth_sg, fraction=0.046, pad=0.04)

        # =========================
        # RIGHT: costmap + waypoints
        # =========================
        #print(costmap_i8.shape)
        cmap_cost = get_costmap_gray_cmap()

        if costmap_i8 is None:
            costmap_vis = np.zeros( (200,200), dtype=np.uint8 )
        else:
            costmap_vis = costmap_i8.astype(np.int16) + 1
            costmap_vis = np.clip(costmap_vis, 0, 255).astype(np.uint8)
        im_cost = ax_costmap.imshow(costmap_vis, cmap=cmap_cost, origin="lower")
        fig.colorbar(im_cost, ax=ax_costmap, fraction=0.046, pad=0.02)

        ax_costmap.set_title(
            f"Data "
        )

        # (a) SLAM SGs (drawPoseSeq)
        if sgs_xyzq_corrected_m is not None and sgs_xyzq_corrected_m.size > 0:
            sgs_xyzq_corrected_px = sgs_xyzq_corrected_m.copy()
            sgs_xyzq_corrected_px[:, 0:2] = sgs_xyzq_corrected_px[:, 0:2] / self.resolution
            self.draw_pose_seq(
                sgs_xyzq_corrected_px,
                rx,
                ry,
                vstep=2,
                pose_marker="o",
                fcolor="b",
                ax=ax_costmap,
            )

        # (b) Pose context (prev robot poses)
        # if pose_context_m is not None and pose_context_m.size > 0:
        #     pose_context_px = pose_context_m.copy()
        #     pose_context_px[0] /= self.resolution
        #     pose_context_px[1] /= self.resolution
        #     self.draw_pose_seq(
        #         pose_context_px,
        #         rx,
        #         ry,
        #         vstep=1,
        #         pose_marker="s",
        #         fcolor="c",
        #         ax=ax_costmap,
        #     )

        # (c) Expert waypoint trajectory (meters → pixels)
        if waypt_label_m is not None and waypt_label_m.size > 0:
            waypt_traj_px = waypt_label_m / self.resolution
            ax_costmap.plot(
                rx + waypt_traj_px[:, 0],
                ry + waypt_traj_px[:, 1],
                marker="s",
                c='orange',
                markersize=4,
                markerfacecolor="limegreen",
                label="Expert waypts",
            )

        # (d) Robot pose
        ax_costmap.plot(
            rx,
            ry,
            marker=">",
            markersize=12,
            markerfacecolor="lawngreen",
            color="g",
            label="Robot",
        )

        # (e) Old SG
        old_sg_px = old_sg_m.copy()
        old_sg_px[0:2] /= self.resolution
        ax_costmap.plot(
            rx + old_sg_px[0],
            ry + old_sg_px[1],
            "mp",
            markersize=8,
            markerfacecolor="m",
            label="SG(old)",
        )

        gt_sg_px = gt_sg_m.copy()
        gt_sg_px[0:2] /= self.resolution
        ax_costmap.plot(
            rx + gt_sg_px[0],
            ry + gt_sg_px[1],
            marker="p",
            c="orange",
            markersize=10,
            markerfacecolor="limegreen",
            label="SG(GT)",
        )

        # (f) Predicted waypoints
        if pred_waypoints_m is not None and pred_waypoints_m.size > 0:
            pred_waypoints_px = pred_waypoints_m[:, :2] / self.resolution
            ax_costmap.plot(
                pred_waypoints_px[:, 0] + rx,
                pred_waypoints_px[:, 1] + ry,
                "cs",
                markersize=4,
                markerfacecolor=(0.3, 0.0, 0.51),
                label="Pred waypts",
            )

        # (g) SG predicted offset
        if pred_pose_diff_m is not None and pred_pose_diff_m.size >= 2:
            pred_pose_diff_px = pred_pose_diff_m[:2] / self.resolution
            ax_costmap.plot(
                pred_pose_diff_px[0] + rx,
                pred_pose_diff_px[1] + ry,
                marker="p",
                c="cyan",
                markersize=12,
                markerfacecolor=(0.3, 0.0, 0.51),
                label="SG(pred)",
            )

        ax_costmap.legend(loc="upper left", framealpha=0.8)

        # =========================
        # Status (SAFE / COLLISION)
        # =========================
        if pred_collprob is None:
            status_str = "ColStatus: N/A"
            box_color = (0.0, 0.0, 1.0)
        elif pred_collprob >= 0.5:
            status_str = "COLLISION %.2f" % pred_collprob
            box_color = (1.0, 0.0, 0.0)
        elif pred_collprob < 0.5:
            status_str = "SAFE %.2f" % pred_collprob
            box_color = (0.0, 0.6, 0.0)
        else:
            raise ValueError

        ax_costmap.text(
            0.5,  # x 중앙
            1.10,  # y: axes 바로 위 (1.0보다 조금 위)
            status_str,
            transform=ax_costmap.transAxes,
            fontsize=16,
            fontweight="bold",
            color="white",
            ha="center",
            va="bottom",
            bbox=dict(
                facecolor=box_color,
                edgecolor="none",
                boxstyle="round,pad=0.4",
            ),
        )

        ax_costmap.set_xlabel("x (px)")
        ax_costmap.set_ylabel("y (px)")
        ax_costmap.set_xlim(0,200)
        ax_costmap.set_ylim(0,200)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path is not None:
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        else:
            plt.close(fig)