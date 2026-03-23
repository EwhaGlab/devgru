import os
import glob
import re
import numpy as np

import cv2  # 필요하면 사용 (여기서는 안 써도 됨)
from PIL import Image  # ★ 이미지 로딩용

from os.path import dirname, abspath
#BASE_DIR = '/home/hankm/python_ws/viznav/depth-nav' #
BASE_DIR = dirname(dirname(abspath(__file__)))

import sys
sys.path.append(BASE_DIR)
import utils.rigid_motion as rm

import visualize.visualizer as visualizer
from visualize.visualizer import Visualizer, CostmapParams

def load_pose_data( pose_file ):
    pose_raw = np.loadtxt(pose_file)
    num_data, _ = pose_raw.shape

    xy0 = pose_raw[0, 4:6]
    quat0 = pose_raw[0, [10, 7, 8, 9]]
    wHb = np.zeros((num_data, 4, 4), dtype=float)
    wHb[0] = rm.quat_to_htm( quat0 )
    wHb[0, 0:2, 3] = xy0
    xy = np.zeros((num_data, 2), dtype=float)
    xy[0, :] = xy0

    for idx in range(1, num_data):
        # MATLAB: xy_line = pose_raw(idx, 5:6)
        xy_line = pose_raw[idx, 4:6]
        # MATLAB: quat = pose_raw(idx, [11, 8:10])
        quat = pose_raw[idx, [10, 7, 8, 9]]
        wHb[idx, :, :] = rm.quat_to_htm(quat)
        wHb[idx, 0:2, 3] = xy_line
        xy[idx, :] = xy_line

    return pose_raw, xy, wHb


def get_topomap_index_strings(topomap_root_dir):
    """rgb*.png 파일들에서 숫자 부분만 뽑아서 리스트로 리턴."""
    rgb_files = sorted(glob.glob(os.path.join(topomap_root_dir, "rgb*.png")))
    topoimg_idxstr = []
    for path in rgb_files:
        name = os.path.basename(path).lower()
        digits_str = re.sub(r"\D", "", name)  # 숫자만 추출
        topoimg_idxstr.append(digits_str)
    return topoimg_idxstr


def pil_read_rgb(path: str) -> np.ndarray:
    """PIL로 RGB 이미지 읽기 -> (H,W,3) uint8 ndarray."""
    return np.array(Image.open(path).convert("RGB"))


def pil_read_depth_mm_to_m(path: str) -> np.ndarray:
    """
    PIL로 depth png 읽고, mm 단위를 m로 변환.
    MATLAB과 동일하게 /1000.0.
    """
    img = Image.open(path)
    depth = np.array(img).astype(np.float32)  # 보통 16-bit or 8-bit
    depth_m = depth / 1000.0
    return depth_m


def main():
    # ============================
    # 1. Hyper params (MATLAB 그대로)
    # ============================
    draw_topomap = True
    context_len = 5

    eta = 1.0
    rho0 = 1.2 / 0.05
    obs_thr = 99
    FPS = 10
    v_max = 0.3
    w_max = 0.6
    ws = 3

    resolution = 0.05
    map_size_m = 10.0
    robot_radius_m = 0.25
    inflation_radius_m = 1.0
    max_range_m = 25.0

    cm_params = CostmapParams(
        resolution=resolution,
        map_size_m=map_size_m,
        robot_radius_m=robot_radius_m,
        inflation_radius_m=inflation_radius_m,
        max_range_m=max_range_m,
    )

    viz = Visualizer(cm_params)

    map_size_px = int(map_size_m / resolution)
    rx = map_size_px / 2.0
    ry = rx

    # ============================
    # 2. 데이터 폴더 설정
    # ============================
    colldata_dir = "/media/data/mydata/former_datasets/colldata/colldata-all"
    bag_dirs = sorted(
        d for d in glob.glob(os.path.join(colldata_dir, "bag_*"))
        if os.path.isdir(d)
    )

    # MATLAB: for bag_idx = 82:length(bag_dirs)  -> Python: index 81부터
    for bag_idx in range(0, 27): #len(bag_dirs)):
        bag_dir = bag_dirs[bag_idx]
        bag_name = os.path.basename(bag_dir)
        print(f"[Bag] {bag_idx+1}/{len(bag_dirs)} : {bag_name}")

        data_path = os.path.join(colldata_dir, bag_name)
        matches = glob.glob(os.path.join(data_path, "T*"))
        if len(matches) == 0:
            print("No T* directory found")
        else:
            topomap_name = os.path.basename(matches[0])

            # data* 디렉토리들
        colldatalist = sorted(
            d for d in glob.glob(os.path.join(data_path, "data*"))
            if os.path.isdir(d) )

        if not colldatalist:
            print(f"  - No data* dirs under {data_path}, skip.")
            continue

        # processed dir
        proc_dir = f"/media/data/results/navdata_collector/colldata/processed/{topomap_name}_coll"

        # navtime_yymmddhhss 추출
        str_split = bag_name.split("_")
        navtime_ymdhs = str_split[-1]
        navtime_tmp = navtime_ymdhs.split("-")
        navtime_ymdh = "-".join(navtime_tmp[:-1])

        # sync metadata dir
        # sync_metadata_dir = os.path.join(
        #     proc_dir,
        #     f"coll_{navtime_ymdh}",
        #     f"bag_{navtime_ymdhs}",
        #     "synced",
        # )

        topomap_root_dir = os.path.join(
            "/home/hankm/python_ws/viznav/depth-nav/deployment/topomaps",
            topomap_name,
        )

        # topo img index list (문자열)
        topoimg_idxstr = get_topomap_index_strings(topomap_root_dir)
        if not topoimg_idxstr:
            print(f"  - No rgb*.png under {topomap_root_dir}, skip.")
            continue

        # ============================
        # 3. Topomap pose data 로드
        # ============================
        topo_odom_file = os.path.join(topomap_root_dir, "topo_odom.txt")
        topo_m2b_file = os.path.join(topomap_root_dir, "topo_tf_m2b.txt")
        topo_m2o_file = os.path.join(topomap_root_dir, "topo_tf_m2o.txt")

        topo_odom_raw, topo_odom_xy, topo_o1Hb = load_pose_data(topo_odom_file)
        topo_m2b_raw, topo_m2b_xy, topo_m1Hb = load_pose_data(topo_m2b_file)
        topo_m2o_raw, topo_m2o_xy, topo_m1Ho = load_pose_data(topo_m2o_file)

        num_node = topo_m2b_raw.shape[0]

        if draw_topomap:
            # nav pose 파일들
            nav_m2b_file = os.path.join(bag_dir, "sync_tf_m2b.txt")
            nav_m2o_file = os.path.join(bag_dir, "sync_tf_m2o.txt")
            nav_odom_file = os.path.join(bag_dir, "sync_odom.txt")

            if os.path.isfile(nav_m2b_file):
                nav_odom_raw, nav_odom_xy, nav_o2Hb = load_pose_data(nav_odom_file)
                nav_m2b_raw, nav_m2b_xy, nav_m2Hb = load_pose_data(nav_m2b_file)
                nav_m2o_raw, nav_m2o_xy, nav_m2Ho = load_pose_data(nav_m2o_file)
                nav_len, _ = nav_odom_raw.shape
            # cannot draw topomap if nav file doesn't exist

        num_coll_data = len(colldatalist)

        # ============================
        # 4. 각 data_k 에 대해 그림 생성
        # ============================
        data_idx = 0
        while data_idx < num_coll_data:
            colldata_path = colldatalist[data_idx]
            colldata_name = os.path.basename(colldata_path)
            print(f"    [Data] {data_idx}/{num_coll_data} : {colldata_name}")

            # 1) context_index
            context_index_path = os.path.join(colldata_path, "context_index.txt")
            if os.path.exists(context_index_path):
                context_index = np.loadtxt(context_index_path, dtype=int)
            else:
                context_index = None

            # 2) label waypoints (GT)
            corrected_waypoints_m_path = os.path.join(colldata_path, "corrected_waypoints_m.txt")
            corrected_waypoints_m = np.loadtxt(corrected_waypoints_m_path)

            # 3) predicted waypoints
            pred_waypoints_m_path = os.path.join(colldata_path, "pred_waypoints_m.txt")
            pred_waypoints_m = np.loadtxt(pred_waypoints_m_path)

            # 4) data_sg_idx
            data_sg_idx_path = os.path.join(colldata_path, "data_sg_idx.txt")

            if os.path.isfile(data_sg_idx_path):
                img_idx, sg_idx, global_sg_idx = np.loadtxt(data_sg_idx_path, dtype=int)
            else:
                img_idx = context_index[-1]
                sg_idx = None
                global_sg_idx = None

            # 5) new subgoal (GT / corrected)
            new_subgoal_m_path = os.path.join(colldata_path, "new_subgoal_m.txt")
            new_subgoal_m = np.loadtxt(new_subgoal_m_path)  # [x,y,qw,qz]

            # 6) old subgoal (pred)
            old_subgoal_m_path = os.path.join(colldata_path, "old_subgoal_m.txt")
            old_subgoal_m = np.loadtxt(old_subgoal_m_path)  # [x,y,qw,qz]

            # 7) pose_context
            pose_context_m_path = os.path.join(colldata_path, "pose_context_m.txt")
            if os.path.exists(pose_context_m_path):
                pose_context_m = np.loadtxt(pose_context_m_path)
            else:
                pose_context_m = None

            # 8) costmap_i8
            costmap_i8_path = os.path.join(colldata_path, "costmap_i8.txt")
            costmap_i8 = np.loadtxt(costmap_i8_path, dtype=np.int32, delimiter=",")

            # 9) topomap 기반 SG 시퀀스
            sgs_xyzq_corrected_m = None
            if draw_topomap and sg_idx is not None:
                o1Hb_curr = topo_o1Hb[sg_idx]
                m1Ho1_curr = topo_m1Ho[sg_idx]
                m1Hsg_curr = topo_m1Hb[sg_idx]

                o2Hb_curr = nav_o2Hb[img_idx]
                m2Ho2_curr = nav_m2Ho[img_idx]
                m2Hb_curr = nav_m2Hb[img_idx]

                sgs_xyzq_corrected_m = np.zeros((num_node, 7), dtype=float)
                m2Hb_inv = np.linalg.inv(m2Hb_curr)
                for next_sg_idx in range(num_node):
                    bHsg = np.matmul(m2Hb_inv, topo_m1Hb[next_sg_idx] )  # 4x4
                    sgs_xyzq_corrected_m[next_sg_idx, 0:2] = bHsg[0:2, 3]
                    sgs_xyzq_corrected_m[next_sg_idx, 3:] = rm.htm_to_quat(bHsg)

            # === 이미지 로딩 (PIL 사용) ===
            depth_img_path = os.path.join(colldata_path, f"depth{img_idx:05d}.png")
            rgb_img_path = os.path.join(colldata_path, f"rgb{img_idx:05d}.png")

            depth_img = pil_read_depth_mm_to_m(depth_img_path)
            rgb_img = pil_read_rgb(rgb_img_path)

            # subgoal RGB/Depth (topomap)
            #sgimg_idx_str = topoimg_idxstr[sg_idx]
            rgb_sg_path = os.path.join(colldata_path, f"rgb_sg.png")
            depth_sg_path = os.path.join(colldata_path, f"depth_sg.png")

            rgb_sg = pil_read_rgb(rgb_sg_path)
            depth_sg = pil_read_depth_mm_to_m(depth_sg_path)

            # ============================
            # 10. Visualizer로 그림 그리기
            # ============================
            str_data_info = f"{bag_name}/{colldata_name}"

            old_sg_m = old_subgoal_m
            gt_sg_m = new_subgoal_m
            pred_pose_diff_m = np.array([])   # 아직 없음
            pred_collprob = None              # collision prob 정보 없으니 1

            outfig_dir = os.path.join(
                "/media/data/results/devgru/colldata_viewer",
                navtime_ymdh,
            )
            os.makedirs(outfig_dir, exist_ok=True)
            outfig_name = os.path.join(outfig_dir, f"data{data_idx:05d}.png")

            corrected_waypoints_m[:, 0:2] *= 2.0
            pred_waypoints_m[:, 0:2] *= 2.0

            viz.draw(
                str_data_info=str_data_info,
                rgb_img=rgb_img,
                rgb_sg=rgb_sg,
                depth_img=depth_img,
                depth_sg=depth_sg,
                costmap_i8=costmap_i8,
                sgs_xyzq_corrected_m=sgs_xyzq_corrected_m,
                pose_context_m=pose_context_m,
                waypt_label_m=corrected_waypoints_m,
                rx=rx,
                ry=ry,
                old_sg_m=old_sg_m,
                gt_sg_m=gt_sg_m,
                pred_waypoints_m=pred_waypoints_m,
                pred_pose_diff_m=pred_pose_diff_m,
                pred_collprob=pred_collprob,
                save_path=outfig_name,
                show=False,
            )

            print(f"      -> saved {outfig_name}")
            data_idx += 1

    print("All done.")


if __name__ == "__main__":
    main()