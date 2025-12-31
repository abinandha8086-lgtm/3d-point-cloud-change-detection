import open3d as o3d
import pyrealsense2 as rs
import numpy as np
import os

# --- PATH INDEPENDENCE LOGIC ---
# This finds the folder where THIS script is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "depth_captures")

def process_video(bag_name):
    bag_path = os.path.join(DATA_DIR, bag_name)
    if not os.path.exists(bag_path):
        print(f"Error: {bag_name} not found in {DATA_DIR}")
        return None

    print(f"Processing: {bag_name}...")
    
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.008, sdf_trunc=0.02,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    
    try:
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False) 
    except Exception as e:
        print(f"Pipeline error: {e}")
        return None

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    align = rs.align(rs.stream.color)

    curr_pose = np.identity(4)
    prev_rgbd = None

    try:
        while True:
            frames = pipeline.wait_for_frames(2000)
            aligned = align.process(frames)
            
            curr_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asanyarray(aligned.get_color_frame().get_data())),
                o3d.geometry.Image(np.asanyarray(aligned.get_depth_frame().get_data())),
                depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)

            if prev_rgbd is not None:
                success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                    curr_rgbd, prev_rgbd, pinhole, np.identity(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    o3d.pipelines.odometry.OdometryOption())
                if success:
                    curr_pose = curr_pose @ np.linalg.inv(trans)

            volume.integrate(curr_rgbd, pinhole, curr_pose)
            prev_rgbd = curr_rgbd
    except RuntimeError: pass
    finally: pipeline.stop()

    pcd = volume.extract_point_cloud()
    pcd, _ = pcd.remove_radius_outlier(nb_points=20, radius=0.03)
    pcd.estimate_normals()
    return pcd

if __name__ == "__main__":
    # Ensure the data directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    p_ref = process_video("reference_video.bag")
    p_cur = process_video("current_video.bag")

    if p_ref and p_cur:
        # Align and detect
        reg = o3d.pipelines.registration.registration_icp(
            p_cur, p_ref, 0.05, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        p_cur.transform(reg.transformation)

        dists = np.asarray(p_cur.compute_point_cloud_distance(p_ref))
        p_cur.paint_uniform_color([0.6, 0.6, 0.6])
        colors = np.asarray(p_cur.colors)
        colors[dists > 0.02] = [1, 0, 0] # Red for changes
        p_cur.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([p_cur])
