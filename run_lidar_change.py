import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import time

# ============= CONFIGURATION =============
BASE_DIR = "/home/abinandha/3d_pc_change/3DCDNet"
SAVE_DIR = os.path.join(BASE_DIR, "depth_captures")
OUTPUT_DIR = os.path.join(BASE_DIR, "change_detection_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Change detection parameters - TUNED FOR CAR DETECTION
CHANGE_THRESHOLD = 1.5  # meters - detect only major changes (car)
MIN_CLUSTER_SIZE = 80   # LOWERED - car only has ~100 points after downsampling
VOXEL_SIZE = 0.05  # voxel size for downsampling

print("="*100)
print(" "*25 + "3D CHANGE DETECTION")
print(" "*20 + "Scene 1 vs Scene 2 (250m × 50m Urban Street)")
print("="*100)

# ============= LOAD POINT CLOUDS =============
def load_point_clouds():
    """Load both point cloud scenes."""
    print("\n[STEP 1] LOADING POINT CLOUDS...")
    
    scene1_path = os.path.join(SAVE_DIR, "ultimate_scene_1_250m_detailed.ply")
    scene2_path = os.path.join(SAVE_DIR, "ultimate_scene_2_250m_3people.ply")
    
    if not os.path.exists(scene1_path) or not os.path.exists(scene2_path):
        print(f"  ✗ ERROR: Point cloud files not found!")
        exit(1)
    
    print(f"  Loading Scene 1...")
    pcd1 = o3d.io.read_point_cloud(scene1_path)
    
    print(f"  Loading Scene 2...")
    pcd2 = o3d.io.read_point_cloud(scene2_path)
    
    print(f"  ✓ Scene 1: {len(pcd1.points):,} points")
    print(f"  ✓ Scene 2: {len(pcd2.points):,} points")
    
    return pcd1, pcd2

# ============= VOXEL DOWNSAMPLING =============
def downsample_point_cloud(pcd, voxel_size):
    """Downsample point cloud for faster processing."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down

# ============= ICP REGISTRATION =============
def register_point_clouds(pcd1, pcd2):
    """Register Scene 2 to Scene 1 using ICP."""
    print("\n[STEP 2] SPATIAL ALIGNMENT (ICP)...")
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source=pcd2,
        target=pcd1,
        max_correspondence_distance=2.0,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    print(f"  ✓ ICP Fitness: {reg_p2p.fitness:.6f}")
    print(f"  ✓ ICP RMSE: {reg_p2p.inlier_rmse:.6f}")
    
    pcd2_aligned = pcd2.transform(reg_p2p.transformation)
    
    return pcd1, pcd2_aligned

# ============= CHANGE DETECTION WITH CLUSTERING =============
def detect_change_points(pcd1, pcd2, threshold=1.5, min_cluster_size=80):
    """
    Detect changed points between two scenes with clustering.
    Only returns large clusters (real changes, not noise).
    """
    print("\n[STEP 3] DETECTING CHANGES WITH CLUSTERING...")
    
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    
    print(f"  Building spatial index from Scene 2...")
    tree2 = KDTree(pts2)
    
    print(f"  Finding points in Scene 1 that are absent in Scene 2...")
    distances, indices = tree2.query(pts1, k=1)
    
    # Points in Scene 1 that are far from Scene 2 = REMOVED objects
    change_mask = distances > threshold
    change_indices_all = np.where(change_mask)[0]
    
    print(f"\n  Initial detection: {len(change_indices_all):,} change points")
    
    if len(change_indices_all) == 0:
        print(f"  ✗ No change points detected!")
        return pts1, np.array([], dtype=int), np.array([])
    
    print(f"  Clustering to filter noise...")
    
    # CLUSTER the change points to find only significant changes
    change_pcd = o3d.geometry.PointCloud()
    change_pcd.points = o3d.utility.Vector3dVector(pts1[change_indices_all])
    
    # DBSCAN clustering: eps=0.3 (tight), min_points=30
    labels = np.array(change_pcd.cluster_dbscan(eps=0.3, min_points=30))
    
    # Filter clusters by size - keep only significant ones
    unique_labels = set(labels)
    significant_change_indices = []
    
    cluster_count = 0
    for label in unique_labels:
        if label == -1:  # Noise
            continue
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)
        
        if cluster_size >= min_cluster_size:
            # Map back to original indices
            actual_indices = change_indices_all[cluster_indices]
            significant_change_indices.extend(actual_indices)
            cluster_count += 1
            print(f"    ✓ Found cluster {cluster_count}: {cluster_size:,} points (SIGNIFICANT)")
        else:
            print(f"    ✗ Filtered: {cluster_size:,} points (too small)")
    
    change_indices = np.array(significant_change_indices, dtype=int)
    
    print(f"\n  ✓ Final detection: {len(change_indices):,} SIGNIFICANT change points")
    print(f"    (Objects in Scene 1 but NOT in Scene 2 - the CAR)")
    
    return pts1, change_indices, distances[change_mask]

# ============= CREATE OUTPUT - SINGLE VISUALIZATION =============
def create_change_visualization(pcd1, pcd2, change_indices):
    """
    Create SINGLE output:
    - Scene 1 background (Gray)
    - Scene 2 overlaid (Blue)
    - CHANGED POINTS in RED (the car - only significant clusters)
    """
    print("\n[STEP 4] CREATING CHANGE VISUALIZATION...")
    
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    
    # Create output point cloud
    output_pcd = o3d.geometry.PointCloud()
    
    # Combine both scenes
    combined_points = np.vstack([pts1, pts2])
    output_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    # Color scheme:
    # - Scene 1 base: Light gray
    # - Scene 2: Light blue
    # - CHANGED (car): BRIGHT RED
    
    colors = np.zeros((len(combined_points), 3))
    
    # Scene 1: Gray background
    colors[:len(pts1)] = [0.4, 0.4, 0.4]
    
    # Scene 2: Blue
    colors[len(pts1):] = [0.2, 0.5, 0.9]
    
    # CHANGE POINTS: BRIGHT RED (only significant clusters = the CAR)
    if len(change_indices) > 0:
        colors[change_indices] = [1.0, 0.0, 0.0]
        print(f"\n  ✓ Colored {len(change_indices):,} points in RED (the car)")
    else:
        print(f"\n  ✗ No change points to color")
    
    output_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\n  Color scheme:")
    print(f"    Gray: Scene 1 (static)")
    print(f"    Blue: Scene 2 (static)")
    print(f"    RED ◀ CHANGED: Car (major change, only significant clusters)")
    
    return output_pcd

# ============= SAVE OUTPUT =============
def save_result(pcd_output, change_count):
    """Save the change detection result."""
    output_path = os.path.join(OUTPUT_DIR, "change_detection_output.ply")
    o3d.io.write_point_cloud(output_path, pcd_output)
    
    print(f"\n  ✓ Saved: {output_path}")
    print(f"    Total points: {len(pcd_output.points):,}")
    print(f"    Changed points (RED - CAR): {change_count:,}")
    
    return output_path

# ============= VISUALIZATION =============
def show_result(pcd_output):
    """Display the change detection result."""
    print("\n[STEP 5] DISPLAYING RESULT...")
    print("  Opening 3D viewer (2560×1440 Ultra-HD)...")
    print("  Controls: Left-click+drag=Rotate | Scroll=Zoom | Right-click+drag=Pan")
    print("  Close window when done...")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Change Detection - RED = Car (Removed)", width=2560, height=1440)
    vis.add_geometry(pcd_output)
    
    opt = vis.get_render_option()
    opt.point_size = 0.6
    opt.background_color = np.array([0, 0, 0])
    
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "="*100)
    
    # Load scenes
    pcd1, pcd2 = load_point_clouds()
    
    # Downsample for speed
    print("\n[DOWNSAMPLING for speed...]")
    pcd1 = downsample_point_cloud(pcd1, VOXEL_SIZE)
    pcd2 = downsample_point_cloud(pcd2, VOXEL_SIZE)
    print(f"  Scene 1: {len(pcd1.points):,} points")
    print(f"  Scene 2: {len(pcd2.points):,} points")
    
    # Register (align) Scene 2 to Scene 1
    pcd1_reg, pcd2_reg = register_point_clouds(pcd1, pcd2)
    
    # Detect what changed (the car) - with clustering to filter noise
    pts1, change_indices, change_distances = detect_change_points(
        pcd1_reg, pcd2_reg, 
        threshold=CHANGE_THRESHOLD,
        min_cluster_size=MIN_CLUSTER_SIZE
    )
    
    # Create single visualization
    output_pcd = create_change_visualization(pcd1_reg, pcd2_reg, change_indices)
    
    # Save result
    output_file = save_result(output_pcd, len(change_indices))
    
    # Show result
    show_result(output_pcd)
    
    # Summary
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*100)
    print("✓ CHANGE DETECTION COMPLETE")
    print("="*100)
    print(f"\nResults:")
    print(f"  Output file: {output_file}")
    print(f"  Processing time: {elapsed_time:.2f} seconds")
    print(f"\nDetection Parameters:")
    print(f"  Distance threshold: {CHANGE_THRESHOLD}m (detect only major changes)")
    print(f"  Min cluster size: {MIN_CLUSTER_SIZE} points (filter noise)")
    print(f"\nColor Legend:")
    print(f"  🔴 RED = Changed points (Car - only in Scene 1, not in Scene 2)")
    print(f"  ⚪ Gray = Scene 1 static points")
    print(f"  🔵 Blue = Scene 2 static points")
    print("\n" + "="*100 + "\n")
