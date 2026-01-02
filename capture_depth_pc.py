import pyrealsense2 as rs
import numpy as np
import cv2
import os

def record_scene(filename, prompt):
    save_path = "/home/abinandha/3d_pc_change/3DCDNet/depth_captures"
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    
    input(f"\n[STEP] {prompt}. Press ENTER to start recording...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_record_to_file(full_path)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Filtering for sharper 3D shapes
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    try:
        profile = pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3) # High Accuracy
        depth_sensor.set_option(rs.option.laser_power, 360) # Force laser for white walls

        print(f"Recording to {filename}... Press 'Q' in the window to stop.")

        while True:
            frames = pipeline.wait_for_frames(5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame: continue

            # Apply filters to the preview so you know if it looks good
            filtered = temporal.process(depth_frame)
            filtered = hole_filling.process(filtered)
            
            depth_image = np.asanyarray(filtered.get_data())
            depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            cv2.putText(depth_cm, f"Recording: {filename}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Live Depth View", depth_cm)
            
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Finished recording {filename}.")

# Sequence of events
if __name__ == "__main__":
    # Scene 1: Empty Wall
    record_scene("reference_video.bag", "Clear the wall (Reference Scene)")
    
    # Scene 2: Wall with Pipes
    record_scene("current_video.bag", "Place objects/pipes on the wall (Current Scene)")
    
    print("\nAll recordings complete. Files are in /depth_captures/")
