import argparse
import os
import sys
import time
from scipy.optimize import minimize

sys.path.append(os.path.abspath("/home/bismuth/Documents/Tese/Depth-Anything"))
import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize
from shared_memory_interface import SharedMemoryInterface
from torchvision.transforms import Compose
from tqdm import tqdm

# Add the path to the depth_anything module (adjust the path accordingly)


# Create the interface
shm_interface = SharedMemoryInterface()
number = 0
# Simple camera intrinsics (assuming focal length = 1 and image center at (width/2, height/2))
f_x = 458.65399169921875
f_y = 457.29598999023438
image_width = 752
image_height = 480
# Assuming the principal point is at the center of the image
# c_x, c_y = image_width / 2, image_height / 2
c_x = 367.21499633789062
c_y = 248.375

# Create the intrinsic matrix (simplified for the example)
K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

# Create a visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window(height=900, width=900)
points = np.random.rand(10, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

vis.add_geometry(pcd)

# Get the view control to adjust camera settings
view_control = vis.get_view_control()

# Set a very large far clipping plane to simulate infinite distance
view_control.set_constant_z_near(0.1)  # Minimum render distance (camera near plane)
view_control.set_constant_z_far(10000)  # Maximum render distance (camera far plane)
view_control.set_lookat([0,0,0])

pcd.points = o3d.utility.Vector3dVector([])  # Reset points
pcd.colors = o3d.utility.Vector3dVector([])  # Reset colors


# Function to convert depth map to point cloud using vectorized operations


def depth_to_point_cloud(raw_image, depth_map):
    """
    Converts a depth map and the corresponding raw image to a 3D point cloud.

    Parameters:
    - raw_image: The input RGB image (height x width x 3)
    - depth_map: The depth map (height x width)
    - c_x, c_y: The principal point (optical center) coordinates
    - f_x, f_y: The focal lengths in x and y directions

    Returns:
    - point_cloud: Open3D point cloud object
    """
    # Normalize depth to [0, 1]
    # Assuming depth is in [0, 255]
    depth = depth_map.astype(np.float32) / 255.0

    scale_factor = scale_point_cloud(depth)
    print(scale_factor)

    # Get image dimensions
    height, width = depth.shape

    # Create mesh grid for pixel coordinates with a step of 10 for downsampling
    u, v = np.meshgrid(np.arange(0, width, 10), np.arange(0, height, 10))  # Downsampling by factor of 10
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth[v, u]  # Use downsampled (u, v) indices directly on depth map

    # Filter out zero depth values (for valid depth only)
    valid = (depth_flat > 0) & (depth_flat < 0.8)
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid] * scale_factor
    # depth_flat = depth_flat[valid] 

    # Compute the 3D points in camera coordinates
    x = (u - c_x) * depth_flat / f_x 
    y = (v - c_y) * depth_flat / f_y
    z = depth_flat

    # Stack to form an (N, 3) array of points in camera coordinates
    points_camera = np.stack((x, y, z), axis=-1)

    # Transform points from camera to world coordinates
    points_world = (camera_rot.T @ points_camera.T).T - camera_pos  # (N, 3)

    # Extract and normalize colors for valid points
    if raw_image.ndim == 2 or raw_image.shape[2] == 1:  # Grayscale case
        colors = np.repeat(raw_image[v, u, np.newaxis], 3, axis=1) / 255.0
    # else:  # RGB case
        # colors = raw_image[v, u] / 255.0
    colors = colors.astype(np.float32)

    # Create the point cloud in world coordinates
    # pcd.points = o3d.utility.Vector3dVector(points_world)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd.points.extend(o3d.utility.Vector3dVector(points_world))
    pcd.colors.extend(o3d.utility.Vector3dVector(colors))
    # print(len(pcd.points))
    # print(len(pcd.colors))
    # return point_cloud
    vis.update_geometry(pcd)
    # Redraw the window
    vis.poll_events()
    vis.update_renderer()

def scale_point_cloud(depth_map):
    """
    Estimate the scaling factor to match the predicted depth map to the SLAM depth values.

    Parameters:
    - slam_points_3d: 3D coordinates from SLAM (N, 3)
    - slam_points_2d: 2D projections from SLAM (N, 2)
    - depth_map: Depth map (height, width) from the algorithm
    - K: Camera intrinsic matrix (3, 3)

    Returns:
    - scaling_factor: Estimated scaling factor for the point cloud depths
    """
    slam_points_2d = projected_points.astype(int)
    slam_points_3d = points
     # Camera rotation and translation
    R = np.array(camera_rot)  # Rotation matrix
        # Translation vector (inverse of the camera position)
    t = -R @ np.array(camera_pos)
    slam_points_camera_coords = []
    

    for point in slam_points_3d:
        # Extract 3D point (world coordinates)
        world_point = np.array([point['x'], point['y'], point['z']])

        # Convert to camera coordinates using rotation and translation
        camera_point = camera_rot @ world_point + t
        slam_points_camera_coords.append(camera_point)

    # Convert the list to a numpy array (N, 3)
    slam_points_camera_coords = np.array(slam_points_camera_coords)

    # Step 1: Extract depths from the depth map corresponding to SLAM 2D projections
    depths_pred = []
    for point_2d in slam_points_2d:
        u, v = int(point_2d[0]), int(point_2d[1])
        # Make sure u, v are within the bounds of the depth map
        if 0 <= u < depth_map.shape[1] and 0 <= v < depth_map.shape[0]:
            depth_pred = depth_map[v, u] / 255.0  # Normalize to [0, 1]
            depths_pred.append(depth_pred)
        else:
            depths_pred.append(0)

    depths_pred = np.array(depths_pred)

    # Step 2: Calculate the real-world depths of the SLAM points
    depths_slam = np.linalg.norm(slam_points_camera_coords, axis=1)  # Euclidean distance from the origin (camera position)
    print(depths_slam)

    # Step 3: Define the loss function (weighted least squares)
    def loss_function(scale_factor):
        scaled_depths = scale_factor * depths_pred
        loss = np.sum(((scaled_depths - depths_slam) ** 2) / depths_slam)
        return loss

    # Step 4: Minimize the loss function to find the best scaling factor
    result = minimize(loss_function, x0=1.0, bounds=[(0, None)])

    scaling_factor = result.x[0]
    return scaling_factor





def project_points(points):
    # Prepare to collect projected 2D points
    projected_points = []

    # Process each point
    for point in points:
        X, Y, Z = point["x"], point["y"], point["z"]
        # Step 1: Apply rotation and translation to get camera space coordinates
        # Camera rotation and translation
        R = np.array(camera_rot)  # Rotation matrix
        # Translation vector (inverse of the camera position)
        t = -R @ np.array(camera_pos)

        # World coordinates in camera space (rotation and translation)
        camera_coords = R @ np.array([X, Y, Z]) + t

        # Step 2: Project the 3D point into 2D image space
        # Using pinhole camera model (no distortion)
        x_cam, y_cam, z_cam = camera_coords
        if z_cam != 0:  # Avoid division by zero
            u = f_x * x_cam / z_cam + c_x
            v = f_y * y_cam / z_cam + c_y
            projected_points.append((u, v))

    # Convert the projected points to a numpy array
    projected_points = np.array(projected_points)
    return projected_points


transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)


DEVICE = "cuda"

depth_anything = (
    DepthAnything.from_pretrained("LiheYoung/depth_anything_vitb14").to(DEVICE).eval()
)


def get_monocular_depth(raw_image):
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

    h, w = image.shape[:2]

    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth = depth_anything(image)

    depth = F.interpolate(depth[None], (h, w), mode="bilinear", align_corners=False)[
        0, 0
    ]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)

    # Create a point cloud from the depth map
    depth_to_point_cloud(raw_image, depth)

    # Visualize the point cloud using Open3D
    # o3d.visualization.draw_geometries([point_cloud])

    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    cv2.imshow("depth, wow", depth)


while True:
    # Read new data
    points, image, camera_pos, camera_rot = shm_interface.read_data()

    projected_points = project_points(points)
    camera_pos = np.array(camera_pos)
    camera_rot= np.array(camera_rot)
    # Assuming you have the image data
    # Read image data (assuming it's grayscale image)
    # image = np.zeros((image_height, image_width), dtype=np.uint8)  # Replace with actual image
    image = image.squeeze()  # Remove the extra dimension (if it's shape (480, 752, 1))
    image_copy = image.copy()

    # Plot the projected points on the image
    for u, v in projected_points:
        if (
            0 <= u < image_width and 0 <= v < image_height
        ):  # Ensure the points are within image bounds
            # Draw points in green
            cv2.circle(image_copy, (int(u), int(v)), 3, (0, 255, 0), -1)

    # Save the image to file
    # cv2.imwrite(f"img/projected_image_{number}.png", image_copy)

    # Save the image to a file (e.g., as PNG)
    # cv2.imwrite(f'img/{number}.png', image)
    # Show the image in a window
    get_monocular_depth(image)
    cv2.imshow("Image Sequence", image_copy)

    # Wait for 1 ms and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # time.sleep(1)
    number += 1

