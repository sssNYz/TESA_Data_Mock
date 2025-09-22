"""
Coordinate projection utilities for the drone detection simulator.

This module implements world-to-pixel projection using the pinhole camera model
with simple 2D rotation for camera yaw/pitch transformations.
"""

import math
from typing import Dict, Tuple, Optional
import numpy as np


def apply_camera_rotation(world_pos_enu: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Apply simple 2D rotation for camera yaw/pitch to convert ENU to camera frame.
    
    This implements Pi-level coordinate transformation without complex 3D transforms.
    Camera frame: +Z forward (optical axis), +X right, +Y down
    World frame: +X east, +Y north, +Z up
    
    Args:
        world_pos_enu: World position in ENU coordinates [x_east, y_north, z_up] (meters)
        yaw_deg: Camera yaw angle in degrees (rotation around Z-axis, 0=north, 90=east)
        pitch_deg: Camera pitch angle in degrees (rotation around X-axis, positive=down)
        
    Returns:
        Position in camera frame [x_right, y_down, z_forward] (meters)
    """
    # Convert angles to radians
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    
    # Extract ENU coordinates
    x_east = world_pos_enu[0]
    y_north = world_pos_enu[1] 
    z_up = world_pos_enu[2]
    
    # Apply yaw rotation (around Z-axis) to align with camera heading
    # Yaw=0: camera faces north, Yaw=90: camera faces east
    # We need to rotate the world coordinates to camera's reference frame
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    
    # Transform ENU to camera-aligned coordinates
    # Camera forward direction after yaw rotation
    forward_east = sin_yaw   # Component of forward in east direction
    forward_north = cos_yaw  # Component of forward in north direction
    
    # Camera right direction after yaw rotation (90° clockwise from forward)
    right_east = cos_yaw     # Component of right in east direction  
    right_north = -sin_yaw   # Component of right in north direction
    
    # Project world position onto camera axes
    x_rotated = right_east * x_east + right_north * y_north    # Right component
    y_rotated = forward_east * x_east + forward_north * y_north # Forward component
    z_rotated = z_up  # Up component unchanged by yaw
    
    # Apply pitch rotation (around X-axis) to tilt camera up/down
    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)
    
    # After yaw rotation: x_rotated=right, y_rotated=forward, z_rotated=up
    # Apply pitch to get camera frame: +X right, +Y down, +Z forward
    x_camera = x_rotated  # Right direction unchanged by pitch
    y_camera = sin_pitch * y_rotated - cos_pitch * z_rotated  # Down (positive Y)
    z_camera = cos_pitch * y_rotated + sin_pitch * z_rotated  # Forward (positive Z)
    
    return np.array([x_camera, y_camera, z_camera])


def project_to_pixels(camera_pos: np.ndarray, focal_px: float, 
                     principal_point: Tuple[float, float]) -> Dict:
    """
    Project camera frame coordinates to pixel coordinates using pinhole camera model.
    
    Args:
        camera_pos: Position in camera frame [x_right, y_down, z_forward] (meters)
        focal_px: Focal length in pixels
        principal_point: Principal point coordinates (cx, cy) in pixels
        
    Returns:
        Dictionary containing:
        - 'pixel_coords': (u, v) pixel coordinates
        - 'depth': Distance from camera in meters (z_forward)
        - 'behind_camera': Boolean indicating if point is behind camera
    """
    x_cam, y_cam, z_cam = camera_pos
    
    # Check if point is behind camera (negative Z in camera frame)
    if z_cam <= 0:
        return {
            'pixel_coords': (0.0, 0.0),
            'depth': z_cam,
            'behind_camera': True
        }
    
    # Apply pinhole projection: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
    # Assuming square pixels (fx = fy = focal_px)
    u = focal_px * (x_cam / z_cam) + principal_point[0]
    v = focal_px * (y_cam / z_cam) + principal_point[1]
    
    return {
        'pixel_coords': (u, v),
        'depth': z_cam,
        'behind_camera': False
    }


def validate_pixel_coordinates(pixel_coords: Tuple[float, float], 
                             image_width: int, image_height: int,
                             margin: float = 0.0) -> Dict:
    """
    Validate pixel coordinates and check image boundary conditions.
    
    Args:
        pixel_coords: Pixel coordinates (u, v)
        image_width: Image width in pixels
        image_height: Image height in pixels
        margin: Optional margin in pixels for boundary checking
        
    Returns:
        Dictionary containing:
        - 'in_bounds': Boolean indicating if coordinates are within image bounds
        - 'clipped_coords': Coordinates clipped to image boundaries
        - 'distance_from_edge': Minimum distance to image edge in pixels
    """
    u, v = pixel_coords
    
    # Check if coordinates are within bounds (including margin)
    in_bounds = (margin <= u <= image_width - margin and 
                margin <= v <= image_height - margin)
    
    # Clip coordinates to image boundaries
    u_clipped = max(0, min(u, image_width))
    v_clipped = max(0, min(v, image_height))
    
    # Calculate minimum distance to image edge
    # For points inside image, this is positive distance to nearest edge
    # For points outside image, this is negative distance from nearest edge
    dist_to_left = u - 0
    dist_to_right = image_width - u
    dist_to_top = v - 0
    dist_to_bottom = image_height - v
    
    distance_from_edge = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
    
    return {
        'in_bounds': in_bounds,
        'clipped_coords': (u_clipped, v_clipped),
        'distance_from_edge': distance_from_edge
    }


def world_to_pixel_projection(world_pos_enu: np.ndarray, 
                            camera_yaw_deg: float, 
                            camera_pitch_deg: float,
                            focal_px: float,
                            principal_point: Tuple[float, float],
                            image_width: int,
                            image_height: int) -> Dict:
    """
    Complete world-to-pixel projection pipeline.
    
    This function combines coordinate transformation, projection, and validation
    to convert world ENU coordinates to pixel coordinates with boundary checking.
    
    Args:
        world_pos_enu: World position in ENU coordinates [x_east, y_north, z_up] (meters)
        camera_yaw_deg: Camera yaw angle in degrees
        camera_pitch_deg: Camera pitch angle in degrees  
        focal_px: Focal length in pixels
        principal_point: Principal point coordinates (cx, cy) in pixels
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Dictionary containing complete projection results:
        - 'pixel_coords': (u, v) pixel coordinates
        - 'depth': Distance from camera in meters
        - 'behind_camera': Boolean indicating if point is behind camera
        - 'in_bounds': Boolean indicating if coordinates are within image bounds
        - 'clipped_coords': Coordinates clipped to image boundaries
        - 'distance_from_edge': Minimum distance to image edge in pixels
        - 'camera_coords': Camera frame coordinates for debugging
    """
    # Step 1: Transform world coordinates to camera frame
    camera_coords = apply_camera_rotation(world_pos_enu, camera_yaw_deg, camera_pitch_deg)
    
    # Step 2: Project to pixel coordinates
    projection_result = project_to_pixels(camera_coords, focal_px, principal_point)
    
    # Step 3: Validate pixel coordinates and check boundaries
    if not projection_result['behind_camera']:
        validation_result = validate_pixel_coordinates(
            projection_result['pixel_coords'], 
            image_width, 
            image_height
        )
    else:
        # If behind camera, mark as out of bounds
        validation_result = {
            'in_bounds': False,
            'clipped_coords': (0.0, 0.0),
            'distance_from_edge': -1.0
        }
    
    # Combine all results
    return {
        'pixel_coords': projection_result['pixel_coords'],
        'depth': projection_result['depth'],
        'behind_camera': projection_result['behind_camera'],
        'in_bounds': validation_result['in_bounds'],
        'clipped_coords': validation_result['clipped_coords'],
        'distance_from_edge': validation_result['distance_from_edge'],
        'camera_coords': camera_coords.tolist()  # For debugging
    }