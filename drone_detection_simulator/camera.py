"""
Camera model implementation for the drone detection simulator.
"""

import math
from typing import Dict, Tuple
import numpy as np

from .config import SimulatorConfig
from .projection import world_to_pixel_projection


class CameraModel:
    """
    Camera model that handles focal length computation and camera metadata generation.
    
    This class implements the pinhole camera model with support for focal length
    calculation from either focal length/sensor dimensions or vertical field of view.
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize the camera model.
        
        Args:
            config: Simulator configuration containing camera parameters
        """
        self.config = config
        self.focal_px = self._compute_focal_px()
        self.principal_point = self._get_principal_point()
    
    def _compute_focal_px(self) -> float:
        """
        Compute focal length in pixels from configuration parameters.
        
        Supports two methods:
        1. From focal length (mm) and sensor height (mm)
        2. From vertical field of view (degrees)
        
        Returns:
            Focal length in pixels
            
        Raises:
            ValueError: If focal length computation parameters are invalid
        """
        if (self.config.focal_length_mm is not None and 
            self.config.sensor_height_mm is not None):
            # Method 1: focal_px = focal_mm * image_height_px / sensor_height_mm
            focal_px = (self.config.focal_length_mm * self.config.image_height_px / 
                       self.config.sensor_height_mm)
            
            if focal_px <= 0:
                raise ValueError("Computed focal length must be positive")
            
            return focal_px
        
        elif self.config.vertical_fov_deg is not None:
            # Method 2: focal_px = image_height_px / (2 * tan(vfov/2))
            vfov_rad = math.radians(self.config.vertical_fov_deg)
            focal_px = self.config.image_height_px / (2.0 * math.tan(vfov_rad / 2.0))
            
            if focal_px <= 0:
                raise ValueError("Computed focal length must be positive")
            
            return focal_px
        
        else:
            raise ValueError(
                "Must specify either (focal_length_mm AND sensor_height_mm) OR vertical_fov_deg"
            )
    
    def _get_principal_point(self) -> Tuple[float, float]:
        """
        Get principal point coordinates, defaulting to image center if not specified.
        
        Returns:
            Principal point coordinates (px, py) in pixels
        """
        if self.config.principal_point_px is not None:
            return self.config.principal_point_px
        else:
            # Default to image center
            return (self.config.image_width_px / 2.0, self.config.image_height_px / 2.0)
    
    def project_world_to_pixels(self, world_pos_enu: np.ndarray) -> Dict:
        """
        Project world position to pixel coordinates using pinhole camera model.
        
        This method is used internally for simulation purposes to generate
        realistic pixel detections from world coordinates.
        
        Args:
            world_pos_enu: World position in ENU coordinates [x, y, z] (meters)
            
        Returns:
            Dictionary containing projection results:
            - 'pixel_coords': (u, v) pixel coordinates
            - 'depth': Distance from camera in meters
            - 'in_view': Boolean indicating if point is in camera view
            - 'behind_camera': Boolean indicating if point is behind camera
            - 'clipped_coords': Coordinates clipped to image boundaries
            - 'distance_from_edge': Minimum distance to image edge in pixels
            - 'camera_coords': Camera frame coordinates for debugging
            
        Note:
            Camera frame: +Z forward, +X right, +Y down
            World frame: +X east, +Y north, +Z up
        """
        # Use the complete projection pipeline from projection utilities
        result = world_to_pixel_projection(
            world_pos_enu=world_pos_enu,
            camera_yaw_deg=self.config.camera_yaw_deg,
            camera_pitch_deg=self.config.camera_pitch_deg,
            focal_px=self.focal_px,
            principal_point=self.principal_point,
            image_width=self.config.image_width_px,
            image_height=self.config.image_height_px
        )
        
        # Convert to the expected format for backward compatibility
        # 'in_view' means point is in front of camera AND within image bounds
        result['in_view'] = not result['behind_camera'] and result['in_bounds']
        
        return result
    
    def get_camera_metadata(self) -> Dict:
        """
        Generate camera metadata for JSON payload.
        
        Returns fixed camera configuration values that would be included
        in detection messages from a Pi-based camera system.
        
        Returns:
            Dictionary containing camera metadata:
            - resolution: [width, height] in pixels
            - focal_px: Focal length in pixels
            - principal_point: [cx, cy] principal point in pixels
            - yaw_deg: Camera yaw angle in degrees
            - pitch_deg: Camera pitch angle in degrees
            - lat_deg: Camera latitude in degrees
            - lon_deg: Camera longitude in degrees
            - alt_m_msl: Camera altitude in meters MSL
        """
        return {
            'resolution': [self.config.image_width_px, self.config.image_height_px],
            'focal_px': self.focal_px,
            'principal_point': [self.principal_point[0], self.principal_point[1]],
            'yaw_deg': self.config.camera_yaw_deg,
            'pitch_deg': self.config.camera_pitch_deg,
            'lat_deg': self.config.camera_lat_deg,
            'lon_deg': self.config.camera_lon_deg,
            'alt_m_msl': self.config.camera_alt_m
        }
    
    def get_focal_length_px(self) -> float:
        """
        Get the computed focal length in pixels.
        
        Returns:
            Focal length in pixels
        """
        return self.focal_px
    
    def get_principal_point_px(self) -> Tuple[float, float]:
        """
        Get the principal point coordinates in pixels.
        
        Returns:
            Principal point coordinates (px, py) in pixels
        """
        return self.principal_point