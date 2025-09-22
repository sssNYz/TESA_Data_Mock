"""
Unit tests for the CameraModel class.
"""

import pytest
import math
import numpy as np
from unittest.mock import Mock

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.camera import CameraModel


class TestCameraModel:
    """Test cases for CameraModel class."""
    
    def test_focal_length_from_fov(self):
        """Test focal length computation from vertical field of view."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            focal_length_mm=None,
            sensor_height_mm=None
        )
        
        camera = CameraModel(config)
        
        # Expected: focal_px = image_height / (2 * tan(vfov/2))
        expected_focal_px = 1080 / (2.0 * math.tan(math.radians(50.0) / 2.0))
        
        assert abs(camera.focal_px - expected_focal_px) < 1e-6
        assert camera.get_focal_length_px() == camera.focal_px
    
    def test_focal_length_from_sensor_params(self):
        """Test focal length computation from focal length and sensor dimensions."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            focal_length_mm=25.0,
            sensor_height_mm=15.0,
            vertical_fov_deg=None
        )
        
        camera = CameraModel(config)
        
        # Expected: focal_px = focal_mm * image_height_px / sensor_height_mm
        expected_focal_px = 25.0 * 1080 / 15.0
        
        assert abs(camera.focal_px - expected_focal_px) < 1e-6
        assert camera.get_focal_length_px() == camera.focal_px
    
    def test_focal_length_edge_cases(self):
        """Test focal length computation edge cases."""
        # Test very small FOV
        config_small_fov = SimulatorConfig(
            image_height_px=1080,
            vertical_fov_deg=1.0
        )
        camera_small = CameraModel(config_small_fov)
        assert camera_small.focal_px > 0
        
        # Test large FOV
        config_large_fov = SimulatorConfig(
            image_height_px=1080,
            vertical_fov_deg=179.0
        )
        camera_large = CameraModel(config_large_fov)
        assert camera_large.focal_px > 0
        
        # Test small sensor
        config_small_sensor = SimulatorConfig(
            image_height_px=1080,
            focal_length_mm=50.0,
            sensor_height_mm=1.0,
            vertical_fov_deg=None
        )
        camera_small_sensor = CameraModel(config_small_sensor)
        assert camera_small_sensor.focal_px > 0
    
    def test_focal_length_validation_errors(self):
        """Test that invalid focal length parameters raise appropriate errors."""
        # Test missing parameters - this should fail at config level
        with pytest.raises(ValueError, match="Must specify either"):
            SimulatorConfig(
                focal_length_mm=None,
                sensor_height_mm=None,
                vertical_fov_deg=None
            )
    
    def test_principal_point_default_center(self):
        """Test that principal point defaults to image center."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            principal_point_px=None
        )
        
        camera = CameraModel(config)
        
        expected_pp = (1920 / 2.0, 1080 / 2.0)
        assert camera.principal_point == expected_pp
        assert camera.get_principal_point_px() == expected_pp
    
    def test_principal_point_custom(self):
        """Test custom principal point specification."""
        custom_pp = (800.0, 600.0)
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            principal_point_px=custom_pp
        )
        
        camera = CameraModel(config)
        
        assert camera.principal_point == custom_pp
        assert camera.get_principal_point_px() == custom_pp
    
    def test_camera_metadata_generation(self):
        """Test camera metadata generation for JSON payload."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            camera_yaw_deg=90.0,
            camera_pitch_deg=10.0,
            camera_lat_deg=13.736717,
            camera_lon_deg=100.523186,
            camera_alt_m=1.5
        )
        
        camera = CameraModel(config)
        metadata = camera.get_camera_metadata()
        
        # Check all required fields are present
        assert 'resolution' in metadata
        assert 'focal_px' in metadata
        assert 'principal_point' in metadata
        assert 'yaw_deg' in metadata
        assert 'pitch_deg' in metadata
        assert 'lat_deg' in metadata
        assert 'lon_deg' in metadata
        assert 'alt_m_msl' in metadata
        
        # Check values
        assert metadata['resolution'] == [1920, 1080]
        assert metadata['focal_px'] == camera.focal_px
        assert metadata['principal_point'] == [camera.principal_point[0], camera.principal_point[1]]
        assert metadata['yaw_deg'] == 90.0
        assert metadata['pitch_deg'] == 10.0
        assert metadata['lat_deg'] == 13.736717
        assert metadata['lon_deg'] == 100.523186
        assert metadata['alt_m_msl'] == 1.5
    
    def test_world_to_pixel_projection_basic(self):
        """Test basic world to pixel projection."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            camera_yaw_deg=0.0,  # Face north for predictable testing
            camera_pitch_deg=0.0  # Level camera for predictable testing
        )
        
        camera = CameraModel(config)
        
        # Test point in front of camera
        world_pos = np.array([0.0, 10.0, 0.0])  # 10m north, at camera altitude
        result = camera.project_world_to_pixels(world_pos)
        
        assert 'pixel_coords' in result
        assert 'depth' in result
        assert 'in_view' in result
        
        # Should project to center of image (no lateral offset)
        u, v = result['pixel_coords']
        assert abs(u - camera.principal_point[0]) < 1e-6
        assert abs(v - camera.principal_point[1]) < 1e-6
        assert result['depth'] == 10.0
        assert result['in_view'] == True
    
    def test_world_to_pixel_projection_behind_camera(self):
        """Test projection of point behind camera."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            camera_yaw_deg=0.0,  # Face north for predictable testing
            camera_pitch_deg=0.0  # Level camera for predictable testing
        )
        
        camera = CameraModel(config)
        
        # Test point behind camera
        world_pos = np.array([0.0, -10.0, 0.0])  # 10m south (behind camera)
        result = camera.project_world_to_pixels(world_pos)
        
        assert result['in_view'] == False
        assert result['depth'] < 0  # Behind camera has negative depth
    
    def test_world_to_pixel_projection_lateral_offset(self):
        """Test projection with lateral offset."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            camera_yaw_deg=0.0,  # Face north for predictable testing
            camera_pitch_deg=0.0  # Level camera for predictable testing
        )
        
        camera = CameraModel(config)
        
        # Test point to the east (right in camera frame)
        world_pos = np.array([5.0, 10.0, 0.0])  # 5m east, 10m north
        result = camera.project_world_to_pixels(world_pos)
        
        u, v = result['pixel_coords']
        # Should be to the right of center
        assert u > camera.principal_point[0]
        assert result['in_view'] == True
        assert result['depth'] == 10.0
    
    def test_world_to_pixel_projection_out_of_bounds(self):
        """Test projection that falls outside image bounds."""
        config = SimulatorConfig(
            image_width_px=100,  # Small image for easier out-of-bounds testing
            image_height_px=100,
            vertical_fov_deg=30.0,  # Narrow FOV
            camera_yaw_deg=0.0,  # Face north for predictable testing
            camera_pitch_deg=0.0  # Level camera for predictable testing
        )
        
        camera = CameraModel(config)
        
        # Test point far to the side
        world_pos = np.array([50.0, 10.0, 0.0])  # Very far east
        result = camera.project_world_to_pixels(world_pos)
        
        # Should be out of view
        assert result['in_view'] == False
    
    def test_focal_length_consistency_between_methods(self):
        """Test that both focal length methods give consistent results for equivalent parameters."""
        # Set up equivalent configurations
        # If focal_mm=25, sensor_height_mm=15, image_height=1080
        # Then focal_px = 25 * 1080 / 15 = 1800
        
        # Now find equivalent FOV: vfov = 2 * atan(image_height / (2 * focal_px))
        focal_px_target = 1800.0
        equivalent_vfov_rad = 2.0 * math.atan(1080 / (2.0 * focal_px_target))
        equivalent_vfov_deg = math.degrees(equivalent_vfov_rad)
        
        config1 = SimulatorConfig(
            image_height_px=1080,
            focal_length_mm=25.0,
            sensor_height_mm=15.0,
            vertical_fov_deg=None
        )
        
        config2 = SimulatorConfig(
            image_height_px=1080,
            focal_length_mm=None,
            sensor_height_mm=None,
            vertical_fov_deg=equivalent_vfov_deg
        )
        
        camera1 = CameraModel(config1)
        camera2 = CameraModel(config2)
        
        # Should give very similar focal lengths
        assert abs(camera1.focal_px - camera2.focal_px) < 1e-6
    
    def test_camera_model_with_different_image_sizes(self):
        """Test camera model with various image dimensions."""
        test_cases = [
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (3840, 2160)
        ]
        
        for width, height in test_cases:
            config = SimulatorConfig(
                image_width_px=width,
                image_height_px=height,
                vertical_fov_deg=60.0
            )
            
            camera = CameraModel(config)
            
            # Focal length should scale with image height
            expected_focal_px = height / (2.0 * math.tan(math.radians(60.0) / 2.0))
            assert abs(camera.focal_px - expected_focal_px) < 1e-6
            
            # Principal point should be at center
            assert camera.principal_point == (width / 2.0, height / 2.0)
            
            # Metadata should reflect correct dimensions
            metadata = camera.get_camera_metadata()
            assert metadata['resolution'] == [width, height]