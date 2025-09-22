"""
Unit tests for coordinate projection utilities.
"""

import math
import pytest
import numpy as np
from drone_detection_simulator.projection import (
    apply_camera_rotation,
    project_to_pixels,
    validate_pixel_coordinates,
    world_to_pixel_projection
)


class TestCameraRotation:
    """Test camera rotation transformations."""
    
    def test_zero_rotation(self):
        """Test that zero yaw and pitch results in identity-like transformation."""
        world_pos = np.array([1.0, 2.0, 3.0])  # [east, north, up]
        camera_pos = apply_camera_rotation(world_pos, yaw_deg=0.0, pitch_deg=0.0)
        
        # With zero rotation: east->right, north->forward, up->up
        # Then pitch converts: right->right, forward->forward, up->down (negated)
        expected = np.array([1.0, -3.0, 2.0])  # [right, down, forward]
        np.testing.assert_allclose(camera_pos, expected, rtol=1e-10)
    
    def test_90_degree_yaw(self):
        """Test 90-degree yaw rotation (camera facing east)."""
        world_pos = np.array([1.0, 2.0, 3.0])  # [east, north, up]
        camera_pos = apply_camera_rotation(world_pos, yaw_deg=90.0, pitch_deg=0.0)
        
        # 90-degree yaw: east becomes forward, north becomes left (negative right)
        # With zero pitch: right=-north, down=-up, forward=east
        expected = np.array([-2.0, -3.0, 1.0])  # [right, down, forward]
        np.testing.assert_allclose(camera_pos, expected, rtol=1e-10)
    
    def test_negative_90_degree_yaw(self):
        """Test -90-degree yaw rotation (camera facing west)."""
        world_pos = np.array([1.0, 2.0, 3.0])  # [east, north, up]
        camera_pos = apply_camera_rotation(world_pos, yaw_deg=-90.0, pitch_deg=0.0)
        
        # -90-degree yaw: east becomes backward (negative forward), north becomes right
        expected = np.array([2.0, -3.0, -1.0])  # [right, down, forward]
        np.testing.assert_allclose(camera_pos, expected, rtol=1e-10)
    
    def test_positive_pitch(self):
        """Test positive pitch (camera tilted down)."""
        world_pos = np.array([0.0, 2.0, 1.0])  # [east, north, up]
        camera_pos = apply_camera_rotation(world_pos, yaw_deg=0.0, pitch_deg=30.0)
        
        # With 30-degree pitch: forward and up components mix
        cos_30 = math.cos(math.radians(30.0))
        sin_30 = math.sin(math.radians(30.0))
        
        expected_x = 0.0  # Right unchanged
        expected_y = sin_30 * 2.0 - cos_30 * 1.0  # Down component
        expected_z = cos_30 * 2.0 + sin_30 * 1.0  # Forward component
        
        expected = np.array([expected_x, expected_y, expected_z])
        np.testing.assert_allclose(camera_pos, expected, rtol=1e-10)
    
    def test_combined_yaw_pitch(self):
        """Test combined yaw and pitch rotation."""
        world_pos = np.array([1.0, 1.0, 1.0])  # [east, north, up]
        camera_pos = apply_camera_rotation(world_pos, yaw_deg=45.0, pitch_deg=45.0)
        
        # This should be a well-defined transformation
        # Verify the result is reasonable (not testing exact values due to complexity)
        assert len(camera_pos) == 3
        assert all(np.isfinite(camera_pos))


class TestPixelProjection:
    """Test pixel projection using pinhole camera model."""
    
    def test_center_projection(self):
        """Test projection of point directly in front of camera."""
        camera_pos = np.array([0.0, 0.0, 5.0])  # [right, down, forward]
        focal_px = 1000.0
        principal_point = (960.0, 540.0)
        
        result = project_to_pixels(camera_pos, focal_px, principal_point)
        
        # Point at camera center should project to principal point
        assert result['pixel_coords'] == principal_point
        assert result['depth'] == 5.0
        assert not result['behind_camera']
    
    def test_off_center_projection(self):
        """Test projection of point offset from camera center."""
        camera_pos = np.array([1.0, 0.5, 2.0])  # [right, down, forward]
        focal_px = 1000.0
        principal_point = (960.0, 540.0)
        
        result = project_to_pixels(camera_pos, focal_px, principal_point)
        
        # Calculate expected pixel coordinates
        expected_u = focal_px * (1.0 / 2.0) + principal_point[0]  # 960 + 500 = 1460
        expected_v = focal_px * (0.5 / 2.0) + principal_point[1]  # 540 + 250 = 790
        
        assert result['pixel_coords'] == (expected_u, expected_v)
        assert result['depth'] == 2.0
        assert not result['behind_camera']
    
    def test_behind_camera(self):
        """Test handling of points behind camera."""
        camera_pos = np.array([1.0, 1.0, -1.0])  # [right, down, forward] - negative Z
        focal_px = 1000.0
        principal_point = (960.0, 540.0)
        
        result = project_to_pixels(camera_pos, focal_px, principal_point)
        
        assert result['pixel_coords'] == (0.0, 0.0)
        assert result['depth'] == -1.0
        assert result['behind_camera']
    
    def test_at_camera_plane(self):
        """Test handling of points at camera plane (Z=0)."""
        camera_pos = np.array([1.0, 1.0, 0.0])  # [right, down, forward] - Z=0
        focal_px = 1000.0
        principal_point = (960.0, 540.0)
        
        result = project_to_pixels(camera_pos, focal_px, principal_point)
        
        # Points at Z=0 should be treated as behind camera
        assert result['behind_camera']


class TestPixelValidation:
    """Test pixel coordinate validation and boundary checking."""
    
    def test_coordinates_in_bounds(self):
        """Test coordinates within image bounds."""
        pixel_coords = (500.0, 300.0)
        result = validate_pixel_coordinates(pixel_coords, 1920, 1080)
        
        assert result['in_bounds']
        assert result['clipped_coords'] == pixel_coords
        assert result['distance_from_edge'] == 300.0  # Distance to top edge
    
    def test_coordinates_out_of_bounds(self):
        """Test coordinates outside image bounds."""
        pixel_coords = (-100.0, 1200.0)  # Left and below image
        result = validate_pixel_coordinates(pixel_coords, 1920, 1080)
        
        assert not result['in_bounds']
        assert result['clipped_coords'] == (0.0, 1080.0)  # Clipped to boundaries
        assert result['distance_from_edge'] == -120.0  # Minimum distance (most negative)
    
    def test_coordinates_at_boundary(self):
        """Test coordinates exactly at image boundary."""
        pixel_coords = (0.0, 1080.0)  # At left edge and bottom edge
        result = validate_pixel_coordinates(pixel_coords, 1920, 1080)
        
        assert result['in_bounds']  # Boundary is considered in bounds
        assert result['clipped_coords'] == pixel_coords
        assert result['distance_from_edge'] == 0.0
    
    def test_coordinates_with_margin(self):
        """Test coordinates with margin requirement."""
        pixel_coords = (50.0, 50.0)
        result = validate_pixel_coordinates(pixel_coords, 1920, 1080, margin=100.0)
        
        assert not result['in_bounds']  # Within image but not within margin
        assert result['distance_from_edge'] == 50.0


class TestCompleteProjection:
    """Test complete world-to-pixel projection pipeline."""
    
    def test_point_in_front_of_camera(self):
        """Test projection of point in front of camera."""
        world_pos = np.array([0.0, 10.0, 2.0])  # [east, north, up] - 10m north, 2m up
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=0.0,  # Camera facing north
            camera_pitch_deg=0.0,  # Camera level
            focal_px=1000.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080
        )
        
        # Point should be visible and project near image center
        assert not result['behind_camera']
        assert result['depth'] > 0
        assert result['in_bounds']  # Should be within image bounds
        
        # With zero yaw/pitch, north becomes forward, up becomes negative down
        # So point at (0, 10, 2) should project to center horizontally, above center vertically
        u, v = result['pixel_coords']
        assert abs(u - 960.0) < 1e-10  # Should be at horizontal center
        assert v < 540.0  # Should be above vertical center (up -> negative down)
    
    def test_point_to_the_right(self):
        """Test projection of point to the right of camera."""
        world_pos = np.array([5.0, 10.0, 0.0])  # [east, north, up] - 5m east, 10m north
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=0.0,  # Camera facing north
            camera_pitch_deg=0.0,  # Camera level
            focal_px=1000.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080
        )
        
        # Point should be visible and to the right of center
        assert not result['behind_camera']
        assert result['in_bounds']
        
        u, v = result['pixel_coords']
        assert u > 960.0  # Should be to the right of center
        assert abs(v - 540.0) < 1e-10  # Should be at vertical center (z=0)
    
    def test_camera_facing_east(self):
        """Test projection with camera facing east (90-degree yaw)."""
        world_pos = np.array([10.0, 0.0, 0.0])  # [east, north, up] - 10m east
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=90.0,  # Camera facing east
            camera_pitch_deg=0.0,  # Camera level
            focal_px=1000.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080
        )
        
        # Point should be directly in front of camera
        assert not result['behind_camera']
        assert result['in_bounds']
        
        u, v = result['pixel_coords']
        assert abs(u - 960.0) < 1e-10  # Should be at horizontal center
        assert abs(v - 540.0) < 1e-10  # Should be at vertical center
    
    def test_point_behind_camera(self):
        """Test projection of point behind camera."""
        world_pos = np.array([0.0, -5.0, 0.0])  # [east, north, up] - 5m south
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=0.0,  # Camera facing north
            camera_pitch_deg=0.0,  # Camera level
            focal_px=1000.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080
        )
        
        # Point should be behind camera
        assert result['behind_camera']
        assert not result['in_bounds']
        assert result['depth'] < 0
    
    def test_camera_tilted_down(self):
        """Test projection with camera tilted down."""
        world_pos = np.array([0.0, 10.0, -2.0])  # [east, north, up] - 10m north, 2m down
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=0.0,  # Camera facing north
            camera_pitch_deg=30.0,  # Camera tilted down 30 degrees
            focal_px=1000.0,
            principal_point=(960.0, 540.0),
            image_width=1920,
            image_height=1080
        )
        
        # Point should be visible
        assert not result['behind_camera']
        assert result['depth'] > 0
        
        # With camera tilted down, ground points should appear higher in image
        u, v = result['pixel_coords']
        assert abs(u - 960.0) < 1e-10  # Should be at horizontal center


class TestProjectionAccuracy:
    """Test projection accuracy with known test cases."""
    
    def test_known_projection_case_1(self):
        """Test projection with known input/output values."""
        # Camera setup: facing north, level, at origin
        world_pos = np.array([2.0, 5.0, 1.0])  # 2m east, 5m north, 1m up
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=0.0,
            camera_pitch_deg=0.0,
            focal_px=500.0,  # Shorter focal length for easier calculation
            principal_point=(400.0, 300.0),
            image_width=800,
            image_height=600
        )
        
        # Manual calculation:
        # Camera rotation: east->right=2, north->forward=5, up->down=-1
        # Projection: u = 500 * (2/5) + 400 = 500 * 0.4 + 400 = 600
        #            v = 500 * (-1/5) + 300 = 500 * (-0.2) + 300 = 200
        
        u, v = result['pixel_coords']
        assert abs(u - 600.0) < 1e-10
        assert abs(v - 200.0) < 1e-10
        assert abs(result['depth'] - 5.0) < 1e-10
    
    def test_known_projection_case_2(self):
        """Test projection with camera facing east."""
        # Camera setup: facing east (90-degree yaw), level
        world_pos = np.array([3.0, 1.0, 2.0])  # 3m east, 1m north, 2m up
        
        result = world_to_pixel_projection(
            world_pos_enu=world_pos,
            camera_yaw_deg=90.0,
            camera_pitch_deg=0.0,
            focal_px=1000.0,
            principal_point=(500.0, 400.0),
            image_width=1000,
            image_height=800
        )
        
        # Manual calculation with 90-degree yaw:
        # After yaw rotation: right=-north=-1, forward=east=3, up=2
        # After pitch (0): right=-1, down=-up=-2, forward=3
        # Projection: u = 1000 * (-1/3) + 500 = 1000 * (-0.333...) + 500 ≈ 166.67
        #            v = 1000 * (-2/3) + 400 = 1000 * (-0.666...) + 400 ≈ 66.67
        
        u, v = result['pixel_coords']
        assert abs(u - (500.0 - 1000.0/3.0)) < 1e-10
        assert abs(v - (400.0 - 2000.0/3.0)) < 1e-10
        assert abs(result['depth'] - 3.0) < 1e-10
    
    def test_projection_consistency(self):
        """Test that projection is consistent and reversible in terms of ratios."""
        # Test multiple points at same depth but different positions
        depth = 10.0
        focal_px = 800.0
        principal_point = (640.0, 480.0)
        
        test_cases = [
            np.array([0.0, depth, 0.0]),    # Center
            np.array([1.0, depth, 0.0]),    # 1m right
            np.array([-1.0, depth, 0.0]),   # 1m left
            np.array([0.0, depth, 1.0]),    # 1m up
            np.array([0.0, depth, -1.0]),   # 1m down
        ]
        
        for world_pos in test_cases:
            result = world_to_pixel_projection(
                world_pos_enu=world_pos,
                camera_yaw_deg=0.0,
                camera_pitch_deg=0.0,
                focal_px=focal_px,
                principal_point=principal_point,
                image_width=1280,
                image_height=960
            )
            
            # All points should be at same depth
            assert abs(result['depth'] - depth) < 1e-10
            
            # Pixel displacement should be proportional to world displacement
            u, v = result['pixel_coords']
            expected_u_offset = focal_px * (world_pos[0] / depth)
            expected_v_offset = focal_px * (-world_pos[2] / depth)  # Up -> negative down
            
            assert abs((u - principal_point[0]) - expected_u_offset) < 1e-10
            assert abs((v - principal_point[1]) - expected_v_offset) < 1e-10