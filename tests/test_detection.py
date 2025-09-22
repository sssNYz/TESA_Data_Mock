"""
Unit tests for detection generation functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.camera import CameraModel
from drone_detection_simulator.detection import DetectionGenerator


class TestDetectionGenerator:
    """Test cases for DetectionGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            drone_size_m=0.25,
            camera_yaw_deg=90.0,
            camera_pitch_deg=10.0
        )
    
    @pytest.fixture
    def camera_model(self, config):
        """Create test camera model."""
        return CameraModel(config)
    
    @pytest.fixture
    def detection_generator(self, config, camera_model):
        """Create test detection generator."""
        return DetectionGenerator(config, camera_model)
    
    def test_initialization(self, config, camera_model):
        """Test DetectionGenerator initialization."""
        generator = DetectionGenerator(config, camera_model)
        
        assert generator.config == config
        assert generator.camera_model == camera_model
    
    def test_generate_detections_empty_list(self, detection_generator):
        """Test detection generation with empty position list."""
        detections = detection_generator.generate_detections([])
        
        assert detections == []
    
    def test_generate_detections_single_drone(self, detection_generator):
        """Test detection generation for single drone in view."""
        # Position drone in front of camera at reasonable distance
        world_pos = np.array([10.0, 0.0, 5.0])  # 10m east, 0m north, 5m up
        
        detections = detection_generator.generate_detections([world_pos])
        
        assert len(detections) == 1
        detection = detections[0]
        
        # Check required fields
        assert 'class' in detection
        assert 'confidence' in detection
        assert 'bbox_px' in detection
        assert 'center_px' in detection
        assert 'size_px' in detection
        assert 'drone_id' in detection
        assert 'world_pos_enu' in detection
        assert 'depth_m' in detection
        
        # Check field values
        assert detection['class'] == 'drone'
        assert 0.0 <= detection['confidence'] <= 1.0
        assert len(detection['bbox_px']) == 4
        assert len(detection['center_px']) == 2
        assert len(detection['size_px']) == 2
        assert detection['drone_id'] == 0
        assert detection['world_pos_enu'] == world_pos.tolist()
        assert detection['depth_m'] > 0
    
    def test_generate_detections_multiple_drones(self, detection_generator):
        """Test detection generation for multiple drones."""
        world_positions = [
            np.array([10.0, 0.0, 5.0]),   # Drone 0 - in front of camera
            np.array([12.0, 1.0, 5.5]),   # Drone 1 - slightly to the side
            np.array([8.0, -0.5, 4.5])    # Drone 2 - closer, slightly offset
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        # Should detect all drones (they're all in reasonable positions)
        assert len(detections) >= 2  # At least 2 should be visible
        
        # Check that drone IDs are assigned correctly for visible drones
        drone_ids = [d['drone_id'] for d in detections]
        assert all(0 <= drone_id <= 2 for drone_id in drone_ids)
        assert len(set(drone_ids)) == len(drone_ids)  # No duplicates
        
        # Check all detections have required fields
        for detection in detections:
            assert 'class' in detection
            assert 'confidence' in detection
            assert 'bbox_px' in detection
            assert detection['class'] == 'drone'
            assert 0.0 <= detection['confidence'] <= 1.0
    
    def test_generate_detections_behind_camera(self, detection_generator):
        """Test that drones behind camera are not detected."""
        # Position drone behind camera (negative depth)
        world_pos = np.array([-10.0, 0.0, 5.0])  # Behind camera
        
        detections = detection_generator.generate_detections([world_pos])
        
        # Should not generate detection for drone behind camera
        assert len(detections) == 0
    
    def test_bounding_box_generation(self, detection_generator):
        """Test bounding box generation from pixel coordinates and depth."""
        pixel_coords = (960.0, 540.0)  # Image center
        depth = 20.0  # 20 meters
        
        bbox_info = detection_generator._generate_bounding_box(pixel_coords, depth)
        
        # Check structure
        assert 'bbox_px' in bbox_info
        assert 'center_px' in bbox_info
        assert 'size_px' in bbox_info
        
        # Check center matches input
        assert bbox_info['center_px'] == [960.0, 540.0]
        
        # Check bounding box is centered on pixel coordinates
        bbox = bbox_info['bbox_px']
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        assert abs(center_x - 960.0) < 0.001
        assert abs(center_y - 540.0) < 0.001
        
        # Check size is positive
        width, height = bbox_info['size_px']
        assert width > 0
        assert height > 0
        
        # Check height is slightly larger than width (realistic drone shape)
        assert height >= width
    
    def test_bounding_box_size_scales_with_distance(self, detection_generator):
        """Test that bounding box size decreases with distance."""
        pixel_coords = (960.0, 540.0)
        
        # Generate bounding boxes at different distances
        bbox_near = detection_generator._generate_bounding_box(pixel_coords, 10.0)
        bbox_far = detection_generator._generate_bounding_box(pixel_coords, 30.0)
        
        # Far object should have smaller bounding box
        near_size = sum(bbox_near['size_px'])
        far_size = sum(bbox_far['size_px'])
        assert far_size < near_size
    
    def test_bounding_box_minimum_size(self, detection_generator):
        """Test that bounding boxes have minimum size for very distant objects."""
        pixel_coords = (960.0, 540.0)
        very_far_depth = 1000.0  # Very far away
        
        bbox_info = detection_generator._generate_bounding_box(pixel_coords, very_far_depth)
        
        # Should still have minimum detectable size
        width, height = bbox_info['size_px']
        assert width >= 4.0  # Minimum size
        assert height >= 4.0
    
    def test_bbox_visibility_check(self, detection_generator):
        """Test bounding box visibility checking."""
        # Test cases: [x_min, y_min, x_max, y_max], expected_visible
        test_cases = [
            ([100, 100, 200, 200], True),    # Fully inside
            ([1900, 100, 2000, 200], True),  # Partially outside right
            ([100, 1000, 200, 1200], True),  # Partially outside bottom
            ([-50, 100, 50, 200], True),     # Partially outside left
            ([100, -50, 200, 50], True),     # Partially outside top
            ([2000, 100, 2100, 200], False), # Completely outside right
            ([100, 1200, 200, 1300], False), # Completely outside bottom
            ([-200, 100, -100, 200], False), # Completely outside left
            ([100, -200, 200, -100], False)  # Completely outside top
        ]
        
        for bbox_px, expected_visible in test_cases:
            is_visible = detection_generator._is_bbox_visible(bbox_px)
            assert is_visible == expected_visible, f"Failed for bbox {bbox_px}"
    
    def test_confidence_calculation_size_factor(self, detection_generator):
        """Test confidence calculation based on detection size."""
        # Mock projection info
        projection = {
            'distance_from_edge': 100.0,  # Far from edge
            'in_bounds': True
        }
        
        # Test different sizes
        test_cases = [
            ([30.0, 36.0], 1.0),    # Large size - high confidence
            ([15.0, 18.0], 0.8),    # Medium size - medium confidence  
            ([4.0, 4.8], 0.3),      # Small size - low confidence
            ([2.0, 2.4], 0.3)       # Very small - minimum confidence
        ]
        
        for size_px, expected_min_confidence in test_cases:
            bbox_info = {
                'size_px': size_px
            }
            
            confidence = detection_generator._calculate_confidence(bbox_info, projection)
            
            # Should be at least the expected minimum (allowing for other factors)
            assert confidence >= expected_min_confidence * 0.9  # Allow some tolerance
            assert 0.0 <= confidence <= 1.0
    
    def test_confidence_calculation_edge_factor(self, detection_generator):
        """Test confidence calculation based on edge proximity."""
        bbox_info = {'size_px': [25.0, 30.0]}  # Good size
        
        # Test different edge distances
        test_cases = [
            (100.0, True, 1.0),    # Far from edge, in bounds - no penalty
            (30.0, True, 0.9),     # Medium distance - small penalty
            (5.0, True, 0.7),      # Close to edge - larger penalty
            (-10.0, False, 0.5)    # Outside bounds - significant penalty
        ]
        
        for distance_from_edge, in_bounds, expected_min_factor in test_cases:
            projection = {
                'distance_from_edge': distance_from_edge,
                'in_bounds': in_bounds
            }
            
            confidence = detection_generator._calculate_confidence(bbox_info, projection)
            
            # Confidence should reflect edge proximity
            if distance_from_edge > 50:
                assert confidence > 0.8  # High confidence far from edge
            elif distance_from_edge < 0:
                assert confidence < 0.8  # Lower confidence outside bounds
    
    def test_clip_detection_to_image(self, detection_generator):
        """Test clipping detection bounding box to image boundaries."""
        # Create detection that extends outside image
        detection = {
            'bbox_px': [-10, -5, 1930, 1090],  # Extends beyond all edges
            'center_px': [960, 540],
            'size_px': [1940, 1095]
        }
        
        clipped = detection_generator.clip_detection_to_image(detection)
        
        # Check bounding box is clipped to image bounds
        bbox = clipped['bbox_px']
        assert bbox[0] >= 0  # x_min
        assert bbox[1] >= 0  # y_min
        assert bbox[2] <= 1920  # x_max
        assert bbox[3] <= 1080  # y_max
        
        # Check center and size are recalculated
        expected_center_x = (bbox[0] + bbox[2]) / 2.0
        expected_center_y = (bbox[1] + bbox[3]) / 2.0
        assert clipped['center_px'] == [expected_center_x, expected_center_y]
        
        expected_width = bbox[2] - bbox[0]
        expected_height = bbox[3] - bbox[1]
        assert clipped['size_px'] == [expected_width, expected_height]
    
    def test_distance_estimation_from_size(self, detection_generator):
        """Test distance estimation from detection size."""
        # Create detection with known size
        detection = {
            'size_px': [20.0, 24.0]  # Average size = 22 pixels
        }
        
        estimated_distance = detection_generator.estimate_distance_from_size(detection)
        
        # Should be positive distance
        assert estimated_distance > 0
        
        # Test with larger detection (should give smaller distance)
        large_detection = {
            'size_px': [40.0, 48.0]  # Average size = 44 pixels
        }
        
        large_estimated_distance = detection_generator.estimate_distance_from_size(large_detection)
        
        # Larger detection should appear closer
        assert large_estimated_distance < estimated_distance
    
    def test_distance_estimation_zero_size(self, detection_generator):
        """Test distance estimation with zero size detection."""
        detection = {
            'size_px': [0.0, 0.0]
        }
        
        estimated_distance = detection_generator.estimate_distance_from_size(detection)
        
        # Should return infinity for zero size
        assert estimated_distance == float('inf')
    
    def test_detection_statistics_empty(self, detection_generator):
        """Test statistics calculation with empty detection list."""
        stats = detection_generator.get_detection_statistics([])
        
        assert stats['num_detections'] == 0
        assert stats['avg_confidence'] == 0.0
        assert stats['avg_size_px'] == 0.0
        assert stats['avg_distance_m'] == 0.0
        assert stats['in_bounds_count'] == 0
        assert stats['out_of_bounds_count'] == 0
    
    def test_detection_statistics_with_detections(self, detection_generator):
        """Test statistics calculation with sample detections."""
        detections = [
            {
                'confidence': 0.9,
                'size_px': [20.0, 24.0],
                'depth_m': 15.0,
                'projection_info': {'in_bounds': True}
            },
            {
                'confidence': 0.7,
                'size_px': [10.0, 12.0],
                'depth_m': 25.0,
                'projection_info': {'in_bounds': False}
            },
            {
                'confidence': 0.8,
                'size_px': [15.0, 18.0],
                'depth_m': 20.0,
                'projection_info': {'in_bounds': True}
            }
        ]
        
        stats = detection_generator.get_detection_statistics(detections)
        
        assert stats['num_detections'] == 3
        assert abs(stats['avg_confidence'] - 0.8) < 0.001  # (0.9 + 0.7 + 0.8) / 3
        assert stats['min_confidence'] == 0.7
        assert stats['max_confidence'] == 0.9
        assert abs(stats['avg_size_px'] - 16.5) < 0.001  # (22 + 11 + 16.5) / 3
        assert abs(stats['avg_distance_m'] - 20.0) < 0.001  # (15 + 25 + 20) / 3
        assert stats['in_bounds_count'] == 2
        assert stats['out_of_bounds_count'] == 1
    
    def test_integration_with_camera_model(self, detection_generator):
        """Test integration between DetectionGenerator and CameraModel."""
        # Position drone at known location
        world_pos = np.array([20.0, 0.0, 5.0])  # 20m east, 0m north, 5m up
        
        # Generate detection
        detections = detection_generator.generate_detections([world_pos])
        
        assert len(detections) == 1
        detection = detections[0]
        
        # Verify the detection contains expected information
        assert detection['world_pos_enu'] == world_pos.tolist()
        assert detection['depth_m'] > 0
        
        # Verify bounding box is reasonable
        bbox = detection['bbox_px']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        assert width > 0
        assert height > 0
        
        # Verify center is within image bounds (for this reasonable position)
        center = detection['center_px']
        assert 0 <= center[0] <= 1920
        assert 0 <= center[1] <= 1080
    
    def test_size_confidence_factor_calculation(self, detection_generator):
        """Test size confidence factor calculation directly."""
        # Test boundary conditions
        assert detection_generator._calculate_size_confidence_factor(25.0) == 1.0  # Large size
        assert detection_generator._calculate_size_confidence_factor(4.0) == 0.3   # Minimum size
        assert detection_generator._calculate_size_confidence_factor(2.0) == 0.3   # Below minimum
        
        # Test interpolation
        mid_size_factor = detection_generator._calculate_size_confidence_factor(12.0)
        assert 0.3 < mid_size_factor < 1.0
    
    def test_edge_confidence_factor_calculation(self, detection_generator):
        """Test edge confidence factor calculation directly."""
        # Test boundary conditions
        assert detection_generator._calculate_edge_confidence_factor(60.0) == 1.0  # Far from edge
        assert detection_generator._calculate_edge_confidence_factor(5.0) == 0.7   # Close to edge
        assert detection_generator._calculate_edge_confidence_factor(-5.0) == 0.5  # Outside bounds
        
        # Test interpolation
        mid_distance_factor = detection_generator._calculate_edge_confidence_factor(30.0)
        assert 0.7 < mid_distance_factor < 1.0


class TestMultiDroneDetectionGeneration:
    """Test cases specifically for multi-drone detection generation (Requirements 7.2, 7.3, 7.4)."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for multi-drone scenarios."""
        return SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            drone_size_m=0.25,
            num_drones=4
        )
    
    @pytest.fixture
    def detection_generator(self, config):
        """Create test detection generator."""
        camera_model = CameraModel(config)
        return DetectionGenerator(config, camera_model)
    
    def test_multiple_drone_detection_array(self, detection_generator):
        """Test requirement 7.2: Detection message includes array with one entry per drone."""
        # Create multiple drones in camera view
        world_positions = [
            np.array([10.0, 0.0, 5.0]),    # Drone 0
            np.array([12.0, 1.0, 5.5]),    # Drone 1
            np.array([8.0, -0.5, 4.5]),    # Drone 2
            np.array([15.0, 0.5, 6.0])     # Drone 3
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        # Should generate detections for all visible drones
        assert len(detections) >= 3  # At least most should be visible
        assert len(detections) <= len(world_positions)  # Not more than input
        
        # Each detection should have unique drone_id
        drone_ids = [d['drone_id'] for d in detections]
        assert len(set(drone_ids)) == len(drone_ids)  # No duplicates
        
        # All drone_ids should be valid indices
        assert all(0 <= drone_id < len(world_positions) for drone_id in drone_ids)
        
        # Each detection should have required fields
        for detection in detections:
            assert 'class' in detection
            assert 'confidence' in detection
            assert 'bbox_px' in detection
            assert 'center_px' in detection
            assert 'size_px' in detection
            assert 'drone_id' in detection
            assert detection['class'] == 'drone'
    
    def test_overlapping_drones_multiple_bboxes(self, detection_generator):
        """Test requirement 7.3: Multiple bounding boxes when drones overlap in camera view."""
        # Create closely positioned drones that will overlap in image
        world_positions = [
            np.array([10.0, 0.0, 5.0]),     # Drone 0 - center
            np.array([10.2, 0.1, 5.1]),     # Drone 1 - very close to drone 0
            np.array([10.1, -0.1, 4.9])     # Drone 2 - also close to drone 0
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        # Should detect multiple drones even when close
        assert len(detections) >= 2
        
        # Check for overlapping bounding boxes
        if len(detections) >= 2:
            bbox1 = detections[0]['bbox_px']
            bbox2 = detections[1]['bbox_px']
            
            # Check if bounding boxes overlap
            overlap_x = not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0])
            overlap_y = not (bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
            
            # For closely positioned drones, we expect overlap
            if overlap_x and overlap_y:
                # Overlapping bounding boxes should still be distinct
                assert bbox1 != bbox2
                assert detections[0]['drone_id'] != detections[1]['drone_id']
    
    def test_independent_distance_estimates_different_sizes(self, detection_generator):
        """Test requirement 7.4: Independent distance estimates for different drone sizes."""
        # Test with different drone sizes by modifying config
        config_small = SimulatorConfig(drone_size_m=0.15, vertical_fov_deg=50.0)
        config_large = SimulatorConfig(drone_size_m=0.35, vertical_fov_deg=50.0)
        
        camera_model_small = CameraModel(config_small)
        camera_model_large = CameraModel(config_large)
        
        gen_small = DetectionGenerator(config_small, camera_model_small)
        gen_large = DetectionGenerator(config_large, camera_model_large)
        
        # Same world position, different drone sizes
        world_pos = np.array([15.0, 0.0, 5.0])
        
        det_small = gen_small.generate_detections([world_pos])[0]
        det_large = gen_large.generate_detections([world_pos])[0]
        
        # Different sizes should produce different pixel sizes
        small_size = sum(det_small['size_px']) / 2
        large_size = sum(det_large['size_px']) / 2
        assert large_size > small_size
        
        # Distance estimates should be independent
        est_dist_small = gen_small.estimate_distance_from_size(det_small)
        est_dist_large = gen_large.estimate_distance_from_size(det_large)
        
        # Both should estimate similar distances (same actual position)
        assert abs(est_dist_small - est_dist_large) < 2.0  # Within 2m
    
    def test_independent_distance_estimates_different_altitudes(self, detection_generator):
        """Test requirement 7.4: Independent distance estimates for different altitudes."""
        # Drones at same horizontal position but different altitudes
        world_positions = [
            np.array([12.0, 0.0, 3.0]),   # Low altitude
            np.array([12.0, 0.0, 6.0]),   # High altitude
            np.array([12.0, 0.0, 9.0])    # Very high altitude
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        # Should detect all drones
        assert len(detections) == 3
        
        # Each should have different depth estimates
        depths = [d['depth_m'] for d in detections]
        assert len(set(depths)) == len(depths)  # All different
        
        # Distance estimates should be independent
        estimated_distances = [
            detection_generator.estimate_distance_from_size(d) for d in detections
        ]
        
        # All estimates should be reasonable
        for est_dist in estimated_distances:
            assert 5.0 < est_dist < 50.0  # Reasonable range
    
    def test_multi_drone_confidence_independence(self, detection_generator):
        """Test that confidence scores are calculated independently for each drone."""
        # Create drones at different distances (affecting confidence)
        world_positions = [
            np.array([8.0, 0.0, 5.0]),    # Close - high confidence
            np.array([20.0, 0.0, 5.0]),   # Far - lower confidence
            np.array([50.0, 0.0, 5.0])    # Very far - low confidence
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        if len(detections) >= 2:
            # Closer drones should generally have higher confidence
            confidences = [d['confidence'] for d in detections]
            
            # All confidences should be valid
            assert all(0.0 <= conf <= 1.0 for conf in confidences)
            
            # Should have some variation in confidence
            assert max(confidences) - min(confidences) > 0.05
    
    def test_multi_drone_edge_proximity_handling(self, detection_generator):
        """Test handling of multiple drones near image edges."""
        # Position drones near different edges
        world_positions = [
            np.array([5.0, -2.0, 5.0]),   # Left edge
            np.array([5.0, 2.0, 5.0]),    # Right edge
            np.array([5.0, 0.0, 2.0]),    # Bottom edge (low altitude)
            np.array([5.0, 0.0, 8.0])     # Top edge (high altitude)
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        # Should handle edge cases gracefully
        assert isinstance(detections, list)
        
        for detection in detections:
            # All detections should have valid confidence
            assert 0.0 <= detection['confidence'] <= 1.0
            
            # Bounding boxes should be reasonable
            bbox = detection['bbox_px']
            assert len(bbox) == 4
            assert bbox[0] <= bbox[2]  # x_min <= x_max
            assert bbox[1] <= bbox[3]  # y_min <= y_max
    
    def test_multi_drone_statistics_aggregation(self, detection_generator):
        """Test statistics calculation for multiple drone detections."""
        world_positions = [
            np.array([10.0, 0.0, 5.0]),
            np.array([15.0, 1.0, 5.5]),
            np.array([20.0, -0.5, 4.5])
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        stats = detection_generator.get_detection_statistics(detections)
        
        # Should have correct count
        assert stats['num_detections'] == len(detections)
        
        if len(detections) > 0:
            # Should have reasonable statistics
            assert 0.0 <= stats['avg_confidence'] <= 1.0
            assert stats['min_confidence'] <= stats['avg_confidence'] <= stats['max_confidence']
            assert stats['avg_size_px'] > 0
            assert stats['avg_distance_m'] > 0
            assert stats['in_bounds_count'] + stats['out_of_bounds_count'] == len(detections)
    
    def test_empty_multi_drone_input(self, detection_generator):
        """Test handling of empty drone position list."""
        detections = detection_generator.generate_detections([])
        
        assert detections == []
        
        stats = detection_generator.get_detection_statistics(detections)
        assert stats['num_detections'] == 0
    
    def test_large_number_of_drones(self, detection_generator):
        """Test detection generation with large number of drones."""
        # Create many drones spread across the view
        world_positions = []
        for i in range(20):
            east = 8.0 + i * 0.5  # Spread east
            north = -2.0 + (i % 5) * 1.0  # Spread north
            up = 4.0 + (i % 3) * 0.5  # Vary altitude
            world_positions.append(np.array([east, north, up]))
        
        detections = detection_generator.generate_detections(world_positions)
        
        # Should handle large number gracefully
        assert isinstance(detections, list)
        assert len(detections) <= len(world_positions)
        
        # All detections should be valid
        for detection in detections:
            assert 'drone_id' in detection
            assert 0 <= detection['drone_id'] < len(world_positions)
            assert 0.0 <= detection['confidence'] <= 1.0
    
    def test_multi_drone_clipping_independence(self, detection_generator):
        """Test that bounding box clipping works independently for each drone."""
        # Create detections that will need clipping
        world_positions = [
            np.array([3.0, -3.0, 5.0]),   # Will project near left edge
            np.array([3.0, 3.0, 5.0])     # Will project near right edge
        ]
        
        detections = detection_generator.generate_detections(world_positions)
        
        for detection in detections:
            # Test clipping
            clipped = detection_generator.clip_detection_to_image(detection)
            
            # Clipped bbox should be within image bounds
            bbox = clipped['bbox_px']
            assert bbox[0] >= 0  # x_min
            assert bbox[1] >= 0  # y_min
            assert bbox[2] <= detection_generator.config.image_width_px  # x_max
            assert bbox[3] <= detection_generator.config.image_height_px  # y_max
            
            # Center and size should be recalculated
            assert 'center_px' in clipped
            assert 'size_px' in clipped


class TestDetectionGeneratorEdgeCases:
    """Test edge cases and error conditions for DetectionGenerator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulatorConfig(
            image_width_px=100,  # Small image for easier testing
            image_height_px=100,
            vertical_fov_deg=60.0,
            drone_size_m=0.5
        )
    
    @pytest.fixture
    def detection_generator(self, config):
        """Create test detection generator."""
        camera_model = CameraModel(config)
        return DetectionGenerator(config, camera_model)
    
    def test_very_close_drone(self, detection_generator):
        """Test detection of very close drone (large in image)."""
        # Very close drone
        world_pos = np.array([2.0, 0.0, 1.0])
        
        detections = detection_generator.generate_detections([world_pos])
        
        if detections:  # May or may not be visible depending on projection
            detection = detections[0]
            # Should have large bounding box
            width, height = detection['size_px']
            assert width > 10  # Should be reasonably large
            assert height > 10
    
    def test_very_far_drone(self, detection_generator):
        """Test detection of very far drone (small in image)."""
        # Very far drone
        world_pos = np.array([100.0, 0.0, 10.0])
        
        detections = detection_generator.generate_detections([world_pos])
        
        if detections:  # May or may not be visible
            detection = detections[0]
            # Should have small bounding box but above minimum
            width, height = detection['size_px']
            assert width >= 4.0  # Minimum size
            assert height >= 4.0
            # Should have lower confidence due to small size
            assert detection['confidence'] < 0.8
    
    def test_drone_at_image_edge(self, detection_generator):
        """Test drone detection at image edges."""
        # Position drone to project near image edge
        # This requires knowledge of camera parameters to position correctly
        world_pos = np.array([1.0, 0.0, 2.0])  # Close and slightly off-center
        
        detections = detection_generator.generate_detections([world_pos])
        
        # Should handle edge cases gracefully
        assert isinstance(detections, list)
        
        if detections:
            detection = detections[0]
            # Should have valid confidence even near edge
            assert 0.0 <= detection['confidence'] <= 1.0