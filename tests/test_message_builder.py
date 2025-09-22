"""
Unit tests for the JSON message builder module.

Tests message formatting, schema compliance, field validation, and processing
latency simulation for Pi-realistic detection messages.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import numpy as np

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.camera import CameraModel
from drone_detection_simulator.message_builder import DetectionMessageBuilder


class TestDetectionMessageBuilder:
    """Test cases for DetectionMessageBuilder class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            camera_lat_deg=13.736717,
            camera_lon_deg=100.523186,
            camera_alt_m=1.5,
            camera_yaw_deg=90.0,
            camera_pitch_deg=10.0,
            processing_latency_ms_mean=50.0,
            processing_latency_ms_jitter=20.0,
            false_positive_rate=0.1,
            deterministic_seed=42
        )
    
    @pytest.fixture
    def camera_model(self, config):
        """Create test camera model."""
        return CameraModel(config)
    
    @pytest.fixture
    def rng(self):
        """Create deterministic random number generator."""
        return np.random.default_rng(42)
    
    @pytest.fixture
    def message_builder(self, config, camera_model, rng):
        """Create test message builder."""
        return DetectionMessageBuilder(config, camera_model, rng)
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detection data."""
        return [
            {
                'class': 'drone',
                'confidence': 0.91,
                'bbox_px': [980.5, 520.3, 1020.7, 580.9],
                'center_px': [1000.6, 550.6],
                'size_px': [40.2, 60.6],
                'drone_id': 0,
                'world_pos_enu': [10.0, 5.0, 5.5],
                'depth_m': 12.5,
                'projection_info': {
                    'in_bounds': True,
                    'distance_from_edge': 100.0
                }
            },
            {
                'class': 'drone',
                'confidence': 0.78,
                'bbox_px': [1200.1, 400.2, 1240.8, 450.9],
                'center_px': [1220.45, 425.55],
                'size_px': [40.7, 50.7],
                'drone_id': 1,
                'world_pos_enu': [15.0, 8.0, 6.0],
                'depth_m': 18.2,
                'projection_info': {
                    'in_bounds': True,
                    'distance_from_edge': 75.0
                }
            }
        ]
    
    def test_initialization(self, config, camera_model, rng):
        """Test message builder initialization."""
        builder = DetectionMessageBuilder(config, camera_model, rng)
        
        assert builder.config == config
        assert builder.camera_model == camera_model
        assert builder.rng == rng
        assert builder._frame_counter == 0
        assert builder._camera_metadata is not None
    
    def test_initialization_default_rng(self, config, camera_model):
        """Test message builder initialization with default RNG."""
        builder = DetectionMessageBuilder(config, camera_model)
        
        assert builder.rng is not None
        assert isinstance(builder.rng, np.random.Generator)
    
    def test_build_detection_message_structure(self, message_builder, sample_detections):
        """Test that detection message has correct structure."""
        timestamp = datetime(2025, 9, 21, 8, 23, 12, 123000, timezone.utc)
        message = message_builder.build_detection_message(sample_detections, timestamp)
        
        # Check top-level structure
        assert 'timestamp_utc' in message
        assert 'frame_id' in message
        assert 'camera' in message
        assert 'detections' in message
        assert 'edge' in message
        
        # Check frame ID increments
        assert message['frame_id'] == 1
        
        # Build another message to test frame counter
        message2 = message_builder.build_detection_message(sample_detections, timestamp)
        assert message2['frame_id'] == 2
    
    def test_timestamp_formatting(self, message_builder, sample_detections):
        """Test timestamp formatting in ISO 8601 format."""
        timestamp = datetime(2025, 9, 21, 8, 23, 12, 123456, timezone.utc)
        message = message_builder.build_detection_message(sample_detections, timestamp)
        
        # Should format with milliseconds precision
        assert message['timestamp_utc'] == '2025-09-21T08:23:12.123Z'
    
    def test_timestamp_default_current_time(self, message_builder, sample_detections):
        """Test that default timestamp uses current time."""
        with patch('drone_detection_simulator.message_builder.datetime') as mock_datetime:
            mock_now = datetime(2025, 9, 21, 10, 30, 45, 678000, timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.timezone = timezone
            
            message = message_builder.build_detection_message(sample_detections)
            
            mock_datetime.now.assert_called_once_with(timezone.utc)
            assert message['timestamp_utc'] == '2025-09-21T10:30:45.678Z'
    
    def test_camera_metadata_structure(self, message_builder, sample_detections):
        """Test camera metadata section structure and values."""
        message = message_builder.build_detection_message(sample_detections)
        camera = message['camera']
        
        # Check required fields
        required_fields = ['resolution', 'focal_px', 'principal_point', 'yaw_deg', 
                          'pitch_deg', 'lat_deg', 'lon_deg', 'alt_m_msl']
        for field in required_fields:
            assert field in camera
        
        # Check specific values from config
        assert camera['resolution'] == [1920, 1080]
        assert camera['yaw_deg'] == 90.0
        assert camera['pitch_deg'] == 10.0
        assert camera['lat_deg'] == 13.736717
        assert camera['lon_deg'] == 100.523186
        assert camera['alt_m_msl'] == 1.5
        
        # Check computed values
        assert isinstance(camera['focal_px'], float)
        assert camera['focal_px'] > 0
        assert isinstance(camera['principal_point'], list)
        assert len(camera['principal_point']) == 2
    
    def test_detections_array_formatting(self, message_builder, sample_detections):
        """Test detections array formatting and field conversion."""
        message = message_builder.build_detection_message(sample_detections)
        detections = message['detections']
        
        assert len(detections) == 2
        
        # Check first detection
        det1 = detections[0]
        assert det1['class'] == 'drone'
        assert det1['confidence'] == 0.91  # Rounded to 3 decimal places
        assert det1['bbox_px'] == [980.5, 520.3, 1020.7, 580.9]  # Rounded to 1 decimal
        assert det1['center_px'] == [1000.6, 550.6]  # Rounded to 1 decimal
        assert det1['size_px'] == [40.2, 60.6]  # Rounded to 1 decimal
        
        # Check that internal simulation data is not exposed
        assert 'drone_id' not in det1
        assert 'world_pos_enu' not in det1
        assert 'depth_m' not in det1
        assert 'projection_info' not in det1
        
        # Check second detection
        det2 = detections[1]
        assert det2['class'] == 'drone'
        assert det2['confidence'] == 0.78
        assert len(det2['bbox_px']) == 4
        assert len(det2['center_px']) == 2
        assert len(det2['size_px']) == 2
    
    def test_edge_metadata_structure(self, message_builder, sample_detections):
        """Test edge metadata section structure."""
        message = message_builder.build_detection_message(sample_detections)
        edge = message['edge']
        
        # Check required fields
        assert 'processing_latency_ms' in edge
        assert 'detector_version' in edge
        
        # Check values
        assert isinstance(edge['processing_latency_ms'], float)
        assert edge['processing_latency_ms'] >= 0
        assert edge['detector_version'] == 'det-v1.2'
    
    def test_processing_latency_simulation(self, config, camera_model):
        """Test processing latency simulation with jitter."""
        rng = np.random.default_rng(42)
        builder = DetectionMessageBuilder(config, camera_model, rng)
        
        latencies = []
        for _ in range(100):
            message = builder.build_detection_message([])
            latencies.append(message['edge']['processing_latency_ms'])
        
        # Check that latencies vary (due to jitter)
        assert len(set(latencies)) > 1
        
        # Check that mean is approximately correct
        mean_latency = np.mean(latencies)
        assert abs(mean_latency - config.processing_latency_ms_mean) < 10.0
        
        # Check that all latencies are non-negative
        assert all(lat >= 0 for lat in latencies)
    
    def test_processing_latency_no_jitter(self, config, camera_model):
        """Test processing latency with zero jitter."""
        config.processing_latency_ms_jitter = 0.0
        rng = np.random.default_rng(42)
        builder = DetectionMessageBuilder(config, camera_model, rng)
        
        message1 = builder.build_detection_message([])
        message2 = builder.build_detection_message([])
        
        # Should be exactly the mean value
        assert message1['edge']['processing_latency_ms'] == config.processing_latency_ms_mean
        assert message2['edge']['processing_latency_ms'] == config.processing_latency_ms_mean
    
    def test_empty_detections_array(self, message_builder):
        """Test message building with empty detections array."""
        message = message_builder.build_detection_message([])
        
        assert message['detections'] == []
        assert 'camera' in message
        assert 'edge' in message
        assert message['frame_id'] == 1
    
    def test_false_positive_generation(self, message_builder):
        """Test false positive detection generation."""
        false_positive = message_builder.build_false_positive_detection()
        
        # Check structure
        assert false_positive['class'] == 'false_drone'
        assert 0.1 <= false_positive['confidence'] <= 0.4
        assert len(false_positive['bbox_px']) == 4
        assert len(false_positive['center_px']) == 2
        assert len(false_positive['size_px']) == 2
        
        # Check bounds
        bbox = false_positive['bbox_px']
        assert 0 <= bbox[0] <= message_builder.config.image_width_px
        assert 0 <= bbox[1] <= message_builder.config.image_height_px
        assert 0 <= bbox[2] <= message_builder.config.image_width_px
        assert 0 <= bbox[3] <= message_builder.config.image_height_px
        
        # Check bbox validity
        assert bbox[0] <= bbox[2]  # x_min <= x_max
        assert bbox[1] <= bbox[3]  # y_min <= y_max
    
    def test_add_false_positives_probability(self, config, camera_model):
        """Test false positive addition based on probability."""
        # Set high false positive rate for testing
        config.false_positive_rate = 1.0  # Always add false positive
        rng = np.random.default_rng(42)
        builder = DetectionMessageBuilder(config, camera_model, rng)
        
        original_detections = []
        updated_detections = builder.add_false_positives(original_detections)
        
        # Should have added one false positive
        assert len(updated_detections) == 1
        assert updated_detections[0]['class'] == 'false_drone'
        assert updated_detections[0]['drone_id'] == -1
    
    def test_add_false_positives_no_probability(self, config, camera_model):
        """Test false positive addition with zero probability."""
        config.false_positive_rate = 0.0  # Never add false positive
        rng = np.random.default_rng(42)
        builder = DetectionMessageBuilder(config, camera_model, rng)
        
        original_detections = []
        updated_detections = builder.add_false_positives(original_detections)
        
        # Should not have added any false positives
        assert len(updated_detections) == 0
    
    def test_json_string_conversion(self, message_builder, sample_detections):
        """Test JSON string conversion."""
        message = message_builder.build_detection_message(sample_detections)
        
        # Test compact JSON
        json_str = message_builder.to_json_string(message)
        assert isinstance(json_str, str)
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed == message
        
        # Test pretty-printed JSON
        pretty_json = message_builder.to_json_string(message, indent=2)
        assert len(pretty_json) > len(json_str)  # Pretty-printed is longer
        
        # Should still be valid JSON
        parsed_pretty = json.loads(pretty_json)
        assert parsed_pretty == message
    
    def test_message_schema_validation_valid(self, message_builder, sample_detections):
        """Test schema validation with valid message."""
        message = message_builder.build_detection_message(sample_detections)
        
        assert message_builder.validate_message_schema(message) is True
    
    def test_message_schema_validation_missing_fields(self, message_builder, sample_detections):
        """Test schema validation with missing required fields."""
        message = message_builder.build_detection_message(sample_detections)
        
        # Test missing top-level fields
        for field in ['timestamp_utc', 'frame_id', 'camera', 'detections', 'edge']:
            invalid_message = message.copy()
            del invalid_message[field]
            assert message_builder.validate_message_schema(invalid_message) is False
        
        # Test missing camera fields
        invalid_message = message.copy()
        del invalid_message['camera']['focal_px']
        assert message_builder.validate_message_schema(invalid_message) is False
        
        # Test missing detection fields
        invalid_message = message.copy()
        del invalid_message['detections'][0]['confidence']
        assert message_builder.validate_message_schema(invalid_message) is False
        
        # Test missing edge fields
        invalid_message = message.copy()
        del invalid_message['edge']['processing_latency_ms']
        assert message_builder.validate_message_schema(invalid_message) is False
    
    def test_message_schema_validation_invalid_formats(self, message_builder, sample_detections):
        """Test schema validation with invalid field formats."""
        message = message_builder.build_detection_message(sample_detections)
        
        # Test invalid detections array
        invalid_message = message.copy()
        invalid_message['detections'] = "not_a_list"
        assert message_builder.validate_message_schema(invalid_message) is False
        
        # Test invalid bbox format
        invalid_message = message.copy()
        invalid_message['detections'][0]['bbox_px'] = [1, 2, 3]  # Should have 4 elements
        assert message_builder.validate_message_schema(invalid_message) is False
        
        # Test invalid center format
        invalid_message = message.copy()
        invalid_message['detections'][0]['center_px'] = [1]  # Should have 2 elements
        assert message_builder.validate_message_schema(invalid_message) is False
        
        # Test invalid size format
        invalid_message = message.copy()
        invalid_message['detections'][0]['size_px'] = [1, 2, 3]  # Should have 2 elements
        assert message_builder.validate_message_schema(invalid_message) is False
    
    def test_frame_counter_operations(self, message_builder):
        """Test frame counter operations."""
        # Initial state
        assert message_builder.get_frame_id() == 0
        
        # After building messages
        message_builder.build_detection_message([])
        assert message_builder.get_frame_id() == 1
        
        message_builder.build_detection_message([])
        assert message_builder.get_frame_id() == 2
        
        # After reset
        message_builder.reset_frame_counter()
        assert message_builder.get_frame_id() == 0
    
    def test_message_size_calculation(self, message_builder, sample_detections):
        """Test message size calculation."""
        message = message_builder.build_detection_message(sample_detections)
        size_bytes = message_builder.get_message_size_bytes(message)
        
        # Should be positive
        assert size_bytes > 0
        
        # Should be reasonable size for JSON message
        assert 500 < size_bytes < 5000  # Typical range for detection messages
        
        # Empty message should be smaller
        empty_message = message_builder.build_detection_message([])
        empty_size = message_builder.get_message_size_bytes(empty_message)
        assert empty_size < size_bytes
    
    def test_deterministic_behavior(self, config, camera_model):
        """Test deterministic behavior with fixed seed."""
        # Create two builders with same seed
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        builder1 = DetectionMessageBuilder(config, camera_model, rng1)
        builder2 = DetectionMessageBuilder(config, camera_model, rng2)
        
        # Generate messages
        message1 = builder1.build_detection_message([])
        message2 = builder2.build_detection_message([])
        
        # Latencies should be the same (deterministic)
        assert message1['edge']['processing_latency_ms'] == message2['edge']['processing_latency_ms']
        
        # False positive generation should be deterministic
        fp1 = builder1.build_false_positive_detection()
        fp2 = builder2.build_false_positive_detection()
        assert fp1 == fp2
    
    def test_coordinate_precision_rounding(self, message_builder):
        """Test that coordinates are properly rounded in output."""
        # Create detection with high precision coordinates
        high_precision_detection = {
            'class': 'drone',
            'confidence': 0.912345678,
            'bbox_px': [980.123456, 520.987654, 1020.555555, 580.777777],
            'center_px': [1000.123456, 550.987654],
            'size_px': [40.123456, 60.987654],
            'drone_id': 0,
            'world_pos_enu': [10.0, 5.0, 5.5],
            'depth_m': 12.5,
            'projection_info': {'in_bounds': True, 'distance_from_edge': 100.0}
        }
        
        message = message_builder.build_detection_message([high_precision_detection])
        detection = message['detections'][0]
        
        # Check rounding
        assert detection['confidence'] == 0.912  # 3 decimal places
        assert detection['bbox_px'] == [980.1, 521.0, 1020.6, 580.8]  # 1 decimal place
        assert detection['center_px'] == [1000.1, 551.0]  # 1 decimal place
        assert detection['size_px'] == [40.1, 61.0]  # 1 decimal place


class TestMultiDroneMessageBuilder:
    """Test cases specifically for multi-drone message building (Requirements 7.2, 7.3)."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for multi-drone scenarios."""
        return SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            num_drones=4,
            processing_latency_ms_mean=50.0,
            processing_latency_ms_jitter=10.0,
            false_positive_rate=0.05,
            deterministic_seed=42
        )
    
    @pytest.fixture
    def message_builder(self, config):
        """Create test message builder."""
        camera_model = CameraModel(config)
        rng = np.random.default_rng(42)
        return DetectionMessageBuilder(config, camera_model, rng)
    
    @pytest.fixture
    def multi_drone_detections(self):
        """Create sample multi-drone detection data."""
        return [
            {
                'class': 'drone',
                'confidence': 0.91,
                'bbox_px': [980.5, 520.3, 1020.7, 580.9],
                'center_px': [1000.6, 550.6],
                'size_px': [40.2, 60.6],
                'drone_id': 0,
                'world_pos_enu': [10.0, 0.0, 5.5],
                'depth_m': 12.5,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 100.0}
            },
            {
                'class': 'drone',
                'confidence': 0.78,
                'bbox_px': [1200.1, 400.2, 1240.8, 450.9],
                'center_px': [1220.45, 425.55],
                'size_px': [40.7, 50.7],
                'drone_id': 1,
                'world_pos_enu': [15.0, 1.0, 6.0],
                'depth_m': 18.2,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 75.0}
            },
            {
                'class': 'drone',
                'confidence': 0.85,
                'bbox_px': [750.3, 600.1, 785.9, 640.8],
                'center_px': [768.1, 620.45],
                'size_px': [35.6, 40.7],
                'drone_id': 2,
                'world_pos_enu': [8.0, -0.5, 4.8],
                'depth_m': 10.1,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 120.0}
            },
            {
                'class': 'drone',
                'confidence': 0.65,
                'bbox_px': [1500.2, 800.5, 1520.8, 825.3],
                'center_px': [1510.5, 812.9],
                'size_px': [20.6, 24.8],
                'drone_id': 3,
                'world_pos_enu': [25.0, 2.0, 7.0],
                'depth_m': 28.5,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 50.0}
            }
        ]
    
    def test_multi_drone_detection_array_structure(self, message_builder, multi_drone_detections):
        """Test requirement 7.2: Detection message includes array with entries for multiple drones."""
        message = message_builder.build_detection_message(multi_drone_detections)
        
        # Should have detections array with all drones
        assert 'detections' in message
        assert len(message['detections']) == 4
        
        # Each detection should have proper structure
        for i, detection in enumerate(message['detections']):
            assert detection['class'] == 'drone'
            assert isinstance(detection['confidence'], float)
            assert len(detection['bbox_px']) == 4
            assert len(detection['center_px']) == 2
            assert len(detection['size_px']) == 2
            
            # Should not expose internal simulation data
            assert 'drone_id' not in detection
            assert 'world_pos_enu' not in detection
            assert 'depth_m' not in detection
            assert 'projection_info' not in detection
    
    def test_multi_drone_overlapping_bboxes_handling(self, message_builder):
        """Test requirement 7.3: Multiple bounding boxes when drones overlap in camera view."""
        # Create overlapping detections
        overlapping_detections = [
            {
                'class': 'drone',
                'confidence': 0.90,
                'bbox_px': [950.0, 500.0, 1000.0, 550.0],  # Overlaps with next
                'center_px': [975.0, 525.0],
                'size_px': [50.0, 50.0],
                'drone_id': 0,
                'world_pos_enu': [10.0, 0.0, 5.0],
                'depth_m': 12.0,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 100.0}
            },
            {
                'class': 'drone',
                'confidence': 0.85,
                'bbox_px': [980.0, 520.0, 1030.0, 570.0],  # Overlaps with previous
                'center_px': [1005.0, 545.0],
                'size_px': [50.0, 50.0],
                'drone_id': 1,
                'world_pos_enu': [10.2, 0.1, 5.1],
                'depth_m': 12.3,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 95.0}
            }
        ]
        
        message = message_builder.build_detection_message(overlapping_detections)
        
        # Should output both detections even when overlapping
        assert len(message['detections']) == 2
        
        # Both should be distinct drone detections
        det1, det2 = message['detections']
        assert det1['class'] == 'drone'
        assert det2['class'] == 'drone'
        assert det1['bbox_px'] != det2['bbox_px']
        assert det1['center_px'] != det2['center_px']
        
        # Verify they actually overlap
        bbox1 = det1['bbox_px']
        bbox2 = det2['bbox_px']
        overlap_x = not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0])
        overlap_y = not (bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
        assert overlap_x and overlap_y  # Should overlap
    
    def test_multi_drone_confidence_preservation(self, message_builder, multi_drone_detections):
        """Test that confidence scores are preserved for all drones."""
        message = message_builder.build_detection_message(multi_drone_detections)
        
        # Extract confidences from input and output
        input_confidences = [d['confidence'] for d in multi_drone_detections]
        output_confidences = [d['confidence'] for d in message['detections']]
        
        # Should preserve all confidence values (rounded to 3 decimal places)
        assert len(output_confidences) == len(input_confidences)
        for i, (input_conf, output_conf) in enumerate(zip(input_confidences, output_confidences)):
            assert abs(output_conf - input_conf) < 0.001, f"Confidence mismatch for drone {i}"
    
    def test_multi_drone_bbox_coordinate_rounding(self, message_builder, multi_drone_detections):
        """Test that bounding box coordinates are properly rounded for all drones."""
        message = message_builder.build_detection_message(multi_drone_detections)
        
        for detection in message['detections']:
            # All coordinates should be rounded to 1 decimal place
            for coord in detection['bbox_px']:
                assert coord == round(coord, 1)
            
            for coord in detection['center_px']:
                assert coord == round(coord, 1)
            
            for size in detection['size_px']:
                assert size == round(size, 1)
    
    def test_multi_drone_with_false_positives(self, message_builder, multi_drone_detections):
        """Test multi-drone detection with false positives added."""
        # Add false positives to multi-drone detections
        detections_with_fp = message_builder.add_false_positives(multi_drone_detections.copy())
        
        # May or may not have added false positive (based on probability)
        assert len(detections_with_fp) >= len(multi_drone_detections)
        
        # Build message
        message = message_builder.build_detection_message(detections_with_fp)
        
        # Should handle mixed real and false positive detections
        assert len(message['detections']) == len(detections_with_fp)
        
        # All detections should be valid
        for detection in message['detections']:
            assert detection['class'] in ['drone', 'false_drone']
            assert 0.0 <= detection['confidence'] <= 1.0
            assert len(detection['bbox_px']) == 4
    
    def test_large_number_of_drones_message(self, message_builder):
        """Test message building with large number of drone detections."""
        # Create many drone detections
        many_detections = []
        for i in range(15):
            detection = {
                'class': 'drone',
                'confidence': 0.7 + (i % 3) * 0.1,
                'bbox_px': [100 + i * 50, 100 + i * 30, 140 + i * 50, 140 + i * 30],
                'center_px': [120 + i * 50, 120 + i * 30],
                'size_px': [40, 40],
                'drone_id': i,
                'world_pos_enu': [10 + i, 0, 5],
                'depth_m': 10 + i,
                'projection_info': {'in_bounds': True, 'distance_from_edge': 100}
            }
            many_detections.append(detection)
        
        message = message_builder.build_detection_message(many_detections)
        
        # Should handle large number of detections
        assert len(message['detections']) == 15
        
        # Message should still be valid
        assert message_builder.validate_message_schema(message)
        
        # Message size should be reasonable
        size_bytes = message_builder.get_message_size_bytes(message)
        assert size_bytes > 0
        assert size_bytes < 50000  # Should not be excessively large
    
    def test_multi_drone_message_schema_validation(self, message_builder, multi_drone_detections):
        """Test that multi-drone messages pass schema validation."""
        message = message_builder.build_detection_message(multi_drone_detections)
        
        # Should pass schema validation
        assert message_builder.validate_message_schema(message) is True
        
        # Test with empty detections
        empty_message = message_builder.build_detection_message([])
        assert message_builder.validate_message_schema(empty_message) is True
        
        # Test with single detection
        single_message = message_builder.build_detection_message([multi_drone_detections[0]])
        assert message_builder.validate_message_schema(single_message) is True
    
    def test_multi_drone_json_serialization(self, message_builder, multi_drone_detections):
        """Test JSON serialization of multi-drone messages."""
        message = message_builder.build_detection_message(multi_drone_detections)
        
        # Should serialize to valid JSON
        json_str = message_builder.to_json_string(message)
        assert isinstance(json_str, str)
        
        # Should be parseable back to same structure
        import json
        parsed = json.loads(json_str)
        assert parsed == message
        
        # Pretty-printed version should also work
        pretty_json = message_builder.to_json_string(message, indent=2)
        parsed_pretty = json.loads(pretty_json)
        assert parsed_pretty == message
    
    def test_multi_drone_frame_counter_consistency(self, message_builder, multi_drone_detections):
        """Test frame counter consistency across multi-drone messages."""
        # Build multiple messages
        message1 = message_builder.build_detection_message(multi_drone_detections)
        message2 = message_builder.build_detection_message(multi_drone_detections[:2])
        message3 = message_builder.build_detection_message([])
        
        # Frame IDs should increment
        assert message1['frame_id'] == 1
        assert message2['frame_id'] == 2
        assert message3['frame_id'] == 3
        
        # Reset and test
        message_builder.reset_frame_counter()
        message4 = message_builder.build_detection_message(multi_drone_detections)
        assert message4['frame_id'] == 1
    
    def test_multi_drone_processing_latency_consistency(self, message_builder, multi_drone_detections):
        """Test that processing latency is consistent per frame regardless of drone count."""
        # Build messages with different numbers of detections
        message_many = message_builder.build_detection_message(multi_drone_detections)
        message_few = message_builder.build_detection_message(multi_drone_detections[:1])
        message_none = message_builder.build_detection_message([])
        
        # All should have processing latency
        assert 'processing_latency_ms' in message_many['edge']
        assert 'processing_latency_ms' in message_few['edge']
        assert 'processing_latency_ms' in message_none['edge']
        
        # All should be non-negative
        assert message_many['edge']['processing_latency_ms'] >= 0
        assert message_few['edge']['processing_latency_ms'] >= 0
        assert message_none['edge']['processing_latency_ms'] >= 0


class TestMessageBuilderIntegration:
    """Integration tests for message builder with other components."""
    
    def test_integration_with_camera_model(self):
        """Test integration with actual camera model."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            focal_length_mm=8.0,
            sensor_height_mm=5.76,
            vertical_fov_deg=None,  # Explicitly set to None to avoid conflict
            camera_lat_deg=13.736717,
            camera_lon_deg=100.523186,
            camera_alt_m=1.5,
            camera_yaw_deg=90.0,
            camera_pitch_deg=10.0
        )
        
        camera_model = CameraModel(config)
        builder = DetectionMessageBuilder(config, camera_model)
        
        message = builder.build_detection_message([])
        
        # Camera metadata should match camera model output
        camera_metadata = camera_model.get_camera_metadata()
        assert message['camera'] == camera_metadata
    
    def test_realistic_message_example(self):
        """Test generation of realistic message matching design specification."""
        config = SimulatorConfig(
            image_width_px=1920,
            image_height_px=1080,
            vertical_fov_deg=50.0,
            camera_lat_deg=13.736717,
            camera_lon_deg=100.523186,
            camera_alt_m=1.5,
            camera_yaw_deg=90.0,
            camera_pitch_deg=10.0,
            processing_latency_ms_mean=42.0,
            processing_latency_ms_jitter=0.0,  # No jitter for predictable test
            deterministic_seed=42
        )
        
        camera_model = CameraModel(config)
        rng = np.random.default_rng(42)
        builder = DetectionMessageBuilder(config, camera_model, rng)
        
        # Create realistic detection
        detection = {
            'class': 'drone',
            'confidence': 0.91,
            'bbox_px': [980, 520, 1020, 580],
            'center_px': [1000, 550],
            'size_px': [40, 60],
            'drone_id': 0,
            'world_pos_enu': [10.0, 5.0, 5.5],
            'depth_m': 12.5,
            'projection_info': {'in_bounds': True, 'distance_from_edge': 100.0}
        }
        
        timestamp = datetime(2025, 9, 21, 8, 23, 12, 123000, timezone.utc)
        message = builder.build_detection_message([detection], timestamp)
        
        # Validate against design specification example
        assert message['timestamp_utc'] == '2025-09-21T08:23:12.123Z'
        assert message['frame_id'] == 1
        
        # Camera section
        camera = message['camera']
        assert camera['resolution'] == [1920, 1080]
        assert camera['yaw_deg'] == 90.0
        assert camera['pitch_deg'] == 10.0
        assert camera['lat_deg'] == 13.736717
        assert camera['lon_deg'] == 100.523186
        assert camera['alt_m_msl'] == 1.5
        
        # Detection section
        det = message['detections'][0]
        assert det['class'] == 'drone'
        assert det['confidence'] == 0.91
        assert det['bbox_px'] == [980.0, 520.0, 1020.0, 580.0]
        assert det['center_px'] == [1000.0, 550.0]
        assert det['size_px'] == [40.0, 60.0]
        
        # Edge section
        edge = message['edge']
        assert edge['processing_latency_ms'] == 42.0
        assert edge['detector_version'] == 'det-v1.2'
        
        # Validate schema
        assert builder.validate_message_schema(message) is True
    
    def test_complete_multi_drone_pipeline_integration(self):
        """Test complete integration of multi-drone pipeline from motion to message."""
        from drone_detection_simulator.motion import MotionGenerator
        from drone_detection_simulator.detection import DetectionGenerator
        
        # Setup multi-drone configuration
        config = SimulatorConfig(
            num_drones=3,
            duration_s=5.0,
            fps=10.0,
            vertical_fov_deg=50.0,
            deterministic_seed=42
        )
        
        # Create pipeline components
        motion_gen = MotionGenerator(config)
        camera_model = CameraModel(config)
        detection_gen = DetectionGenerator(config, camera_model)
        message_builder = DetectionMessageBuilder(config, camera_model)
        
        # Simulate one frame
        time_s = 2.5
        world_positions = motion_gen.get_positions_at_time(time_s)
        detections = detection_gen.generate_detections(world_positions)
        message = message_builder.build_detection_message(detections)
        
        # Verify complete pipeline
        assert len(world_positions) == 3  # Input: 3 drone positions
        assert len(detections) >= 1  # Should detect at least some drones
        assert len(message['detections']) == len(detections)  # Output: same number
        
        # Message should be valid
        assert message_builder.validate_message_schema(message)
        
        # Should contain expected structure
        assert 'timestamp_utc' in message
        assert 'frame_id' in message
        assert 'camera' in message
        assert 'detections' in message
        assert 'edge' in message
        
        # All detections should be drone class
        for detection in message['detections']:
            assert detection['class'] == 'drone'