"""
Tests for deterministic testing support and validation utilities.

This module implements comprehensive tests for deterministic behavior,
ground truth validation, and motion constraints verification.
Tests requirement 5.4: deterministic random seed option for reproducible test runs.
"""

import pytest
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from unittest.mock import patch

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.simulator import DroneSimulator
from drone_detection_simulator.motion import MotionGenerator
from drone_detection_simulator.camera import CameraModel
from drone_detection_simulator.detection import DetectionGenerator
from drone_detection_simulator.noise import NoiseModel


class TestUtilities:
    """Test utilities for validation with known ground truth."""
    
    @staticmethod
    def create_deterministic_config(seed: int = 42, **overrides) -> SimulatorConfig:
        """
        Create a deterministic configuration for testing.
        
        Args:
            seed: Random seed for reproducible behavior
            **overrides: Configuration parameters to override
            
        Returns:
            SimulatorConfig with deterministic settings
        """
        base_config = {
            'deterministic_seed': seed,
            'offline_mode': True,
            'duration_s': 2.0,
            'fps': 10.0,
            'num_drones': 2,
            'vertical_fov_deg': 50.0,
            'path_span_m': 20.0,
            'speed_mps': 3.0,
            'path_altitude_agl_m': 5.0,
            'pixel_centroid_sigma_px': 1.0,
            'bbox_size_sigma_px': 2.0,
            'confidence_noise': 0.05,
            'miss_rate_small': 0.02,
            'false_positive_rate': 0.01
        }
        base_config.update(overrides)
        return SimulatorConfig(**base_config)
    
    @staticmethod
    def extract_detection_messages(captured_output: List[str]) -> List[Dict[str, Any]]:
        """
        Extract and parse JSON detection messages from captured output.
        
        Args:
            captured_output: List of captured print statements
            
        Returns:
            List of parsed JSON detection messages
        """
        messages = []
        for output in captured_output:
            if output.strip():
                try:
                    parsed = json.loads(output)
                    if 'timestamp_utc' in parsed and 'detections' in parsed:
                        messages.append(parsed)
                except json.JSONDecodeError:
                    pass  # Skip non-JSON output
        return messages
    
    @staticmethod
    def validate_detection_message_structure(message: Dict[str, Any]) -> bool:
        """
        Validate that a detection message has the expected structure.
        
        Args:
            message: Detection message to validate
            
        Returns:
            True if message structure is valid
        """
        required_fields = ['timestamp_utc', 'frame_id', 'camera', 'detections', 'edge']
        
        # Check top-level fields
        for field in required_fields:
            if field not in message:
                return False
        
        # Check camera metadata
        camera_fields = ['resolution', 'focal_px', 'principal_point', 'yaw_deg', 'pitch_deg', 
                        'lat_deg', 'lon_deg', 'alt_m_msl']
        for field in camera_fields:
            if field not in message['camera']:
                return False
        
        # Check detections array
        if not isinstance(message['detections'], list):
            return False
        
        for detection in message['detections']:
            detection_fields = ['class', 'confidence', 'bbox_px', 'center_px', 'size_px']
            for field in detection_fields:
                if field not in detection:
                    return False
        
        # Check edge metadata
        edge_fields = ['processing_latency_ms', 'detector_version']
        for field in edge_fields:
            if field not in message['edge']:
                return False
        
        return True
    
    @staticmethod
    def calculate_motion_smoothness(positions: List[np.ndarray], fps: float) -> Dict[str, float]:
        """
        Calculate motion smoothness metrics for a path.
        
        Args:
            positions: List of 3D position vectors
            fps: Frames per second
            
        Returns:
            Dictionary with smoothness metrics
        """
        if len(positions) < 3:
            return {'max_acceleration': 0.0, 'avg_acceleration': 0.0, 'jerk_metric': 0.0}
        
        frame_time = 1.0 / fps
        velocities = []
        accelerations = []
        
        # Calculate velocities
        for i in range(1, len(positions)):
            velocity = (positions[i] - positions[i-1]) / frame_time
            velocities.append(velocity)
        
        # Calculate accelerations
        for i in range(1, len(velocities)):
            acceleration = (velocities[i] - velocities[i-1]) / frame_time
            accelerations.append(acceleration)
        
        # Calculate metrics
        lateral_accelerations = [abs(acc[0]) for acc in accelerations]  # East component
        max_acceleration = max(lateral_accelerations) if lateral_accelerations else 0.0
        avg_acceleration = np.mean(lateral_accelerations) if lateral_accelerations else 0.0
        
        # Calculate jerk (rate of change of acceleration) as smoothness metric
        jerks = []
        for i in range(1, len(accelerations)):
            jerk = np.linalg.norm(accelerations[i] - accelerations[i-1]) / frame_time
            jerks.append(jerk)
        
        jerk_metric = np.mean(jerks) if jerks else 0.0
        
        return {
            'max_acceleration': max_acceleration,
            'avg_acceleration': avg_acceleration,
            'jerk_metric': jerk_metric
        }
    
    @staticmethod
    def validate_pixel_bounds(detections: List[Dict[str, Any]], image_width: int, image_height: int) -> bool:
        """
        Validate that all detections are within pixel bounds.
        
        Args:
            detections: List of detection dictionaries
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            True if all detections are within bounds
        """
        for detection in detections:
            # Check center coordinates
            if 'center_px' in detection:
                center_x, center_y = detection['center_px']
                if not (0 <= center_x < image_width and 0 <= center_y < image_height):
                    return False
            
            # Check bounding box coordinates
            if 'bbox_px' in detection:
                x1, y1, x2, y2 = detection['bbox_px']
                if not (0 <= x1 < image_width and 0 <= y1 < image_height and
                       0 < x2 <= image_width and 0 < y2 <= image_height):
                    return False
                if not (x1 < x2 and y1 < y2):
                    return False
        
        return True
    
    @staticmethod
    def calculate_detection_consistency(messages1: List[Dict], messages2: List[Dict]) -> Dict[str, float]:
        """
        Calculate consistency metrics between two sets of detection messages.
        
        Args:
            messages1: First set of detection messages
            messages2: Second set of detection messages
            
        Returns:
            Dictionary with consistency metrics
        """
        if len(messages1) != len(messages2):
            return {'frame_count_match': False, 'detection_count_consistency': 0.0, 
                   'position_consistency': 0.0, 'confidence_consistency': 0.0}
        
        detection_count_matches = 0
        position_differences = []
        confidence_differences = []
        
        for msg1, msg2 in zip(messages1, messages2):
            det1 = msg1['detections']
            det2 = msg2['detections']
            
            # Check detection count consistency
            if len(det1) == len(det2):
                detection_count_matches += 1
                
                # Compare individual detections
                for d1, d2 in zip(det1, det2):
                    # Position consistency
                    if 'center_px' in d1 and 'center_px' in d2:
                        pos_diff = np.linalg.norm(np.array(d1['center_px']) - np.array(d2['center_px']))
                        position_differences.append(pos_diff)
                    
                    # Confidence consistency
                    if 'confidence' in d1 and 'confidence' in d2:
                        conf_diff = abs(d1['confidence'] - d2['confidence'])
                        confidence_differences.append(conf_diff)
        
        return {
            'frame_count_match': True,
            'detection_count_consistency': detection_count_matches / len(messages1),
            'position_consistency': 1.0 - np.mean(position_differences) if position_differences else 1.0,
            'confidence_consistency': 1.0 - np.mean(confidence_differences) if confidence_differences else 1.0
        }


class TestDeterministicBehavior:
    """Test deterministic behavior with fixed random seeds."""
    
    def test_deterministic_simulation_identical_output(self):
        """Test that identical configurations produce identical output."""
        config = TestUtilities.create_deterministic_config(seed=123)
        
        # Capture output from both runs
        outputs1 = []
        outputs2 = []
        
        def capture_output1(*args, **kwargs):
            outputs1.append(args[0] if args else "")
        
        def capture_output2(*args, **kwargs):
            outputs2.append(args[0] if args else "")
        
        # Run simulation twice with same configuration
        with patch('builtins.print', side_effect=capture_output1):
            simulator1 = DroneSimulator(config)
            results1 = simulator1.run()
        
        with patch('builtins.print', side_effect=capture_output2):
            simulator2 = DroneSimulator(config)
            results2 = simulator2.run()
        
        # Extract detection messages
        messages1 = TestUtilities.extract_detection_messages(outputs1)
        messages2 = TestUtilities.extract_detection_messages(outputs2)
        
        # Verify identical output
        assert len(messages1) == len(messages2)
        assert len(messages1) > 0, "No detection messages found"
        
        # Calculate consistency metrics
        consistency = TestUtilities.calculate_detection_consistency(messages1, messages2)
        
        assert consistency['frame_count_match'] is True
        assert consistency['detection_count_consistency'] == 1.0
        assert consistency['position_consistency'] > 0.999  # Allow tiny floating point differences
        assert consistency['confidence_consistency'] > 0.999
    
    def test_deterministic_motion_generation(self):
        """Test that motion generation is deterministic."""
        config = TestUtilities.create_deterministic_config(seed=456)
        
        # Create two motion generators with same config
        motion_gen1 = MotionGenerator(config)
        motion_gen2 = MotionGenerator(config)
        
        # Verify identical paths
        assert len(motion_gen1.paths) == len(motion_gen2.paths)
        
        for path1, path2 in zip(motion_gen1.paths, motion_gen2.paths):
            assert len(path1) == len(path2)
            for pos1, pos2 in zip(path1, path2):
                assert np.allclose(pos1, pos2, atol=1e-12)
    
    def test_deterministic_noise_application(self):
        """Test that noise application is deterministic."""
        config = TestUtilities.create_deterministic_config(seed=789)
        
        rng1 = np.random.default_rng(789)
        rng2 = np.random.default_rng(789)
        
        noise_model1 = NoiseModel(config, rng1)
        noise_model2 = NoiseModel(config, rng2)
        
        clean_detection = {
            'center_px': [960.0, 540.0],
            'size_px': [40.0, 60.0],
            'confidence': 0.8,
            'class': 'drone'
        }
        
        # Apply noise multiple times
        for _ in range(10):
            noisy1 = noise_model1.apply_detection_noise(clean_detection.copy())
            noisy2 = noise_model2.apply_detection_noise(clean_detection.copy())
            
            assert np.allclose(noisy1['center_px'], noisy2['center_px'], atol=1e-12)
            assert np.allclose(noisy1['size_px'], noisy2['size_px'], atol=1e-12)
            assert abs(noisy1['confidence'] - noisy2['confidence']) < 1e-12
    
    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different output."""
        config1 = TestUtilities.create_deterministic_config(seed=111)
        config2 = TestUtilities.create_deterministic_config(seed=222)
        
        outputs1 = []
        outputs2 = []
        
        def capture_output1(*args, **kwargs):
            outputs1.append(args[0] if args else "")
        
        def capture_output2(*args, **kwargs):
            outputs2.append(args[0] if args else "")
        
        # Run simulations with different seeds
        with patch('builtins.print', side_effect=capture_output1):
            simulator1 = DroneSimulator(config1)
            results1 = simulator1.run()
        
        with patch('builtins.print', side_effect=capture_output2):
            simulator2 = DroneSimulator(config2)
            results2 = simulator2.run()
        
        # Extract detection messages
        messages1 = TestUtilities.extract_detection_messages(outputs1)
        messages2 = TestUtilities.extract_detection_messages(outputs2)
        
        assert len(messages1) == len(messages2)
        assert len(messages1) > 0
        
        # Should have different output with different seeds
        consistency = TestUtilities.calculate_detection_consistency(messages1, messages2)
        
        # Detection counts might be similar, but positions should be different
        assert consistency['position_consistency'] < 0.9  # Should be significantly different
    
    def test_deterministic_false_positive_generation(self):
        """Test deterministic false positive generation."""
        config = TestUtilities.create_deterministic_config(
            seed=333,
            false_positive_rate=0.5  # High rate to ensure generation
        )
        
        rng1 = np.random.default_rng(333)
        rng2 = np.random.default_rng(333)
        
        noise_model1 = NoiseModel(config, rng1)
        noise_model2 = NoiseModel(config, rng2)
        
        # Generate false positives multiple times
        for _ in range(20):
            fp1 = noise_model1.generate_false_positive()
            fp2 = noise_model2.generate_false_positive()
            
            # Both should generate or both should not generate
            assert (fp1 is None) == (fp2 is None)
            
            if fp1 is not None and fp2 is not None:
                assert np.allclose(fp1['center_px'], fp2['center_px'], atol=1e-12)
                assert np.allclose(fp1['size_px'], fp2['size_px'], atol=1e-12)
                assert abs(fp1['confidence'] - fp2['confidence']) < 1e-12


class TestMotionConstraints:
    """Test motion smoothness and acceleration constraints."""
    
    def test_smooth_motion_constraints(self):
        """Test that motion satisfies smoothness constraints."""
        config = TestUtilities.create_deterministic_config(
            seed=444,
            duration_s=5.0,
            fps=30.0,
            speed_mps=4.0,
            max_lateral_accel_mps2=2.0
        )
        
        motion_gen = MotionGenerator(config)
        
        for drone_idx, path in enumerate(motion_gen.paths):
            # Calculate smoothness metrics
            smoothness = TestUtilities.calculate_motion_smoothness(path, config.fps)
            
            # Verify acceleration constraints
            assert smoothness['max_acceleration'] <= config.max_lateral_accel_mps2 + 0.5, \
                f"Drone {drone_idx} exceeds acceleration limit: {smoothness['max_acceleration']}"
            
            # Verify reasonable jerk (smoothness)
            assert smoothness['jerk_metric'] < 10.0, \
                f"Drone {drone_idx} has excessive jerk: {smoothness['jerk_metric']}"
    
    def test_bounded_pixel_movement(self):
        """Test that pixel detections remain within image bounds."""
        config = TestUtilities.create_deterministic_config(seed=555)
        
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            simulator = DroneSimulator(config)
            results = simulator.run()
        
        # Extract and validate detection messages
        messages = TestUtilities.extract_detection_messages(captured_output)
        assert len(messages) > 0
        
        for message in messages:
            # Validate message structure
            assert TestUtilities.validate_detection_message_structure(message)
            
            # Validate pixel bounds
            detections = message['detections']
            assert TestUtilities.validate_pixel_bounds(
                detections, config.image_width_px, config.image_height_px
            ), f"Detections out of bounds in frame {message['frame_id']}"
    
    def test_motion_continuity(self):
        """Test that motion is continuous without teleporting."""
        config = TestUtilities.create_deterministic_config(
            seed=666,
            duration_s=4.0,
            fps=25.0,
            speed_mps=3.0
        )
        
        motion_gen = MotionGenerator(config)
        frame_time = 1.0 / config.fps
        max_reasonable_speed = config.speed_mps * 2.0  # Allow margin for acceleration
        
        for drone_idx, path in enumerate(motion_gen.paths):
            for i in range(1, len(path)):
                displacement = np.linalg.norm(path[i] - path[i-1])
                max_displacement = max_reasonable_speed * frame_time
                
                assert displacement <= max_displacement + 0.1, \
                    f"Drone {drone_idx} teleported at frame {i}: {displacement}m > {max_displacement}m"
    
    def test_altitude_consistency(self):
        """Test that drone altitudes remain consistent."""
        config = TestUtilities.create_deterministic_config(
            seed=777,
            num_drones=3,
            path_altitude_agl_m=8.0
        )
        
        motion_gen = MotionGenerator(config)
        
        for drone_idx, path in enumerate(motion_gen.paths):
            expected_altitude = config.path_altitude_agl_m + drone_idx * 0.5
            
            for frame_idx, position in enumerate(path):
                altitude = position[2]  # Z component (up)
                
                assert abs(altitude - expected_altitude) < 0.01, \
                    f"Drone {drone_idx} altitude inconsistent at frame {frame_idx}: " \
                    f"{altitude} != {expected_altitude}"


class TestGroundTruthValidation:
    """Test validation against known ground truth scenarios."""
    
    def test_known_camera_projection(self):
        """Test camera projection with known ground truth values."""
        config = TestUtilities.create_deterministic_config(
            seed=888,
            vertical_fov_deg=60.0,
            image_width_px=1920,
            image_height_px=1080
        )
        
        camera_model = CameraModel(config)
        
        # Known ground truth: object at (0, 0, 10) should project to image center
        world_position = np.array([0.0, 0.0, 10.0])  # 10m in front of camera
        
        projection_result = camera_model.project_world_to_pixels(world_position)
        
        # Should project near image center
        center_x = config.image_width_px / 2.0
        center_y = config.image_height_px / 2.0
        
        projected_x, projected_y = projection_result['pixel_coordinates']
        
        # Allow some tolerance for camera model specifics
        assert abs(projected_x - center_x) < 50, \
            f"X projection error: {projected_x} vs expected {center_x}"
        assert abs(projected_y - center_y) < 50, \
            f"Y projection error: {projected_y} vs expected {center_y}"
    
    def test_known_motion_trajectory(self):
        """Test motion generation with known trajectory parameters."""
        config = TestUtilities.create_deterministic_config(
            seed=999,
            duration_s=10.0,
            fps=20.0,
            path_span_m=30.0,
            speed_mps=3.0,
            num_drones=1
        )
        
        motion_gen = MotionGenerator(config)
        path = motion_gen.paths[0]
        
        # Verify start and end positions
        start_pos = path[0]
        end_pos = path[-1]
        
        # Should start from left (negative east) and end at right (positive east)
        assert start_pos[0] < 0, f"Start position not on left: {start_pos[0]}"
        assert end_pos[0] > 0, f"End position not on right: {end_pos[0]}"
        
        # Total east displacement should be approximately path_span_m
        east_displacement = end_pos[0] - start_pos[0]
        assert abs(east_displacement - config.path_span_m) < 2.0, \
            f"East displacement error: {east_displacement} vs expected {config.path_span_m}"
        
        # Altitude should remain constant
        altitudes = [pos[2] for pos in path]
        assert all(abs(alt - config.path_altitude_agl_m) < 0.01 for alt in altitudes), \
            "Altitude not constant during flight"
    
    def test_detection_size_distance_relationship(self):
        """Test that detection size correlates with distance as expected."""
        config = TestUtilities.create_deterministic_config(
            seed=1000,
            drone_size_m=0.5,  # Larger drone for clearer size relationship
            path_altitude_agl_m=10.0
        )
        
        camera_model = CameraModel(config)
        detection_gen = DetectionGenerator(config, camera_model)
        
        # Test objects at different distances
        positions_near = [np.array([0.0, 0.0, 5.0])]   # 5m away
        positions_far = [np.array([0.0, 0.0, 20.0])]   # 20m away
        
        detections_near = detection_gen.generate_detections(positions_near)
        detections_far = detection_gen.generate_detections(positions_far)
        
        if detections_near and detections_far:
            size_near = detections_near[0]['size_px']
            size_far = detections_far[0]['size_px']
            
            area_near = size_near[0] * size_near[1]
            area_far = size_far[0] * size_far[1]
            
            # Nearer object should appear larger
            assert area_near > area_far, \
                f"Size-distance relationship violated: near={area_near}, far={area_far}"
    
    def test_end_to_end_consistency_validation(self):
        """Test end-to-end simulation consistency with known parameters."""
        config = TestUtilities.create_deterministic_config(
            seed=1111,
            duration_s=3.0,
            fps=15.0,
            num_drones=2
        )
        
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            simulator = DroneSimulator(config)
            results = simulator.run()
        
        # Validate simulation results
        expected_frames = int(config.duration_s * config.fps)
        assert results['simulation']['frames_processed'] == expected_frames
        assert results['simulation']['completion_rate'] == 1.0
        
        # Extract and validate messages
        messages = TestUtilities.extract_detection_messages(captured_output)
        assert len(messages) == expected_frames
        
        # Validate message consistency
        frame_ids = [msg['frame_id'] for msg in messages]
        assert frame_ids == list(range(len(frame_ids))), "Frame IDs not sequential"
        
        # Validate detection bounds and structure
        for message in messages:
            assert TestUtilities.validate_detection_message_structure(message)
            assert TestUtilities.validate_pixel_bounds(
                message['detections'], config.image_width_px, config.image_height_px
            )
        
        # Validate timing consistency
        timestamps = [msg['timestamp_utc'] for msg in messages]
        assert len(set(timestamps)) == len(timestamps), "Duplicate timestamps found"


if __name__ == "__main__":
    pytest.main([__file__])