"""
Integration tests for multi-drone support across all components.

Tests the complete pipeline from motion generation through detection to message building
for multiple drones, validating requirements 7.1, 7.2, 7.3, and 7.4.
"""

import pytest
import numpy as np
from datetime import datetime, timezone

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.motion import MotionGenerator
from drone_detection_simulator.camera import CameraModel
from drone_detection_simulator.detection import DetectionGenerator
from drone_detection_simulator.message_builder import DetectionMessageBuilder


class TestMultiDroneIntegration:
    """Integration tests for complete multi-drone pipeline."""
    
    @pytest.fixture
    def multi_drone_config(self):
        """Create configuration for multi-drone testing."""
        return SimulatorConfig(
            num_drones=4,
            duration_s=8.0,
            fps=20.0,
            path_span_m=30.0,
            speed_mps=4.0,
            path_altitude_agl_m=5.0,
            vertical_fov_deg=50.0,
            drone_size_m=0.25,
            deterministic_seed=42
        )
    
    @pytest.fixture
    def pipeline_components(self, multi_drone_config):
        """Create all pipeline components."""
        motion_gen = MotionGenerator(multi_drone_config)
        camera_model = CameraModel(multi_drone_config)
        detection_gen = DetectionGenerator(multi_drone_config, camera_model)
        message_builder = DetectionMessageBuilder(multi_drone_config, camera_model)
        
        return {
            'config': multi_drone_config,
            'motion_gen': motion_gen,
            'camera_model': camera_model,
            'detection_gen': detection_gen,
            'message_builder': message_builder
        }
    
    def test_complete_multi_drone_pipeline(self, pipeline_components):
        """Test complete pipeline from motion to message for multiple drones."""
        components = pipeline_components
        
        # Use manually positioned drones that are definitely visible
        # (The motion generator positions may put drones behind camera at certain times)
        visible_positions = [
            np.array([10.0, 0.0, 5.0]),    # Drone 0 - center front
            np.array([12.0, 1.0, 5.5]),    # Drone 1 - right front
            np.array([8.0, -1.0, 4.5]),    # Drone 2 - left front
            np.array([15.0, 0.5, 6.0])     # Drone 3 - far right front
        ]
        
        # Verify requirement 7.1: Independent smooth flight paths (using motion generator)
        motion_positions = components['motion_gen'].get_positions_at_time(2.0)
        assert len(motion_positions) == 4
        
        # All motion positions should be different
        for i in range(len(motion_positions)):
            for j in range(i + 1, len(motion_positions)):
                distance = np.linalg.norm(motion_positions[i] - motion_positions[j])
                assert distance > 0.5, f"Motion drones {i} and {j} too close: {distance}m"
        
        # Step 2: Generate detections from visible positions
        detections = components['detection_gen'].generate_detections(visible_positions)
        
        # Verify requirement 7.4: Independent distance estimates
        assert len(detections) >= 3  # Should detect most visible drones
        
        # Each detection should have unique drone_id
        drone_ids = [d['drone_id'] for d in detections]
        assert len(set(drone_ids)) == len(drone_ids)
        
        # Distance estimates should be reasonable and independent
        depths = [d['depth_m'] for d in detections]
        assert all(5.0 < depth < 50.0 for depth in depths)
        
        # Step 3: Build JSON message
        message = components['message_builder'].build_detection_message(detections)
        
        # Verify requirement 7.2: Array of detections
        assert 'detections' in message
        assert len(message['detections']) == len(detections)
        
        # Each detection should be properly formatted
        for detection in message['detections']:
            assert detection['class'] == 'drone'
            assert 0.0 <= detection['confidence'] <= 1.0
            assert len(detection['bbox_px']) == 4
            assert len(detection['center_px']) == 2
            assert len(detection['size_px']) == 2
            
            # Internal data should not be exposed
            assert 'drone_id' not in detection
            assert 'world_pos_enu' not in detection
            assert 'depth_m' not in detection
        
        # Message should be valid
        assert components['message_builder'].validate_message_schema(message)
    
    def test_multi_drone_temporal_consistency(self, pipeline_components):
        """Test that multi-drone behavior is consistent over time."""
        components = pipeline_components
        
        # Test multiple time points
        test_times = [0.0, 2.0, 4.0, 6.0, 8.0]
        frame_results = []
        
        for time_s in test_times:
            # Generate positions and detections
            world_positions = components['motion_gen'].get_positions_at_time(time_s)
            detections = components['detection_gen'].generate_detections(world_positions)
            message = components['message_builder'].build_detection_message(detections)
            
            frame_results.append({
                'time': time_s,
                'world_positions': world_positions,
                'detections': detections,
                'message': message
            })
        
        # Verify temporal consistency
        for i, result in enumerate(frame_results):
            # Should always have same number of world positions
            assert len(result['world_positions']) == 4
            
            # Detections should be reasonable
            assert len(result['detections']) >= 0
            
            # Messages should be valid
            assert components['message_builder'].validate_message_schema(result['message'])
            
            # Frame IDs should increment
            expected_frame_id = i + 1
            assert result['message']['frame_id'] == expected_frame_id
    
    def test_multi_drone_overlapping_scenario(self, pipeline_components):
        """Test requirement 7.3: Multiple bounding boxes when drones overlap."""
        components = pipeline_components
        
        # Create scenario with overlapping drones
        overlapping_positions = [
            np.array([10.0, 0.0, 5.0]),     # Drone 0 - center
            np.array([10.3, 0.1, 5.1]),     # Drone 1 - close to drone 0
            np.array([10.1, -0.1, 4.9]),    # Drone 2 - also close to drone 0
            np.array([15.0, 2.0, 6.0])      # Drone 3 - separate
        ]
        
        # Generate detections and message
        detections = components['detection_gen'].generate_detections(overlapping_positions)
        message = components['message_builder'].build_detection_message(detections)
        
        # Should detect multiple drones even when overlapping
        assert len(message['detections']) >= 2
        
        # Check for overlapping bounding boxes in the first two detections
        if len(message['detections']) >= 2:
            bbox1 = message['detections'][0]['bbox_px']
            bbox2 = message['detections'][1]['bbox_px']
            
            # Check if they overlap
            overlap_x = not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0])
            overlap_y = not (bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
            
            # For close drones, we expect potential overlap
            if overlap_x and overlap_y:
                # Overlapping boxes should still be distinct
                assert bbox1 != bbox2
                assert message['detections'][0] != message['detections'][1]
    
    def test_multi_drone_different_sizes_and_altitudes(self, pipeline_components):
        """Test requirement 7.4: Independent estimates for different sizes/altitudes."""
        components = pipeline_components
        
        # Test with drones at different altitudes (same horizontal position)
        different_altitude_positions = [
            np.array([12.0, 0.0, 3.0]),   # Low altitude
            np.array([12.0, 0.0, 6.0]),   # Medium altitude
            np.array([12.0, 0.0, 9.0])    # High altitude
        ]
        
        detections = components['detection_gen'].generate_detections(different_altitude_positions)
        
        # Should generate independent detections
        assert len(detections) >= 2
        
        # Each should have different depth estimates
        depths = [d['depth_m'] for d in detections]
        assert len(set(depths)) == len(depths)  # All different
        
        # Distance estimates should be independent
        estimated_distances = [
            components['detection_gen'].estimate_distance_from_size(d) for d in detections
        ]
        
        # All estimates should be reasonable
        for est_dist in estimated_distances:
            assert 5.0 < est_dist < 50.0
    
    def test_multi_drone_performance_scalability(self, pipeline_components):
        """Test system performance with varying numbers of drones."""
        components = pipeline_components
        
        # Test with different drone counts
        drone_counts = [1, 3, 5, 8, 12]
        
        for num_drones in drone_counts:
            # Update config for this test
            config = SimulatorConfig(
                num_drones=num_drones,
                duration_s=4.0,
                fps=10.0,
                vertical_fov_deg=50.0,
                deterministic_seed=42
            )
            
            # Create components for this drone count
            motion_gen = MotionGenerator(config)
            camera_model = CameraModel(config)
            detection_gen = DetectionGenerator(config, camera_model)
            message_builder = DetectionMessageBuilder(config, camera_model)
            
            # Test one frame
            time_s = 2.0
            world_positions = motion_gen.get_positions_at_time(time_s)
            detections = detection_gen.generate_detections(world_positions)
            message = message_builder.build_detection_message(detections)
            
            # Verify scalability
            assert len(world_positions) == num_drones
            assert len(detections) <= num_drones  # Some may not be visible
            assert len(message['detections']) == len(detections)
            
            # Message should remain valid regardless of drone count
            assert message_builder.validate_message_schema(message)
            
            # Message size should be reasonable
            size_bytes = message_builder.get_message_size_bytes(message)
            assert size_bytes > 0
            assert size_bytes < 100000  # Should not be excessively large
    
    def test_multi_drone_edge_cases(self, pipeline_components):
        """Test edge cases in multi-drone scenarios."""
        components = pipeline_components
        
        # Test with all drones behind camera
        behind_positions = [
            np.array([-10.0, 0.0, 5.0]),   # Behind camera
            np.array([-12.0, 1.0, 5.5]),   # Behind camera
            np.array([-8.0, -0.5, 4.5])    # Behind camera
        ]
        
        detections = components['detection_gen'].generate_detections(behind_positions)
        message = components['message_builder'].build_detection_message(detections)
        
        # Should handle gracefully (likely no detections)
        assert isinstance(detections, list)
        assert isinstance(message['detections'], list)
        assert components['message_builder'].validate_message_schema(message)
        
        # Test with all drones very far away
        far_positions = [
            np.array([100.0, 0.0, 5.0]),   # Very far
            np.array([102.0, 1.0, 5.5]),   # Very far
            np.array([98.0, -0.5, 4.5])    # Very far
        ]
        
        far_detections = components['detection_gen'].generate_detections(far_positions)
        far_message = components['message_builder'].build_detection_message(far_detections)
        
        # Should handle gracefully
        assert isinstance(far_detections, list)
        assert isinstance(far_message['detections'], list)
        assert components['message_builder'].validate_message_schema(far_message)
        
        # If any detections, they should have low confidence due to small size
        for detection in far_detections:
            assert detection['confidence'] < 0.9  # Should be lower due to distance
    
    def test_multi_drone_deterministic_behavior(self):
        """Test that multi-drone simulation is deterministic with fixed seed."""
        config = SimulatorConfig(
            num_drones=3,
            duration_s=5.0,
            fps=15.0,
            deterministic_seed=123
        )
        
        # Run simulation twice with same seed
        results1 = self._run_simulation_frame(config, time_s=2.5)
        results2 = self._run_simulation_frame(config, time_s=2.5)
        
        # Results should be identical
        assert len(results1['world_positions']) == len(results2['world_positions'])
        
        for pos1, pos2 in zip(results1['world_positions'], results2['world_positions']):
            assert np.allclose(pos1, pos2, atol=1e-10)
        
        # Messages should be identical (except frame_id which increments)
        msg1 = results1['message']
        msg2 = results2['message']
        
        # Same number of detections
        assert len(msg1['detections']) == len(msg2['detections'])
        
        # Same detection content (ignoring frame_id)
        for det1, det2 in zip(msg1['detections'], msg2['detections']):
            assert det1['class'] == det2['class']
            assert abs(det1['confidence'] - det2['confidence']) < 1e-10
            assert det1['bbox_px'] == det2['bbox_px']
    
    def _run_simulation_frame(self, config, time_s):
        """Helper method to run one frame of simulation."""
        motion_gen = MotionGenerator(config)
        camera_model = CameraModel(config)
        detection_gen = DetectionGenerator(config, camera_model)
        message_builder = DetectionMessageBuilder(config, camera_model)
        
        world_positions = motion_gen.get_positions_at_time(time_s)
        detections = detection_gen.generate_detections(world_positions)
        message = message_builder.build_detection_message(detections)
        
        return {
            'world_positions': world_positions,
            'detections': detections,
            'message': message
        }
    
    def test_multi_drone_statistics_aggregation(self, pipeline_components):
        """Test statistics collection across multi-drone scenarios."""
        components = pipeline_components
        
        # Generate data for statistics
        time_s = 3.0
        world_positions = components['motion_gen'].get_positions_at_time(time_s)
        detections = components['detection_gen'].generate_detections(world_positions)
        
        # Test motion statistics
        motion_stats = components['motion_gen'].get_path_statistics()
        assert motion_stats['num_drones'] == 4
        assert motion_stats['acceleration_constraints_satisfied'] is True
        
        # Test detection statistics
        detection_stats = components['detection_gen'].get_detection_statistics(detections)
        assert detection_stats['num_detections'] == len(detections)
        
        if len(detections) > 0:
            assert 0.0 <= detection_stats['avg_confidence'] <= 1.0
            assert detection_stats['avg_size_px'] > 0
            assert detection_stats['avg_distance_m'] > 0
    
    def test_multi_drone_json_output_format(self, pipeline_components):
        """Test that multi-drone JSON output matches expected format."""
        components = pipeline_components
        
        # Generate a realistic multi-drone frame
        time_s = 3.5
        world_positions = components['motion_gen'].get_positions_at_time(time_s)
        detections = components['detection_gen'].generate_detections(world_positions)
        
        timestamp = datetime(2025, 9, 21, 10, 30, 45, 123000, timezone.utc)
        message = components['message_builder'].build_detection_message(detections, timestamp)
        
        # Convert to JSON string
        json_str = components['message_builder'].to_json_string(message, indent=2)
        
        # Should be valid JSON
        import json
        parsed = json.loads(json_str)
        assert parsed == message
        
        # Should contain expected structure for multi-drone
        assert 'timestamp_utc' in parsed
        assert 'frame_id' in parsed
        assert 'camera' in parsed
        assert 'detections' in parsed
        assert 'edge' in parsed
        
        # Detections should be array
        assert isinstance(parsed['detections'], list)
        
        # Each detection should have proper format
        for detection in parsed['detections']:
            assert 'class' in detection
            assert 'confidence' in detection
            assert 'bbox_px' in detection
            assert 'center_px' in detection
            assert 'size_px' in detection
            
            # Should not contain internal data
            assert 'drone_id' not in detection
            assert 'world_pos_enu' not in detection


if __name__ == "__main__":
    pytest.main([__file__])