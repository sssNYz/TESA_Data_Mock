"""
Integration tests for the complete simulation cycle.

Tests the DroneSimulator class and main simulation orchestrator functionality,
validating requirements 5.1 and 5.3.
"""

import pytest
import tempfile
import json
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.simulator import DroneSimulator, main


class TestSimulatorIntegration:
    """Integration tests for complete simulation cycle."""
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration for integration testing."""
        return SimulatorConfig(
            # Short simulation for fast testing
            duration_s=2.0,
            fps=10.0,
            num_drones=2,
            
            # Offline mode for testing
            offline_mode=True,
            
            # Deterministic behavior
            deterministic_seed=42,
            
            # Reasonable parameters
            vertical_fov_deg=50.0,
            path_span_m=20.0,
            speed_mps=3.0,
            path_altitude_agl_m=5.0,
            
            # Reduced noise for predictable testing
            pixel_centroid_sigma_px=0.5,
            bbox_size_sigma_px=1.0,
            confidence_noise=0.02,
            miss_rate_small=0.01,
            false_positive_rate=0.005
        )
    
    @pytest.fixture
    def mqtt_config(self):
        """Create configuration for MQTT testing."""
        return SimulatorConfig(
            duration_s=1.0,
            fps=5.0,
            num_drones=1,
            offline_mode=False,  # Enable MQTT
            deterministic_seed=42,
            mqtt_host="localhost",
            mqtt_port=1883,
            mqtt_topic="test/detections"
        )
    
    def test_simulator_initialization(self, test_config):
        """Test requirement 5.1: Proper initialization of all components."""
        simulator = DroneSimulator(test_config)
        
        # Verify all components are initialized
        assert simulator.config == test_config
        assert simulator.camera_model is not None
        assert simulator.motion_generator is not None
        assert simulator.detection_generator is not None
        assert simulator.noise_model is not None
        assert simulator.message_builder is not None
        assert simulator.mqtt_publisher is not None
        assert simulator.simulation_loop is not None
        
        # Verify random number generator setup
        assert simulator.rng is not None
        
        # Verify statistics initialization
        assert simulator.stats is not None
        assert simulator.stats['frames_processed'] == 0
        
        # Verify shutdown handling
        assert not simulator.running
        assert not simulator.shutdown_event.is_set()
    
    def test_complete_simulation_cycle_offline(self, test_config):
        """Test requirement 5.1: Complete simulation cycle in offline mode."""
        # Capture printed output
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            simulator = DroneSimulator(test_config)
            results = simulator.run()
        
        # Verify simulation completed successfully
        assert results is not None
        assert 'simulation' in results
        assert 'detections' in results
        assert 'publishing' in results
        assert 'timing' in results
        
        # Verify expected number of frames processed
        expected_frames = int(test_config.duration_s * test_config.fps)
        assert results['simulation']['frames_processed'] == expected_frames
        assert results['simulation']['completion_rate'] == 1.0
        
        # Verify detections were generated
        assert results['detections']['total_generated'] >= 0
        
        # Verify offline mode worked (JSON messages printed)
        assert results['publishing']['offline_mode'] is True
        assert len(captured_output) > 0
        
        # Verify printed messages are valid JSON
        valid_json_count = 0
        for output in captured_output:
            if output.strip():
                try:
                    parsed = json.loads(output)
                    # Verify basic message structure
                    assert 'timestamp_utc' in parsed
                    assert 'frame_id' in parsed
                    assert 'camera' in parsed
                    assert 'detections' in parsed
                    assert 'edge' in parsed
                    valid_json_count += 1
                except json.JSONDecodeError:
                    pass  # Skip non-JSON output (like log messages)
        
        assert valid_json_count > 0, "No valid JSON messages found in output"
    
    def test_simulation_with_mqtt_connection_failure(self, mqtt_config):
        """Test graceful handling of MQTT connection failures."""
        # Mock MQTT client to simulate connection failure
        with patch('paho.mqtt.client.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_client.connect.side_effect = Exception("Connection failed")
            
            simulator = DroneSimulator(mqtt_config)
            results = simulator.run()
            
            # Simulation should complete even with MQTT failure
            assert results is not None
            assert results['simulation']['completion_rate'] == 1.0
            
            # Should have attempted connection
            mock_client.connect.assert_called()
    
    def test_simulation_graceful_shutdown(self, test_config):
        """Test graceful shutdown during simulation."""
        simulator = DroneSimulator(test_config)
        
        # Start simulation in separate thread
        results = {}
        exception = {}
        
        def run_simulation():
            try:
                results['data'] = simulator.run()
            except Exception as e:
                exception['error'] = e
        
        # Start simulation
        sim_thread = threading.Thread(target=run_simulation)
        sim_thread.start()
        
        # Wait a bit then shutdown
        time.sleep(0.1)
        simulator.shutdown()
        
        # Wait for completion
        sim_thread.join(timeout=5.0)
        
        # Should complete without errors
        assert 'error' not in exception
        assert 'data' in results
        assert results['data'] is not None
    
    def test_simulation_error_handling(self, test_config):
        """Test error handling during simulation."""
        simulator = DroneSimulator(test_config)
        
        # Mock a component to raise an error during frame processing
        original_generate = simulator.detection_generator.generate_detections
        
        def failing_generate(positions):
            if simulator.stats['frames_processed'] == 5:  # Fail on 6th frame
                raise Exception("Simulated detection failure")
            return original_generate(positions)
        
        simulator.detection_generator.generate_detections = failing_generate
        
        # Simulation should continue despite error
        results = simulator.run()
        
        # Should have processed some frames before and after error
        assert results['simulation']['frames_processed'] > 0
        # May not complete all frames due to error, but should not crash
    
    def test_simulation_statistics_accuracy(self, test_config):
        """Test accuracy of simulation statistics collection."""
        simulator = DroneSimulator(test_config)
        results = simulator.run()
        
        # Verify statistics consistency
        stats = results
        
        # Frame statistics
        expected_frames = int(test_config.duration_s * test_config.fps)
        assert stats['simulation']['frames_processed'] == expected_frames
        assert stats['simulation']['duration_s'] == test_config.duration_s
        assert stats['simulation']['fps'] == test_config.fps
        
        # Detection statistics should be reasonable
        total_detections = stats['detections']['total_generated']
        assert total_detections >= 0
        
        if total_detections > 0:
            avg_per_frame = stats['detections']['avg_per_frame']
            assert avg_per_frame == total_detections / expected_frames
        
        # Publishing statistics
        assert stats['publishing']['messages_published'] == expected_frames
        assert stats['publishing']['publish_failures'] == 0  # Offline mode
        assert stats['publishing']['success_rate'] == 1.0
        
        # Configuration verification
        assert stats['configuration']['num_drones'] == test_config.num_drones
        assert stats['configuration']['deterministic_seed'] == test_config.deterministic_seed
    
    def test_simulation_timing_accuracy(self, test_config):
        """Test timing accuracy of simulation loop."""
        # Use higher FPS for more precise timing test
        timing_config = SimulatorConfig(
            duration_s=1.0,
            fps=20.0,
            num_drones=1,
            offline_mode=True,
            deterministic_seed=42
        )
        
        simulator = DroneSimulator(timing_config)
        start_time = time.time()
        results = simulator.run()
        end_time = time.time()
        
        # Verify timing statistics
        timing_stats = results['timing']
        assert 'actual_fps' in timing_stats
        assert 'timing_error_ms' in timing_stats
        
        # Actual runtime should be close to expected duration
        actual_runtime = end_time - start_time
        expected_runtime = timing_config.duration_s
        
        # Allow some tolerance for processing overhead
        assert actual_runtime >= expected_runtime
        assert actual_runtime < expected_runtime + 1.0  # Max 1 second overhead
        
        # FPS should be reasonably close to target
        actual_fps = timing_stats['actual_fps']
        target_fps = timing_config.fps
        fps_error = abs(actual_fps - target_fps) / target_fps
        assert fps_error < 0.2  # Within 20% of target FPS
    
    def test_simulation_deterministic_behavior(self):
        """Test requirement 5.4: Deterministic behavior with fixed seed."""
        config = SimulatorConfig(
            duration_s=1.0,
            fps=10.0,
            num_drones=2,
            offline_mode=True,
            deterministic_seed=123
        )
        
        # Capture output from both runs
        outputs1 = []
        outputs2 = []
        
        def capture_output1(*args, **kwargs):
            outputs1.append(args[0] if args else "")
        
        def capture_output2(*args, **kwargs):
            outputs2.append(args[0] if args else "")
        
        # Run simulation twice with same seed
        with patch('builtins.print', side_effect=capture_output1):
            simulator1 = DroneSimulator(config)
            results1 = simulator1.run()
        
        with patch('builtins.print', side_effect=capture_output2):
            simulator2 = DroneSimulator(config)
            results2 = simulator2.run()
        
        # Results should be identical (except frame_id increments)
        assert results1['simulation']['frames_processed'] == results2['simulation']['frames_processed']
        assert results1['detections']['total_generated'] == results2['detections']['total_generated']
        
        # Parse JSON outputs and compare (ignoring frame_id)
        json_messages1 = []
        json_messages2 = []
        
        for output in outputs1:
            if output.strip():
                try:
                    json_messages1.append(json.loads(output))
                except json.JSONDecodeError:
                    pass
        
        for output in outputs2:
            if output.strip():
                try:
                    json_messages2.append(json.loads(output))
                except json.JSONDecodeError:
                    pass
        
        assert len(json_messages1) == len(json_messages2)
        
        # Compare messages (ignoring frame_id which increments)
        for msg1, msg2 in zip(json_messages1, json_messages2):
            # Same number of detections
            assert len(msg1['detections']) == len(msg2['detections'])
            
            # Same detection content
            for det1, det2 in zip(msg1['detections'], msg2['detections']):
                assert det1['class'] == det2['class']
                assert abs(det1['confidence'] - det2['confidence']) < 1e-10
                # Bounding boxes should be identical
                for coord1, coord2 in zip(det1['bbox_px'], det2['bbox_px']):
                    assert abs(coord1 - coord2) < 1e-10
    
    def test_simulation_progress_tracking(self, test_config):
        """Test simulation progress tracking functionality."""
        simulator = DroneSimulator(test_config)
        
        # Initially not running
        assert not simulator.is_running()
        assert simulator.get_progress() == 0.0
        
        # Track progress during simulation
        progress_values = []
        
        def track_progress():
            while simulator.is_running():
                progress_values.append(simulator.get_progress())
                time.sleep(0.05)
        
        # Start progress tracking
        progress_thread = threading.Thread(target=track_progress)
        progress_thread.start()
        
        # Run simulation
        results = simulator.run()
        
        # Wait for progress tracking to complete
        progress_thread.join(timeout=2.0)
        
        # Verify progress tracking
        assert len(progress_values) > 0
        assert all(0.0 <= p <= 1.0 for p in progress_values)
        
        # Progress should generally increase
        if len(progress_values) > 1:
            assert progress_values[-1] >= progress_values[0]
    
    def test_main_function_with_config_file(self):
        """Test main function with configuration file loading."""
        # Create temporary config file
        config_data = {
            'duration_s': 0.5,
            'fps': 5.0,
            'num_drones': 1,
            'offline_mode': True,
            'deterministic_seed': 42
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Capture output
            captured_output = []
            
            def mock_print(*args, **kwargs):
                captured_output.append(args[0] if args else "")
            
            with patch('builtins.print', side_effect=mock_print):
                results = main(config_path)
            
            # Verify results
            assert results is not None
            assert results['simulation']['duration_s'] == 0.5
            assert results['simulation']['fps'] == 5.0
            assert results['configuration']['num_drones'] == 1
            
            # Should have generated JSON output
            assert len(captured_output) > 0
            
        finally:
            # Clean up temp file
            Path(config_path).unlink()
    
    def test_main_function_with_overrides(self):
        """Test main function with configuration overrides."""
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            results = main(
                config_path=None,
                duration_s=0.5,
                fps=8.0,
                num_drones=2,
                offline_mode=True,
                deterministic_seed=99
            )
        
        # Verify overrides were applied
        assert results['simulation']['duration_s'] == 0.5
        assert results['simulation']['fps'] == 8.0
        assert results['configuration']['num_drones'] == 2
        assert results['configuration']['deterministic_seed'] == 99
        
        # Should have generated output
        assert len(captured_output) > 0
    
    def test_simulation_component_integration(self, test_config):
        """Test integration between all simulation components."""
        simulator = DroneSimulator(test_config)
        
        # Verify component integration
        # Camera model should be shared between detection generator and message builder
        assert simulator.detection_generator.camera_model is simulator.camera_model
        assert simulator.message_builder.camera_model is simulator.camera_model
        
        # Random number generator should be shared for deterministic behavior
        # (Note: components may create their own RNG instances, but seed should be consistent)
        
        # Configuration should be consistent across components
        assert simulator.motion_generator.config is simulator.config
        assert simulator.detection_generator.config is simulator.config
        assert simulator.noise_model.config is simulator.config
        assert simulator.message_builder.config is simulator.config
        assert simulator.mqtt_publisher.config is simulator.config
        
        # Run one frame to test integration
        results = simulator.run()
        
        # Verify successful integration
        assert results['simulation']['frames_processed'] > 0
        assert results['publishing']['success_rate'] == 1.0
    
    def test_simulation_memory_usage(self, test_config):
        """Test that simulation doesn't have excessive memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run simulation
        simulator = DroneSimulator(test_config)
        results = simulator.run()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for short simulation)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        # Verify simulation completed successfully
        assert results['simulation']['completion_rate'] == 1.0
    
    def test_simulation_cleanup(self, test_config):
        """Test proper cleanup after simulation."""
        simulator = DroneSimulator(test_config)
        
        # Run simulation
        results = simulator.run()
        
        # Verify cleanup state
        assert not simulator.is_running()
        assert simulator.shutdown_event.is_set() or not simulator.running
        
        # MQTT publisher should be disconnected
        assert not simulator.mqtt_publisher.connected
        
        # Should be able to run again after cleanup
        simulator2 = DroneSimulator(test_config)
        results2 = simulator2.run()
        
        assert results2['simulation']['completion_rate'] == 1.0


class TestSimulatorErrorHandling:
    """Test error handling in simulation scenarios."""
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # Test invalid FPS
        with pytest.raises(ValueError):
            config = SimulatorConfig(fps=-1.0)
            DroneSimulator(config)
        
        # Test invalid duration
        with pytest.raises(ValueError):
            config = SimulatorConfig(duration_s=0.0)
            DroneSimulator(config)
        
        # Test invalid drone count
        with pytest.raises(ValueError):
            config = SimulatorConfig(num_drones=0)
            DroneSimulator(config)
    
    def test_component_initialization_failure(self):
        """Test handling of component initialization failures."""
        config = SimulatorConfig(
            duration_s=1.0,
            fps=10.0,
            offline_mode=True,
            # Invalid focal length configuration
            focal_length_mm=None,
            sensor_height_mm=None,
            vertical_fov_deg=None
        )
        
        with pytest.raises(ValueError):
            DroneSimulator(config)
    
    def test_mqtt_configuration_errors(self):
        """Test MQTT configuration error handling."""
        config = SimulatorConfig(
            duration_s=0.5,
            fps=5.0,
            offline_mode=False,
            mqtt_port=0  # Invalid port
        )
        
        with pytest.raises(ValueError):
            DroneSimulator(config)


if __name__ == "__main__":
    pytest.main([__file__])