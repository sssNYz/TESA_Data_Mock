"""
Tests for simulator error handling and recovery.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.simulator import DroneSimulator
from drone_detection_simulator.error_handling import SimulationError, ConfigurationError


class TestSimulatorErrorHandling:
    """Test error handling in DroneSimulator."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulatorConfig(
            duration_s=1.0,
            fps=2.0,
            offline_mode=True,
            deterministic_seed=42
        )
    
    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid configuration."""
        # Create invalid config
        with pytest.raises(ValueError):  # Should fail during config validation
            invalid_config = SimulatorConfig(duration_s=-1.0)
    
    def test_initialization_component_failure(self, config):
        """Test initialization when component initialization fails."""
        with patch('drone_detection_simulator.simulator.CameraModel', side_effect=Exception("Camera init failed")):
            with pytest.raises(SimulationError, match="Camera model initialization failed"):
                DroneSimulator(config)
    
    def test_initialization_statistics_tracking(self, config):
        """Test that initialization statistics are tracked."""
        simulator = DroneSimulator(config)
        
        assert 'initialization_time_s' in simulator.stats
        assert simulator.stats['initialization_time_s'] > 0
        assert len(simulator.initialization_errors) == 0
    
    @patch('drone_detection_simulator.simulator.CameraModel')
    @patch('drone_detection_simulator.simulator.MotionGenerator')
    def test_partial_initialization_failure(self, mock_motion, mock_camera, config):
        """Test behavior when some components fail to initialize."""
        # Camera succeeds, motion fails
        mock_camera.return_value = Mock()
        mock_motion.side_effect = Exception("Motion init failed")
        
        with pytest.raises(SimulationError, match="Motion generator initialization failed"):
            DroneSimulator(config)
    
    def test_signal_handler_setup_failure(self, config, caplog):
        """Test graceful handling of signal handler setup failure."""
        with patch('signal.signal', side_effect=OSError("Signal setup failed")):
            # Should not fail initialization, just log warning
            simulator = DroneSimulator(config)
            assert "Failed to setup signal handlers" in caplog.text
    
    def test_simulator_state_validation(self, config):
        """Test simulator state validation."""
        simulator = DroneSimulator(config)
        
        # Valid state
        assert simulator._validate_simulator_state() == True
        
        # Missing component
        delattr(simulator, 'camera_model')
        assert simulator._validate_simulator_state() == False
    
    def test_mqtt_connection_failure_handling(self, config, caplog):
        """Test handling of MQTT connection failures."""
        config.offline_mode = False  # Enable MQTT mode
        
        with patch('drone_detection_simulator.simulator.MQTTPublisher') as mock_publisher_class:
            mock_publisher = Mock()
            mock_publisher.connect.return_value = False
            mock_publisher_class.return_value = mock_publisher
            
            simulator = DroneSimulator(config)
            
            with caplog.at_level("ERROR"):
                result = simulator.run()
            
            assert "Failed to connect to MQTT broker" in caplog.text
            assert result['publishing']['messages_published'] == 0
    
    def test_simulation_loop_failure(self, config):
        """Test handling of simulation loop failures."""
        simulator = DroneSimulator(config)
        
        with patch.object(simulator.simulation_loop, 'run', side_effect=Exception("Loop failed")):
            with pytest.raises(SimulationError, match="Simulation loop failed"):
                simulator.run()
    
    def test_frame_processing_error_recovery(self, config, caplog):
        """Test error recovery during frame processing."""
        simulator = DroneSimulator(config)
        
        # Mock motion generator to fail on some frames
        original_get_positions = simulator.motion_generator.get_positions_at_time
        call_count = 0
        
        def failing_get_positions(time_s):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second frame
                raise Exception("Motion generation failed")
            return original_get_positions(time_s)
        
        simulator.motion_generator.get_positions_at_time = failing_get_positions
        
        with caplog.at_level("WARNING"):
            result = simulator.run()
        
        # Should continue processing other frames
        assert result['simulation']['frames_processed'] > 0
        assert "motion generation for frame" in caplog.text
        assert len(simulator.runtime_errors) > 0
    
    def test_detection_processing_error_recovery(self, config, caplog):
        """Test error recovery during detection processing."""
        simulator = DroneSimulator(config)
        
        # Mock noise model to fail occasionally
        original_apply_noise = simulator.noise_model.apply_detection_noise
        
        def failing_apply_noise(detection):
            if detection.get('test_fail', False):
                raise Exception("Noise application failed")
            return original_apply_noise(detection)
        
        simulator.noise_model.apply_detection_noise = failing_apply_noise
        
        # Mock detection generator to include failing detection
        original_generate = simulator.detection_generator.generate_detections
        
        def generate_with_failure(positions):
            detections = original_generate(positions)
            if detections:
                detections[0]['test_fail'] = True  # Mark first detection to fail
            return detections
        
        simulator.detection_generator.generate_detections = generate_with_failure
        
        with caplog.at_level("WARNING"):
            result = simulator.run()
        
        # Should continue processing
        assert "Error processing detection" in caplog.text
        assert len(simulator.runtime_errors) > 0
    
    def test_message_building_failure_recovery(self, config, caplog):
        """Test recovery from message building failures."""
        simulator = DroneSimulator(config)
        
        # Mock message builder to fail
        simulator.message_builder.build_detection_message = Mock(side_effect=Exception("Message build failed"))
        
        with caplog.at_level("ERROR"):
            result = simulator.run()
        
        assert "message building for frame" in caplog.text
        assert result['simulation']['frame_errors'] > 0
    
    def test_publish_failure_handling(self, config):
        """Test handling of publish failures."""
        simulator = DroneSimulator(config)
        
        # Mock publisher to fail
        simulator.mqtt_publisher.publish_detection = Mock(return_value=False)
        
        result = simulator.run()
        
        # Should track publish failures
        assert result['publishing']['publish_failures'] > 0
        assert result['publishing']['messages_published'] == 0
    
    def test_keyboard_interrupt_handling(self, config, caplog):
        """Test graceful handling of keyboard interrupt."""
        simulator = DroneSimulator(config)
        
        # Mock simulation loop to raise KeyboardInterrupt
        with patch.object(simulator.simulation_loop, 'run', side_effect=KeyboardInterrupt()):
            with caplog.at_level("INFO"):
                result = simulator.run()
            
            assert "Simulation interrupted by user" in caplog.text
            assert 'simulation' in result
    
    def test_cleanup_error_handling(self, config, caplog):
        """Test error handling during cleanup."""
        simulator = DroneSimulator(config)
        
        # Mock components to fail during cleanup
        simulator.mqtt_publisher.disconnect = Mock(side_effect=Exception("Disconnect failed"))
        simulator.simulation_loop.shutdown = Mock(side_effect=Exception("Shutdown failed"))
        
        with caplog.at_level("ERROR"):
            simulator._cleanup()
        
        assert "Error stopping simulation loop" in caplog.text
        assert "Error disconnecting MQTT publisher" in caplog.text
        assert "Cleanup completed with 2 errors" in caplog.text
        assert simulator.running == False  # Should still set running to False
    
    def test_cleanup_critical_error(self, config, caplog):
        """Test cleanup with critical error."""
        simulator = DroneSimulator(config)
        
        # Mock hasattr to fail (simulating critical error)
        with patch('builtins.hasattr', side_effect=Exception("Critical error")):
            with caplog.at_level("ERROR"):
                simulator._cleanup()
            
            assert "Critical error during cleanup" in caplog.text
            assert simulator.running == False
    
    def test_error_summary_in_results(self, config):
        """Test that error summary is included in results when errors occur."""
        simulator = DroneSimulator(config)
        
        # Add some runtime errors
        simulator.runtime_errors = [
            ValueError("Error 1"),
            ValueError("Error 2"),
            TypeError("Error 3")
        ]
        
        result = simulator.run()
        
        assert 'errors' in result
        assert result['errors']['total_errors'] == 3
        assert result['errors']['error_counts']['ValueError'] == 2
        assert result['errors']['error_counts']['TypeError'] == 1
    
    def test_statistics_tracking_with_errors(self, config):
        """Test that statistics are properly tracked even with errors."""
        simulator = DroneSimulator(config)
        
        # Mock some components to cause errors
        simulator.mqtt_publisher.publish_detection = Mock(return_value=False)
        
        result = simulator.run()
        
        # Should have comprehensive statistics
        assert 'simulation' in result
        assert 'detections' in result
        assert 'publishing' in result
        assert 'timing' in result
        
        assert result['simulation']['frames_processed'] >= 0
        assert result['publishing']['publish_failures'] > 0
        assert result['simulation']['total_runtime_s'] > 0
    
    def test_progress_logging_with_errors(self, config, caplog):
        """Test that progress is logged even when errors occur."""
        # Use longer duration to ensure progress logging
        config.duration_s = 2.0
        config.fps = 5.0
        
        simulator = DroneSimulator(config)
        
        with caplog.at_level("INFO"):
            result = simulator.run()
        
        # Should have progress messages
        progress_messages = [record for record in caplog.records if "Progress:" in record.message]
        assert len(progress_messages) > 0
    
    def test_shutdown_during_simulation(self, config):
        """Test shutdown request during simulation."""
        simulator = DroneSimulator(config)
        
        # Mock simulation loop to call shutdown after first frame
        original_run = simulator.simulation_loop.run
        
        def run_with_shutdown(process_frame_func):
            # Process one frame then shutdown
            from drone_detection_simulator.timing import FrameInfo
            frame_info = FrameInfo(
                frame_id=1,
                simulation_time_s=0.0,
                timestamp_utc="2023-01-01T00:00:00Z",
                processing_latency_ms=50.0
            )
            process_frame_func(frame_info)
            simulator.shutdown()
            return {'actual_fps': 1.0, 'timing_error_ms': 0.0}
        
        simulator.simulation_loop.run = run_with_shutdown
        
        result = simulator.run()
        
        assert simulator.running == False
        assert simulator.shutdown_event.is_set()
    
    def test_component_health_monitoring(self, config):
        """Test monitoring of component health."""
        config.offline_mode = False
        simulator = DroneSimulator(config)
        
        # Test MQTT publisher health check
        simulator.mqtt_publisher.is_healthy = Mock(return_value=False)
        
        # Should still validate as True but log warning
        with patch('drone_detection_simulator.simulator.logger') as mock_logger:
            result = simulator._validate_simulator_state()
            assert result == True
            mock_logger.warning.assert_called_with("MQTT publisher is not in a healthy state")
    
    def test_should_continue_without_mqtt(self, config):
        """Test decision logic for continuing without MQTT."""
        simulator = DroneSimulator(config)
        
        # Offline mode should continue
        simulator.config.offline_mode = True
        assert simulator._should_continue_without_mqtt() == True
        
        # Online mode should not continue (current implementation)
        simulator.config.offline_mode = False
        assert simulator._should_continue_without_mqtt() == False


class TestSimulatorRecoveryScenarios:
    """Test various recovery scenarios."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulatorConfig(
            duration_s=0.5,
            fps=4.0,
            offline_mode=True,
            deterministic_seed=42
        )
    
    def test_intermittent_motion_failures(self, config, caplog):
        """Test recovery from intermittent motion generation failures."""
        simulator = DroneSimulator(config)
        
        # Mock motion generator to fail every other call
        original_method = simulator.motion_generator.get_positions_at_time
        call_count = 0
        
        def intermittent_failure(time_s):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception(f"Intermittent failure {call_count}")
            return original_method(time_s)
        
        simulator.motion_generator.get_positions_at_time = intermittent_failure
        
        with caplog.at_level("WARNING"):
            result = simulator.run()
        
        # Should process some frames successfully
        assert result['simulation']['frames_processed'] > 0
        assert len(simulator.runtime_errors) > 0
        
        # Should have multiple error messages
        error_messages = [record for record in caplog.records if "motion generation" in record.message]
        assert len(error_messages) > 0
    
    def test_cascading_component_failures(self, config, caplog):
        """Test handling of cascading component failures."""
        simulator = DroneSimulator(config)
        
        # Make multiple components fail
        simulator.motion_generator.get_positions_at_time = Mock(side_effect=Exception("Motion failed"))
        simulator.detection_generator.generate_detections = Mock(side_effect=Exception("Detection failed"))
        simulator.noise_model.apply_detection_noise = Mock(side_effect=Exception("Noise failed"))
        
        with caplog.at_level("WARNING"):
            result = simulator.run()
        
        # Should handle all failures gracefully
        assert len(simulator.runtime_errors) > 0
        assert result['simulation']['frame_errors'] > 0
        
        # Should have error messages from multiple components
        assert any("motion generation" in record.message for record in caplog.records)
    
    def test_memory_pressure_simulation(self, config):
        """Test behavior under simulated memory pressure."""
        simulator = DroneSimulator(config)
        
        # Mock components to simulate memory issues
        def memory_pressure_detection(positions):
            # Simulate memory allocation failure
            if len(positions) > 0:
                raise MemoryError("Simulated memory pressure")
            return []
        
        simulator.detection_generator.generate_detections = memory_pressure_detection
        
        result = simulator.run()
        
        # Should handle memory errors gracefully
        assert len(simulator.runtime_errors) > 0
        assert result['simulation']['frames_processed'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])