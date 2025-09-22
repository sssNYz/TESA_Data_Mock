"""
Simplified performance tests for the drone detection simulator.

Tests core performance optimization, monitoring, and validation functionality.
"""

import unittest
import time
import gc
from unittest.mock import patch
import numpy as np

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.performance import (
    PerformanceMonitor, PerformanceOptimizer, FrameRateValidator,
    create_performance_context, cleanup_performance_context
)
from drone_detection_simulator.simulator import DroneSimulator


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = SimulatorConfig(
            fps=10.0,
            duration_s=1.0,
            offline_mode=True,
            deterministic_seed=42
        )
        self.monitor = PerformanceMonitor(self.test_config, max_samples=100)
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        self.assertEqual(self.monitor.config, self.test_config)
        self.assertEqual(self.monitor.max_samples, 100)
        self.assertEqual(self.monitor.metrics.target_fps, self.test_config.fps)
        self.assertFalse(self.monitor.monitoring_active)
        self.assertEqual(self.monitor.target_frame_time, 1.0 / self.test_config.fps)
    
    def test_monitor_start_stop(self):
        """Test starting and stopping performance monitoring."""
        # Initially not monitoring
        self.assertFalse(self.monitor.monitoring_active)
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        self.assertIsNotNone(self.monitor.monitor_thread)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_frame_timing_recording(self):
        """Test frame timing recording."""
        self.monitor.start_monitoring()
        
        try:
            # Record several frames
            for i in range(3):
                frame_start = self.monitor.record_frame_start()
                time.sleep(0.02)  # 20ms processing time
                self.monitor.record_frame_end(frame_start)
            
            # Check recorded data
            metrics = self.monitor.get_current_metrics()
            self.assertEqual(metrics.frames_processed, 3)
            self.assertEqual(len(metrics.frame_times), 3)
            self.assertEqual(len(metrics.frame_intervals), 2)  # N-1 intervals
            self.assertEqual(len(metrics.processing_times), 3)
            
            # Check processing times are reasonable
            for proc_time in metrics.processing_times:
                self.assertGreaterEqual(proc_time, 0.015)  # At least 15ms
                self.assertLessEqual(proc_time, 0.030)     # At most 30ms
        
        finally:
            self.monitor.stop_monitoring()
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        self.monitor.start_monitoring()
        
        try:
            # Record some frames
            for i in range(2):
                frame_start = self.monitor.record_frame_start()
                time.sleep(0.02)
                self.monitor.record_frame_end(frame_start)
            
            # Get performance summary
            summary = self.monitor.get_performance_summary()
            
            # Check summary structure
            self.assertIn('frames_processed', summary)
            self.assertIn('timing', summary)
            self.assertIn('processing', summary)
            self.assertIn('memory', summary)
            
            # Check timing analysis
            timing = summary['timing']
            self.assertIn('actual_fps', timing)
            self.assertIn('fps_error_percent', timing)
            
            # Check processing analysis
            processing = summary['processing']
            self.assertIn('mean_processing_time_ms', processing)
            self.assertIn('frames_dropped', processing)
        
        finally:
            self.monitor.stop_monitoring()


class TestPerformanceOptimizer(unittest.TestCase):
    """Test cases for PerformanceOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = SimulatorConfig(fps=15.0, duration_s=2.0)
        self.optimizer = PerformanceOptimizer(self.test_config)
    
    def test_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        self.assertEqual(self.optimizer.config, self.test_config)
        self.assertFalse(self.optimizer.optimization_active)
        self.assertIsNone(self.optimizer.gc_threshold_original)
    
    def test_enable_disable_optimizations(self):
        """Test enabling and disabling optimizations."""
        # Store original GC threshold
        original_threshold = gc.get_threshold()
        
        # Enable optimizations
        self.optimizer.enable_optimizations()
        self.assertTrue(self.optimizer.optimization_active)
        self.assertEqual(self.optimizer.gc_threshold_original, original_threshold)
        
        # GC threshold should be modified
        current_threshold = gc.get_threshold()
        self.assertNotEqual(current_threshold, original_threshold)
        
        # Disable optimizations
        self.optimizer.disable_optimizations()
        self.assertFalse(self.optimizer.optimization_active)
        
        # GC threshold should be restored
        restored_threshold = gc.get_threshold()
        self.assertEqual(restored_threshold, original_threshold)
    
    def test_memory_optimization(self):
        """Test memory optimization functionality."""
        # Create some objects to be garbage collected
        large_objects = []
        for i in range(100):
            large_objects.append([0] * 100)
        
        # Clear references
        large_objects.clear()
        
        # Perform memory optimization
        result = self.optimizer.optimize_memory_usage()
        
        # Check optimization results
        self.assertIn('memory_before_mb', result)
        self.assertIn('memory_after_mb', result)
        self.assertIn('memory_freed_mb', result)
        self.assertIn('objects_collected', result)
        
        # Should have collected some objects
        self.assertGreaterEqual(result['objects_collected'], 0)
        self.assertGreaterEqual(result['memory_freed_mb'], 0)
    
    def test_memory_optimization_threshold(self):
        """Test memory optimization threshold logic."""
        # Low memory usage - should not optimize
        self.assertFalse(self.optimizer.should_optimize_memory(100.0))  # 100 MB
        
        # High memory usage - should optimize
        self.assertTrue(self.optimizer.should_optimize_memory(600.0))  # 600 MB


class TestFrameRateValidator(unittest.TestCase):
    """Test cases for FrameRateValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = FrameRateValidator(target_fps=10.0, tolerance_percent=10.0)
    
    def test_validator_initialization(self):
        """Test frame rate validator initialization."""
        self.assertEqual(self.validator.target_fps, 10.0)
        self.assertEqual(self.validator.tolerance_percent, 10.0)
        self.assertEqual(self.validator.target_interval, 0.1)  # 1/10 seconds
        self.assertAlmostEqual(self.validator.tolerance_interval, 0.01, places=3)  # 10% of 0.1
    
    def test_validate_good_frame_timing(self):
        """Test validation of good frame timing."""
        # Generate good frame intervals (close to target)
        target_interval = 0.1
        intervals = [target_interval + (i * 0.001) for i in range(-2, 3)]  # Small variations
        
        result = self.validator.validate_frame_timing(intervals)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['frames_analyzed'], 5)
        self.assertLess(abs(result['actual_fps'] - 10.0), 2.0)  # Within 2 FPS
        self.assertLess(result['fps_error_percent'], 20.0)  # Within 20%
    
    def test_validate_bad_frame_timing(self):
        """Test validation of bad frame timing."""
        # Generate bad frame intervals (far from target)
        intervals = [0.05, 0.2, 0.03, 0.25, 0.04]  # Very inconsistent
        
        result = self.validator.validate_frame_timing(intervals)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['frames_analyzed'], 5)
        self.assertGreater(result['fps_error_percent'], 10.0)
    
    def test_validate_empty_intervals(self):
        """Test validation with empty intervals."""
        result = self.validator.validate_frame_timing([])
        
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
        self.assertEqual(result['frames_analyzed'], 0)


class TestPerformanceContext(unittest.TestCase):
    """Test cases for performance context management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = SimulatorConfig(
            fps=20.0,
            duration_s=0.5,
            offline_mode=True,
            deterministic_seed=42
        )
    
    def test_create_performance_context(self):
        """Test performance context creation."""
        context = create_performance_context(self.test_config, enable_optimizations=True)
        
        try:
            # Check context structure
            self.assertIn('monitor', context)
            self.assertIn('optimizer', context)
            self.assertIn('validator', context)
            
            # Check components are initialized
            monitor = context['monitor']
            optimizer = context['optimizer']
            validator = context['validator']
            
            self.assertIsInstance(monitor, PerformanceMonitor)
            self.assertIsInstance(optimizer, PerformanceOptimizer)
            self.assertIsInstance(validator, FrameRateValidator)
            
            # Check monitoring is active
            self.assertTrue(monitor.monitoring_active)
            
            # Check optimizations are enabled
            self.assertTrue(optimizer.optimization_active)
            
        finally:
            # Cleanup
            cleanup_performance_context(context)
    
    def test_cleanup_performance_context(self):
        """Test performance context cleanup."""
        context = create_performance_context(self.test_config, enable_optimizations=True)
        
        # Simulate some activity
        monitor = context['monitor']
        for i in range(2):
            frame_start = monitor.record_frame_start()
            time.sleep(0.01)
            monitor.record_frame_end(frame_start)
        
        # Cleanup and get results
        results = cleanup_performance_context(context)
        
        # Check results structure
        self.assertIn('performance_summary', results)
        self.assertIn('timing_validation', results)
        self.assertIn('real_time_validation', results)
        self.assertIn('optimization_recommendations', results)
        self.assertIn('final_memory_optimization', results)
        
        # Check monitoring is stopped
        self.assertFalse(monitor.monitoring_active)
        
        # Check optimizations are disabled
        optimizer = context['optimizer']
        self.assertFalse(optimizer.optimization_active)


class TestSimulatorPerformanceIntegration(unittest.TestCase):
    """Integration tests for simulator performance features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.performance_config = SimulatorConfig(
            duration_s=0.5,
            fps=10.0,
            num_drones=1,
            offline_mode=True,
            deterministic_seed=42,
            # Reduced complexity for performance testing
            pixel_centroid_sigma_px=0.5,
            bbox_size_sigma_px=1.0,
            false_positive_rate=0.01
        )
    
    def test_simulator_performance_monitoring(self):
        """Test simulator with performance monitoring enabled."""
        # Capture printed output
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            simulator = DroneSimulator(self.performance_config)
            results = simulator.run()
        
        # Check that performance results are included
        self.assertIn('performance', results)
        
        performance = results['performance']
        self.assertIn('performance_summary', performance)
        self.assertIn('timing_validation', performance)
        self.assertIn('real_time_validation', performance)
        self.assertIn('optimization_recommendations', performance)
        
        # Check timing validation
        timing_validation = performance['timing_validation']
        self.assertIn('valid', timing_validation)
        self.assertIn('actual_fps', timing_validation)
        self.assertIn('fps_error_percent', timing_validation)
        
        # Performance should be reasonable for this simple test
        self.assertLess(timing_validation['fps_error_percent'], 100.0)  # Allow generous tolerance
    
    def test_simulator_memory_optimization(self):
        """Test simulator memory optimization during runs."""
        # Slightly longer duration
        config = SimulatorConfig(
            duration_s=1.0,
            fps=10.0,
            num_drones=1,
            offline_mode=True,
            deterministic_seed=42
        )
        
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            simulator = DroneSimulator(config)
            results = simulator.run()
        
        # Should complete successfully
        self.assertEqual(results['simulation']['completion_rate'], 1.0)
        
        # Should have performance monitoring results
        self.assertIn('performance', results)
        
        # Check memory optimization occurred
        performance = results['performance']
        final_optimization = performance['final_memory_optimization']
        self.assertIn('objects_collected', final_optimization)
        self.assertGreaterEqual(final_optimization['objects_collected'], 0)
    
    def test_performance_validation_structure(self):
        """Test that performance validation produces correct structure."""
        captured_output = []
        
        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")
        
        with patch('builtins.print', side_effect=mock_print):
            simulator = DroneSimulator(self.performance_config)
            results = simulator.run()
        
        # Should have performance results
        self.assertIn('performance', results)
        
        # Check structure of performance results
        performance = results['performance']
        
        # Performance summary structure
        summary = performance['performance_summary']
        self.assertIn('frames_processed', summary)
        self.assertIn('timing', summary)
        self.assertIn('processing', summary)
        self.assertIn('memory', summary)
        
        # Timing validation structure
        timing_val = performance['timing_validation']
        self.assertIn('valid', timing_val)
        self.assertIn('frames_analyzed', timing_val)
        self.assertIn('target_fps', timing_val)
        self.assertIn('actual_fps', timing_val)
        
        # Real-time validation structure
        rt_val = performance['real_time_validation']
        self.assertIn('real_time_capable', rt_val)
        self.assertIn('processing_times', rt_val)
        
        # Recommendations structure
        recommendations = performance['optimization_recommendations']
        self.assertIsInstance(recommendations, list)


if __name__ == '__main__':
    unittest.main()