"""
Unit tests for timing and frame rate control.
"""

import unittest
import time
import threading
from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np

from drone_detection_simulator.timing import (
    TimingController, SimulationLoop, FrameInfo,
    format_timestamp_utc, sleep_with_interrupt
)
from drone_detection_simulator.config import SimulatorConfig


class TestTimingController(unittest.TestCase):
    """Test cases for TimingController class."""
    
    def test_initialization(self):
        """Test TimingController initialization."""
        config = SimulatorConfig(fps=10.0, duration_s=5.0)
        controller = TimingController(config)
        
        self.assertEqual(controller.config, config)
        self.assertAlmostEqual(controller.target_frame_interval_s, 0.1, places=6)
        self.assertEqual(controller.total_frames, 50)
        self.assertEqual(controller.current_frame_id, 0)
        self.assertIsNone(controller.simulation_start_time)
    
    def test_start_simulation(self):
        """Test simulation start initialization."""
        config = SimulatorConfig(fps=15.0, duration_s=2.0)
        controller = TimingController(config)
        
        controller.start_simulation()
        
        self.assertIsNotNone(controller.simulation_start_time)
        self.assertEqual(controller.current_frame_id, 0)
        self.assertEqual(controller.last_frame_time, controller.simulation_start_time)
        self.assertEqual(len(controller.frame_times), 0)
        self.assertEqual(len(controller.actual_intervals), 0)
    
    def test_frame_info_generation(self):
        """Test frame info generation with proper timing."""
        config = SimulatorConfig(fps=20.0, duration_s=1.0)
        controller = TimingController(config)
        
        controller.start_simulation()
        
        with patch('time.sleep'):  # Speed up test
            # Get first frame
            frame1 = controller.get_next_frame_info()
            self.assertIsNotNone(frame1)
            self.assertEqual(frame1.frame_id, 0)
            self.assertEqual(frame1.simulation_time_s, 0.0)
            self.assertIsInstance(frame1.timestamp_utc, datetime)
            self.assertEqual(frame1.timestamp_utc.tzinfo, timezone.utc)
            self.assertGreaterEqual(frame1.processing_latency_ms, 0)
            
            # Get second frame
            frame2 = controller.get_next_frame_info()
            self.assertIsNotNone(frame2)
            self.assertEqual(frame2.frame_id, 1)
            self.assertAlmostEqual(frame2.simulation_time_s, 0.05, places=6)
    
    def test_processing_latency_generation(self):
        """Test processing latency generation."""
        config = SimulatorConfig(
            processing_latency_ms_mean=50.0,
            processing_latency_ms_jitter=10.0
        )
        controller = TimingController(config)
        
        # Generate latency values
        latencies = [controller._generate_processing_latency() for _ in range(10)]
        
        # Check that all values are non-negative
        for latency in latencies:
            self.assertGreaterEqual(latency, 0)
    
    def test_processing_latency_no_jitter(self):
        """Test processing latency without jitter."""
        config = SimulatorConfig(
            processing_latency_ms_mean=25.0,
            processing_latency_ms_jitter=0.0
        )
        controller = TimingController(config)
        
        # All latencies should be exactly the mean
        for _ in range(5):
            latency = controller._generate_processing_latency()
            self.assertEqual(latency, 25.0)
    
    def test_simulation_completion(self):
        """Test simulation completion detection."""
        config = SimulatorConfig(fps=10.0, duration_s=0.2)  # 2 frames
        controller = TimingController(config)
        
        controller.start_simulation()
        
        with patch('time.sleep'):
            # Should get 2 frames
            frame1 = controller.get_next_frame_info()
            self.assertIsNotNone(frame1)
            self.assertFalse(controller.is_simulation_complete())
            
            frame2 = controller.get_next_frame_info()
            self.assertIsNotNone(frame2)
            # After getting the second frame, simulation should be complete
            self.assertTrue(controller.is_simulation_complete())
            
            # Third call should return None
            frame3 = controller.get_next_frame_info()
            self.assertIsNone(frame3)
            self.assertTrue(controller.is_simulation_complete())
    
    def test_timing_statistics_empty(self):
        """Test timing statistics with no frames processed."""
        config = SimulatorConfig(fps=15.0)
        controller = TimingController(config)
        
        stats = controller.get_timing_statistics()
        
        self.assertEqual(stats['target_fps'], 15.0)
        self.assertAlmostEqual(stats['target_interval_s'], 1.0/15.0, places=6)
        self.assertEqual(stats['frames_processed'], 0)
        self.assertEqual(stats['actual_fps'], 0.0)
        self.assertEqual(stats['mean_interval_s'], 0.0)
        self.assertEqual(stats['interval_std_s'], 0.0)
        self.assertEqual(stats['timing_error_ms'], 0.0)


class TestSimulationLoop(unittest.TestCase):
    """Test cases for SimulationLoop class."""
    
    def test_initialization(self):
        """Test SimulationLoop initialization."""
        config = SimulatorConfig(fps=5.0, duration_s=1.0)
        loop = SimulationLoop(config)
        
        self.assertEqual(loop.config, config)
        self.assertIsInstance(loop.timing_controller, TimingController)
        self.assertFalse(loop.running)
        self.assertFalse(loop.shutdown_event.is_set())
    
    def test_frame_callback_execution(self):
        """Test that frame callback is called for each frame."""
        config = SimulatorConfig(fps=20.0, duration_s=0.15)  # 3 frames
        loop = SimulationLoop(config)
        
        callback_calls = []
        
        def frame_callback(frame_info):
            callback_calls.append(frame_info)
        
        with patch('time.sleep'):  # Speed up test
            stats = loop.run(frame_callback)
        
        # Should have called callback for each frame
        self.assertEqual(len(callback_calls), 3)
        for call in callback_calls:
            self.assertIsInstance(call, FrameInfo)
        
        # Frame IDs should be sequential
        frame_ids = [call.frame_id for call in callback_calls]
        self.assertEqual(frame_ids, [0, 1, 2])
        
        # Should return timing statistics
        self.assertIsInstance(stats, dict)
        self.assertIn('target_fps', stats)
    
    def test_progress_tracking(self):
        """Test simulation progress tracking."""
        config = SimulatorConfig(fps=10.0, duration_s=0.4)  # 4 frames
        loop = SimulationLoop(config)
        
        # Before starting
        self.assertEqual(loop.get_progress(), 0.0)
        
        progress_values = []
        
        def frame_callback(frame_info):
            progress_values.append(loop.get_progress())
        
        with patch('time.sleep'):
            loop.run(frame_callback)
        
        # Progress should increase with each frame
        self.assertEqual(len(progress_values), 4)
        self.assertAlmostEqual(progress_values[0], 0.25, places=2)  # After frame 0
        self.assertAlmostEqual(progress_values[1], 0.50, places=2)  # After frame 1
        self.assertAlmostEqual(progress_values[2], 0.75, places=2)  # After frame 2
        self.assertAlmostEqual(progress_values[3], 1.00, places=2)  # After frame 3


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_format_timestamp_utc(self):
        """Test UTC timestamp formatting."""
        # Test with UTC datetime
        dt_utc = datetime(2025, 9, 22, 14, 30, 45, 123456, timezone.utc)
        formatted = format_timestamp_utc(dt_utc)
        self.assertEqual(formatted, "2025-09-22T14:30:45.123Z")
        
        # Test with naive datetime (should assume UTC)
        dt_naive = datetime(2025, 9, 22, 14, 30, 45, 123456)
        formatted = format_timestamp_utc(dt_naive)
        self.assertEqual(formatted, "2025-09-22T14:30:45.123Z")
    
    def test_sleep_with_interrupt_normal(self):
        """Test normal sleep without interruption."""
        start_time = time.time()
        result = sleep_with_interrupt(0.05)  # 50ms
        end_time = time.time()
        
        self.assertTrue(result)  # Completed normally
        self.assertGreaterEqual(end_time - start_time, 0.04)  # At least 40ms
    
    def test_sleep_with_interrupt_no_event(self):
        """Test sleep without interrupt event (normal sleep)."""
        start_time = time.time()
        result = sleep_with_interrupt(0.05, None)
        end_time = time.time()
        
        self.assertTrue(result)
        self.assertGreaterEqual(end_time - start_time, 0.04)


class TestFrameInfo(unittest.TestCase):
    """Test cases for FrameInfo dataclass."""
    
    def test_frame_info_creation(self):
        """Test FrameInfo creation and attributes."""
        timestamp = datetime.now(timezone.utc)
        frame_info = FrameInfo(
            frame_id=42,
            timestamp_utc=timestamp,
            simulation_time_s=2.5,
            processing_latency_ms=35.7
        )
        
        self.assertEqual(frame_info.frame_id, 42)
        self.assertEqual(frame_info.timestamp_utc, timestamp)
        self.assertEqual(frame_info.simulation_time_s, 2.5)
        self.assertEqual(frame_info.processing_latency_ms, 35.7)


class TestTimingAccuracy(unittest.TestCase):
    """Integration tests for timing accuracy."""
    
    def test_frame_rate_accuracy(self):
        """Test that frame timing is calculated correctly."""
        config = SimulatorConfig(fps=10.0, duration_s=0.5)  # 5 frames
        controller = TimingController(config)
        
        controller.start_simulation()
        
        # Process all frames
        frames = []
        with patch('time.sleep'):  # Control timing for test
            while not controller.is_simulation_complete():
                frame = controller.get_next_frame_info()
                if frame:
                    frames.append(frame)
        
        # Check frame timing
        self.assertEqual(len(frames), 5)
        
        # Check simulation times are correct
        expected_times = [i * 0.1 for i in range(5)]  # 0.0, 0.1, 0.2, 0.3, 0.4
        actual_times = [f.simulation_time_s for f in frames]
        
        for expected, actual in zip(expected_times, actual_times):
            self.assertAlmostEqual(actual, expected, places=6)
    
    def test_latency_simulation_statistics(self):
        """Test that latency simulation produces reasonable values."""
        config = SimulatorConfig(
            fps=50.0,  # 50 frames for testing
            duration_s=1.0,
            processing_latency_ms_mean=40.0,
            processing_latency_ms_jitter=8.0
        )
        rng = np.random.default_rng(12345)  # Fixed seed
        controller = TimingController(config, rng)
        
        controller.start_simulation()
        
        latencies = []
        with patch('time.sleep'):
            while not controller.is_simulation_complete():
                frame = controller.get_next_frame_info()
                if frame:
                    latencies.append(frame.processing_latency_ms)
        
        # Statistical validation
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        
        # Mean should be reasonably close to target
        self.assertGreater(mean_latency, 30.0)  # At least 30ms
        self.assertLess(mean_latency, 50.0)     # At most 50ms
        
        # All values should be non-negative
        self.assertTrue(np.all(latencies >= 0))


if __name__ == '__main__':
    unittest.main()