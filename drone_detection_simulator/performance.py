"""
Performance optimization and monitoring for the drone detection simulator.

This module provides performance monitoring, optimization utilities, and
validation tools to ensure the simulator runs at target FPS with efficient
memory usage for long-running simulations.
"""

import time
import gc
import threading
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from .config import SimulatorConfig
from .logging_config import SimulatorLogger

# Optional dependency handling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create mock psutil for basic functionality
    class MockProcess:
        def __init__(self, pid):
            self.pid = pid
        
        def memory_info(self):
            # Return mock memory info
            class MockMemoryInfo:
                rss = 100 * 1024 * 1024  # 100 MB default
            return MockMemoryInfo()
        
        def cpu_percent(self, interval=None):
            return 50.0  # Mock 50% CPU usage
    
    class MockPsutil:
        @staticmethod
        def Process(pid):
            return MockProcess(pid)
        
        class NoSuchProcess(Exception):
            pass
    
    psutil = MockPsutil()


logger = SimulatorLogger.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for simulation monitoring."""
    
    # Frame timing metrics
    frame_times: List[float] = field(default_factory=list)
    frame_intervals: List[float] = field(default_factory=list)
    processing_times: List[float] = field(default_factory=list)
    
    # Memory metrics
    memory_samples: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    
    # FPS metrics
    target_fps: float = 0.0
    actual_fps: float = 0.0
    fps_variance: float = 0.0
    
    # Performance statistics
    frames_processed: int = 0
    frames_dropped: int = 0
    timing_violations: int = 0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.frame_times.clear()
        self.frame_intervals.clear()
        self.processing_times.clear()
        self.memory_samples.clear()
        self.peak_memory_mb = 0.0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.timing_violations = 0


class PerformanceMonitor:
    """
    Real-time performance monitoring for simulation.
    
    Tracks frame timing, memory usage, CPU utilization, and other
    performance metrics during simulation execution.
    """
    
    def __init__(self, config: SimulatorConfig, max_samples: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            config: Simulator configuration
            max_samples: Maximum number of samples to keep in memory
        """
        self.config = config
        self.max_samples = max_samples
        self.metrics = PerformanceMetrics()
        self.metrics.target_fps = config.fps
        
        # Circular buffers for efficient memory usage
        self.frame_times = deque(maxlen=max_samples)
        self.frame_intervals = deque(maxlen=max_samples)
        self.processing_times = deque(maxlen=max_samples)
        self.memory_samples = deque(maxlen=max_samples)
        
        # Process monitoring
        try:
            self.process = psutil.Process(os.getpid())
        except (AttributeError, OSError):
            self.process = None
            if not PSUTIL_AVAILABLE:
                logger.warning("psutil not available, using mock process monitoring")
        
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Performance thresholds
        self.target_frame_time = 1.0 / config.fps
        self.timing_tolerance = 0.1  # 10% tolerance for timing violations
        
        logger.debug(f"Performance monitor initialized with target FPS: {config.fps}")
    
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("Performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        logger.debug("Performance monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active and not self.stop_event.is_set():
            try:
                # Sample memory usage
                if self.process:
                    try:
                        memory_info = self.process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)
                        self.memory_samples.append(memory_mb)
                        
                        # Update peak memory
                        if memory_mb > self.metrics.peak_memory_mb:
                            self.metrics.peak_memory_mb = memory_mb
                        
                        # Sample CPU usage (non-blocking)
                        cpu_percent = self.process.cpu_percent(interval=None)
                        self.metrics.cpu_usage_percent = cpu_percent
                    except (psutil.NoSuchProcess, AttributeError):
                        break
                    except Exception as e:
                        logger.debug(f"Error sampling process metrics: {e}")
                
                # Sleep for monitoring interval
                self.stop_event.wait(0.1)  # 100ms monitoring interval
                
            except Exception as e:
                logger.warning(f"Error in performance monitoring: {e}")
                break
    
    def record_frame_start(self) -> float:
        """
        Record the start of frame processing.
        
        Returns:
            Timestamp for frame start
        """
        frame_start = time.time()
        self.frame_times.append(frame_start)
        
        # Calculate frame interval if we have previous frame
        if len(self.frame_times) >= 2:
            interval = frame_start - self.frame_times[-2]
            self.frame_intervals.append(interval)
            
            # Check for timing violations
            expected_interval = self.target_frame_time
            if abs(interval - expected_interval) > (expected_interval * self.timing_tolerance):
                self.metrics.timing_violations += 1
        
        return frame_start
    
    def record_frame_end(self, frame_start: float) -> None:
        """
        Record the end of frame processing.
        
        Args:
            frame_start: Timestamp from record_frame_start()
        """
        frame_end = time.time()
        processing_time = frame_end - frame_start
        self.processing_times.append(processing_time)
        self.metrics.frames_processed += 1
        
        # Check if processing took too long
        if processing_time > self.target_frame_time:
            self.metrics.frames_dropped += 1
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics.
        
        Returns:
            Current performance metrics
        """
        # Update metrics from circular buffers
        self.metrics.frame_times = list(self.frame_times)
        self.metrics.frame_intervals = list(self.frame_intervals)
        self.metrics.processing_times = list(self.processing_times)
        self.metrics.memory_samples = list(self.memory_samples)
        
        # Calculate FPS statistics
        if len(self.frame_intervals) > 0:
            intervals = np.array(self.frame_intervals)
            mean_interval = np.mean(intervals)
            self.metrics.actual_fps = 1.0 / mean_interval if mean_interval > 0 else 0.0
            self.metrics.fps_variance = np.var(1.0 / intervals) if len(intervals) > 1 else 0.0
        
        # Update current memory usage
        if self.process:
            try:
                memory_info = self.process.memory_info()
                self.metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            except (psutil.NoSuchProcess, AttributeError):
                pass
        
        return self.metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary containing performance analysis
        """
        metrics = self.get_current_metrics()
        
        # Frame timing analysis
        timing_analysis = {}
        if metrics.frame_intervals:
            intervals = np.array(metrics.frame_intervals)
            timing_analysis = {
                'target_fps': metrics.target_fps,
                'actual_fps': metrics.actual_fps,
                'fps_error_percent': abs(metrics.actual_fps - metrics.target_fps) / metrics.target_fps * 100,
                'fps_variance': metrics.fps_variance,
                'mean_frame_interval_ms': np.mean(intervals) * 1000,
                'frame_interval_std_ms': np.std(intervals) * 1000,
                'timing_violations': metrics.timing_violations,
                'timing_violation_rate': metrics.timing_violations / max(1, metrics.frames_processed)
            }
        
        # Processing time analysis
        processing_analysis = {}
        if metrics.processing_times:
            processing_times = np.array(metrics.processing_times)
            processing_analysis = {
                'mean_processing_time_ms': np.mean(processing_times) * 1000,
                'max_processing_time_ms': np.max(processing_times) * 1000,
                'processing_time_std_ms': np.std(processing_times) * 1000,
                'frames_dropped': metrics.frames_dropped,
                'drop_rate_percent': metrics.frames_dropped / max(1, metrics.frames_processed) * 100
            }
        
        # Memory analysis
        memory_analysis = {}
        if metrics.memory_samples:
            memory_samples = np.array(metrics.memory_samples)
            memory_analysis = {
                'current_memory_mb': metrics.memory_usage_mb,
                'peak_memory_mb': metrics.peak_memory_mb,
                'mean_memory_mb': np.mean(memory_samples),
                'memory_std_mb': np.std(memory_samples),
                'memory_growth_mb': memory_samples[-1] - memory_samples[0] if len(memory_samples) > 1 else 0.0
            }
        
        return {
            'frames_processed': metrics.frames_processed,
            'cpu_usage_percent': metrics.cpu_usage_percent,
            'timing': timing_analysis,
            'processing': processing_analysis,
            'memory': memory_analysis
        }


class PerformanceOptimizer:
    """
    Performance optimization utilities for the simulator.
    
    Provides methods to optimize memory usage, reduce garbage collection
    overhead, and improve real-time performance.
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize performance optimizer.
        
        Args:
            config: Simulator configuration
        """
        self.config = config
        self.gc_threshold_original = None
        self.optimization_active = False
        
        logger.debug("Performance optimizer initialized")
    
    def enable_optimizations(self) -> None:
        """Enable performance optimizations."""
        if self.optimization_active:
            return
        
        # Store original GC thresholds
        self.gc_threshold_original = gc.get_threshold()
        
        # Optimize garbage collection for real-time performance
        # Increase thresholds to reduce GC frequency during simulation
        gc.set_threshold(2000, 20, 20)  # Increased from default (700, 10, 10)
        
        # Disable automatic garbage collection during critical sections
        # (will be manually triggered at appropriate times)
        
        self.optimization_active = True
        logger.info("Performance optimizations enabled")
    
    def disable_optimizations(self) -> None:
        """Disable performance optimizations and restore defaults."""
        if not self.optimization_active:
            return
        
        # Restore original GC thresholds
        if self.gc_threshold_original:
            gc.set_threshold(*self.gc_threshold_original)
        
        # Re-enable automatic garbage collection
        gc.enable()
        
        self.optimization_active = False
        logger.info("Performance optimizations disabled")
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Perform memory optimization.
        
        Returns:
            Dictionary containing optimization results
        """
        # Get memory usage before optimization
        memory_before = 0.0
        memory_after = 0.0
        
        try:
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)
        except (AttributeError, OSError):
            memory_before = 100.0  # Default value if psutil unavailable
        
        # Force garbage collection
        collected_objects = gc.collect()
        
        # Get memory usage after optimization
        try:
            process = psutil.Process(os.getpid())
            memory_after = process.memory_info().rss / (1024 * 1024)
        except (AttributeError, OSError):
            memory_after = memory_before  # No change if psutil unavailable
        
        memory_freed = memory_before - memory_after
        
        optimization_results = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_freed_mb': memory_freed,
            'objects_collected': collected_objects
        }
        
        if memory_freed > 1.0:  # Only log if significant memory was freed
            logger.info(f"Memory optimization freed {memory_freed:.1f} MB, "
                       f"collected {collected_objects} objects")
        
        return optimization_results
    
    def should_optimize_memory(self, current_memory_mb: float) -> bool:
        """
        Determine if memory optimization should be performed.
        
        Args:
            current_memory_mb: Current memory usage in MB
            
        Returns:
            True if memory optimization is recommended
        """
        # Optimize if memory usage exceeds threshold
        memory_threshold_mb = 500  # 500 MB threshold
        return current_memory_mb > memory_threshold_mb
    
    def get_optimization_recommendations(self, performance_summary: Dict[str, Any]) -> List[str]:
        """
        Get performance optimization recommendations.
        
        Args:
            performance_summary: Performance summary from PerformanceMonitor
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # FPS recommendations
        timing = performance_summary.get('timing', {})
        fps_error = timing.get('fps_error_percent', 0)
        if fps_error > 10:  # More than 10% FPS error
            recommendations.append(
                f"FPS error is {fps_error:.1f}%. Consider reducing simulation complexity "
                "or increasing frame interval tolerance."
            )
        
        # Processing time recommendations
        processing = performance_summary.get('processing', {})
        drop_rate = processing.get('drop_rate_percent', 0)
        if drop_rate > 5:  # More than 5% frame drops
            recommendations.append(
                f"Frame drop rate is {drop_rate:.1f}%. Consider optimizing frame processing "
                "or reducing target FPS."
            )
        
        # Memory recommendations
        memory = performance_summary.get('memory', {})
        memory_growth = memory.get('memory_growth_mb', 0)
        if memory_growth > 50:  # More than 50 MB growth
            recommendations.append(
                f"Memory usage increased by {memory_growth:.1f} MB. Consider enabling "
                "periodic garbage collection or reducing data retention."
            )
        
        peak_memory = memory.get('peak_memory_mb', 0)
        if peak_memory > 1000:  # More than 1 GB peak memory
            recommendations.append(
                f"Peak memory usage is {peak_memory:.1f} MB. Consider reducing buffer sizes "
                "or enabling memory optimizations."
            )
        
        # CPU recommendations
        cpu_usage = performance_summary.get('cpu_usage_percent', 0)
        if cpu_usage > 80:  # More than 80% CPU usage
            recommendations.append(
                f"CPU usage is {cpu_usage:.1f}%. Consider reducing simulation complexity "
                "or distributing processing across multiple cores."
            )
        
        return recommendations


class FrameRateValidator:
    """
    Validates frame rate consistency and timing accuracy.
    
    Provides methods to validate that the simulation maintains
    consistent frame timing within acceptable tolerances.
    """
    
    def __init__(self, target_fps: float, tolerance_percent: float = 10.0):
        """
        Initialize frame rate validator.
        
        Args:
            target_fps: Target frames per second
            tolerance_percent: Acceptable tolerance percentage
        """
        self.target_fps = target_fps
        self.tolerance_percent = tolerance_percent
        self.target_interval = 1.0 / target_fps
        self.tolerance_interval = self.target_interval * (tolerance_percent / 100.0)
        
        logger.debug(f"Frame rate validator initialized: {target_fps} FPS ±{tolerance_percent}%")
    
    def validate_frame_timing(self, frame_intervals: List[float]) -> Dict[str, Any]:
        """
        Validate frame timing consistency.
        
        Args:
            frame_intervals: List of frame intervals in seconds
            
        Returns:
            Dictionary containing validation results
        """
        if not frame_intervals:
            return {
                'valid': False,
                'error': 'No frame intervals provided',
                'frames_analyzed': 0
            }
        
        intervals = np.array(frame_intervals)
        
        # Calculate statistics
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        actual_fps = 1.0 / mean_interval if mean_interval > 0 else 0.0
        
        # Check timing violations
        timing_violations = np.abs(intervals - self.target_interval) > self.tolerance_interval
        violation_count = np.sum(timing_violations)
        violation_rate = violation_count / len(intervals)
        
        # Determine if timing is valid
        fps_error_percent = abs(actual_fps - self.target_fps) / self.target_fps * 100
        timing_valid = (
            fps_error_percent <= self.tolerance_percent and
            violation_rate <= 0.1  # Allow up to 10% of frames to violate timing
        )
        
        return {
            'valid': timing_valid,
            'frames_analyzed': len(intervals),
            'target_fps': self.target_fps,
            'actual_fps': actual_fps,
            'fps_error_percent': fps_error_percent,
            'mean_interval_ms': mean_interval * 1000,
            'interval_std_ms': std_interval * 1000,
            'timing_violations': int(violation_count),
            'violation_rate_percent': violation_rate * 100,
            'tolerance_percent': self.tolerance_percent
        }
    
    def validate_real_time_performance(self, 
                                     frame_intervals: List[float],
                                     processing_times: List[float]) -> Dict[str, Any]:
        """
        Validate real-time performance requirements.
        
        Args:
            frame_intervals: List of frame intervals in seconds
            processing_times: List of processing times in seconds
            
        Returns:
            Dictionary containing real-time validation results
        """
        if not frame_intervals or not processing_times:
            return {
                'valid': False,
                'error': 'Insufficient data for validation',
                'real_time_capable': False
            }
        
        # Validate frame timing
        timing_validation = self.validate_frame_timing(frame_intervals)
        
        # Check processing time constraints
        processing_times = np.array(processing_times)
        max_processing_time = np.max(processing_times)
        mean_processing_time = np.mean(processing_times)
        
        # Real-time constraint: processing time should be less than frame interval
        real_time_violations = processing_times > self.target_interval
        real_time_violation_count = np.sum(real_time_violations)
        real_time_violation_rate = real_time_violation_count / len(processing_times)
        
        # Determine real-time capability
        real_time_capable = (
            timing_validation['valid'] and
            real_time_violation_rate <= 0.05 and  # Allow up to 5% violations
            mean_processing_time < (self.target_interval * 0.8)  # 80% utilization max
        )
        
        return {
            'valid': timing_validation['valid'],
            'real_time_capable': real_time_capable,
            'timing_validation': timing_validation,
            'processing_times': {
                'mean_ms': mean_processing_time * 1000,
                'max_ms': max_processing_time * 1000,
                'target_max_ms': self.target_interval * 1000,
                'real_time_violations': int(real_time_violation_count),
                'violation_rate_percent': real_time_violation_rate * 100
            }
        }


def create_performance_context(config: SimulatorConfig, 
                             enable_optimizations: bool = True) -> Dict[str, Any]:
    """
    Create a performance monitoring and optimization context.
    
    Args:
        config: Simulator configuration
        enable_optimizations: Whether to enable performance optimizations
        
    Returns:
        Dictionary containing performance monitoring objects
    """
    monitor = PerformanceMonitor(config)
    optimizer = PerformanceOptimizer(config)
    validator = FrameRateValidator(config.fps)
    
    if enable_optimizations:
        optimizer.enable_optimizations()
    
    monitor.start_monitoring()
    
    return {
        'monitor': monitor,
        'optimizer': optimizer,
        'validator': validator
    }


def cleanup_performance_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up performance monitoring context and return final results.
    
    Args:
        context: Performance context from create_performance_context()
        
    Returns:
        Dictionary containing final performance results
    """
    monitor = context['monitor']
    optimizer = context['optimizer']
    validator = context['validator']
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get final performance summary
    performance_summary = monitor.get_performance_summary()
    
    # Validate performance
    metrics = monitor.get_current_metrics()
    timing_validation = validator.validate_frame_timing(metrics.frame_intervals)
    real_time_validation = validator.validate_real_time_performance(
        metrics.frame_intervals, metrics.processing_times
    )
    
    # Get optimization recommendations
    recommendations = optimizer.get_optimization_recommendations(performance_summary)
    
    # Perform final memory optimization
    final_optimization = optimizer.optimize_memory_usage()
    
    # Disable optimizations
    optimizer.disable_optimizations()
    
    return {
        'performance_summary': performance_summary,
        'timing_validation': timing_validation,
        'real_time_validation': real_time_validation,
        'optimization_recommendations': recommendations,
        'final_memory_optimization': final_optimization
    }