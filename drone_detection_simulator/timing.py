"""
Timing and frame rate control for the drone detection simulator.
"""

import time
import threading
from datetime import datetime, timezone
from typing import Optional, Callable, Any
import numpy as np
from dataclasses import dataclass

from .config import SimulatorConfig


@dataclass
class FrameInfo:
    """Information about a simulation frame."""
    frame_id: int
    timestamp_utc: datetime
    simulation_time_s: float
    processing_latency_ms: float


class TimingController:
    """
    Controls frame timing, latency simulation, and timestamp generation.
    
    This class manages the simulation loop timing to maintain accurate frame rates,
    simulates processing latency with configurable jitter, and generates frame IDs
    and UTC timestamps for each frame.
    """
    
    def __init__(self, config: SimulatorConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize timing controller.
        
        Args:
            config: Simulator configuration
            rng: Random number generator for latency jitter (optional)
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Frame timing parameters
        self.target_frame_interval_s = 1.0 / config.fps
        self.total_frames = int(config.duration_s * config.fps)
        
        # Frame tracking
        self.current_frame_id = 0
        self.simulation_start_time = None
        self.last_frame_time = None
        
        # Timing statistics
        self.frame_times = []
        self.actual_intervals = []
        
    def start_simulation(self) -> None:
        """Initialize simulation timing."""
        self.simulation_start_time = time.time()
        self.last_frame_time = self.simulation_start_time
        self.current_frame_id = 0
        self.frame_times.clear()
        self.actual_intervals.clear()
        
    def get_next_frame_info(self) -> Optional[FrameInfo]:
        """
        Get information for the next frame, handling timing control.
        
        Returns:
            FrameInfo for the next frame, or None if simulation is complete
        """
        if self.current_frame_id >= self.total_frames:
            return None
            
        # Calculate target time for this frame
        target_time = self.simulation_start_time + (self.current_frame_id * self.target_frame_interval_s)
        current_time = time.time()
        
        # Sleep if we're ahead of schedule
        if current_time < target_time:
            sleep_time = target_time - current_time
            time.sleep(sleep_time)
            current_time = time.time()
        
        # Record actual timing
        if self.last_frame_time is not None:
            actual_interval = current_time - self.last_frame_time
            self.actual_intervals.append(actual_interval)
        
        self.frame_times.append(current_time)
        self.last_frame_time = current_time
        
        # Generate frame info
        simulation_time_s = self.current_frame_id * self.target_frame_interval_s
        timestamp_utc = datetime.now(timezone.utc)
        processing_latency_ms = self._generate_processing_latency()
        
        frame_info = FrameInfo(
            frame_id=self.current_frame_id,
            timestamp_utc=timestamp_utc,
            simulation_time_s=simulation_time_s,
            processing_latency_ms=processing_latency_ms
        )
        
        self.current_frame_id += 1
        return frame_info
    
    def _generate_processing_latency(self) -> float:
        """
        Generate processing latency with configurable jitter.
        
        Returns:
            Processing latency in milliseconds
        """
        mean_latency = self.config.processing_latency_ms_mean
        jitter = self.config.processing_latency_ms_jitter
        
        if jitter > 0:
            # Use normal distribution for jitter, clamped to positive values
            latency = self.rng.normal(mean_latency, jitter)
            latency = max(0.0, latency)  # Ensure non-negative
        else:
            latency = mean_latency
            
        return latency
    
    def get_timing_statistics(self) -> dict:
        """
        Get timing accuracy statistics.
        
        Returns:
            Dictionary containing timing statistics
        """
        if not self.actual_intervals:
            return {
                'target_fps': self.config.fps,
                'target_interval_s': self.target_frame_interval_s,
                'frames_processed': 0,
                'actual_fps': 0.0,
                'mean_interval_s': 0.0,
                'interval_std_s': 0.0,
                'timing_error_ms': 0.0
            }
        
        actual_intervals = np.array(self.actual_intervals)
        mean_interval = np.mean(actual_intervals)
        std_interval = np.std(actual_intervals)
        actual_fps = 1.0 / mean_interval if mean_interval > 0 else 0.0
        timing_error_ms = abs(mean_interval - self.target_frame_interval_s) * 1000
        
        return {
            'target_fps': self.config.fps,
            'target_interval_s': self.target_frame_interval_s,
            'frames_processed': len(self.actual_intervals),
            'actual_fps': actual_fps,
            'mean_interval_s': mean_interval,
            'interval_std_s': std_interval,
            'timing_error_ms': timing_error_ms
        }
    
    def is_simulation_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.current_frame_id >= self.total_frames


class SimulationLoop:
    """
    Main simulation loop with accurate frame timing.
    
    This class provides a framework for running the simulation loop with precise
    timing control, frame callbacks, and graceful shutdown handling.
    """
    
    def __init__(self, config: SimulatorConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize simulation loop.
        
        Args:
            config: Simulator configuration
            rng: Random number generator (optional)
        """
        self.config = config
        self.timing_controller = TimingController(config, rng)
        self.running = False
        self.shutdown_event = threading.Event()
        
    def run(self, frame_callback: Callable[[FrameInfo], Any]) -> dict:
        """
        Run the simulation loop with frame callback.
        
        Args:
            frame_callback: Function called for each frame with FrameInfo
            
        Returns:
            Dictionary containing timing statistics
        """
        self.running = True
        self.shutdown_event.clear()
        
        try:
            self.timing_controller.start_simulation()
            
            while self.running and not self.shutdown_event.is_set():
                frame_info = self.timing_controller.get_next_frame_info()
                
                if frame_info is None:
                    # Simulation complete
                    break
                
                # Call frame processing callback
                try:
                    frame_callback(frame_info)
                except Exception as e:
                    print(f"Error in frame callback: {e}")
                    # Continue with next frame
                
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break
                    
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
        finally:
            self.running = False
            
        return self.timing_controller.get_timing_statistics()
    
    def shutdown(self) -> None:
        """Request graceful shutdown of simulation loop."""
        self.running = False
        self.shutdown_event.set()
    
    def get_progress(self) -> float:
        """
        Get simulation progress as fraction (0.0 to 1.0).
        
        Returns:
            Progress fraction
        """
        if self.timing_controller.total_frames == 0:
            return 1.0
        return min(1.0, self.timing_controller.current_frame_id / self.timing_controller.total_frames)


def format_timestamp_utc(dt: datetime) -> str:
    """
    Format datetime as UTC timestamp string for JSON messages.
    
    Args:
        dt: Datetime object (should be timezone-aware)
        
    Returns:
        ISO format timestamp string with milliseconds
    """
    # Ensure timezone is UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)
    
    # Format with milliseconds
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


def sleep_with_interrupt(duration_s: float, interrupt_event: Optional[threading.Event] = None) -> bool:
    """
    Sleep for specified duration with optional interrupt capability.
    
    Args:
        duration_s: Sleep duration in seconds
        interrupt_event: Optional event to interrupt sleep
        
    Returns:
        True if sleep completed normally, False if interrupted
    """
    if interrupt_event is None:
        time.sleep(duration_s)
        return True
    
    return not interrupt_event.wait(duration_s)