#!/usr/bin/env python3
"""
Demonstration of timing and frame rate control functionality.

This example shows how to use the TimingController and SimulationLoop
to create accurate frame timing with processing latency simulation.
"""

import time
from drone_detection_simulator import (
    SimulatorConfig, TimingController, SimulationLoop, 
    FrameInfo, format_timestamp_utc
)


def timing_controller_demo():
    """Demonstrate TimingController usage."""
    print("=== TimingController Demo ===")
    
    # Create configuration for short demo
    config = SimulatorConfig(
        fps=5.0,  # 5 frames per second
        duration_s=2.0,  # 2 second simulation
        processing_latency_ms_mean=30.0,
        processing_latency_ms_jitter=10.0
    )
    
    print(f"Configuration: {config.fps} FPS for {config.duration_s}s")
    print(f"Expected frames: {int(config.fps * config.duration_s)}")
    print(f"Target interval: {1.0/config.fps:.3f}s")
    print()
    
    # Create timing controller
    controller = TimingController(config)
    controller.start_simulation()
    
    frame_count = 0
    start_time = time.time()
    
    print("Processing frames...")
    while not controller.is_simulation_complete():
        frame_info = controller.get_next_frame_info()
        if frame_info is None:
            break
            
        frame_count += 1
        timestamp_str = format_timestamp_utc(frame_info.timestamp_utc)
        
        print(f"Frame {frame_info.frame_id}: "
              f"sim_time={frame_info.simulation_time_s:.3f}s, "
              f"latency={frame_info.processing_latency_ms:.1f}ms, "
              f"timestamp={timestamp_str}")
        
        # Simulate some processing time
        time.sleep(0.05)
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Get timing statistics
    stats = controller.get_timing_statistics()
    
    print(f"\nTiming Statistics:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Actual duration: {actual_duration:.3f}s")
    print(f"  Target FPS: {stats['target_fps']:.1f}")
    print(f"  Actual FPS: {stats['actual_fps']:.1f}")
    print(f"  Timing error: {stats['timing_error_ms']:.1f}ms")
    print()


def simulation_loop_demo():
    """Demonstrate SimulationLoop usage."""
    print("=== SimulationLoop Demo ===")
    
    # Create configuration
    config = SimulatorConfig(
        fps=10.0,  # 10 frames per second
        duration_s=1.5,  # 1.5 second simulation
        processing_latency_ms_mean=25.0,
        processing_latency_ms_jitter=5.0
    )
    
    print(f"Configuration: {config.fps} FPS for {config.duration_s}s")
    print(f"Expected frames: {int(config.fps * config.duration_s)}")
    print()
    
    # Create simulation loop
    loop = SimulationLoop(config)
    
    # Frame processing callback
    processed_frames = []
    
    def frame_callback(frame_info: FrameInfo):
        processed_frames.append(frame_info)
        
        # Show progress
        progress = loop.get_progress()
        print(f"Frame {frame_info.frame_id}: "
              f"progress={progress:.1%}, "
              f"latency={frame_info.processing_latency_ms:.1f}ms")
        
        # Simulate frame processing
        time.sleep(0.02)  # 20ms processing time
    
    print("Running simulation loop...")
    start_time = time.time()
    
    # Run the simulation
    stats = loop.run(frame_callback)
    
    end_time = time.time()
    actual_duration = end_time - start_time
    
    print(f"\nSimulation Complete!")
    print(f"  Frames processed: {len(processed_frames)}")
    print(f"  Actual duration: {actual_duration:.3f}s")
    print(f"  Target FPS: {stats['target_fps']:.1f}")
    print(f"  Actual FPS: {stats['actual_fps']:.1f}")
    print(f"  Timing error: {stats['timing_error_ms']:.1f}ms")
    print()


def latency_simulation_demo():
    """Demonstrate processing latency simulation."""
    print("=== Latency Simulation Demo ===")
    
    # Test different latency configurations
    configs = [
        ("No Jitter", 40.0, 0.0),
        ("Low Jitter", 40.0, 5.0),
        ("High Jitter", 40.0, 15.0),
    ]
    
    for name, mean_latency, jitter in configs:
        print(f"\n{name} (mean={mean_latency}ms, jitter={jitter}ms):")
        
        config = SimulatorConfig(
            fps=20.0,
            duration_s=0.5,  # Short test
            processing_latency_ms_mean=mean_latency,
            processing_latency_ms_jitter=jitter
        )
        
        controller = TimingController(config)
        controller.start_simulation()
        
        latencies = []
        while not controller.is_simulation_complete():
            frame_info = controller.get_next_frame_info()
            if frame_info:
                latencies.append(frame_info.processing_latency_ms)
        
        if latencies:
            import statistics
            mean_actual = statistics.mean(latencies)
            stdev_actual = statistics.stdev(latencies) if len(latencies) > 1 else 0
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            print(f"  Samples: {len(latencies)}")
            print(f"  Mean: {mean_actual:.1f}ms (target: {mean_latency:.1f}ms)")
            print(f"  Std Dev: {stdev_actual:.1f}ms (target: {jitter:.1f}ms)")
            print(f"  Range: {min_latency:.1f}ms - {max_latency:.1f}ms")


def main():
    """Run all timing demos."""
    print("Drone Detection Simulator - Timing System Demo")
    print("=" * 50)
    print()
    
    try:
        timing_controller_demo()
        simulation_loop_demo()
        latency_simulation_demo()
        
        print("All demos completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        raise


if __name__ == "__main__":
    main()