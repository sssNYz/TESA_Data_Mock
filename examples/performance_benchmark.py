#!/usr/bin/env python3
"""
Performance benchmark for the drone detection simulator.

This script demonstrates the performance optimization and monitoring
features of the simulator, running various scenarios to test
real-time performance capabilities.
"""

import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import the simulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.simulator import DroneSimulator


def run_performance_benchmark():
    """Run performance benchmarks with different configurations."""
    
    print("Drone Detection Simulator - Performance Benchmark")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Low Load (10 FPS, 1 drone)',
            'config': SimulatorConfig(
                duration_s=2.0,
                fps=10.0,
                num_drones=1,
                offline_mode=True,
                deterministic_seed=42
            )
        },
        {
            'name': 'Medium Load (20 FPS, 2 drones)',
            'config': SimulatorConfig(
                duration_s=2.0,
                fps=20.0,
                num_drones=2,
                offline_mode=True,
                deterministic_seed=42
            )
        },
        {
            'name': 'High Load (30 FPS, 3 drones)',
            'config': SimulatorConfig(
                duration_s=2.0,
                fps=30.0,
                num_drones=3,
                offline_mode=True,
                deterministic_seed=42
            )
        },
        {
            'name': 'Stress Test (50 FPS, 5 drones)',
            'config': SimulatorConfig(
                duration_s=1.0,
                fps=50.0,
                num_drones=5,
                offline_mode=True,
                deterministic_seed=42,
                # Higher noise for more processing
                pixel_centroid_sigma_px=2.0,
                bbox_size_sigma_px=3.0,
                false_positive_rate=0.02
            )
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\nRunning: {scenario['name']}")
        print("-" * 40)
        
        # Suppress output during benchmark
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()):
            start_time = time.time()
            simulator = DroneSimulator(scenario['config'])
            sim_results = simulator.run()
            end_time = time.time()
        
        # Extract performance metrics
        actual_duration = end_time - start_time
        expected_duration = scenario['config'].duration_s
        
        performance = sim_results.get('performance', {})
        timing_validation = performance.get('timing_validation', {})
        performance_summary = performance.get('performance_summary', {})
        
        # Calculate key metrics
        target_fps = scenario['config'].fps
        actual_fps = timing_validation.get('actual_fps', 0)
        fps_error = timing_validation.get('fps_error_percent', 0)
        
        # Handle case where fps_error might be None
        if fps_error is None:
            fps_error = 0
        frames_processed = sim_results['simulation']['frames_processed']
        
        # Memory metrics
        memory_info = performance_summary.get('memory', {})
        peak_memory = memory_info.get('peak_memory_mb', 0)
        memory_growth = memory_info.get('memory_growth_mb', 0)
        
        # Processing metrics
        processing_info = performance_summary.get('processing', {})
        mean_processing_time = processing_info.get('mean_processing_time_ms', 0)
        frames_dropped = processing_info.get('frames_dropped', 0)
        
        # Store results
        result = {
            'scenario': scenario['name'],
            'target_fps': target_fps,
            'actual_fps': actual_fps,
            'fps_error': fps_error,  # Changed key name to match usage below
            'frames_processed': frames_processed,
            'actual_duration_s': actual_duration,
            'expected_duration_s': expected_duration,
            'timing_overhead_s': actual_duration - expected_duration,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': memory_growth,
            'mean_processing_time_ms': mean_processing_time,
            'frames_dropped': frames_dropped,
            'timing_valid': timing_validation.get('valid', False),
            'real_time_capable': performance.get('real_time_validation', {}).get('real_time_capable', False)
        }
        results.append(result)
        
        # Print results
        print(f"Target FPS: {target_fps:.1f}")
        print(f"Actual FPS: {actual_fps:.2f}")
        print(f"FPS Error: {fps_error:.1f}%")
        print(f"Frames Processed: {frames_processed}")
        print(f"Duration: {actual_duration:.2f}s (expected: {expected_duration:.1f}s)")
        print(f"Timing Overhead: {result['timing_overhead_s']:.2f}s")
        print(f"Peak Memory: {peak_memory:.1f} MB")
        print(f"Mean Processing Time: {mean_processing_time:.1f} ms")
        print(f"Frames Dropped: {frames_dropped}")
        print(f"Timing Valid: {result['timing_valid']}")
        print(f"Real-time Capable: {result['real_time_capable']}")
        
        # Performance assessment
        if fps_error < 10 and result['timing_valid']:
            print("✅ EXCELLENT performance")
        elif fps_error < 25 and actual_fps > target_fps * 0.8:
            print("✅ GOOD performance")
        elif fps_error < 50:
            print("⚠️  ACCEPTABLE performance")
        else:
            print("❌ POOR performance")
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    
    print(f"{'Scenario':<25} {'Target FPS':<10} {'Actual FPS':<10} {'Error %':<8} {'Status':<12}")
    print("-" * 70)
    
    for result in results:
        status = "EXCELLENT" if result['fps_error'] < 10 and result['timing_valid'] else \
                 "GOOD" if result['fps_error'] < 25 else \
                 "ACCEPTABLE" if result['fps_error'] < 50 else "POOR"
        
        print(f"{result['scenario']:<25} {result['target_fps']:<10.1f} "
              f"{result['actual_fps']:<10.2f} {result['fps_error']:<8.1f} {status:<12}")
    
    # Overall assessment
    excellent_count = sum(1 for r in results if r['fps_error'] < 10 and r['timing_valid'])
    good_count = sum(1 for r in results if 10 <= r['fps_error'] < 25)
    
    print(f"\nOverall Performance:")
    print(f"  Excellent: {excellent_count}/{len(results)} scenarios")
    print(f"  Good: {good_count}/{len(results)} scenarios")
    
    if excellent_count >= len(results) * 0.75:
        print("🎉 System shows EXCELLENT overall performance!")
    elif excellent_count + good_count >= len(results) * 0.75:
        print("👍 System shows GOOD overall performance!")
    else:
        print("⚠️  System may need performance optimization.")
    
    # Performance recommendations
    print(f"\nPerformance Recommendations:")
    
    high_error_scenarios = [r for r in results if r['fps_error'] > 25]
    if high_error_scenarios:
        print("  - Consider reducing FPS or drone count for high-load scenarios")
    
    high_memory_scenarios = [r for r in results if r['peak_memory_mb'] > 500]
    if high_memory_scenarios:
        print("  - Consider enabling memory optimization for memory-intensive scenarios")
    
    slow_scenarios = [r for r in results if r['timing_overhead_s'] > 0.5]
    if slow_scenarios:
        print("  - Consider optimizing processing pipeline for better real-time performance")
    
    if not (high_error_scenarios or high_memory_scenarios or slow_scenarios):
        print("  - No specific optimizations needed, performance is good!")
    
    return results


if __name__ == "__main__":
    try:
        results = run_performance_benchmark()
        print(f"\nBenchmark completed successfully!")
    except KeyboardInterrupt:
        print(f"\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        sys.exit(1)