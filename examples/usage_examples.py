#!/usr/bin/env python3
"""
Usage examples for the drone detection simulator.

This script demonstrates various ways to use the simulator programmatically
and via the command-line interface.
"""

import sys
import subprocess
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_detection_simulator import SimulatorConfig, DroneSimulator, cli_main


def example_1_basic_programmatic():
    """Example 1: Basic programmatic usage with default configuration."""
    print("=" * 60)
    print("Example 1: Basic Programmatic Usage")
    print("=" * 60)
    
    # Create default configuration
    config = SimulatorConfig(
        duration_s=3.0,  # Short duration for demo
        offline_mode=True,  # Print JSON instead of MQTT
        deterministic_seed=42  # Reproducible results
    )
    
    # Create and run simulator
    simulator = DroneSimulator(config)
    results = simulator.run()
    
    print(f"Simulation completed:")
    print(f"  Frames processed: {results['simulation']['frames_processed']}")
    print(f"  Detections generated: {results['detections']['total_generated']}")
    print(f"  Actual FPS: {results['timing']['actual_fps']:.2f}")
    print()


def example_2_custom_configuration():
    """Example 2: Custom configuration with multiple drones."""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = SimulatorConfig(
        # Camera setup
        image_width_px=1280,
        image_height_px=720,
        vertical_fov_deg=45.0,
        
        # Multiple drones
        num_drones=2,
        duration_s=4.0,
        fps=10.0,
        
        # Custom motion
        speed_mps=6.0,
        path_span_m=50.0,
        
        # Testing setup
        deterministic_seed=123,
        offline_mode=True
    )
    
    simulator = DroneSimulator(config)
    results = simulator.run()
    
    print(f"Multi-drone simulation completed:")
    print(f"  Drones: {config.num_drones}")
    print(f"  Frames: {results['simulation']['frames_processed']}")
    print(f"  Total detections: {results['detections']['total_generated']}")
    print(f"  Avg detections per frame: {results['detections']['avg_per_frame']:.1f}")
    print()


def example_3_configuration_from_file():
    """Example 3: Load configuration from file."""
    print("=" * 60)
    print("Example 3: Configuration from File")
    print("=" * 60)
    
    # Load configuration from YAML file
    config_path = Path("examples/testing_deterministic.yaml")
    if config_path.exists():
        config = SimulatorConfig.from_yaml(config_path)
        
        # Override some parameters
        config_dict = config.to_dict()
        config_dict.update({
            'duration_s': 2.0,  # Shorter for demo
            'offline_mode': True
        })
        config = SimulatorConfig.from_dict(config_dict)
        
        simulator = DroneSimulator(config)
        results = simulator.run()
        
        print(f"File-based configuration completed:")
        print(f"  Config file: {config_path}")
        print(f"  Duration: {config.duration_s}s")
        print(f"  Frames: {results['simulation']['frames_processed']}")
        print()
    else:
        print(f"Configuration file not found: {config_path}")
        print()


def example_4_cli_programmatic():
    """Example 4: Using CLI programmatically."""
    print("=" * 60)
    print("Example 4: CLI Programmatic Usage")
    print("=" * 60)
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Set up CLI arguments
        sys.argv = [
            'drone_detection_simulator',
            '--duration', '2',
            '--fps', '10',
            '--seed', '42',
            '--offline',
            '--quiet'
        ]
        
        # Run CLI
        result = cli_main()
        print(f"CLI execution completed with exit code: {result}")
        print()
        
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def example_5_cli_subprocess():
    """Example 5: Using CLI via subprocess."""
    print("=" * 60)
    print("Example 5: CLI via Subprocess")
    print("=" * 60)
    
    try:
        # Run CLI command
        cmd = [
            sys.executable, '-m', 'drone_detection_simulator',
            '--duration', '2',
            '--num-drones', '2',
            '--seed', '999',
            '--offline',
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        print(f"Subprocess execution:")
        print(f"  Exit code: {result.returncode}")
        print(f"  Output lines: {len(result.stdout.splitlines())}")
        
        if result.returncode == 0:
            # Count JSON messages
            json_lines = [line for line in result.stdout.splitlines() if line.strip().startswith('{')]
            print(f"  JSON messages: {len(json_lines)}")
            
            # Parse first message to show structure
            if json_lines:
                first_msg = json.loads(json_lines[0])
                print(f"  First frame ID: {first_msg['frame_id']}")
                print(f"  Detections in first frame: {len(first_msg['detections'])}")
        else:
            print(f"  Error: {result.stderr}")
        
        print()
        
    except subprocess.TimeoutExpired:
        print("Subprocess timed out")
        print()
    except Exception as e:
        print(f"Subprocess error: {e}")
        print()


def example_6_validation_and_config():
    """Example 6: Configuration validation and inspection."""
    print("=" * 60)
    print("Example 6: Configuration Validation")
    print("=" * 60)
    
    # Test valid configuration
    try:
        config = SimulatorConfig(
            vertical_fov_deg=50.0,
            num_drones=3,
            duration_s=10.0
        )
        config.validate()
        print("✓ Valid configuration created and validated")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
    
    # Test invalid configuration
    try:
        invalid_config = SimulatorConfig(
            vertical_fov_deg=200.0,  # Invalid FOV
            num_drones=0  # Invalid drone count
        )
        invalid_config.validate()
        print("✗ Invalid configuration should have failed")
    except Exception as e:
        print(f"✓ Invalid configuration correctly rejected: {e}")
    
    # Show configuration export
    config = SimulatorConfig(duration_s=5.0, offline_mode=True)
    config_dict = config.to_dict()
    print(f"✓ Configuration exported to dict with {len(config_dict)} parameters")
    print()


def example_7_different_scenarios():
    """Example 7: Different simulation scenarios."""
    print("=" * 60)
    print("Example 7: Different Scenarios")
    print("=" * 60)
    
    scenarios = [
        {
            'name': 'High Performance',
            'config': SimulatorConfig(
                fps=30.0, duration_s=2.0, pixel_centroid_sigma_px=0.5,
                offline_mode=True, deterministic_seed=1
            )
        },
        {
            'name': 'Noisy Environment',
            'config': SimulatorConfig(
                pixel_centroid_sigma_px=3.0, miss_rate_small=0.1,
                false_positive_rate=0.05, duration_s=2.0,
                offline_mode=True, deterministic_seed=2
            )
        },
        {
            'name': 'Multi-Drone',
            'config': SimulatorConfig(
                num_drones=4, duration_s=3.0, path_span_m=60.0,
                offline_mode=True, deterministic_seed=3
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"Running {scenario['name']} scenario...")
        simulator = DroneSimulator(scenario['config'])
        results = simulator.run()
        
        print(f"  Frames: {results['simulation']['frames_processed']}")
        print(f"  Detections: {results['detections']['total_generated']}")
        print(f"  False positives: {results['detections']['false_positives']}")
        print(f"  Missed detections: {results['detections']['missed_detections']}")
        print()


def main():
    """Run all examples."""
    print("Drone Detection Simulator - Usage Examples")
    print("==========================================")
    print()
    
    examples = [
        example_1_basic_programmatic,
        example_2_custom_configuration,
        example_3_configuration_from_file,
        example_4_cli_programmatic,
        example_5_cli_subprocess,
        example_6_validation_and_config,
        example_7_different_scenarios
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Example {i} failed: {e}")
            print()
    
    print("All examples completed!")
    print()
    print("To run individual examples:")
    print("  python examples/usage_examples.py")
    print()
    print("To run CLI directly:")
    print("  python -m drone_detection_simulator --help")
    print("  python -m drone_detection_simulator --list-examples")
    print("  python -m drone_detection_simulator --config examples/multi_drone.yaml")


if __name__ == "__main__":
    main()