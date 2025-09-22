#!/usr/bin/env python3
"""
Example script demonstrating configuration loading and validation.
"""

import sys
from pathlib import Path

# Add parent directory to path to import the simulator package
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_detection_simulator.config import SimulatorConfig


def main():
    """Demonstrate configuration loading and validation."""
    
    print("=== Drone Detection Simulator Configuration Demo ===\n")
    
    # Test 1: Default configuration
    print("1. Testing default configuration:")
    try:
        config = SimulatorConfig()
        print("   ✓ Default configuration is valid")
        print(f"   - Image size: {config.image_width_px}x{config.image_height_px}")
        print(f"   - Vertical FOV: {config.vertical_fov_deg}°")
        print(f"   - Camera position: ({config.camera_lat_deg}, {config.camera_lon_deg})")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Load from YAML file
    print("\n2. Testing YAML configuration loading:")
    yaml_path = Path(__file__).parent / "config_example.yaml"
    try:
        config = SimulatorConfig.from_yaml(yaml_path)
        print("   ✓ YAML configuration loaded successfully")
        print(f"   - Number of drones: {config.num_drones}")
        print(f"   - Simulation duration: {config.duration_s}s")
        print(f"   - MQTT topic: {config.mqtt_topic}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Configuration validation
    print("\n3. Testing configuration validation:")
    try:
        # This should fail - invalid FOV
        invalid_config = SimulatorConfig(vertical_fov_deg=200.0)
        print("   ✗ Validation failed to catch invalid FOV")
    except ValueError as e:
        print(f"   ✓ Validation correctly caught error: {e}")
    
    # Test 4: Dictionary conversion
    print("\n4. Testing dictionary conversion:")
    try:
        config = SimulatorConfig(image_width_px=1280, principal_point_px=(640, 360))
        config_dict = config.to_dict()
        print("   ✓ Configuration converted to dictionary")
        print(f"   - Principal point as list: {config_dict['principal_point_px']}")
        
        # Convert back
        restored_config = SimulatorConfig.from_dict(config_dict)
        print(f"   - Principal point restored as tuple: {restored_config.principal_point_px}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n=== Configuration demo completed ===")


if __name__ == "__main__":
    main()