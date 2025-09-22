#!/usr/bin/env python3
"""
Example demonstrating deterministic testing support for the drone detection simulator.

This example shows how to use the deterministic seed configuration to create
reproducible test runs for validation and debugging purposes.
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.simulator import DroneSimulator


def demonstrate_deterministic_behavior():
    """Demonstrate that deterministic seeds produce identical results."""
    print("=== Deterministic Testing Support Example ===\n")
    
    # Create a deterministic configuration
    config = SimulatorConfig(
        deterministic_seed=42,  # Fixed seed for reproducible results
        offline_mode=True,      # Print JSON instead of MQTT
        duration_s=2.0,         # Short simulation for demo
        fps=10.0,               # 10 frames per second
        num_drones=2,           # Two drones for interesting motion
        path_span_m=15.0,       # 15m flight path
        speed_mps=3.0,          # 3 m/s speed
        miss_rate_small=0.0,    # Ensure all detections are generated
        false_positive_rate=0.05  # Add some false positives
    )
    
    print(f"Configuration:")
    print(f"  - Deterministic seed: {config.deterministic_seed}")
    print(f"  - Duration: {config.duration_s}s at {config.fps} FPS")
    print(f"  - Number of drones: {config.num_drones}")
    print(f"  - Offline mode: {config.offline_mode}")
    print()
    
    # Run simulation twice with same configuration
    print("Running first simulation...")
    outputs1 = run_simulation_with_capture(config)
    
    print("Running second simulation with same seed...")
    outputs2 = run_simulation_with_capture(config)
    
    # Parse JSON messages from both runs
    messages1 = extract_json_messages(outputs1)
    messages2 = extract_json_messages(outputs2)
    
    print(f"First run: {len(messages1)} messages")
    print(f"Second run: {len(messages2)} messages")
    
    # Compare results
    if len(messages1) != len(messages2):
        print("❌ Different number of messages - determinism failed!")
        return False
    
    identical_frames = 0
    for i, (msg1, msg2) in enumerate(zip(messages1, messages2)):
        if messages_identical(msg1, msg2):
            identical_frames += 1
        else:
            print(f"Frame {i} differs between runs")
    
    match_rate = identical_frames / len(messages1) if messages1 else 0
    print(f"Identical frames: {identical_frames}/{len(messages1)} ({match_rate:.1%})")
    
    if match_rate == 1.0:
        print("✅ Perfect deterministic behavior achieved!")
        return True
    else:
        print("❌ Deterministic behavior failed!")
        return False


def demonstrate_different_seeds():
    """Demonstrate that different seeds produce different results."""
    print("\n=== Different Seeds Produce Different Results ===\n")
    
    # Create configurations with different seeds
    config1 = SimulatorConfig(
        deterministic_seed=123,
        offline_mode=True,
        duration_s=1.0,
        fps=5.0,
        num_drones=1
    )
    
    config2 = SimulatorConfig(
        deterministic_seed=456,  # Different seed
        offline_mode=True,
        duration_s=1.0,
        fps=5.0,
        num_drones=1
    )
    
    print("Running simulation with seed 123...")
    outputs1 = run_simulation_with_capture(config1)
    
    print("Running simulation with seed 456...")
    outputs2 = run_simulation_with_capture(config2)
    
    messages1 = extract_json_messages(outputs1)
    messages2 = extract_json_messages(outputs2)
    
    if len(messages1) != len(messages2):
        print("Different number of messages (expected with different seeds)")
        return True
    
    # Compare first frame detections
    if messages1 and messages2:
        det1 = messages1[0].get('detections', [])
        det2 = messages2[0].get('detections', [])
        
        if det1 and det2:
            pos1 = det1[0].get('center_px', [0, 0])
            pos2 = det2[0].get('center_px', [0, 0])
            
            distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
            print(f"First detection position difference: {distance:.1f} pixels")
            
            if distance > 1.0:  # Should be different
                print("✅ Different seeds produce different results!")
                return True
            else:
                print("❌ Different seeds produced similar results")
                return False
    
    print("No detections to compare")
    return True


def demonstrate_validation_utilities():
    """Demonstrate validation utilities for testing."""
    print("\n=== Validation Utilities Example ===\n")
    
    config = SimulatorConfig(
        deterministic_seed=789,
        offline_mode=True,
        duration_s=1.5,
        fps=8.0,
        num_drones=1,
        max_lateral_accel_mps2=1.5
    )
    
    print("Running simulation for validation...")
    outputs = run_simulation_with_capture(config)
    messages = extract_json_messages(outputs)
    
    print(f"Generated {len(messages)} detection messages")
    
    # Validate message structure
    valid_messages = 0
    for i, message in enumerate(messages):
        if validate_message_structure(message):
            valid_messages += 1
        else:
            print(f"Invalid message structure at frame {i}")
    
    print(f"Valid message structures: {valid_messages}/{len(messages)}")
    
    # Validate pixel bounds
    total_detections = 0
    bounds_violations = 0
    
    for message in messages:
        for detection in message.get('detections', []):
            total_detections += 1
            
            # Check center coordinates
            if 'center_px' in detection:
                x, y = detection['center_px']
                if not (0 <= x < config.image_width_px and 0 <= y < config.image_height_px):
                    bounds_violations += 1
            
            # Check bounding box
            if 'bbox_px' in detection:
                x1, y1, x2, y2 = detection['bbox_px']
                if not (0 <= x1 < x2 <= config.image_width_px and 
                       0 <= y1 < y2 <= config.image_height_px):
                    bounds_violations += 1
    
    print(f"Pixel bounds validation: {total_detections} detections, {bounds_violations} violations")
    
    if bounds_violations == 0:
        print("✅ All detections within pixel bounds!")
    else:
        print("❌ Some detections outside pixel bounds!")
    
    return bounds_violations == 0


def run_simulation_with_capture(config: SimulatorConfig):
    """Run simulation and capture printed output."""
    captured_output = []
    
    def mock_print(*args, **kwargs):
        captured_output.append(args[0] if args else "")
    
    with patch('builtins.print', side_effect=mock_print):
        simulator = DroneSimulator(config)
        results = simulator.run()
    
    return captured_output


def extract_json_messages(outputs):
    """Extract JSON detection messages from captured output."""
    messages = []
    for output in outputs:
        if output.strip():
            try:
                msg = json.loads(output)
                if 'detections' in msg and 'timestamp_utc' in msg:
                    messages.append(msg)
            except json.JSONDecodeError:
                pass
    return messages


def messages_identical(msg1, msg2, tolerance=1e-10):
    """Check if two messages are identical within tolerance."""
    det1 = msg1.get('detections', [])
    det2 = msg2.get('detections', [])
    
    if len(det1) != len(det2):
        return False
    
    for d1, d2 in zip(det1, det2):
        # Compare positions
        if 'center_px' in d1 and 'center_px' in d2:
            pos_diff = sum(abs(a - b) for a, b in zip(d1['center_px'], d2['center_px']))
            if pos_diff > tolerance:
                return False
        
        # Compare confidence
        if 'confidence' in d1 and 'confidence' in d2:
            if abs(d1['confidence'] - d2['confidence']) > tolerance:
                return False
    
    return True


def validate_message_structure(message):
    """Validate detection message structure."""
    required_fields = ['timestamp_utc', 'frame_id', 'camera', 'detections', 'edge']
    
    for field in required_fields:
        if field not in message:
            return False
    
    if not isinstance(message['detections'], list):
        return False
    
    for detection in message['detections']:
        detection_fields = ['class', 'confidence', 'bbox_px', 'center_px', 'size_px']
        for field in detection_fields:
            if field not in detection:
                return False
    
    return True


def main():
    """Main example function."""
    print("Drone Detection Simulator - Deterministic Testing Example")
    print("=" * 60)
    
    success = True
    
    # Demonstrate deterministic behavior
    success &= demonstrate_deterministic_behavior()
    
    # Demonstrate different seeds
    success &= demonstrate_different_seeds()
    
    # Demonstrate validation utilities
    success &= demonstrate_validation_utilities()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All deterministic testing examples completed successfully!")
        print("\nKey takeaways:")
        print("  - Use deterministic_seed parameter for reproducible tests")
        print("  - Same seed = identical results")
        print("  - Different seeds = different results")
        print("  - Validation utilities help verify correctness")
        return 0
    else:
        print("❌ Some examples failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())