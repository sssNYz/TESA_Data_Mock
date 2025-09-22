#!/usr/bin/env python3
"""
Example script demonstrating the JSON message builder functionality.

This script shows how to use the DetectionMessageBuilder to create
Pi-realistic detection messages with proper schema formatting.
"""

import json
from datetime import datetime, timezone
import numpy as np

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.camera import CameraModel
from drone_detection_simulator.message_builder import DetectionMessageBuilder


def main():
    """Demonstrate JSON message builder functionality."""
    print("=== Drone Detection Simulator - JSON Message Builder Demo ===\n")
    
    # Create configuration
    config = SimulatorConfig(
        image_width_px=1920,
        image_height_px=1080,
        vertical_fov_deg=50.0,
        camera_lat_deg=13.736717,
        camera_lon_deg=100.523186,
        camera_alt_m=1.5,
        camera_yaw_deg=90.0,
        camera_pitch_deg=10.0,
        processing_latency_ms_mean=45.0,
        processing_latency_ms_jitter=15.0,
        false_positive_rate=0.05,
        deterministic_seed=42
    )
    
    # Create camera model and message builder
    camera_model = CameraModel(config)
    rng = np.random.default_rng(42)
    message_builder = DetectionMessageBuilder(config, camera_model, rng)
    
    print("1. Camera Configuration:")
    print(f"   - Resolution: {config.image_width_px}x{config.image_height_px}")
    print(f"   - Vertical FOV: {config.vertical_fov_deg}°")
    print(f"   - Focal length: {camera_model.get_focal_length_px():.1f} pixels")
    print(f"   - Camera position: ({config.camera_lat_deg}, {config.camera_lon_deg})")
    print(f"   - Camera orientation: yaw={config.camera_yaw_deg}°, pitch={config.camera_pitch_deg}°")
    print()
    
    # Create sample detections (simulating DetectionGenerator output)
    sample_detections = [
        {
            'class': 'drone',
            'confidence': 0.91,
            'bbox_px': [980.5, 520.3, 1020.7, 580.9],
            'center_px': [1000.6, 550.6],
            'size_px': [40.2, 60.6],
            'drone_id': 0,
            'world_pos_enu': [10.0, 5.0, 5.5],
            'depth_m': 12.5,
            'projection_info': {
                'in_bounds': True,
                'distance_from_edge': 100.0
            }
        },
        {
            'class': 'drone',
            'confidence': 0.78,
            'bbox_px': [1200.1, 400.2, 1240.8, 450.9],
            'center_px': [1220.45, 425.55],
            'size_px': [40.7, 50.7],
            'drone_id': 1,
            'world_pos_enu': [15.0, 8.0, 6.0],
            'depth_m': 18.2,
            'projection_info': {
                'in_bounds': True,
                'distance_from_edge': 75.0
            }
        }
    ]
    
    print("2. Sample Detection Data:")
    for i, det in enumerate(sample_detections):
        print(f"   Drone {i+1}:")
        print(f"     - Confidence: {det['confidence']:.2f}")
        print(f"     - Bounding box: {det['bbox_px']}")
        print(f"     - Center: {det['center_px']}")
        print(f"     - Size: {det['size_px']} pixels")
        print(f"     - Distance: {det['depth_m']:.1f}m")
    print()
    
    # Build detection message
    timestamp = datetime(2025, 9, 21, 8, 23, 12, 123000, timezone.utc)
    message = message_builder.build_detection_message(sample_detections, timestamp)
    
    print("3. Generated JSON Message:")
    json_str = message_builder.to_json_string(message, indent=2)
    print(json_str)
    print()
    
    # Validate schema
    is_valid = message_builder.validate_message_schema(message)
    print(f"4. Schema Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    print()
    
    # Show message statistics
    size_bytes = message_builder.get_message_size_bytes(message)
    print("5. Message Statistics:")
    print(f"   - Frame ID: {message['frame_id']}")
    print(f"   - Number of detections: {len(message['detections'])}")
    print(f"   - Message size: {size_bytes} bytes")
    print(f"   - Processing latency: {message['edge']['processing_latency_ms']:.1f}ms")
    print()
    
    # Demonstrate false positive generation
    print("6. False Positive Generation:")
    detections_with_fp = message_builder.add_false_positives(sample_detections.copy())
    if len(detections_with_fp) > len(sample_detections):
        fp_message = message_builder.build_detection_message(detections_with_fp)
        print(f"   - Added {len(detections_with_fp) - len(sample_detections)} false positive(s)")
        print(f"   - Total detections: {len(fp_message['detections'])}")
        
        # Show false positive details
        for det in fp_message['detections']:
            if det['class'] == 'false_drone':
                print(f"   - False positive: confidence={det['confidence']:.3f}, "
                      f"center={det['center_px']}, size={det['size_px']}")
    else:
        print("   - No false positive generated (probability-based)")
    print()
    
    # Demonstrate multiple messages with frame counter
    print("7. Frame Counter Demonstration:")
    for i in range(3):
        msg = message_builder.build_detection_message([])
        print(f"   Frame {msg['frame_id']}: {msg['timestamp_utc']}")
    print()
    
    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()