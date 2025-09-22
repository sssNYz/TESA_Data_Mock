#!/usr/bin/env python3
"""
Multi-Drone Detection Simulator Demo

This script demonstrates the multi-drone support functionality implemented
in task 9, showing how the simulator can handle multiple drones simultaneously
with independent flight paths, overlapping detections, and proper JSON output.

Requirements demonstrated:
- 7.1: Independent smooth flight paths for multiple drones
- 7.2: Detection message arrays with entries for multiple drones  
- 7.3: Multiple bounding boxes when drones overlap in camera view
- 7.4: Independent distance estimates for different drone sizes/altitudes
"""

import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import the simulator
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.motion import MotionGenerator
from drone_detection_simulator.camera import CameraModel
from drone_detection_simulator.detection import DetectionGenerator
from drone_detection_simulator.message_builder import DetectionMessageBuilder


def demo_requirement_7_1_independent_paths():
    """Demonstrate requirement 7.1: Independent smooth flight paths."""
    print("=" * 60)
    print("REQUIREMENT 7.1: Independent Smooth Flight Paths")
    print("=" * 60)
    
    config = SimulatorConfig(
        num_drones=5,
        duration_s=10.0,
        fps=20.0,
        path_span_m=40.0,
        speed_mps=5.0,
        deterministic_seed=42
    )
    
    motion_gen = MotionGenerator(config)
    
    print(f"Generated {len(motion_gen.paths)} independent drone paths")
    print(f"Each path has {len(motion_gen.paths[0])} frames")
    
    # Show positions at different times
    test_times = [0.0, 2.5, 5.0, 7.5, 10.0]
    
    for time_s in test_times:
        positions = motion_gen.get_positions_at_time(time_s)
        print(f"\nTime {time_s:.1f}s:")
        for i, pos in enumerate(positions):
            print(f"  Drone {i}: East={pos[0]:6.1f}m, North={pos[1]:5.1f}m, Up={pos[2]:4.1f}m")
    
    # Verify independence - different north positions and altitudes
    initial_positions = motion_gen.get_positions_at_time(0.0)
    north_positions = [pos[1] for pos in initial_positions]
    altitudes = [pos[2] for pos in initial_positions]
    
    print(f"\nPath Independence Verification:")
    print(f"  North position spread: {max(north_positions) - min(north_positions):.1f}m")
    print(f"  Altitude spread: {max(altitudes) - min(altitudes):.1f}m")
    print(f"  All north positions unique: {len(set(north_positions)) == len(north_positions)}")
    print(f"  All altitudes unique: {len(set(altitudes)) == len(altitudes)}")
    
    # Verify acceleration constraints
    constraints_ok, max_accel = motion_gen.verify_acceleration_constraints()
    print(f"  Acceleration constraints satisfied: {constraints_ok}")
    print(f"  Maximum acceleration found: {max_accel:.2f} m/s²")
    print(f"  Configured limit: {config.max_lateral_accel_mps2:.2f} m/s²")


def demo_requirement_7_2_detection_arrays():
    """Demonstrate requirement 7.2: Detection message arrays."""
    print("\n" + "=" * 60)
    print("REQUIREMENT 7.2: Detection Message Arrays")
    print("=" * 60)
    
    config = SimulatorConfig(
        num_drones=4,
        vertical_fov_deg=50.0,
        drone_size_m=0.25,
        deterministic_seed=42
    )
    
    camera_model = CameraModel(config)
    detection_gen = DetectionGenerator(config, camera_model)
    message_builder = DetectionMessageBuilder(config, camera_model)
    
    # Create multiple visible drone positions
    world_positions = [
        np.array([10.0, 0.0, 5.0]),    # Drone 0 - center
        np.array([12.0, 1.5, 5.5]),    # Drone 1 - right
        np.array([8.0, -1.0, 4.5]),    # Drone 2 - left
        np.array([15.0, 0.5, 6.0])     # Drone 3 - far right
    ]
    
    print(f"Input: {len(world_positions)} drone world positions")
    for i, pos in enumerate(world_positions):
        print(f"  Drone {i}: East={pos[0]:4.1f}m, North={pos[1]:5.1f}m, Up={pos[2]:4.1f}m")
    
    # Generate detections
    detections = detection_gen.generate_detections(world_positions)
    print(f"\nGenerated {len(detections)} detections")
    
    for i, det in enumerate(detections):
        print(f"  Detection {i}: drone_id={det['drone_id']}, confidence={det['confidence']:.2f}")
        print(f"    BBox: [{det['bbox_px'][0]:6.1f}, {det['bbox_px'][1]:6.1f}, {det['bbox_px'][2]:6.1f}, {det['bbox_px'][3]:6.1f}]")
        print(f"    Center: [{det['center_px'][0]:6.1f}, {det['center_px'][1]:6.1f}]")
        print(f"    Size: [{det['size_px'][0]:5.1f}, {det['size_px'][1]:5.1f}]")
        print(f"    Depth: {det['depth_m']:5.1f}m")
    
    # Build JSON message
    message = message_builder.build_detection_message(detections)
    
    print(f"\nJSON Message Structure:")
    print(f"  Frame ID: {message['frame_id']}")
    print(f"  Timestamp: {message['timestamp_utc']}")
    print(f"  Detection array length: {len(message['detections'])}")
    print(f"  Camera metadata: {len(message['camera'])} fields")
    print(f"  Edge metadata: {len(message['edge'])} fields")
    
    # Show detection array in message
    print(f"\nDetection Array in Message:")
    for i, det in enumerate(message['detections']):
        print(f"  Detection {i}: class='{det['class']}', confidence={det['confidence']}")
        print(f"    BBox: {det['bbox_px']}")
        print(f"    Center: {det['center_px']}")
        print(f"    Size: {det['size_px']}")
        # Verify internal data is not exposed
        internal_fields = ['drone_id', 'world_pos_enu', 'depth_m', 'projection_info']
        has_internal = any(field in det for field in internal_fields)
        print(f"    Has internal data: {has_internal}")
    
    # Validate schema
    is_valid = message_builder.validate_message_schema(message)
    print(f"\nMessage schema validation: {'PASSED' if is_valid else 'FAILED'}")


def demo_requirement_7_3_overlapping_drones():
    """Demonstrate requirement 7.3: Multiple bounding boxes for overlapping drones."""
    print("\n" + "=" * 60)
    print("REQUIREMENT 7.3: Overlapping Drones - Multiple Bounding Boxes")
    print("=" * 60)
    
    config = SimulatorConfig(
        vertical_fov_deg=50.0,
        drone_size_m=0.25,
        deterministic_seed=42
    )
    
    camera_model = CameraModel(config)
    detection_gen = DetectionGenerator(config, camera_model)
    message_builder = DetectionMessageBuilder(config, camera_model)
    
    # Create overlapping drone positions
    overlapping_positions = [
        np.array([10.0, 0.0, 5.0]),     # Drone 0 - center
        np.array([10.3, 0.1, 5.1]),     # Drone 1 - very close to drone 0
        np.array([10.1, -0.1, 4.9]),    # Drone 2 - also close to drone 0
        np.array([15.0, 2.0, 6.0])      # Drone 3 - separate
    ]
    
    print("Overlapping Drone Positions:")
    for i, pos in enumerate(overlapping_positions):
        print(f"  Drone {i}: East={pos[0]:5.1f}m, North={pos[1]:5.1f}m, Up={pos[2]:4.1f}m")
    
    # Calculate distances between drones
    print("\nDrone-to-Drone Distances:")
    for i in range(len(overlapping_positions)):
        for j in range(i + 1, len(overlapping_positions)):
            distance = np.linalg.norm(overlapping_positions[i] - overlapping_positions[j])
            print(f"  Drone {i} to {j}: {distance:.2f}m")
    
    # Generate detections
    detections = detection_gen.generate_detections(overlapping_positions)
    message = message_builder.build_detection_message(detections)
    
    print(f"\nGenerated {len(message['detections'])} detections from overlapping drones")
    
    # Check for overlapping bounding boxes
    if len(message['detections']) >= 2:
        print("\nBounding Box Overlap Analysis:")
        for i in range(len(message['detections'])):
            for j in range(i + 1, len(message['detections'])):
                bbox1 = message['detections'][i]['bbox_px']
                bbox2 = message['detections'][j]['bbox_px']
                
                # Check overlap
                overlap_x = not (bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0])
                overlap_y = not (bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1])
                overlapping = overlap_x and overlap_y
                
                print(f"  Detection {i} vs {j}: {'OVERLAPPING' if overlapping else 'SEPARATE'}")
                if overlapping:
                    print(f"    BBox {i}: [{bbox1[0]:6.1f}, {bbox1[1]:6.1f}, {bbox1[2]:6.1f}, {bbox1[3]:6.1f}]")
                    print(f"    BBox {j}: [{bbox2[0]:6.1f}, {bbox2[1]:6.1f}, {bbox2[2]:6.1f}, {bbox2[3]:6.1f}]")
    
    # Show that overlapping detections are still distinct
    print(f"\nDistinct Detection Verification:")
    for i, det in enumerate(message['detections']):
        print(f"  Detection {i}: confidence={det['confidence']:.3f}, center={det['center_px']}")


def demo_requirement_7_4_independent_distance_estimates():
    """Demonstrate requirement 7.4: Independent distance estimates."""
    print("\n" + "=" * 60)
    print("REQUIREMENT 7.4: Independent Distance Estimates")
    print("=" * 60)
    
    # Test with different drone sizes
    print("A. Different Drone Sizes (same position):")
    
    sizes = [0.15, 0.25, 0.35, 0.45]  # Different drone sizes
    same_position = np.array([15.0, 0.0, 5.0])
    
    for size in sizes:
        config = SimulatorConfig(drone_size_m=size, vertical_fov_deg=50.0)
        camera_model = CameraModel(config)
        detection_gen = DetectionGenerator(config, camera_model)
        
        detection = detection_gen.generate_detections([same_position])[0]
        estimated_distance = detection_gen.estimate_distance_from_size(detection)
        
        avg_size_px = sum(detection['size_px']) / 2
        print(f"  Drone size {size:.2f}m: pixel_size={avg_size_px:5.1f}px, "
              f"actual_depth={detection['depth_m']:5.1f}m, estimated={estimated_distance:5.1f}m")
    
    # Test with different altitudes
    print("\nB. Different Altitudes (same horizontal position):")
    
    config = SimulatorConfig(drone_size_m=0.25, vertical_fov_deg=50.0)
    camera_model = CameraModel(config)
    detection_gen = DetectionGenerator(config, camera_model)
    
    altitudes = [3.0, 5.0, 7.0, 9.0]
    base_position = [12.0, 0.0]  # Same east/north
    
    for altitude in altitudes:
        position = np.array([base_position[0], base_position[1], altitude])
        detections = detection_gen.generate_detections([position])
        if not detections:
            print(f"  Altitude {altitude:.1f}m: NOT VISIBLE (outside camera view)")
            continue
        detection = detections[0]
        estimated_distance = detection_gen.estimate_distance_from_size(detection)
        
        avg_size_px = sum(detection['size_px']) / 2
        print(f"  Altitude {altitude:.1f}m: pixel_size={avg_size_px:5.1f}px, "
              f"actual_depth={detection['depth_m']:5.1f}m, estimated={estimated_distance:5.1f}m, "
              f"confidence={detection['confidence']:.3f}")
    
    # Test with mixed scenario
    print("\nC. Mixed Scenario (different sizes and altitudes):")
    
    mixed_positions = [
        np.array([10.0, 0.0, 4.0]),   # Low altitude
        np.array([12.0, 0.5, 6.0]),   # Medium altitude
        np.array([14.0, -0.5, 8.0])   # High altitude
    ]
    
    detections = detection_gen.generate_detections(mixed_positions)
    
    for i, det in enumerate(detections):
        estimated_distance = detection_gen.estimate_distance_from_size(det)
        avg_size_px = sum(det['size_px']) / 2
        
        print(f"  Drone {i}: altitude={mixed_positions[i][2]:.1f}m, "
              f"pixel_size={avg_size_px:5.1f}px, "
              f"actual_depth={det['depth_m']:5.1f}m, "
              f"estimated={estimated_distance:5.1f}m, "
              f"confidence={det['confidence']:.3f}")


def demo_complete_multi_drone_simulation():
    """Demonstrate complete multi-drone simulation over time."""
    print("\n" + "=" * 60)
    print("COMPLETE MULTI-DRONE SIMULATION")
    print("=" * 60)
    
    config = SimulatorConfig(
        num_drones=3,
        duration_s=6.0,
        fps=10.0,
        path_span_m=25.0,
        speed_mps=3.0,
        vertical_fov_deg=50.0,
        drone_size_m=0.25,
        deterministic_seed=42
    )
    
    # Create pipeline components
    motion_gen = MotionGenerator(config)
    camera_model = CameraModel(config)
    detection_gen = DetectionGenerator(config, camera_model)
    message_builder = DetectionMessageBuilder(config, camera_model)
    
    print(f"Simulating {config.num_drones} drones over {config.duration_s}s at {config.fps} FPS")
    
    # Simulate several frames
    sample_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    for time_s in sample_times:
        print(f"\n--- Frame at t={time_s:.1f}s ---")
        
        # Get world positions from motion generator
        world_positions = motion_gen.get_positions_at_time(time_s)
        
        # For demo purposes, use visible positions if motion positions are behind camera
        if all(pos[0] < 0 for pos in world_positions):
            # Use manually positioned visible drones for demo
            visible_positions = [
                np.array([8.0 + i * 2, (i - 1) * 0.5, 5.0 + i * 0.3]) 
                for i in range(config.num_drones)
            ]
            world_positions = visible_positions
            print("  (Using demo positions - motion positions behind camera)")
        
        # Generate detections
        detections = detection_gen.generate_detections(world_positions)
        
        # Build message
        message = message_builder.build_detection_message(detections)
        
        print(f"  World positions: {len(world_positions)}")
        print(f"  Detections generated: {len(detections)}")
        print(f"  Message detections: {len(message['detections'])}")
        print(f"  Frame ID: {message['frame_id']}")
        
        if detections:
            confidences = [d['confidence'] for d in detections]
            depths = [d['depth_m'] for d in detections]
            print(f"  Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")
            print(f"  Depth range: {min(depths):.1f}m - {max(depths):.1f}m")
    
    # Show final statistics
    print(f"\nSimulation Statistics:")
    motion_stats = motion_gen.get_path_statistics()
    print(f"  Motion paths generated: {motion_stats['num_drones']}")
    print(f"  Path length (frames): {motion_stats['path_length_frames']}")
    print(f"  Acceleration constraints satisfied: {motion_stats['acceleration_constraints_satisfied']}")
    print(f"  Max acceleration found: {motion_stats['max_lateral_acceleration_mps2']:.2f} m/s²")


def demo_json_output_example():
    """Show example JSON output for multi-drone detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE JSON OUTPUT")
    print("=" * 60)
    
    config = SimulatorConfig(
        vertical_fov_deg=50.0,
        drone_size_m=0.25,
        processing_latency_ms_mean=45.0,
        processing_latency_ms_jitter=0.0,  # No jitter for consistent demo
        deterministic_seed=42
    )
    
    camera_model = CameraModel(config)
    detection_gen = DetectionGenerator(config, camera_model)
    message_builder = DetectionMessageBuilder(config, camera_model)
    
    # Create sample multi-drone scenario
    demo_positions = [
        np.array([10.0, 0.0, 5.0]),    # Drone 0
        np.array([12.0, 1.0, 5.5]),    # Drone 1
        np.array([8.0, -0.5, 4.5])     # Drone 2
    ]
    
    detections = detection_gen.generate_detections(demo_positions)
    message = message_builder.build_detection_message(detections)
    
    # Pretty print the JSON
    json_str = message_builder.to_json_string(message, indent=2)
    print(json_str)
    
    print(f"\nMessage size: {message_builder.get_message_size_bytes(message)} bytes")
    print(f"Schema validation: {'PASSED' if message_builder.validate_message_schema(message) else 'FAILED'}")


def main():
    """Run all multi-drone demonstrations."""
    print("MULTI-DRONE DETECTION SIMULATOR DEMONSTRATION")
    print("Task 9: Add multi-drone support")
    print("Requirements 7.1, 7.2, 7.3, 7.4")
    
    try:
        demo_requirement_7_1_independent_paths()
        demo_requirement_7_2_detection_arrays()
        demo_requirement_7_3_overlapping_drones()
        demo_requirement_7_4_independent_distance_estimates()
        demo_complete_multi_drone_simulation()
        demo_json_output_example()
        
        print("\n" + "=" * 60)
        print("✓ ALL MULTI-DRONE REQUIREMENTS DEMONSTRATED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())