"""
Validation utilities for testing drone detection simulator.

This module provides reusable utilities for validating simulation output,
motion constraints, and deterministic behavior across different test scenarios.
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from drone_detection_simulator.config import SimulatorConfig


class ValidationUtils:
    """Utility class for validation functions used across tests."""
    
    @staticmethod
    def validate_motion_smoothness(positions: List[np.ndarray], 
                                 max_acceleration: float, 
                                 fps: float,
                                 tolerance: float = 0.5) -> Tuple[bool, Dict[str, float]]:
        """
        Validate that motion satisfies smoothness and acceleration constraints.
        
        Args:
            positions: List of 3D position vectors (ENU coordinates)
            max_acceleration: Maximum allowed lateral acceleration (m/s²)
            fps: Frames per second
            tolerance: Tolerance for acceleration constraint (m/s²)
            
        Returns:
            Tuple of (constraints_satisfied, metrics_dict)
        """
        if len(positions) < 3:
            return True, {'max_acceleration': 0.0, 'avg_acceleration': 0.0}
        
        frame_time = 1.0 / fps
        velocities = []
        accelerations = []
        
        # Calculate velocities
        for i in range(1, len(positions)):
            velocity = (positions[i] - positions[i-1]) / frame_time
            velocities.append(velocity)
        
        # Calculate accelerations (lateral component only)
        lateral_accelerations = []
        for i in range(1, len(velocities)):
            acceleration = (velocities[i] - velocities[i-1]) / frame_time
            lateral_accel = abs(acceleration[0])  # East component
            lateral_accelerations.append(lateral_accel)
            accelerations.append(acceleration)
        
        if not lateral_accelerations:
            return True, {'max_acceleration': 0.0, 'avg_acceleration': 0.0}
        
        max_accel = max(lateral_accelerations)
        avg_accel = np.mean(lateral_accelerations)
        
        # Calculate jerk for smoothness assessment
        jerks = []
        for i in range(1, len(accelerations)):
            jerk = np.linalg.norm(accelerations[i] - accelerations[i-1]) / frame_time
            jerks.append(jerk)
        
        avg_jerk = np.mean(jerks) if jerks else 0.0
        
        constraints_satisfied = max_accel <= max_acceleration + tolerance
        
        metrics = {
            'max_acceleration': max_accel,
            'avg_acceleration': avg_accel,
            'avg_jerk': avg_jerk,
            'constraint_violation': max(0, max_accel - max_acceleration)
        }
        
        return constraints_satisfied, metrics
    
    @staticmethod
    def validate_pixel_coordinates(detections: List[Dict[str, Any]], 
                                 image_width: int, 
                                 image_height: int) -> Tuple[bool, List[str]]:
        """
        Validate that all pixel coordinates are within image bounds.
        
        Args:
            detections: List of detection dictionaries
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        
        for i, detection in enumerate(detections):
            # Validate center coordinates
            if 'center_px' in detection:
                center_x, center_y = detection['center_px']
                if not (0 <= center_x < image_width):
                    errors.append(f"Detection {i}: center_x {center_x} out of bounds [0, {image_width})")
                if not (0 <= center_y < image_height):
                    errors.append(f"Detection {i}: center_y {center_y} out of bounds [0, {image_height})")
            
            # Validate bounding box coordinates
            if 'bbox_px' in detection:
                x1, y1, x2, y2 = detection['bbox_px']
                
                if not (0 <= x1 < image_width):
                    errors.append(f"Detection {i}: bbox x1 {x1} out of bounds [0, {image_width})")
                if not (0 <= y1 < image_height):
                    errors.append(f"Detection {i}: bbox y1 {y1} out of bounds [0, {image_height})")
                if not (0 < x2 <= image_width):
                    errors.append(f"Detection {i}: bbox x2 {x2} out of bounds (0, {image_width}]")
                if not (0 < y2 <= image_height):
                    errors.append(f"Detection {i}: bbox y2 {y2} out of bounds (0, {image_height}]")
                
                if x1 >= x2:
                    errors.append(f"Detection {i}: invalid bbox x1 {x1} >= x2 {x2}")
                if y1 >= y2:
                    errors.append(f"Detection {i}: invalid bbox y1 {y1} >= y2 {y2}")
            
            # Validate size consistency
            if 'size_px' in detection:
                width, height = detection['size_px']
                if width <= 0:
                    errors.append(f"Detection {i}: invalid width {width} <= 0")
                if height <= 0:
                    errors.append(f"Detection {i}: invalid height {height} <= 0")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_detection_message_schema(message: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate detection message against expected schema.
        
        Args:
            message: Detection message dictionary
            
        Returns:
            Tuple of (schema_valid, list_of_errors)
        """
        errors = []
        
        # Check required top-level fields
        required_fields = ['timestamp_utc', 'frame_id', 'camera', 'detections', 'edge']
        for field in required_fields:
            if field not in message:
                errors.append(f"Missing required field: {field}")
        
        # Validate camera metadata
        if 'camera' in message:
            camera = message['camera']
            camera_fields = ['resolution', 'focal_px', 'principal_point', 'yaw_deg', 
                           'pitch_deg', 'lat_deg', 'lon_deg', 'alt_m_msl']
            for field in camera_fields:
                if field not in camera:
                    errors.append(f"Missing camera field: {field}")
        
        # Validate detections array
        if 'detections' in message:
            if not isinstance(message['detections'], list):
                errors.append("detections field must be a list")
            else:
                for i, detection in enumerate(message['detections']):
                    detection_fields = ['class', 'confidence', 'bbox_px', 'center_px', 'size_px']
                    for field in detection_fields:
                        if field not in detection:
                            errors.append(f"Detection {i} missing field: {field}")
                    
                    # Validate confidence range
                    if 'confidence' in detection:
                        conf = detection['confidence']
                        if not (0.0 <= conf <= 1.0):
                            errors.append(f"Detection {i} confidence {conf} out of range [0, 1]")
        
        # Validate edge metadata
        if 'edge' in message:
            edge = message['edge']
            edge_fields = ['processing_latency_ms', 'detector_version']
            for field in edge_fields:
                if field not in edge:
                    errors.append(f"Missing edge field: {field}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def compare_detection_sequences(sequence1: List[Dict[str, Any]], 
                                  sequence2: List[Dict[str, Any]],
                                  position_tolerance: float = 1e-6,
                                  confidence_tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Compare two sequences of detection messages for consistency.
        
        Args:
            sequence1: First sequence of detection messages
            sequence2: Second sequence of detection messages
            position_tolerance: Tolerance for position comparisons
            confidence_tolerance: Tolerance for confidence comparisons
            
        Returns:
            Dictionary with comparison results and metrics
        """
        if len(sequence1) != len(sequence2):
            return {
                'sequences_match': False,
                'length_mismatch': True,
                'length1': len(sequence1),
                'length2': len(sequence2)
            }
        
        frame_matches = 0
        detection_count_matches = 0
        position_differences = []
        confidence_differences = []
        
        for i, (msg1, msg2) in enumerate(zip(sequence1, sequence2)):
            det1 = msg1.get('detections', [])
            det2 = msg2.get('detections', [])
            
            # Check detection count
            if len(det1) == len(det2):
                detection_count_matches += 1
                
                # Compare individual detections
                frame_position_diffs = []
                frame_confidence_diffs = []
                
                for d1, d2 in zip(det1, det2):
                    # Compare positions
                    if 'center_px' in d1 and 'center_px' in d2:
                        pos_diff = np.linalg.norm(np.array(d1['center_px']) - np.array(d2['center_px']))
                        frame_position_diffs.append(pos_diff)
                        position_differences.append(pos_diff)
                    
                    # Compare confidence
                    if 'confidence' in d1 and 'confidence' in d2:
                        conf_diff = abs(d1['confidence'] - d2['confidence'])
                        frame_confidence_diffs.append(conf_diff)
                        confidence_differences.append(conf_diff)
                
                # Check if frame matches within tolerance
                if (all(diff <= position_tolerance for diff in frame_position_diffs) and
                    all(diff <= confidence_tolerance for diff in frame_confidence_diffs)):
                    frame_matches += 1
        
        total_frames = len(sequence1)
        
        return {
            'sequences_match': frame_matches == total_frames,
            'length_mismatch': False,
            'frame_match_rate': frame_matches / total_frames if total_frames > 0 else 0.0,
            'detection_count_match_rate': detection_count_matches / total_frames if total_frames > 0 else 0.0,
            'avg_position_difference': np.mean(position_differences) if position_differences else 0.0,
            'max_position_difference': np.max(position_differences) if position_differences else 0.0,
            'avg_confidence_difference': np.mean(confidence_differences) if confidence_differences else 0.0,
            'max_confidence_difference': np.max(confidence_differences) if confidence_differences else 0.0,
            'total_frames': total_frames,
            'matching_frames': frame_matches
        }
    
    @staticmethod
    def extract_motion_path_from_messages(messages: List[Dict[str, Any]], 
                                        drone_index: int = 0) -> List[np.ndarray]:
        """
        Extract motion path for a specific drone from detection messages.
        
        Args:
            messages: List of detection messages
            drone_index: Index of drone to extract path for
            
        Returns:
            List of 2D pixel positions for the specified drone
        """
        path = []
        
        for message in messages:
            detections = message.get('detections', [])
            if len(detections) > drone_index:
                detection = detections[drone_index]
                if 'center_px' in detection:
                    center = detection['center_px']
                    path.append(np.array([center[0], center[1]]))
        
        return path
    
    @staticmethod
    def calculate_pixel_motion_smoothness(pixel_path: List[np.ndarray], 
                                        fps: float) -> Dict[str, float]:
        """
        Calculate smoothness metrics for pixel-space motion.
        
        Args:
            pixel_path: List of 2D pixel coordinates
            fps: Frames per second
            
        Returns:
            Dictionary with pixel motion smoothness metrics
        """
        if len(pixel_path) < 3:
            return {'max_pixel_velocity': 0.0, 'avg_pixel_velocity': 0.0, 'pixel_jerk': 0.0}
        
        frame_time = 1.0 / fps
        velocities = []
        accelerations = []
        
        # Calculate pixel velocities
        for i in range(1, len(pixel_path)):
            velocity = (pixel_path[i] - pixel_path[i-1]) / frame_time
            velocities.append(velocity)
        
        # Calculate pixel accelerations
        for i in range(1, len(velocities)):
            acceleration = (velocities[i] - velocities[i-1]) / frame_time
            accelerations.append(acceleration)
        
        # Calculate metrics
        velocity_magnitudes = [np.linalg.norm(v) for v in velocities]
        acceleration_magnitudes = [np.linalg.norm(a) for a in accelerations]
        
        # Calculate jerk (rate of change of acceleration)
        jerks = []
        for i in range(1, len(accelerations)):
            jerk = np.linalg.norm(accelerations[i] - accelerations[i-1]) / frame_time
            jerks.append(jerk)
        
        return {
            'max_pixel_velocity': max(velocity_magnitudes) if velocity_magnitudes else 0.0,
            'avg_pixel_velocity': np.mean(velocity_magnitudes) if velocity_magnitudes else 0.0,
            'max_pixel_acceleration': max(acceleration_magnitudes) if acceleration_magnitudes else 0.0,
            'avg_pixel_acceleration': np.mean(acceleration_magnitudes) if acceleration_magnitudes else 0.0,
            'pixel_jerk': np.mean(jerks) if jerks else 0.0
        }
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], 
                         output_path: str,
                         test_name: str) -> None:
        """
        Save test results to file for analysis and debugging.
        
        Args:
            results: Test results dictionary
            output_path: Path to save results
            test_name: Name of the test
        """
        output_file = Path(output_path) / f"{test_name}_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    @staticmethod
    def load_test_baseline(baseline_path: str, test_name: str) -> Optional[Dict[str, Any]]:
        """
        Load baseline test results for comparison.
        
        Args:
            baseline_path: Path to baseline results
            test_name: Name of the test
            
        Returns:
            Baseline results dictionary or None if not found
        """
        baseline_file = Path(baseline_path) / f"{test_name}_baseline.json"
        
        if not baseline_file.exists():
            return None
        
        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    @staticmethod
    def create_deterministic_test_config(seed: int = 42, **overrides) -> SimulatorConfig:
        """
        Create a standard deterministic configuration for testing.
        
        Args:
            seed: Random seed for reproducible behavior
            **overrides: Configuration parameters to override
            
        Returns:
            SimulatorConfig with deterministic settings optimized for testing
        """
        base_config = {
            'deterministic_seed': seed,
            'offline_mode': True,
            'duration_s': 2.0,
            'fps': 10.0,
            'num_drones': 2,
            'vertical_fov_deg': 50.0,
            'path_span_m': 20.0,
            'speed_mps': 3.0,
            'path_altitude_agl_m': 5.0,
            'max_lateral_accel_mps2': 2.0,
            'pixel_centroid_sigma_px': 1.0,
            'bbox_size_sigma_px': 2.0,
            'confidence_noise': 0.05,
            'miss_rate_small': 0.02,
            'false_positive_rate': 0.01,
            'processing_latency_ms_mean': 50.0,
            'processing_latency_ms_jitter': 10.0
        }
        base_config.update(overrides)
        return SimulatorConfig(**base_config)


class GroundTruthValidator:
    """Validator for known ground truth scenarios."""
    
    @staticmethod
    def validate_camera_projection_accuracy(camera_model, 
                                          test_points: List[Tuple[np.ndarray, Tuple[float, float]]],
                                          tolerance: float = 10.0) -> Tuple[bool, List[str]]:
        """
        Validate camera projection against known ground truth points.
        
        Args:
            camera_model: CameraModel instance
            test_points: List of (world_position, expected_pixel_coords) tuples
            tolerance: Tolerance in pixels for projection accuracy
            
        Returns:
            Tuple of (all_accurate, list_of_errors)
        """
        errors = []
        
        for i, (world_pos, expected_pixel) in enumerate(test_points):
            result = camera_model.project_world_to_pixels(world_pos)
            
            if 'pixel_coordinates' not in result:
                errors.append(f"Test point {i}: No pixel coordinates in projection result")
                continue
            
            actual_pixel = result['pixel_coordinates']
            expected_x, expected_y = expected_pixel
            actual_x, actual_y = actual_pixel
            
            error_x = abs(actual_x - expected_x)
            error_y = abs(actual_y - expected_y)
            
            if error_x > tolerance:
                errors.append(f"Test point {i}: X projection error {error_x:.2f} > {tolerance}")
            
            if error_y > tolerance:
                errors.append(f"Test point {i}: Y projection error {error_y:.2f} > {tolerance}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_motion_trajectory_accuracy(motion_generator,
                                          expected_start_pos: np.ndarray,
                                          expected_end_pos: np.ndarray,
                                          tolerance: float = 1.0) -> Tuple[bool, List[str]]:
        """
        Validate motion trajectory against expected start and end positions.
        
        Args:
            motion_generator: MotionGenerator instance
            expected_start_pos: Expected starting position
            expected_end_pos: Expected ending position
            tolerance: Tolerance in meters for position accuracy
            
        Returns:
            Tuple of (trajectory_accurate, list_of_errors)
        """
        errors = []
        
        if not motion_generator.paths:
            errors.append("No motion paths generated")
            return False, errors
        
        path = motion_generator.paths[0]  # Check first drone
        
        if len(path) == 0:
            errors.append("Empty motion path")
            return False, errors
        
        actual_start = path[0]
        actual_end = path[-1]
        
        start_error = np.linalg.norm(actual_start - expected_start_pos)
        end_error = np.linalg.norm(actual_end - expected_end_pos)
        
        if start_error > tolerance:
            errors.append(f"Start position error {start_error:.2f}m > {tolerance}m")
        
        if end_error > tolerance:
            errors.append(f"End position error {end_error:.2f}m > {tolerance}m")
        
        return len(errors) == 0, errors