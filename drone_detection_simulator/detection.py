"""
Detection generation for drone detection simulator.

This module provides pixel detection generation that projects world positions
to pixel coordinates and generates realistic bounding boxes with confidence scoring.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import SimulatorConfig
from .camera import CameraModel


class DetectionGenerator:
    """
    Generates realistic pixel detections from world drone positions.
    
    Projects world positions to pixel coordinates and creates bounding boxes
    with confidence scores based on detection size and visibility factors.
    """
    
    def __init__(self, config: SimulatorConfig, camera_model: CameraModel):
        """
        Initialize detection generator.
        
        Args:
            config: Simulator configuration
            camera_model: Camera model for projection calculations
        """
        self.config = config
        self.camera_model = camera_model
    
    def generate_detections(self, world_positions: List[np.ndarray]) -> List[Dict]:
        """
        Generate pixel detections from world drone positions.
        
        Args:
            world_positions: List of ENU world positions for each drone
            
        Returns:
            List of detection dictionaries, one per visible drone
        """
        detections = []
        
        for drone_idx, world_pos in enumerate(world_positions):
            detection = self._generate_single_detection(world_pos, drone_idx)
            if detection is not None:
                detections.append(detection)
        
        return detections
    
    def _generate_single_detection(self, world_pos: np.ndarray, drone_idx: int) -> Optional[Dict]:
        """
        Generate a single detection from world position.
        
        Args:
            world_pos: ENU world position [east, north, up] in meters
            drone_idx: Index of the drone for identification
            
        Returns:
            Detection dictionary or None if not visible
        """
        # Project world position to pixel coordinates
        projection = self.camera_model.project_world_to_pixels(world_pos)
        
        # Skip if behind camera or too far out of bounds
        if projection['behind_camera']:
            return None
        
        # Get pixel coordinates and depth
        pixel_coords = projection['pixel_coords']
        depth = projection['depth']
        
        # Generate bounding box from projected position
        bbox_info = self._generate_bounding_box(pixel_coords, depth)
        
        # Skip if bounding box is completely outside image
        if not self._is_bbox_visible(bbox_info['bbox_px']):
            return None
        
        # Calculate confidence score based on size and visibility
        confidence = self._calculate_confidence(bbox_info, projection)
        
        # Create detection dictionary
        detection = {
            'class': 'drone',
            'confidence': confidence,
            'bbox_px': bbox_info['bbox_px'],
            'center_px': bbox_info['center_px'],
            'size_px': bbox_info['size_px'],
            'drone_id': drone_idx,
            'world_pos_enu': world_pos.tolist(),
            'depth_m': depth,
            'projection_info': {
                'in_bounds': projection['in_bounds'],
                'distance_from_edge': projection['distance_from_edge']
            }
        }
        
        return detection
    
    def _generate_bounding_box(self, pixel_coords: Tuple[float, float], depth: float) -> Dict:
        """
        Generate realistic bounding box from pixel center and depth.
        
        Args:
            pixel_coords: Center pixel coordinates (u, v)
            depth: Distance from camera in meters
            
        Returns:
            Dictionary containing bounding box information:
            - bbox_px: [x_min, y_min, x_max, y_max] bounding box
            - center_px: [u, v] center coordinates
            - size_px: [width, height] bounding box size
        """
        u, v = pixel_coords
        
        # Calculate apparent size based on drone size and distance
        # Using pinhole camera model: pixel_size = focal_px * object_size / depth
        focal_px = self.camera_model.get_focal_length_px()
        drone_size_m = self.config.drone_size_m
        
        # Calculate pixel size (assuming square drone for simplicity)
        pixel_size = focal_px * drone_size_m / depth
        
        # Add some variation to make it more realistic (slightly rectangular)
        width_px = pixel_size * 1.0  # Width factor
        height_px = pixel_size * 1.2  # Height factor (drones are often taller)
        
        # Ensure minimum size for detectability
        min_size_px = 4.0
        width_px = max(width_px, min_size_px)
        height_px = max(height_px, min_size_px)
        
        # Calculate bounding box corners
        half_width = width_px / 2.0
        half_height = height_px / 2.0
        
        x_min = u - half_width
        y_min = v - half_height
        x_max = u + half_width
        y_max = v + half_height
        
        return {
            'bbox_px': [x_min, y_min, x_max, y_max],
            'center_px': [u, v],
            'size_px': [width_px, height_px]
        }
    
    def _is_bbox_visible(self, bbox_px: List[float]) -> bool:
        """
        Check if bounding box has any visible portion in the image.
        
        Args:
            bbox_px: Bounding box [x_min, y_min, x_max, y_max]
            
        Returns:
            True if any part of the bounding box is visible
        """
        x_min, y_min, x_max, y_max = bbox_px
        
        # Check if bounding box overlaps with image bounds
        image_width = self.config.image_width_px
        image_height = self.config.image_height_px
        
        # No overlap if completely outside image bounds
        if (x_max < 0 or x_min > image_width or 
            y_max < 0 or y_min > image_height):
            return False
        
        # Has some overlap
        return True
    
    def _calculate_confidence(self, bbox_info: Dict, projection: Dict) -> float:
        """
        Calculate detection confidence based on size and visibility factors.
        
        Args:
            bbox_info: Bounding box information
            projection: Projection information from camera model
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence starts high for clear detections
        base_confidence = 0.95
        
        # Factor 1: Size-based confidence (smaller objects are less confident)
        width_px, height_px = bbox_info['size_px']
        avg_size_px = (width_px + height_px) / 2.0
        
        # Confidence decreases for very small objects
        size_factor = self._calculate_size_confidence_factor(avg_size_px)
        
        # Factor 2: Edge proximity (objects near edges are less confident)
        edge_factor = self._calculate_edge_confidence_factor(projection['distance_from_edge'])
        
        # Factor 3: Bounds factor (partially out of bounds reduces confidence)
        bounds_factor = 1.0 if projection['in_bounds'] else 0.8
        
        # Combine factors
        confidence = base_confidence * size_factor * edge_factor * bounds_factor
        
        # Clamp to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _calculate_size_confidence_factor(self, avg_size_px: float) -> float:
        """
        Calculate confidence factor based on detection size.
        
        Args:
            avg_size_px: Average size of detection in pixels
            
        Returns:
            Size confidence factor between 0.0 and 1.0
        """
        # Define size thresholds
        min_confident_size = 20.0  # Pixels - above this size, full confidence
        min_detectable_size = 4.0   # Pixels - below this size, very low confidence
        
        if avg_size_px >= min_confident_size:
            return 1.0
        elif avg_size_px <= min_detectable_size:
            return 0.3  # Still detectable but low confidence
        else:
            # Linear interpolation between thresholds
            factor = (avg_size_px - min_detectable_size) / (min_confident_size - min_detectable_size)
            return 0.3 + 0.7 * factor  # Scale from 0.3 to 1.0
    
    def _calculate_edge_confidence_factor(self, distance_from_edge: float) -> float:
        """
        Calculate confidence factor based on distance from image edge.
        
        Args:
            distance_from_edge: Distance from nearest image edge in pixels
            
        Returns:
            Edge confidence factor between 0.0 and 1.0
        """
        # Define edge proximity thresholds
        safe_distance = 50.0    # Pixels - beyond this, no edge penalty
        critical_distance = 10.0 # Pixels - below this, significant penalty
        
        if distance_from_edge >= safe_distance:
            return 1.0
        elif distance_from_edge <= 0:
            # Outside image bounds
            return 0.5
        elif distance_from_edge <= critical_distance:
            # Very close to edge
            return 0.7
        else:
            # Linear interpolation between critical and safe distances
            factor = (distance_from_edge - critical_distance) / (safe_distance - critical_distance)
            return 0.7 + 0.3 * factor  # Scale from 0.7 to 1.0
    
    def clip_detection_to_image(self, detection: Dict) -> Dict:
        """
        Clip detection bounding box to image boundaries.
        
        Args:
            detection: Detection dictionary with bbox_px
            
        Returns:
            Detection dictionary with clipped bounding box
        """
        bbox_px = detection['bbox_px']
        x_min, y_min, x_max, y_max = bbox_px
        
        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.config.image_width_px, x_max)
        y_max = min(self.config.image_height_px, y_max)
        
        # Update detection
        clipped_detection = detection.copy()
        clipped_detection['bbox_px'] = [x_min, y_min, x_max, y_max]
        
        # Recalculate center and size
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        clipped_detection['center_px'] = [center_x, center_y]
        clipped_detection['size_px'] = [width, height]
        
        return clipped_detection
    
    def estimate_distance_from_size(self, detection: Dict) -> float:
        """
        Estimate distance from detection size (reverse of projection).
        
        This simulates what a real detector might do to estimate distance
        from the apparent size of the detected object.
        
        Args:
            detection: Detection dictionary with size_px
            
        Returns:
            Estimated distance in meters
        """
        width_px, height_px = detection['size_px']
        avg_size_px = (width_px + height_px) / 2.0
        
        # Use pinhole camera model in reverse: depth = focal_px * object_size / pixel_size
        focal_px = self.camera_model.get_focal_length_px()
        drone_size_m = self.config.drone_size_m
        
        if avg_size_px > 0:
            estimated_distance = focal_px * drone_size_m / avg_size_px
        else:
            estimated_distance = float('inf')
        
        return estimated_distance
    
    def get_detection_statistics(self, detections: List[Dict]) -> Dict:
        """
        Calculate statistics about generated detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary containing detection statistics
        """
        if not detections:
            return {
                'num_detections': 0,
                'avg_confidence': 0.0,
                'avg_size_px': 0.0,
                'avg_distance_m': 0.0,
                'in_bounds_count': 0,
                'out_of_bounds_count': 0
            }
        
        confidences = [d['confidence'] for d in detections]
        sizes = [sum(d['size_px']) / 2.0 for d in detections]  # Average of width and height
        distances = [d['depth_m'] for d in detections]
        in_bounds = [d['projection_info']['in_bounds'] for d in detections]
        
        return {
            'num_detections': len(detections),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'avg_size_px': sum(sizes) / len(sizes),
            'min_size_px': min(sizes),
            'max_size_px': max(sizes),
            'avg_distance_m': sum(distances) / len(distances),
            'min_distance_m': min(distances),
            'max_distance_m': max(distances),
            'in_bounds_count': sum(in_bounds),
            'out_of_bounds_count': len(in_bounds) - sum(in_bounds)
        }