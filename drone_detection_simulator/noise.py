"""
Noise model implementation for realistic detector behavior.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import SimulatorConfig


class NoiseModel:
    """
    Noise model that applies realistic detection noise and implements detector behavior.
    
    This class simulates the imperfections of real drone detection systems by:
    - Adding noise to pixel coordinates and bounding box dimensions
    - Implementing miss detection probability based on object size
    - Generating false positive detections
    - Varying confidence scores realistically
    """
    
    def __init__(self, config: SimulatorConfig, rng: np.random.Generator):
        """
        Initialize the noise model.
        
        Args:
            config: Simulator configuration containing noise parameters
            rng: Random number generator for deterministic behavior
        """
        self.config = config
        self.rng = rng
    
    def apply_detection_noise(self, clean_detection: Dict) -> Dict:
        """
        Apply realistic noise to a clean detection.
        
        Adds noise to:
        - Pixel coordinates (center_px)
        - Bounding box dimensions (size_px and bbox_px)
        - Confidence score
        
        Args:
            clean_detection: Clean detection dictionary with keys:
                - center_px: [x, y] center coordinates in pixels
                - size_px: [width, height] bounding box size in pixels
                - bbox_px: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score (0-1)
                - class: Detection class (e.g., "drone")
        
        Returns:
            Noisy detection dictionary with same structure but modified values
        """
        noisy_detection = clean_detection.copy()
        
        # Apply pixel coordinate noise to center
        if 'center_px' in clean_detection:
            center_x, center_y = clean_detection['center_px']
            
            # Add Gaussian noise to center coordinates
            noise_x = self.rng.normal(0, self.config.pixel_centroid_sigma_px)
            noise_y = self.rng.normal(0, self.config.pixel_centroid_sigma_px)
            
            noisy_center_x = center_x + noise_x
            noisy_center_y = center_y + noise_y
            
            # Clamp to image boundaries
            noisy_center_x = np.clip(noisy_center_x, 0, self.config.image_width_px - 1)
            noisy_center_y = np.clip(noisy_center_y, 0, self.config.image_height_px - 1)
            
            noisy_detection['center_px'] = [noisy_center_x, noisy_center_y]
        
        # Apply bounding box size noise
        if 'size_px' in clean_detection:
            width, height = clean_detection['size_px']
            
            # Add Gaussian noise to size (ensure positive values)
            width_noise = self.rng.normal(0, self.config.bbox_size_sigma_px)
            height_noise = self.rng.normal(0, self.config.bbox_size_sigma_px)
            
            noisy_width = max(1.0, width + width_noise)  # Minimum size of 1 pixel
            noisy_height = max(1.0, height + height_noise)
            
            noisy_detection['size_px'] = [noisy_width, noisy_height]
            
            # Update bounding box coordinates based on noisy center and size
            if 'center_px' in noisy_detection:
                center_x, center_y = noisy_detection['center_px']
                half_width = noisy_width / 2.0
                half_height = noisy_height / 2.0
                
                x1 = center_x - half_width
                y1 = center_y - half_height
                x2 = center_x + half_width
                y2 = center_y + half_height
                
                # Clamp bounding box to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(self.config.image_width_px - 1, x2)
                y2 = min(self.config.image_height_px - 1, y2)
                
                noisy_detection['bbox_px'] = [x1, y1, x2, y2]
        
        # Apply confidence noise
        if 'confidence' in clean_detection:
            base_confidence = clean_detection['confidence']
            
            # Add Gaussian noise to confidence
            confidence_noise = self.rng.normal(0, self.config.confidence_noise)
            noisy_confidence = base_confidence + confidence_noise
            
            # Clamp to valid range [0, 1]
            noisy_confidence = np.clip(noisy_confidence, 0.0, 1.0)
            
            noisy_detection['confidence'] = noisy_confidence
        
        return noisy_detection
    
    def should_miss_detection(self, detection: Dict) -> bool:
        """
        Determine if a detection should be missed based on object size.
        
        Implements size-based miss detection probability where smaller objects
        are more likely to be missed by the detector.
        
        Args:
            detection: Detection dictionary containing size_px information
        
        Returns:
            True if detection should be missed, False otherwise
        """
        if 'size_px' not in detection:
            return False
        
        width, height = detection['size_px']
        object_area = width * height
        
        # Define size thresholds for miss rate calculation
        # Small objects (< 400 pixels²) have higher miss rate
        small_object_threshold = 400.0  # 20x20 pixels
        
        if object_area < small_object_threshold:
            # Apply configured miss rate for small objects
            miss_probability = self.config.miss_rate_small
        else:
            # Larger objects have much lower miss rate
            miss_probability = self.config.miss_rate_small * 0.1
        
        # Additional size-based scaling: very small objects are even more likely to be missed
        if object_area < 100.0:  # 10x10 pixels
            miss_probability = min(1.0, miss_probability * 2.0)
        
        return self.rng.random() < miss_probability
    
    def generate_false_positive(self) -> Optional[Dict]:
        """
        Generate a false positive detection if probability triggers.
        
        Creates false positive detections with:
        - Random locations within the image
        - Random but realistic bounding box sizes
        - Low confidence scores
        - "false_drone" class label
        
        Returns:
            False positive detection dictionary or None if no false positive generated
        """
        if self.rng.random() >= self.config.false_positive_rate:
            return None
        
        # Generate random center location
        center_x = self.rng.uniform(0, self.config.image_width_px - 1)
        center_y = self.rng.uniform(0, self.config.image_height_px - 1)
        
        # Generate random but realistic size (typically smaller than real drones)
        # False positives tend to be noise or small objects
        min_size = 5.0
        max_size = 30.0
        width = self.rng.uniform(min_size, max_size)
        height = self.rng.uniform(min_size, max_size)
        
        # Calculate bounding box coordinates
        half_width = width / 2.0
        half_height = height / 2.0
        
        x1 = max(0, center_x - half_width)
        y1 = max(0, center_y - half_height)
        x2 = min(self.config.image_width_px - 1, center_x + half_width)
        y2 = min(self.config.image_height_px - 1, center_y + half_height)
        
        # False positives have low confidence scores
        # Typically between 0.1 and 0.4 to distinguish from real detections
        confidence = self.rng.uniform(0.1, 0.4)
        
        return {
            'class': 'false_drone',
            'confidence': confidence,
            'bbox_px': [x1, y1, x2, y2],
            'center_px': [center_x, center_y],
            'size_px': [width, height]
        }
    
    def apply_confidence_variation(self, base_confidence: float, object_size_px: Tuple[float, float]) -> float:
        """
        Apply size-based confidence variation to a detection.
        
        Larger, more visible objects get higher confidence scores,
        while smaller objects get lower confidence scores.
        
        Args:
            base_confidence: Base confidence score (0-1)
            object_size_px: Object size as (width, height) in pixels
        
        Returns:
            Modified confidence score with size-based variation
        """
        width, height = object_size_px
        object_area = width * height
        
        # Size-based confidence scaling
        # Objects larger than 1000 pixels² (e.g., 32x32) get confidence boost
        # Objects smaller than 100 pixels² (e.g., 10x10) get confidence penalty
        large_threshold = 1000.0
        small_threshold = 100.0
        
        if object_area > large_threshold:
            # Boost confidence for large objects (up to +0.1)
            size_factor = min(0.1, (object_area - large_threshold) / 5000.0)
            modified_confidence = base_confidence + size_factor
        elif object_area < small_threshold:
            # Reduce confidence for small objects (up to -0.2)
            size_factor = min(0.2, (small_threshold - object_area) / small_threshold * 0.2)
            modified_confidence = base_confidence - size_factor
        else:
            # Medium-sized objects keep base confidence
            modified_confidence = base_confidence
        
        # Apply additional random variation
        noise = self.rng.normal(0, self.config.confidence_noise)
        modified_confidence += noise
        
        # Clamp to valid range
        return np.clip(modified_confidence, 0.0, 1.0)
    
    def get_noise_statistics(self) -> Dict:
        """
        Get current noise model statistics for monitoring and testing.
        
        Returns:
            Dictionary containing noise model parameters and statistics
        """
        return {
            'pixel_centroid_sigma_px': self.config.pixel_centroid_sigma_px,
            'bbox_size_sigma_px': self.config.bbox_size_sigma_px,
            'confidence_noise': self.config.confidence_noise,
            'miss_rate_small': self.config.miss_rate_small,
            'false_positive_rate': self.config.false_positive_rate,
            'image_dimensions': [self.config.image_width_px, self.config.image_height_px]
        }