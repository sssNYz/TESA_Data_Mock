"""
JSON message builder for drone detection simulator.

This module provides functionality to build detection messages in the Pi-realistic
JSON schema format, including camera metadata, detection data, and edge metadata
with processing latency simulation.
"""

import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import numpy as np

from .config import SimulatorConfig
from .camera import CameraModel


class DetectionMessageBuilder:
    """
    Builds JSON detection messages matching Pi-realistic schema.
    
    Creates standardized detection messages that include timestamp, camera metadata,
    detection data, and edge processing metadata with simulated latency.
    """
    
    def __init__(self, config: SimulatorConfig, camera_model: CameraModel, rng: Optional[np.random.Generator] = None):
        """
        Initialize the message builder.
        
        Args:
            config: Simulator configuration
            camera_model: Camera model for metadata generation
            rng: Random number generator for latency simulation (optional)
        """
        self.config = config
        self.camera_model = camera_model
        self.rng = rng if rng is not None else np.random.default_rng()
        self._frame_counter = 0
        
        # Cache camera metadata since it's fixed
        self._camera_metadata = self.camera_model.get_camera_metadata()
    
    def build_detection_message(self, detections: List[Dict], timestamp_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Build a complete detection message in Pi-realistic JSON format.
        
        Args:
            detections: List of detection dictionaries from DetectionGenerator
            timestamp_utc: UTC timestamp for the detection (defaults to current time)
            
        Returns:
            Complete detection message dictionary ready for JSON serialization
        """
        if timestamp_utc is None:
            timestamp_utc = datetime.now(timezone.utc)
        
        # Increment frame counter
        self._frame_counter += 1
        
        # Build the message structure
        message = {
            'timestamp_utc': self._format_timestamp(timestamp_utc),
            'frame_id': self._frame_counter,
            'camera': self._build_camera_metadata(),
            'detections': self._build_detections_array(detections),
            'edge': self._build_edge_metadata()
        }
        
        return message
    
    def _format_timestamp(self, timestamp_utc: datetime) -> str:
        """
        Format timestamp in ISO 8601 format with milliseconds.
        
        Args:
            timestamp_utc: UTC datetime object
            
        Returns:
            ISO 8601 formatted timestamp string
        """
        # Format with milliseconds precision
        return timestamp_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    def _build_camera_metadata(self) -> Dict[str, Any]:
        """
        Build camera metadata section for the JSON message.
        
        Returns fixed camera configuration values that would be included
        in detection messages from a Pi-based camera system.
        
        Returns:
            Camera metadata dictionary
        """
        return self._camera_metadata.copy()
    
    def _build_detections_array(self, detections: List[Dict]) -> List[Dict[str, Any]]:
        """
        Build detections array from detection dictionaries.
        
        Converts internal detection format to Pi-realistic JSON schema format,
        focusing on pixel-space detections without exposing internal simulation data.
        
        Args:
            detections: List of detection dictionaries from DetectionGenerator
            
        Returns:
            List of detection dictionaries in JSON schema format
        """
        json_detections = []
        
        for detection in detections:
            # Convert to Pi-realistic format (pixel detections only)
            json_detection = {
                'class': detection['class'],
                'confidence': round(detection['confidence'], 3),
                'bbox_px': [round(coord, 1) for coord in detection['bbox_px']],
                'center_px': [round(coord, 1) for coord in detection['center_px']],
                'size_px': [round(size, 1) for size in detection['size_px']]
            }
            
            json_detections.append(json_detection)
        
        return json_detections
    
    def _build_edge_metadata(self) -> Dict[str, Any]:
        """
        Build edge metadata section with processing latency simulation.
        
        Simulates processing latency that would be present in a real Pi-based
        detection system, including configurable jitter.
        
        Returns:
            Edge metadata dictionary
        """
        # Simulate processing latency with jitter
        latency_ms = self._simulate_processing_latency()
        
        return {
            'processing_latency_ms': round(latency_ms, 1),
            'detector_version': 'det-v1.2'
        }
    
    def _simulate_processing_latency(self) -> float:
        """
        Simulate realistic processing latency with jitter.
        
        Uses configured mean latency and jitter values to generate
        realistic processing times that would occur on Pi hardware.
        
        Returns:
            Processing latency in milliseconds
        """
        mean_latency = self.config.processing_latency_ms_mean
        jitter = self.config.processing_latency_ms_jitter
        
        # Generate latency with normal distribution
        if jitter > 0:
            latency = self.rng.normal(mean_latency, jitter)
            # Ensure latency is non-negative
            latency = max(0.0, latency)
        else:
            latency = mean_latency
        
        return latency
    
    def build_false_positive_detection(self) -> Dict[str, Any]:
        """
        Build a false positive detection for testing robustness.
        
        Creates a detection with random location and low confidence
        that doesn't follow realistic physics-based motion patterns.
        
        Returns:
            False positive detection dictionary
        """
        # Random location within image bounds
        center_x = self.rng.uniform(0, self.config.image_width_px)
        center_y = self.rng.uniform(0, self.config.image_height_px)
        
        # Random size (typically smaller than real drones)
        width = self.rng.uniform(8, 25)
        height = self.rng.uniform(8, 25)
        
        # Calculate bounding box
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        
        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.config.image_width_px, x_max)
        y_max = min(self.config.image_height_px, y_max)
        
        # Low confidence for false positives
        confidence = self.rng.uniform(0.1, 0.4)
        
        return {
            'class': 'false_drone',
            'confidence': round(confidence, 3),
            'bbox_px': [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)],
            'center_px': [round(center_x, 1), round(center_y, 1)],
            'size_px': [round(x_max - x_min, 1), round(y_max - y_min, 1)]
        }
    
    def add_false_positives(self, detections: List[Dict]) -> List[Dict]:
        """
        Add false positive detections based on configured probability.
        
        Args:
            detections: Existing list of real detections
            
        Returns:
            Updated list of detections including false positives
        """
        # Check if we should add a false positive
        if self.rng.random() < self.config.false_positive_rate:
            false_positive = self.build_false_positive_detection()
            # Convert to internal format for consistency
            false_positive_internal = {
                'class': false_positive['class'],
                'confidence': false_positive['confidence'],
                'bbox_px': false_positive['bbox_px'],
                'center_px': false_positive['center_px'],
                'size_px': false_positive['size_px'],
                'drone_id': -1,  # Special ID for false positives
                'world_pos_enu': [0, 0, 0],  # No real world position
                'depth_m': 0,  # No real depth
                'projection_info': {
                    'in_bounds': True,
                    'distance_from_edge': 50.0
                }
            }
            detections.append(false_positive_internal)
        
        return detections
    
    def to_json_string(self, message: Dict[str, Any], indent: Optional[int] = None) -> str:
        """
        Convert message dictionary to JSON string.
        
        Args:
            message: Message dictionary to serialize
            indent: JSON indentation (None for compact, int for pretty-printed)
            
        Returns:
            JSON string representation of the message
        """
        return json.dumps(message, indent=indent, ensure_ascii=False)
    
    def validate_message_schema(self, message: Dict[str, Any]) -> bool:
        """
        Validate that message conforms to expected Pi-realistic schema.
        
        Args:
            message: Message dictionary to validate
            
        Returns:
            True if message is valid, False otherwise
        """
        try:
            # Check required top-level fields
            required_fields = ['timestamp_utc', 'frame_id', 'camera', 'detections', 'edge']
            for field in required_fields:
                if field not in message:
                    return False
            
            # Validate camera metadata
            camera = message['camera']
            camera_required = ['resolution', 'focal_px', 'principal_point', 'yaw_deg', 
                             'pitch_deg', 'lat_deg', 'lon_deg', 'alt_m_msl']
            for field in camera_required:
                if field not in camera:
                    return False
            
            # Validate detections array
            detections = message['detections']
            if not isinstance(detections, list):
                return False
            
            for detection in detections:
                detection_required = ['class', 'confidence', 'bbox_px', 'center_px', 'size_px']
                for field in detection_required:
                    if field not in detection:
                        return False
                
                # Validate bbox format
                bbox = detection['bbox_px']
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    return False
                
                # Validate center format
                center = detection['center_px']
                if not (isinstance(center, list) and len(center) == 2):
                    return False
                
                # Validate size format
                size = detection['size_px']
                if not (isinstance(size, list) and len(size) == 2):
                    return False
            
            # Validate edge metadata
            edge = message['edge']
            edge_required = ['processing_latency_ms', 'detector_version']
            for field in edge_required:
                if field not in edge:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_frame_id(self) -> int:
        """
        Get current frame ID counter.
        
        Returns:
            Current frame ID
        """
        return self._frame_counter
    
    def reset_frame_counter(self) -> None:
        """Reset frame counter to 0."""
        self._frame_counter = 0
    
    def get_message_size_bytes(self, message: Dict[str, Any]) -> int:
        """
        Calculate approximate message size in bytes.
        
        Args:
            message: Message dictionary
            
        Returns:
            Approximate size in bytes when serialized to JSON
        """
        json_str = self.to_json_string(message)
        return len(json_str.encode('utf-8'))