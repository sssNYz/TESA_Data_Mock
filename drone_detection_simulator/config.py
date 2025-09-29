"""
Configuration management for the drone detection simulator.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
import yaml
import json
from pathlib import Path


@dataclass
class SimulatorConfig:
    """Configuration parameters for the drone detection simulator."""
    
    # Camera intrinsics (Pi config values)
    image_width_px: int = 1920
    image_height_px: int = 1080
    focal_length_mm: Optional[float] = None
    sensor_height_mm: Optional[float] = None
    vertical_fov_deg: Optional[float] = 50.0
    principal_point_px: Optional[Tuple[float, float]] = None  # Defaults to center
    
    # Camera extrinsics & geodetics (Pi install config)
    camera_lat_deg: float = 13.736717
    camera_lon_deg: float = 100.523186
    camera_alt_m: float = 1.5
    camera_yaw_deg: float = 90.0  # Simplified: no roll for Pi mount
    camera_pitch_deg: float = 10.0
    
    # Simulation parameters (for generating realistic pixel motion)
    drone_size_m: float = 0.25  # Used internally for projection
    num_drones: int = 1
    path_altitude_agl_m: float = 5.5
    path_span_m: float = 40.0
    speed_mps: float = 5.0
    max_lateral_accel_mps2: float = 1.5
    
    # Timing
    duration_s: float = 12.0
    fps: float = 15.0
    
    # Pi-level noise (detector realism)
    pixel_centroid_sigma_px: float = 1.0
    bbox_size_sigma_px: float = 2.0
    confidence_noise: float = 0.05
    miss_rate_small: float = 0.03
    false_positive_rate: float = 0.01
    processing_latency_ms_mean: float = 50.0
    processing_latency_ms_jitter: float = 20.0
    
    # MQTT settings
    mqtt_host: str = "192.168.1.103"
    mqtt_port: int = 1883
    mqtt_topic: str = "sensors/cam01/detections"
    mqtt_qos: int = 0
    retain: bool = False
    client_id: str = ""
    
    # Testing options
    deterministic_seed: Optional[int] = None
    offline_mode: bool = False  # Print JSON instead of MQTT
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters and raise clear errors for invalid values.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Validate focal length specification (mutually exclusive)
        focal_specs = sum([
            self.focal_length_mm is not None and self.sensor_height_mm is not None,
            self.vertical_fov_deg is not None
        ])
        
        if focal_specs == 0:
            raise ValueError(
                "Must specify either (focal_length_mm AND sensor_height_mm) OR vertical_fov_deg"
            )
        elif focal_specs > 1:
            raise ValueError(
                "Cannot specify both focal length/sensor and vertical FOV - choose one method"
            )
        
        # Validate image dimensions
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("Image dimensions must be positive")
        
        # Validate focal length parameters
        if self.focal_length_mm is not None and self.focal_length_mm <= 0:
            raise ValueError("Focal length must be positive")
        
        if self.sensor_height_mm is not None and self.sensor_height_mm <= 0:
            raise ValueError("Sensor height must be positive")
        
        # Validate FOV
        if self.vertical_fov_deg is not None:
            if not (1.0 <= self.vertical_fov_deg <= 179.0):
                raise ValueError("Vertical FOV must be between 1 and 179 degrees")
        
        # Validate principal point
        if self.principal_point_px is not None:
            px, py = self.principal_point_px
            if not (0 <= px <= self.image_width_px and 0 <= py <= self.image_height_px):
                raise ValueError("Principal point must be within image bounds")
        
        # Validate geodetic coordinates
        if not (-90.0 <= self.camera_lat_deg <= 90.0):
            raise ValueError("Camera latitude must be between -90 and 90 degrees")
        
        if not (-180.0 <= self.camera_lon_deg <= 180.0):
            raise ValueError("Camera longitude must be between -180 and 180 degrees")
        
        # Validate positive distances and speeds
        if self.camera_alt_m < 0:
            raise ValueError("Camera altitude must be non-negative")
        
        if self.drone_size_m <= 0:
            raise ValueError("Drone size must be positive")
        
        if self.path_altitude_agl_m <= 0:
            raise ValueError("Path altitude must be positive")
        
        if self.path_span_m <= 0:
            raise ValueError("Path span must be positive")
        
        if self.speed_mps <= 0:
            raise ValueError("Speed must be positive")
        
        if self.max_lateral_accel_mps2 <= 0:
            raise ValueError("Maximum lateral acceleration must be positive")
        
        # Validate timing parameters
        if self.duration_s <= 0:
            raise ValueError("Duration must be positive")
        
        if self.fps <= 0:
            raise ValueError("FPS must be positive")
        
        # Validate drone count
        if self.num_drones <= 0:
            raise ValueError("Number of drones must be positive")
        
        # Validate noise parameters
        if self.pixel_centroid_sigma_px < 0:
            raise ValueError("Pixel centroid sigma must be non-negative")
        
        if self.bbox_size_sigma_px < 0:
            raise ValueError("Bounding box size sigma must be non-negative")
        
        if not (0.0 <= self.confidence_noise <= 1.0):
            raise ValueError("Confidence noise must be between 0 and 1")
        
        if not (0.0 <= self.miss_rate_small <= 1.0):
            raise ValueError("Miss rate must be between 0 and 1")
        
        if not (0.0 <= self.false_positive_rate <= 1.0):
            raise ValueError("False positive rate must be between 0 and 1")
        
        if self.processing_latency_ms_mean < 0:
            raise ValueError("Processing latency mean must be non-negative")
        
        if self.processing_latency_ms_jitter < 0:
            raise ValueError("Processing latency jitter must be non-negative")
        
        # Validate MQTT parameters
        if self.mqtt_port <= 0 or self.mqtt_port > 65535:
            raise ValueError("MQTT port must be between 1 and 65535")
        
        if not (0 <= self.mqtt_qos <= 2):
            raise ValueError("MQTT QoS must be 0, 1, or 2")
        
        if not self.mqtt_topic.strip():
            raise ValueError("MQTT topic cannot be empty")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulatorConfig':
        """
        Create SimulatorConfig from dictionary with proper error handling.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            SimulatorConfig instance
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If parameter types are incorrect
        """
        try:
            # Handle principal_point_px conversion from list to tuple
            if 'principal_point_px' in config_dict and config_dict['principal_point_px'] is not None:
                pp = config_dict['principal_point_px']
                if isinstance(pp, (list, tuple)) and len(pp) == 2:
                    config_dict['principal_point_px'] = tuple(pp)
                else:
                    raise ValueError("principal_point_px must be a 2-element list or tuple")
            
            return cls(**config_dict)
        except TypeError as e:
            raise TypeError(f"Invalid parameter type in configuration: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create configuration: {e}")
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'SimulatorConfig':
        """
        Load configuration from YAML file with proper error handling.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            SimulatorConfig instance
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration is invalid
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                config_dict = {}
            
            return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {yaml_path}: {e}")
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'SimulatorConfig':
        """
        Load configuration from JSON file with proper error handling.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            SimulatorConfig instance
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
            ValueError: If configuration is invalid
        """
        json_path = Path(json_path)
        
        if not json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to parse JSON file {json_path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {json_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path where to save YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path where to save JSON configuration
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)