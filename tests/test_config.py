"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from drone_detection_simulator.config import SimulatorConfig


class TestSimulatorConfig:
    """Test cases for SimulatorConfig class."""
    
    def test_default_config_is_valid(self):
        """Test that default configuration is valid."""
        config = SimulatorConfig()
        # Should not raise any exceptions
        assert config.image_width_px == 1920
        assert config.image_height_px == 1080
        assert config.vertical_fov_deg == 50.0
    
    def test_focal_length_validation_both_methods(self):
        """Test that specifying both focal length methods raises error."""
        with pytest.raises(ValueError, match="Cannot specify both focal length"):
            SimulatorConfig(
                focal_length_mm=25.0,
                sensor_height_mm=5.0,
                vertical_fov_deg=50.0
            )
    
    def test_focal_length_validation_no_method(self):
        """Test that specifying no focal length method raises error."""
        with pytest.raises(ValueError, match="Must specify either"):
            SimulatorConfig(vertical_fov_deg=None)
    
    def test_focal_length_method_valid(self):
        """Test valid focal length/sensor specification."""
        config = SimulatorConfig(
            focal_length_mm=25.0,
            sensor_height_mm=5.0,
            vertical_fov_deg=None
        )
        assert config.focal_length_mm == 25.0
        assert config.sensor_height_mm == 5.0
    
    def test_fov_method_valid(self):
        """Test valid FOV specification."""
        config = SimulatorConfig(vertical_fov_deg=60.0)
        assert config.vertical_fov_deg == 60.0
    
    def test_image_dimensions_validation(self):
        """Test image dimension validation."""
        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            SimulatorConfig(image_width_px=0)
        
        with pytest.raises(ValueError, match="Image dimensions must be positive"):
            SimulatorConfig(image_height_px=-100)
    
    def test_focal_length_positive_validation(self):
        """Test focal length must be positive."""
        with pytest.raises(ValueError, match="Focal length must be positive"):
            SimulatorConfig(
                focal_length_mm=-5.0,
                sensor_height_mm=5.0,
                vertical_fov_deg=None
            )
    
    def test_sensor_height_positive_validation(self):
        """Test sensor height must be positive."""
        with pytest.raises(ValueError, match="Sensor height must be positive"):
            SimulatorConfig(
                focal_length_mm=25.0,
                sensor_height_mm=0.0,
                vertical_fov_deg=None
            )
    
    def test_fov_range_validation(self):
        """Test FOV range validation."""
        with pytest.raises(ValueError, match="Vertical FOV must be between 1 and 179"):
            SimulatorConfig(vertical_fov_deg=0.5)
        
        with pytest.raises(ValueError, match="Vertical FOV must be between 1 and 179"):
            SimulatorConfig(vertical_fov_deg=180.0)
    
    def test_principal_point_validation(self):
        """Test principal point bounds validation."""
        with pytest.raises(ValueError, match="Principal point must be within image bounds"):
            SimulatorConfig(principal_point_px=(2000, 540))  # Outside width
        
        with pytest.raises(ValueError, match="Principal point must be within image bounds"):
            SimulatorConfig(principal_point_px=(960, 1200))  # Outside height
    
    def test_latitude_validation(self):
        """Test latitude range validation."""
        with pytest.raises(ValueError, match="Camera latitude must be between -90 and 90"):
            SimulatorConfig(camera_lat_deg=95.0)
        
        with pytest.raises(ValueError, match="Camera latitude must be between -90 and 90"):
            SimulatorConfig(camera_lat_deg=-100.0)
    
    def test_longitude_validation(self):
        """Test longitude range validation."""
        with pytest.raises(ValueError, match="Camera longitude must be between -180 and 180"):
            SimulatorConfig(camera_lon_deg=185.0)
        
        with pytest.raises(ValueError, match="Camera longitude must be between -180 and 180"):
            SimulatorConfig(camera_lon_deg=-200.0)
    
    def test_positive_distance_validation(self):
        """Test positive distance parameters."""
        with pytest.raises(ValueError, match="Camera altitude must be non-negative"):
            SimulatorConfig(camera_alt_m=-1.0)
        
        with pytest.raises(ValueError, match="Drone size must be positive"):
            SimulatorConfig(drone_size_m=0.0)
        
        with pytest.raises(ValueError, match="Path altitude must be positive"):
            SimulatorConfig(path_altitude_agl_m=-5.0)
        
        with pytest.raises(ValueError, match="Path span must be positive"):
            SimulatorConfig(path_span_m=0.0)
        
        with pytest.raises(ValueError, match="Speed must be positive"):
            SimulatorConfig(speed_mps=-2.0)
        
        with pytest.raises(ValueError, match="Maximum lateral acceleration must be positive"):
            SimulatorConfig(max_lateral_accel_mps2=0.0)
    
    def test_timing_validation(self):
        """Test timing parameter validation."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            SimulatorConfig(duration_s=0.0)
        
        with pytest.raises(ValueError, match="FPS must be positive"):
            SimulatorConfig(fps=-10.0)
    
    def test_drone_count_validation(self):
        """Test drone count validation."""
        with pytest.raises(ValueError, match="Number of drones must be positive"):
            SimulatorConfig(num_drones=0)
    
    def test_noise_parameter_validation(self):
        """Test noise parameter validation."""
        with pytest.raises(ValueError, match="Pixel centroid sigma must be non-negative"):
            SimulatorConfig(pixel_centroid_sigma_px=-1.0)
        
        with pytest.raises(ValueError, match="Bounding box size sigma must be non-negative"):
            SimulatorConfig(bbox_size_sigma_px=-2.0)
        
        with pytest.raises(ValueError, match="Confidence noise must be between 0 and 1"):
            SimulatorConfig(confidence_noise=1.5)
        
        with pytest.raises(ValueError, match="Miss rate must be between 0 and 1"):
            SimulatorConfig(miss_rate_small=-0.1)
        
        with pytest.raises(ValueError, match="False positive rate must be between 0 and 1"):
            SimulatorConfig(false_positive_rate=2.0)
        
        with pytest.raises(ValueError, match="Processing latency mean must be non-negative"):
            SimulatorConfig(processing_latency_ms_mean=-10.0)
        
        with pytest.raises(ValueError, match="Processing latency jitter must be non-negative"):
            SimulatorConfig(processing_latency_ms_jitter=-5.0)
    
    def test_mqtt_validation(self):
        """Test MQTT parameter validation."""
        with pytest.raises(ValueError, match="MQTT port must be between 1 and 65535"):
            SimulatorConfig(mqtt_port=0)
        
        with pytest.raises(ValueError, match="MQTT port must be between 1 and 65535"):
            SimulatorConfig(mqtt_port=70000)
        
        with pytest.raises(ValueError, match="MQTT QoS must be 0, 1, or 2"):
            SimulatorConfig(mqtt_qos=3)
        
        with pytest.raises(ValueError, match="MQTT topic cannot be empty"):
            SimulatorConfig(mqtt_topic="")
        
        with pytest.raises(ValueError, match="MQTT topic cannot be empty"):
            SimulatorConfig(mqtt_topic="   ")
    
    def test_from_dict_valid(self):
        """Test creating config from valid dictionary."""
        config_dict = {
            "image_width_px": 1280,
            "image_height_px": 720,
            "vertical_fov_deg": 45.0,
            "camera_lat_deg": 40.0,
            "camera_lon_deg": -74.0
        }
        
        config = SimulatorConfig.from_dict(config_dict)
        assert config.image_width_px == 1280
        assert config.image_height_px == 720
        assert config.vertical_fov_deg == 45.0
        assert config.camera_lat_deg == 40.0
        assert config.camera_lon_deg == -74.0
    
    def test_from_dict_principal_point_conversion(self):
        """Test principal point conversion from list to tuple."""
        config_dict = {
            "principal_point_px": [640, 360]
        }
        
        config = SimulatorConfig.from_dict(config_dict)
        assert config.principal_point_px == (640, 360)
    
    def test_from_dict_invalid_principal_point(self):
        """Test invalid principal point format."""
        config_dict = {
            "principal_point_px": [640]  # Only one element
        }
        
        with pytest.raises(ValueError, match="principal_point_px must be a 2-element"):
            SimulatorConfig.from_dict(config_dict)
    
    def test_from_dict_type_error(self):
        """Test type error handling in from_dict."""
        config_dict = {
            "image_width_px": "not_an_int"
        }
        
        with pytest.raises(TypeError, match="Invalid parameter type"):
            SimulatorConfig.from_dict(config_dict)
    
    def test_yaml_loading(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "image_width_px": 1280,
            "image_height_px": 720,
            "vertical_fov_deg": 45.0,
            "camera_lat_deg": 40.0,
            "camera_lon_deg": -74.0,
            "principal_point_px": [640, 360]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            config = SimulatorConfig.from_yaml(yaml_path)
            assert config.image_width_px == 1280
            assert config.image_height_px == 720
            assert config.principal_point_px == (640, 360)
        finally:
            Path(yaml_path).unlink()
    
    def test_yaml_file_not_found(self):
        """Test YAML file not found error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            SimulatorConfig.from_yaml("nonexistent.yaml")
    
    def test_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            yaml_path = f.name
        
        try:
            config = SimulatorConfig.from_yaml(yaml_path)
            # Should use defaults
            assert config.image_width_px == 1920
        finally:
            Path(yaml_path).unlink()
    
    def test_json_loading(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "image_width_px": 1280,
            "image_height_px": 720,
            "vertical_fov_deg": 45.0,
            "camera_lat_deg": 40.0,
            "camera_lon_deg": -74.0,
            "principal_point_px": [640, 360]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            json_path = f.name
        
        try:
            config = SimulatorConfig.from_json(json_path)
            assert config.image_width_px == 1280
            assert config.image_height_px == 720
            assert config.principal_point_px == (640, 360)
        finally:
            Path(json_path).unlink()
    
    def test_json_file_not_found(self):
        """Test JSON file not found error."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            SimulatorConfig.from_json("nonexistent.json")
    
    def test_to_dict_conversion(self):
        """Test converting config to dictionary."""
        config = SimulatorConfig(
            image_width_px=1280,
            principal_point_px=(640, 360)
        )
        
        config_dict = config.to_dict()
        assert config_dict["image_width_px"] == 1280
        assert config_dict["principal_point_px"] == [640, 360]  # Tuple converted to list
    
    def test_yaml_saving(self):
        """Test saving configuration to YAML file."""
        config = SimulatorConfig(
            image_width_px=1280,
            image_height_px=720,
            principal_point_px=(640, 360)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            config.to_yaml(yaml_path)
            
            # Load back and verify
            loaded_config = SimulatorConfig.from_yaml(yaml_path)
            assert loaded_config.image_width_px == 1280
            assert loaded_config.image_height_px == 720
            assert loaded_config.principal_point_px == (640, 360)
        finally:
            Path(yaml_path).unlink()
    
    def test_json_saving(self):
        """Test saving configuration to JSON file."""
        config = SimulatorConfig(
            image_width_px=1280,
            image_height_px=720,
            principal_point_px=(640, 360)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            config.to_json(json_path)
            
            # Load back and verify
            loaded_config = SimulatorConfig.from_json(json_path)
            assert loaded_config.image_width_px == 1280
            assert loaded_config.image_height_px == 720
            assert loaded_config.principal_point_px == (640, 360)
        finally:
            Path(json_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])