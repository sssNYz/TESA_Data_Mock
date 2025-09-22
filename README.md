# Drone Detection Simulator

A Python-based drone detection simulator that generates realistic camera-based drone detections using proper geometric calculations and publishes them via MQTT.

## Project Structure

```
drone-detection-simulator/
├── drone_detection_simulator/     # Main package
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration management
│   └── simulator.py              # Main simulator class (placeholder)
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_config.py           # Configuration tests
├── examples/                     # Example configurations and scripts
│   ├── config_example.yaml      # Example YAML configuration
│   └── test_config_loading.py   # Configuration demo script
├── requirements.txt              # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Configuration

The simulator uses a comprehensive configuration system that supports both YAML and JSON formats. Configuration parameters include:

### Camera Parameters
- Image dimensions and focal length (via FOV or focal length/sensor size)
- Camera position (latitude, longitude, altitude)
- Camera orientation (yaw, pitch)

### Simulation Parameters
- Drone properties (size, count, flight path)
- Timing (duration, frame rate)
- Noise and realism settings

### MQTT Settings
- Broker connection details
- Topic and QoS settings

### Example Usage

```python
from drone_detection_simulator.config import SimulatorConfig

# Load from YAML file
config = SimulatorConfig.from_yaml("config.yaml")

# Create with custom parameters
config = SimulatorConfig(
    image_width_px=1280,
    image_height_px=720,
    vertical_fov_deg=45.0,
    num_drones=3
)

# Validate configuration
config.validate()  # Raises ValueError if invalid
```

## Testing

Run the test suite:
```bash
python3 -m pytest tests/ -v
```

Run the configuration demo:
```bash
python3 examples/test_config_loading.py
```

## Configuration Validation

The configuration system includes comprehensive validation:
- Mutually exclusive focal length specification methods
- Range validation for angles, coordinates, and physical parameters
- Type checking and error handling
- Clear error messages for invalid configurations

## Requirements Satisfied

This implementation satisfies the following requirements:
- **6.1**: Configurable camera intrinsics via focal length/sensor or FOV
- **6.4**: Parameter validation with clear error messages for invalid values

## Next Steps

This is the foundation for the drone detection simulator. Subsequent tasks will implement:
- Camera projection and coordinate transformations
- Drone motion generation
- Detection simulation with noise
- MQTT publishing
- Complete simulation orchestration# TESA_Data_Mock
