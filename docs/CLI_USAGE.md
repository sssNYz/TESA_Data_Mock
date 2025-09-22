# Command-Line Interface Usage Guide

The Drone Detection Simulator provides a comprehensive command-line interface for running simulations with various configurations and scenarios.

## Installation and Basic Usage

### Running the Simulator

The simulator can be run in several ways:

```bash
# Run as a Python module (recommended)
python -m drone_detection_simulator

# Run the simulator script directly
python drone_detection_simulator/simulator.py

# Run with the CLI module
python drone_detection_simulator/cli.py
```

### Basic Examples

```bash
# Run with default configuration
python -m drone_detection_simulator

# Run in offline mode (print JSON instead of MQTT)
python -m drone_detection_simulator --offline

# Run with custom duration and frame rate
python -m drone_detection_simulator --duration 30 --fps 10

# Run with multiple drones
python -m drone_detection_simulator --num-drones 3 --duration 20
```

## Configuration Files

### Loading Configuration Files

```bash
# Load YAML configuration
python -m drone_detection_simulator --config examples/multi_drone.yaml

# Load JSON configuration
python -m drone_detection_simulator --config examples/minimal_config.json

# Override specific parameters from config file
python -m drone_detection_simulator --config examples/multi_drone.yaml --duration 30 --offline
```

### Available Example Configurations

Use `--list-examples` to see all available example configurations:

```bash
python -m drone_detection_simulator --list-examples
```

#### Example Configurations

1. **config_example.yaml** - Basic example with detailed comments
2. **multi_drone.yaml** - Multiple drones with overlapping flight paths
3. **high_performance.yaml** - Optimized for high frame rate and minimal noise
4. **noisy_environment.yaml** - Challenging conditions with high noise
5. **testing_deterministic.yaml** - Reproducible testing configuration
6. **long_duration.yaml** - Extended simulation for endurance testing
7. **minimal_config.json** - Minimal JSON configuration example

## Command-Line Parameters

### Basic Simulation Parameters

```bash
--duration SECONDS, -d SECONDS    # Simulation duration (default: 12.0)
--fps FPS, -f FPS                 # Frames per second (default: 15.0)
--num-drones COUNT, -n COUNT      # Number of drones (default: 1)
```

### Camera Parameters

```bash
--camera-lat DEGREES              # Camera latitude
--camera-lon DEGREES              # Camera longitude  
--camera-alt METERS               # Camera altitude
--camera-yaw DEGREES              # Camera yaw (heading from north)
--camera-pitch DEGREES            # Camera pitch (tilt angle)
--vertical-fov DEGREES            # Vertical field of view
```

### Drone Motion Parameters

```bash
--altitude METERS                 # Drone flight altitude AGL
--speed MPS                       # Drone speed in m/s
--span METERS                     # Flight path span
```

### MQTT Parameters

```bash
--mqtt-host HOST                  # MQTT broker hostname
--mqtt-port PORT                  # MQTT broker port
--mqtt-topic TOPIC                # MQTT topic for publishing
--mqtt-qos {0,1,2}               # MQTT Quality of Service level
```

### Noise and Realism Parameters

```bash
--pixel-noise SIGMA               # Pixel coordinate noise std dev
--bbox-noise SIGMA                # Bounding box size noise std dev
--miss-rate RATE                  # Detection miss rate (0.0-1.0)
--false-positive-rate RATE        # False positive rate (0.0-1.0)
```

### Testing and Debugging Options

```bash
--offline                         # Print JSON instead of MQTT
--seed SEED                       # Random seed for deterministic behavior
--verbose, -v                     # Enable verbose logging
--quiet, -q                       # Suppress output except errors
```

### Utility Commands

```bash
--validate-config                 # Validate configuration and exit
--print-config                    # Print effective configuration and exit
--list-examples                   # List available example configurations
--version                         # Show version information
```

## Usage Scenarios

### Development and Testing

```bash
# Quick test with deterministic output
python -m drone_detection_simulator --seed 42 --offline --duration 5

# Validate a configuration file
python -m drone_detection_simulator --config my_config.yaml --validate-config

# Print effective configuration after overrides
python -m drone_detection_simulator --config base.yaml --duration 30 --print-config
```

### Performance Testing

```bash
# High-performance scenario
python -m drone_detection_simulator --config examples/high_performance.yaml

# Custom high-FPS test
python -m drone_detection_simulator --fps 30 --duration 10 --pixel-noise 0.5
```

### Multi-Drone Scenarios

```bash
# Multiple drones with custom parameters
python -m drone_detection_simulator --num-drones 4 --duration 25 --span 60

# Load multi-drone configuration
python -m drone_detection_simulator --config examples/multi_drone.yaml
```

### Challenging Conditions

```bash
# Noisy environment testing
python -m drone_detection_simulator --config examples/noisy_environment.yaml

# Custom noisy conditions
python -m drone_detection_simulator --pixel-noise 3.0 --miss-rate 0.1 --false-positive-rate 0.05
```

### Long-Duration Testing

```bash
# Extended simulation
python -m drone_detection_simulator --config examples/long_duration.yaml

# Custom long-duration test
python -m drone_detection_simulator --duration 300 --fps 5 --num-drones 2
```

## Configuration File Format

### YAML Format

```yaml
# Camera intrinsics
image_width_px: 1920
image_height_px: 1080
vertical_fov_deg: 50.0
principal_point_px: [960, 540]

# Camera position
camera_lat_deg: 13.736717
camera_lon_deg: 100.523186
camera_alt_m: 1.5
camera_yaw_deg: 90.0
camera_pitch_deg: 10.0

# Simulation parameters
drone_size_m: 0.25
num_drones: 1
path_altitude_agl_m: 5.5
path_span_m: 40.0
speed_mps: 5.0
max_lateral_accel_mps2: 1.5

# Timing
duration_s: 12.0
fps: 15.0

# Noise parameters
pixel_centroid_sigma_px: 1.0
bbox_size_sigma_px: 2.0
confidence_noise: 0.05
miss_rate_small: 0.03
false_positive_rate: 0.01
processing_latency_ms_mean: 50.0
processing_latency_ms_jitter: 20.0

# MQTT settings
mqtt_host: "localhost"
mqtt_port: 1883
mqtt_topic: "sensors/cam01/detections"
mqtt_qos: 0
retain: false
client_id: ""

# Testing options
deterministic_seed: 42
offline_mode: false
```

### JSON Format

```json
{
  "image_width_px": 1920,
  "image_height_px": 1080,
  "vertical_fov_deg": 50.0,
  "principal_point_px": [960, 540],
  
  "camera_lat_deg": 13.736717,
  "camera_lon_deg": 100.523186,
  "camera_alt_m": 1.5,
  "camera_yaw_deg": 90.0,
  "camera_pitch_deg": 10.0,
  
  "num_drones": 1,
  "duration_s": 12.0,
  "fps": 15.0,
  
  "mqtt_host": "localhost",
  "mqtt_port": 1883,
  "mqtt_topic": "sensors/cam01/detections",
  
  "deterministic_seed": 42,
  "offline_mode": false
}
```

## Output and Logging

### Logging Levels

- **Default**: INFO level with progress updates
- **Verbose** (`-v`): DEBUG level with detailed component information
- **Quiet** (`-q`): ERROR level only

### Simulation Results

The simulator outputs comprehensive statistics at completion:

```
Simulation Results:
==================================================
Frames processed: 180
Detections generated: 156
Messages published: 156
Publish success rate: 100.0%
Actual FPS: 14.98
Timing accuracy: ±2.1ms
```

### JSON Output Format (Offline Mode)

When running in offline mode (`--offline`), detection messages are printed as JSON:

```json
{
  "timestamp_utc": "2025-09-21T08:23:12.123Z",
  "frame_id": 12345,
  "camera": {
    "resolution": [1920, 1080],
    "focal_px": 900.0,
    "principal_point": [960.0, 540.0],
    "yaw_deg": 90.0,
    "pitch_deg": 10.0,
    "lat_deg": 13.736717,
    "lon_deg": 100.523186,
    "alt_m_msl": 1.50
  },
  "detections": [
    {
      "class": "drone",
      "confidence": 0.91,
      "bbox_px": [980, 520, 1020, 580],
      "center_px": [1000, 550],
      "size_px": [40, 60]
    }
  ],
  "edge": {
    "processing_latency_ms": 42,
    "detector_version": "det-v1.2"
  }
}
```

## Error Handling

### Configuration Errors

The CLI provides clear error messages for configuration issues:

```bash
# Invalid configuration file
Error: Configuration file not found: missing_config.yaml

# Invalid parameter values
Error loading configuration: Vertical FOV must be between 1 and 179 degrees

# Validation errors
Configuration validation: FAILED
Error: Cannot specify both focal length/sensor and vertical FOV - choose one method
```

### Runtime Errors

Runtime errors are logged with appropriate detail level:

```bash
# MQTT connection failure
ERROR - Failed to connect to MQTT broker: Connection refused

# Simulation interruption
INFO - Simulation interrupted by user
```

## Integration Examples

### Shell Scripts

```bash
#!/bin/bash
# Run multiple test scenarios

echo "Running basic test..."
python -m drone_detection_simulator --config examples/testing_deterministic.yaml

echo "Running multi-drone test..."
python -m drone_detection_simulator --config examples/multi_drone.yaml --duration 15

echo "Running performance test..."
python -m drone_detection_simulator --config examples/high_performance.yaml
```

### Python Integration

```python
from drone_detection_simulator import cli_main
import sys

# Run CLI programmatically
sys.argv = ['drone_detection_simulator', '--config', 'my_config.yaml', '--offline']
result = cli_main()
```

## Troubleshooting

### Common Issues

1. **MQTT Connection Failed**
   - Check if MQTT broker is running
   - Verify host and port settings
   - Use `--offline` mode for testing without MQTT

2. **Configuration Validation Failed**
   - Use `--validate-config` to check configuration
   - Check parameter ranges and required combinations
   - Refer to example configurations

3. **Performance Issues**
   - Reduce FPS or image resolution
   - Use fewer drones
   - Enable `--quiet` mode to reduce logging overhead

4. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path and package installation
   - Verify YAML/JSON file syntax

### Getting Help

```bash
# Show help message
python -m drone_detection_simulator --help

# List available examples
python -m drone_detection_simulator --list-examples

# Validate configuration
python -m drone_detection_simulator --config my_config.yaml --validate-config
```