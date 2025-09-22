# Drone Detection Simulator - Command Line Interface

The Drone Detection Simulator provides a comprehensive command-line interface for generating realistic camera-based drone detections with various configuration options and scenarios.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings
python -m drone_detection_simulator

# Run in offline mode (print JSON instead of MQTT)
python -m drone_detection_simulator --offline

# Run with custom parameters
python -m drone_detection_simulator --duration 30 --fps 10 --num-drones 2
```

## Features

- **Flexible Configuration**: Load settings from YAML/JSON files or override via command line
- **Multiple Scenarios**: Pre-built configurations for different testing scenarios
- **Offline Mode**: Print JSON output for testing without MQTT broker
- **Deterministic Testing**: Fixed random seeds for reproducible results
- **Comprehensive Validation**: Built-in configuration validation and error reporting
- **Rich Documentation**: Extensive help and examples

## Available Example Configurations

| Configuration | Description |
|---------------|-------------|
| `config_example.yaml` | Basic example with detailed comments |
| `multi_drone.yaml` | Multiple drones with overlapping flight paths |
| `high_performance.yaml` | Optimized for high frame rate and minimal noise |
| `noisy_environment.yaml` | Challenging conditions with high noise |
| `testing_deterministic.yaml` | Reproducible testing configuration |
| `long_duration.yaml` | Extended simulation for endurance testing |
| `minimal_config.json` | Minimal JSON configuration example |

## Command Line Options

### Basic Usage
```bash
python -m drone_detection_simulator [OPTIONS]
```

### Configuration
- `--config FILE` - Load configuration from YAML or JSON file
- `--validate-config` - Validate configuration and exit
- `--print-config` - Print effective configuration and exit
- `--list-examples` - List available example configurations

### Simulation Parameters
- `--duration SECONDS` - Simulation duration (default: 12.0)
- `--fps FPS` - Frames per second (default: 15.0)
- `--num-drones COUNT` - Number of drones (default: 1)

### Camera Settings
- `--camera-lat DEGREES` - Camera latitude
- `--camera-lon DEGREES` - Camera longitude
- `--camera-alt METERS` - Camera altitude
- `--camera-yaw DEGREES` - Camera yaw angle
- `--camera-pitch DEGREES` - Camera pitch angle
- `--vertical-fov DEGREES` - Vertical field of view

### Motion Parameters
- `--altitude METERS` - Drone flight altitude AGL
- `--speed MPS` - Drone speed in m/s
- `--span METERS` - Flight path span

### MQTT Settings
- `--mqtt-host HOST` - MQTT broker hostname
- `--mqtt-port PORT` - MQTT broker port
- `--mqtt-topic TOPIC` - MQTT topic for publishing
- `--mqtt-qos {0,1,2}` - MQTT Quality of Service level

### Noise and Realism
- `--pixel-noise SIGMA` - Pixel coordinate noise std dev
- `--bbox-noise SIGMA` - Bounding box size noise std dev
- `--miss-rate RATE` - Detection miss rate (0.0-1.0)
- `--false-positive-rate RATE` - False positive rate (0.0-1.0)

### Testing and Debugging
- `--offline` - Print JSON instead of MQTT
- `--seed SEED` - Random seed for deterministic behavior
- `--verbose` - Enable verbose logging
- `--quiet` - Suppress output except errors

## Usage Examples

### Basic Examples
```bash
# Default simulation
python -m drone_detection_simulator

# Quick test with offline output
python -m drone_detection_simulator --offline --duration 5 --seed 42

# Multi-drone scenario
python -m drone_detection_simulator --num-drones 3 --duration 20
```

### Configuration File Examples
```bash
# Load YAML configuration
python -m drone_detection_simulator --config examples/multi_drone.yaml

# Load JSON configuration
python -m drone_detection_simulator --config examples/minimal_config.json

# Override parameters from config file
python -m drone_detection_simulator --config examples/multi_drone.yaml --duration 30 --offline
```

### Testing and Validation
```bash
# Validate configuration
python -m drone_detection_simulator --config my_config.yaml --validate-config

# Print effective configuration
python -m drone_detection_simulator --config my_config.yaml --print-config

# Deterministic testing
python -m drone_detection_simulator --seed 42 --offline --duration 5
```

### Performance Testing
```bash
# High-performance scenario
python -m drone_detection_simulator --config examples/high_performance.yaml

# Custom high-FPS test
python -m drone_detection_simulator --fps 30 --duration 10 --pixel-noise 0.5
```

### Challenging Conditions
```bash
# Noisy environment
python -m drone_detection_simulator --config examples/noisy_environment.yaml

# Custom noisy conditions
python -m drone_detection_simulator --pixel-noise 3.0 --miss-rate 0.1 --false-positive-rate 0.05
```

## Output Formats

### JSON Detection Messages
When running in offline mode, the simulator outputs JSON detection messages:

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

### Simulation Results
At completion, the simulator outputs comprehensive statistics:

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

## Configuration File Format

### YAML Example
```yaml
# Camera intrinsics
image_width_px: 1920
image_height_px: 1080
vertical_fov_deg: 50.0

# Camera position
camera_lat_deg: 13.736717
camera_lon_deg: 100.523186
camera_alt_m: 1.5
camera_yaw_deg: 90.0
camera_pitch_deg: 10.0

# Simulation parameters
num_drones: 1
duration_s: 12.0
fps: 15.0

# MQTT settings
mqtt_host: "localhost"
mqtt_port: 1883
mqtt_topic: "sensors/cam01/detections"

# Testing options
deterministic_seed: 42
offline_mode: false
```

### JSON Example
```json
{
  "image_width_px": 1920,
  "image_height_px": 1080,
  "vertical_fov_deg": 50.0,
  "camera_lat_deg": 13.736717,
  "camera_lon_deg": 100.523186,
  "camera_alt_m": 1.5,
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

## Programmatic Usage

### Python API
```python
from drone_detection_simulator import SimulatorConfig, DroneSimulator

# Create configuration
config = SimulatorConfig(
    duration_s=10.0,
    num_drones=2,
    offline_mode=True,
    deterministic_seed=42
)

# Run simulation
simulator = DroneSimulator(config)
results = simulator.run()

print(f"Processed {results['simulation']['frames_processed']} frames")
print(f"Generated {results['detections']['total_generated']} detections")
```

### CLI Integration
```python
from drone_detection_simulator import cli_main
import sys

# Run CLI programmatically
sys.argv = ['drone_detection_simulator', '--config', 'my_config.yaml', '--offline']
result = cli_main()
```

## Error Handling

The CLI provides clear error messages for common issues:

### Configuration Errors
```bash
Error: Configuration file not found: missing_config.yaml
Error: Vertical FOV must be between 1 and 179 degrees
Error: Cannot specify both focal length/sensor and vertical FOV
```

### Runtime Errors
```bash
ERROR - Failed to connect to MQTT broker: Connection refused
INFO - Simulation interrupted by user
```

## Integration Examples

### Shell Scripts
```bash
#!/bin/bash
# Run test suite

echo "Running basic test..."
python -m drone_detection_simulator --config examples/testing_deterministic.yaml

echo "Running multi-drone test..."
python -m drone_detection_simulator --config examples/multi_drone.yaml --duration 15

echo "Running performance test..."
python -m drone_detection_simulator --config examples/high_performance.yaml
```

### Docker Integration
```dockerfile
FROM python:3.9-slim

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

# Run simulation
CMD ["python", "-m", "drone_detection_simulator", "--config", "examples/multi_drone.yaml"]
```

## Troubleshooting

### Common Issues

1. **MQTT Connection Failed**
   - Check if MQTT broker is running
   - Verify host and port settings
   - Use `--offline` mode for testing

2. **Configuration Validation Failed**
   - Use `--validate-config` to check configuration
   - Check parameter ranges and combinations
   - Refer to example configurations

3. **Performance Issues**
   - Reduce FPS or image resolution
   - Use fewer drones
   - Enable `--quiet` mode

### Getting Help
```bash
# Show help
python -m drone_detection_simulator --help

# List examples
python -m drone_detection_simulator --list-examples

# Validate configuration
python -m drone_detection_simulator --config my_config.yaml --validate-config
```

## Advanced Usage

### Batch Processing
```bash
# Process multiple configurations
for config in examples/*.yaml; do
    echo "Processing $config..."
    python -m drone_detection_simulator --config "$config" --offline --quiet > "output_$(basename $config .yaml).json"
done
```

### Performance Monitoring
```bash
# Monitor resource usage
time python -m drone_detection_simulator --config examples/high_performance.yaml --verbose
```

### Automated Testing
```bash
# Regression testing
python -m drone_detection_simulator --seed 42 --offline --duration 5 > expected_output.json
python -m drone_detection_simulator --seed 42 --offline --duration 5 > actual_output.json
diff expected_output.json actual_output.json
```

For more detailed information, see:
- [CLI Usage Guide](CLI_USAGE.md)
- [JSON Schema Reference](JSON_SCHEMA.md)
- [Usage Examples](../examples/usage_examples.py)