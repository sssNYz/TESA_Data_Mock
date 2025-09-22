# JSON Schema Reference

This document provides the complete JSON schema for the drone detection simulator's detection messages and configuration files.

## Detection Message Schema

The simulator outputs detection messages in the following JSON format:

### Complete Detection Message

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

### Schema Definition

#### Root Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp_utc` | string | Yes | ISO 8601 UTC timestamp of detection |
| `frame_id` | integer | Yes | Sequential frame identifier |
| `camera` | object | Yes | Camera metadata and configuration |
| `detections` | array | Yes | Array of detection objects |
| `edge` | object | Yes | Edge processing metadata |

#### Camera Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resolution` | array[integer] | Yes | Image resolution [width, height] in pixels |
| `focal_px` | number | Yes | Focal length in pixels |
| `principal_point` | array[number] | Yes | Principal point [x, y] in pixels |
| `yaw_deg` | number | Yes | Camera yaw angle in degrees (heading from north) |
| `pitch_deg` | number | Yes | Camera pitch angle in degrees (tilt) |
| `lat_deg` | number | Yes | Camera latitude in degrees |
| `lon_deg` | number | Yes | Camera longitude in degrees |
| `alt_m_msl` | number | Yes | Camera altitude in meters above mean sea level |

#### Detection Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `class` | string | Yes | Object class ("drone", "false_drone", "unknown") |
| `confidence` | number | Yes | Detection confidence score (0.0-1.0) |
| `bbox_px` | array[integer] | Yes | Bounding box [x_min, y_min, x_max, y_max] in pixels |
| `center_px` | array[number] | Yes | Detection center [x, y] in pixels |
| `size_px` | array[number] | Yes | Detection size [width, height] in pixels |

#### Edge Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `processing_latency_ms` | number | Yes | Processing latency in milliseconds |
| `detector_version` | string | Yes | Detector version identifier |

### JSON Schema (JSON Schema Draft 7)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Drone Detection Message",
  "type": "object",
  "required": ["timestamp_utc", "frame_id", "camera", "detections", "edge"],
  "properties": {
    "timestamp_utc": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 UTC timestamp of detection"
    },
    "frame_id": {
      "type": "integer",
      "minimum": 0,
      "description": "Sequential frame identifier"
    },
    "camera": {
      "type": "object",
      "required": ["resolution", "focal_px", "principal_point", "yaw_deg", "pitch_deg", "lat_deg", "lon_deg", "alt_m_msl"],
      "properties": {
        "resolution": {
          "type": "array",
          "items": {"type": "integer", "minimum": 1},
          "minItems": 2,
          "maxItems": 2,
          "description": "Image resolution [width, height] in pixels"
        },
        "focal_px": {
          "type": "number",
          "minimum": 0,
          "description": "Focal length in pixels"
        },
        "principal_point": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2,
          "description": "Principal point [x, y] in pixels"
        },
        "yaw_deg": {
          "type": "number",
          "minimum": -180,
          "maximum": 180,
          "description": "Camera yaw angle in degrees"
        },
        "pitch_deg": {
          "type": "number",
          "minimum": -90,
          "maximum": 90,
          "description": "Camera pitch angle in degrees"
        },
        "lat_deg": {
          "type": "number",
          "minimum": -90,
          "maximum": 90,
          "description": "Camera latitude in degrees"
        },
        "lon_deg": {
          "type": "number",
          "minimum": -180,
          "maximum": 180,
          "description": "Camera longitude in degrees"
        },
        "alt_m_msl": {
          "type": "number",
          "description": "Camera altitude in meters above mean sea level"
        }
      }
    },
    "detections": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["class", "confidence", "bbox_px", "center_px", "size_px"],
        "properties": {
          "class": {
            "type": "string",
            "enum": ["drone", "false_drone", "unknown"],
            "description": "Object class"
          },
          "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Detection confidence score"
          },
          "bbox_px": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 4,
            "maxItems": 4,
            "description": "Bounding box [x_min, y_min, x_max, y_max] in pixels"
          },
          "center_px": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
            "description": "Detection center [x, y] in pixels"
          },
          "size_px": {
            "type": "array",
            "items": {"type": "number", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
            "description": "Detection size [width, height] in pixels"
          }
        }
      }
    },
    "edge": {
      "type": "object",
      "required": ["processing_latency_ms", "detector_version"],
      "properties": {
        "processing_latency_ms": {
          "type": "number",
          "minimum": 0,
          "description": "Processing latency in milliseconds"
        },
        "detector_version": {
          "type": "string",
          "description": "Detector version identifier"
        }
      }
    }
  }
}
```

## Configuration File Schema

### YAML Configuration Schema

```yaml
# Camera intrinsics (choose one focal length method)
image_width_px: 1920                    # integer, > 0
image_height_px: 1080                   # integer, > 0
focal_length_mm: 25.0                   # number, > 0 (requires sensor_height_mm)
sensor_height_mm: 5.0                   # number, > 0 (requires focal_length_mm)
vertical_fov_deg: 50.0                  # number, 1-179 (alternative to focal/sensor)
principal_point_px: [960, 540]         # array[number], optional, defaults to center

# Camera extrinsics
camera_lat_deg: 13.736717               # number, -90 to 90
camera_lon_deg: 100.523186              # number, -180 to 180
camera_alt_m: 1.5                       # number, >= 0
camera_yaw_deg: 90.0                    # number, -180 to 180
camera_pitch_deg: 10.0                  # number, -90 to 90

# Simulation parameters
drone_size_m: 0.25                      # number, > 0
num_drones: 1                           # integer, > 0
path_altitude_agl_m: 5.5                # number, > 0
path_span_m: 40.0                       # number, > 0
speed_mps: 5.0                          # number, > 0
max_lateral_accel_mps2: 1.5             # number, > 0

# Timing
duration_s: 12.0                        # number, > 0
fps: 15.0                               # number, > 0

# Noise parameters
pixel_centroid_sigma_px: 1.0            # number, >= 0
bbox_size_sigma_px: 2.0                 # number, >= 0
confidence_noise: 0.05                  # number, 0-1
miss_rate_small: 0.03                   # number, 0-1
false_positive_rate: 0.01               # number, 0-1
processing_latency_ms_mean: 50.0        # number, >= 0
processing_latency_ms_jitter: 20.0      # number, >= 0

# MQTT settings
mqtt_host: "localhost"                  # string, non-empty
mqtt_port: 1883                         # integer, 1-65535
mqtt_topic: "sensors/cam01/detections"  # string, non-empty
mqtt_qos: 0                             # integer, 0-2
retain: false                           # boolean
client_id: ""                           # string

# Testing options
deterministic_seed: 42                  # integer or null
offline_mode: false                     # boolean
```

### JSON Configuration Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Drone Detection Simulator Configuration",
  "type": "object",
  "properties": {
    "image_width_px": {
      "type": "integer",
      "minimum": 1,
      "default": 1920
    },
    "image_height_px": {
      "type": "integer", 
      "minimum": 1,
      "default": 1080
    },
    "focal_length_mm": {
      "type": ["number", "null"],
      "minimum": 0,
      "exclusiveMinimum": true
    },
    "sensor_height_mm": {
      "type": ["number", "null"],
      "minimum": 0,
      "exclusiveMinimum": true
    },
    "vertical_fov_deg": {
      "type": ["number", "null"],
      "minimum": 1,
      "maximum": 179,
      "default": 50.0
    },
    "principal_point_px": {
      "type": ["array", "null"],
      "items": {"type": "number"},
      "minItems": 2,
      "maxItems": 2
    },
    "camera_lat_deg": {
      "type": "number",
      "minimum": -90,
      "maximum": 90,
      "default": 13.736717
    },
    "camera_lon_deg": {
      "type": "number",
      "minimum": -180,
      "maximum": 180,
      "default": 100.523186
    },
    "camera_alt_m": {
      "type": "number",
      "minimum": 0,
      "default": 1.5
    },
    "camera_yaw_deg": {
      "type": "number",
      "minimum": -180,
      "maximum": 180,
      "default": 90.0
    },
    "camera_pitch_deg": {
      "type": "number",
      "minimum": -90,
      "maximum": 90,
      "default": 10.0
    },
    "drone_size_m": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 0.25
    },
    "num_drones": {
      "type": "integer",
      "minimum": 1,
      "default": 1
    },
    "path_altitude_agl_m": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 5.5
    },
    "path_span_m": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 40.0
    },
    "speed_mps": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 5.0
    },
    "max_lateral_accel_mps2": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 1.5
    },
    "duration_s": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 12.0
    },
    "fps": {
      "type": "number",
      "minimum": 0,
      "exclusiveMinimum": true,
      "default": 15.0
    },
    "pixel_centroid_sigma_px": {
      "type": "number",
      "minimum": 0,
      "default": 1.0
    },
    "bbox_size_sigma_px": {
      "type": "number",
      "minimum": 0,
      "default": 2.0
    },
    "confidence_noise": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.05
    },
    "miss_rate_small": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.03
    },
    "false_positive_rate": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "default": 0.01
    },
    "processing_latency_ms_mean": {
      "type": "number",
      "minimum": 0,
      "default": 50.0
    },
    "processing_latency_ms_jitter": {
      "type": "number",
      "minimum": 0,
      "default": 20.0
    },
    "mqtt_host": {
      "type": "string",
      "minLength": 1,
      "default": "localhost"
    },
    "mqtt_port": {
      "type": "integer",
      "minimum": 1,
      "maximum": 65535,
      "default": 1883
    },
    "mqtt_topic": {
      "type": "string",
      "minLength": 1,
      "default": "sensors/cam01/detections"
    },
    "mqtt_qos": {
      "type": "integer",
      "minimum": 0,
      "maximum": 2,
      "default": 0
    },
    "retain": {
      "type": "boolean",
      "default": false
    },
    "client_id": {
      "type": "string",
      "default": ""
    },
    "deterministic_seed": {
      "type": ["integer", "null"],
      "default": null
    },
    "offline_mode": {
      "type": "boolean",
      "default": false
    }
  }
}
```

## Validation Examples

### Python Validation

```python
import json
import jsonschema

# Load detection message schema
with open('detection_schema.json', 'r') as f:
    detection_schema = json.load(f)

# Validate detection message
detection_message = {
    "timestamp_utc": "2025-09-21T08:23:12.123Z",
    "frame_id": 12345,
    # ... rest of message
}

try:
    jsonschema.validate(detection_message, detection_schema)
    print("Detection message is valid")
except jsonschema.ValidationError as e:
    print(f"Validation error: {e.message}")
```

### Command-Line Validation

```bash
# Validate configuration file
python -m drone_detection_simulator --config my_config.yaml --validate-config

# Print effective configuration for inspection
python -m drone_detection_simulator --config my_config.yaml --print-config
```

## Common Validation Errors

### Configuration Errors

1. **Focal Length Specification**
   ```
   Error: Must specify either (focal_length_mm AND sensor_height_mm) OR vertical_fov_deg
   ```

2. **Parameter Range Errors**
   ```
   Error: Vertical FOV must be between 1 and 179 degrees
   Error: Camera latitude must be between -90 and 90 degrees
   ```

3. **Type Errors**
   ```
   Error: principal_point_px must be a 2-element list or tuple
   Error: MQTT port must be between 1 and 65535
   ```

### Detection Message Errors

1. **Missing Required Fields**
   ```
   ValidationError: 'timestamp_utc' is a required property
   ```

2. **Invalid Data Types**
   ```
   ValidationError: 0.91 is not of type 'string' (for class field)
   ```

3. **Value Range Errors**
   ```
   ValidationError: 1.5 is greater than the maximum of 1 (for confidence)
   ```

## Schema Evolution

The JSON schema follows semantic versioning principles:

- **Major version changes**: Breaking changes to required fields or data types
- **Minor version changes**: Addition of optional fields or enum values
- **Patch version changes**: Documentation updates or constraint clarifications

Current schema version: `1.0.0`

### Backward Compatibility

The simulator maintains backward compatibility for:
- Configuration file formats
- Detection message structure
- Core field names and types

### Future Extensions

Planned schema extensions:
- Additional detection classes
- Enhanced camera metadata
- Performance metrics
- Multi-camera support