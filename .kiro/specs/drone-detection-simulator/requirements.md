# Requirements Document

## Introduction

This feature implements a Python-based drone detection simulator that simulates realistic detections from a fixed camera using proper geometric calculations and publishes detection data over MQTT. The simulator uses a pinhole camera model with real camera pose geometry to generate smooth drone movement across the camera's field of view, computing accurate distance estimates from object size in pixels.

## Requirements

### Requirement 1

**User Story:** As a developer testing drone detection systems, I want a realistic simulation of camera-based drone detections, so that I can validate detection algorithms and data processing pipelines without requiring physical drones.

#### Acceptance Criteria

1. WHEN the simulator is configured with camera parameters THEN the system SHALL compute focal length in pixels using either focal length/sensor dimensions or vertical field of view
2. WHEN a drone moves through the camera view THEN the system SHALL use pinhole camera projection to calculate accurate distance from measured object size
3. WHEN generating detections THEN the system SHALL apply realistic noise to pixel coordinates and bounding box dimensions
4. WHEN the drone is far from camera THEN the system SHALL apply miss detection probability based on object size

### Requirement 2

**User Story:** As a system integrator, I want the simulator to publish standardized JSON detection messages over MQTT, so that I can integrate with existing detection processing systems.

#### Acceptance Criteria

1. WHEN a detection occurs THEN the system SHALL publish a JSON message containing timestamp, camera metadata, detection data, world position estimates, and processing metadata
2. WHEN publishing THEN the system SHALL use configurable MQTT broker settings including host, port, topic, QoS, and retain flags
3. WHEN network issues occur THEN the system SHALL handle MQTT connection failures gracefully
4. WHEN configured THEN the system SHALL support packet drop simulation for realistic network conditions

### Requirement 3

**User Story:** As a simulation operator, I want to configure realistic camera positioning and drone flight paths, so that I can test various operational scenarios.

#### Acceptance Criteria

1. WHEN configuring camera THEN the system SHALL accept geodetic position (lat/lon/altitude), orientation (yaw/pitch/roll), and intrinsic parameters
2. WHEN defining drone path THEN the system SHALL generate smooth left-to-right motion with configurable altitude, span, speed, and acceleration limits
3. WHEN drone moves THEN the system SHALL enforce maximum lateral acceleration to prevent unrealistic teleporting motion
4. WHEN drone exits camera view THEN the system SHALL continue world motion simulation but may miss detections naturally

### Requirement 4

**User Story:** As a geospatial application developer, I want accurate coordinate transformations between camera, world, and geodetic reference frames, so that I can correlate detections with real-world positions.

#### Acceptance Criteria

1. WHEN transforming coordinates THEN the system SHALL convert between camera frame (+Z forward, +X right, +Y down) and ENU world frame (+X east, +Y north, +Z up)
2. WHEN computing world positions THEN the system SHALL apply camera pose rotation matrix using yaw (heading from north), pitch (tilt), and roll angles
3. WHEN converting to geodetic THEN the system SHALL transform ENU coordinates to WGS84 latitude/longitude/altitude using local tangent plane approximation
4. WHEN estimating target position THEN the system SHALL provide both camera-relative and world geodetic coordinates in detection messages

### Requirement 5

**User Story:** As a software developer, I want well-structured, testable code with clear interfaces, so that I can maintain, extend, and validate the simulator functionality.

#### Acceptance Criteria

1. WHEN implementing functions THEN the system SHALL provide specific named functions for focal length computation, coordinate transformations, projection, and noise application
2. WHEN processing data THEN the system SHALL use pure functions without side effects where possible for testability
3. WHEN validating THEN the system SHALL include deterministic random seed option for reproducible test runs
4. WHEN running THEN the system SHALL support offline mode that prints JSON without MQTT for testing purposes

### Requirement 6

**User Story:** As a system administrator, I want configurable simulation parameters, so that I can adapt the simulator to different camera setups and operational requirements.

#### Acceptance Criteria

1. WHEN configuring THEN the system SHALL accept camera intrinsics via either focal length/sensor size or vertical field of view
2. WHEN setting up THEN the system SHALL support configurable noise parameters for pixel accuracy, bounding box size, focal length bias, and detection miss rates
3. WHEN running THEN the system SHALL accept timing parameters including frame rate, duration, and processing latency simulation
4. WHEN initializing THEN the system SHALL validate configuration parameters and raise clear errors for invalid values

### Requirement 7

**User Story:** As a developer, I want the simulator to generate detections for multiple drones simultaneously, so that I can test system performance and handling of concurrent targets.

#### Acceptance Criteria

1. WHEN multiple drones are configured THEN the simulator SHALL generate independent smooth flight paths for each drone
2. WHEN publishing a frame THEN the detection message SHALL include an array of detections with one entry per drone
3. WHEN drones overlap in the camera field of view THEN the simulator SHALL output multiple bounding boxes within the same frame
4. WHEN drones have different sizes or altitudes THEN the system SHALL compute independent distance estimates for each target

### Requirement 8

**User Story:** As a system tester, I want the simulator to occasionally produce false positive detections, so that I can validate filtering and robustness of detection pipelines.

#### Acceptance Criteria

1. WHEN simulating detections THEN the system SHALL insert non-drone objects with low confidence scores at configurable probability
2. WHEN generating false positives THEN the simulator SHALL vary bounding box size and location randomly within the image
3. WHEN publishing JSON THEN false positives SHALL be clearly labeled with "class": "unknown" or "class": "false_drone"
4. WHEN false positives occur THEN they SHALL not follow realistic physics-based motion patterns

### Requirement 9

**User Story:** As a backend integrator, I want the simulator to introduce variable message latency, so that I can test how the system handles real-world network timing issues.

#### Acceptance Criteria

1. WHEN publishing MQTT messages THEN the simulator SHALL add configurable random jitter to processing latency values
2. WHEN jitter is applied THEN the message timestamp SHALL remain the original detection time, while delivery order may vary
3. WHEN packet delays exceed thresholds THEN the simulator SHALL still publish messages but out-of-order delivery MAY occur
4. WHEN latency simulation is enabled THEN the system SHALL support both mean latency and jitter variance parameters