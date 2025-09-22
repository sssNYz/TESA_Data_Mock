# Implementation Plan

- [x] 1. Set up project structure and configuration management
  - Create directory structure for simulator package
  - Implement SimulatorConfig dataclass with validation
  - Add configuration loading from dict/YAML with proper error handling
  - Write unit tests for configuration validation
  - _Requirements: 6.1, 6.4_

- [x] 2. Implement core camera model and focal length computation
  - Create CameraModel class with focal length calculation from FOV or focal/sensor parameters
  - Implement principal point computation with center default
  - Add camera metadata generation for JSON payload
  - Write unit tests for focal length calculations with both input methods
  - _Requirements: 1.1, 6.1_

- [x] 3. Create basic coordinate projection utilities
  - Implement world-to-pixel projection using pinhole camera model
  - Create simple 2D rotation for camera yaw/pitch (Pi-level, no complex 3D transforms)
  - Add pixel coordinate validation and image boundary checking
  - Write unit tests for projection accuracy with known test cases
  - _Requirements: 1.2, 4.1_

- [x] 4. Implement smooth drone motion generation
  - Create MotionGenerator class for physics-based ENU path planning
  - Generate left-to-right motion with configurable span, speed, and altitude
  - Enforce maximum lateral acceleration constraints to prevent teleporting
  - Write unit tests to verify smooth motion and acceleration limits
  - _Requirements: 3.2, 3.3_

- [x] 5. Build pixel detection generation system
  - Create DetectionGenerator that projects world positions to pixel coordinates
  - Generate realistic bounding boxes from projected drone positions
  - Implement confidence scoring based on detection size and visibility
  - Write unit tests for detection generation and bounding box computation
  - _Requirements: 1.2, 1.3_

- [x] 6. Add detector-level noise and realism
  - Implement NoiseModel for pixel coordinate and bounding box size noise
  - Add confidence variation and miss detection probability based on object size
  - Create false positive generation with random locations and low confidence
  - Write unit tests for noise statistical properties and miss rate behavior
  - _Requirements: 1.3, 1.4, 8.1, 8.2_

- [x] 7. Create JSON message builder
  - Implement detection message formatting matching Pi-realistic schema
  - Build camera metadata section with fixed configuration values
  - Add edge metadata with processing latency simulation
  - Write unit tests for JSON schema compliance and field validation
  - _Requirements: 2.1, 5.2_

- [x] 8. Implement MQTT publishing system
  - Create MQTTPublisher class with connection management and retry logic
  - Add configurable QoS, retain flags, and topic publishing
  - Implement graceful connection failure handling
  - Write unit tests with mock MQTT broker for publish verification
  - _Requirements: 2.2, 2.3_

- [x] 9. Add multi-drone support
  - Extend MotionGenerator to handle multiple independent drone paths
  - Update DetectionGenerator to process multiple drones per frame
  - Modify JSON builder to output detection arrays
  - Write unit tests for multi-drone detection generation and path independence
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 10. Implement timing and frame rate control
  - Create simulation loop with accurate frame timing at target FPS
  - Add processing latency simulation with configurable jitter
  - Implement frame ID tracking and UTC timestamp generation
  - Write unit tests for timing accuracy and latency simulation
  - _Requirements: 9.1, 9.4_

- [x] 11. Build main simulation orchestrator
  - Create DroneSimulator class that coordinates all components
  - Implement main simulation loop with proper initialization and cleanup
  - Add offline mode for testing (print JSON instead of MQTT)
  - Write integration tests for complete simulation cycle
  - _Requirements: 5.1, 5.3_

- [x] 12. Add deterministic testing support
  - Implement random seed configuration for reproducible runs
  - Create test utilities for validation with known ground truth
  - Add assertion tests for smooth motion and bounded pixel movement
  - Write end-to-end tests with fixed seed to verify consistent output
  - _Requirements: 5.4_

- [x] 13. Create command-line interface and example usage
  - Implement CLI with configuration file loading and parameter overrides
  - Add example configuration files for different scenarios
  - Create main() function demonstrating typical usage patterns
  - Write documentation with usage examples and JSON schema reference
  - _Requirements: 6.2, 6.3_

- [x] 14. Add comprehensive error handling and logging
  - Implement proper exception handling for configuration, MQTT, and runtime errors
  - Add structured logging with configurable levels for debugging
  - Create graceful shutdown handling for MQTT connections
  - Write tests for error conditions and recovery behavior
  - _Requirements: 2.3, 6.4_

- [x] 15. Performance optimization and validation
  - Optimize simulation loop for real-time performance at target FPS
  - Add performance monitoring and frame rate validation
  - Implement memory usage optimization for long-running simulations
  - Write performance tests to ensure consistent frame timing
  - _Requirements: 5.1_