"""
Command-line interface for the drone detection simulator.

This module provides a comprehensive CLI with configuration file loading,
parameter overrides, and various usage patterns for different scenarios.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from .config import SimulatorConfig
from .simulator import DroneSimulator


def create_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Drone Detection Simulator - Generate realistic camera-based drone detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m drone_detection_simulator

  # Run with custom configuration file
  python -m drone_detection_simulator --config examples/multi_drone.yaml

  # Run in offline mode (print JSON instead of MQTT)
  python -m drone_detection_simulator --offline

  # Override specific parameters
  python -m drone_detection_simulator --duration 30 --fps 10 --num-drones 3

  # Run deterministic simulation for testing
  python -m drone_detection_simulator --seed 42 --offline --duration 5

  # High-performance scenario
  python -m drone_detection_simulator --config examples/high_performance.yaml

  # Noisy environment testing
  python -m drone_detection_simulator --config examples/noisy_environment.yaml
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        metavar="FILE",
        help="Path to configuration file (YAML or JSON)"
    )
    
    # Basic simulation parameters
    parser.add_argument(
        "--duration", "-d",
        type=float,
        metavar="SECONDS",
        help="Simulation duration in seconds (default: 12.0)"
    )
    
    parser.add_argument(
        "--fps", "-f",
        type=float,
        metavar="FPS",
        help="Frames per second (default: 15.0)"
    )
    
    parser.add_argument(
        "--num-drones", "-n",
        type=int,
        metavar="COUNT",
        help="Number of drones to simulate (default: 1)"
    )
    
    # Camera parameters
    camera_group = parser.add_argument_group("camera parameters")
    camera_group.add_argument(
        "--camera-lat",
        type=float,
        metavar="DEGREES",
        help="Camera latitude in degrees"
    )
    
    camera_group.add_argument(
        "--camera-lon",
        type=float,
        metavar="DEGREES",
        help="Camera longitude in degrees"
    )
    
    camera_group.add_argument(
        "--camera-alt",
        type=float,
        metavar="METERS",
        help="Camera altitude in meters"
    )
    
    camera_group.add_argument(
        "--camera-yaw",
        type=float,
        metavar="DEGREES",
        help="Camera yaw angle in degrees (heading from north)"
    )
    
    camera_group.add_argument(
        "--camera-pitch",
        type=float,
        metavar="DEGREES",
        help="Camera pitch angle in degrees"
    )
    
    camera_group.add_argument(
        "--vertical-fov",
        type=float,
        metavar="DEGREES",
        help="Vertical field of view in degrees"
    )
    
    # Drone motion parameters
    motion_group = parser.add_argument_group("drone motion parameters")
    motion_group.add_argument(
        "--altitude",
        type=float,
        metavar="METERS",
        help="Drone flight altitude AGL in meters"
    )
    
    motion_group.add_argument(
        "--speed",
        type=float,
        metavar="MPS",
        help="Drone speed in meters per second"
    )
    
    motion_group.add_argument(
        "--span",
        type=float,
        metavar="METERS",
        help="Flight path span in meters"
    )
    
    # MQTT parameters
    mqtt_group = parser.add_argument_group("MQTT parameters")
    mqtt_group.add_argument(
        "--mqtt-host",
        type=str,
        metavar="HOST",
        help="MQTT broker hostname (default: localhost)"
    )
    
    mqtt_group.add_argument(
        "--mqtt-port",
        type=int,
        metavar="PORT",
        help="MQTT broker port (default: 1883)"
    )
    
    mqtt_group.add_argument(
        "--mqtt-topic",
        type=str,
        metavar="TOPIC",
        help="MQTT topic for publishing detections"
    )
    
    mqtt_group.add_argument(
        "--mqtt-qos",
        type=int,
        choices=[0, 1, 2],
        help="MQTT Quality of Service level (0, 1, or 2)"
    )
    
    # Noise and realism parameters
    noise_group = parser.add_argument_group("noise and realism parameters")
    noise_group.add_argument(
        "--pixel-noise",
        type=float,
        metavar="SIGMA",
        help="Pixel coordinate noise standard deviation"
    )
    
    noise_group.add_argument(
        "--bbox-noise",
        type=float,
        metavar="SIGMA",
        help="Bounding box size noise standard deviation"
    )
    
    noise_group.add_argument(
        "--miss-rate",
        type=float,
        metavar="RATE",
        help="Detection miss rate for small objects (0.0-1.0)"
    )
    
    noise_group.add_argument(
        "--false-positive-rate",
        type=float,
        metavar="RATE",
        help="False positive detection rate (0.0-1.0)"
    )
    
    # Testing and debugging options
    test_group = parser.add_argument_group("testing and debugging options")
    test_group.add_argument(
        "--offline",
        action="store_true",
        help="Run in offline mode (print JSON instead of MQTT)"
    )
    
    test_group.add_argument(
        "--seed",
        type=int,
        metavar="SEED",
        help="Random seed for deterministic behavior"
    )
    
    test_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    test_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    test_group.add_argument(
        "--log-file",
        type=str,
        metavar="FILE",
        help="Path to log file for detailed logging"
    )
    
    # Utility commands
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration file and exit"
    )
    
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print effective configuration and exit"
    )
    
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="List available example configurations"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def setup_logging(verbose: bool = False, quiet: bool = False, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration based on verbosity level.
    
    Args:
        verbose: Enable verbose logging
        quiet: Suppress all output except errors
        log_file: Optional path to log file
    """
    from .logging_config import SimulatorLogger, LogLevel
    from pathlib import Path
    
    # Determine log level
    if quiet:
        level = LogLevel.ERROR
    elif verbose:
        level = LogLevel.DEBUG
    else:
        level = LogLevel.INFO
    
    # Setup logging
    SimulatorLogger.setup_logging(
        level=level,
        log_file=Path(log_file) if log_file else None,
        console_output=True,
        verbose=verbose,
        quiet=quiet
    )


def build_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build configuration overrides from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}
    
    # Basic simulation parameters
    if args.duration is not None:
        overrides['duration_s'] = args.duration
    if args.fps is not None:
        overrides['fps'] = args.fps
    if args.num_drones is not None:
        overrides['num_drones'] = args.num_drones
    
    # Camera parameters
    if args.camera_lat is not None:
        overrides['camera_lat_deg'] = args.camera_lat
    if args.camera_lon is not None:
        overrides['camera_lon_deg'] = args.camera_lon
    if args.camera_alt is not None:
        overrides['camera_alt_m'] = args.camera_alt
    if args.camera_yaw is not None:
        overrides['camera_yaw_deg'] = args.camera_yaw
    if args.camera_pitch is not None:
        overrides['camera_pitch_deg'] = args.camera_pitch
    if args.vertical_fov is not None:
        overrides['vertical_fov_deg'] = args.vertical_fov
    
    # Drone motion parameters
    if args.altitude is not None:
        overrides['path_altitude_agl_m'] = args.altitude
    if args.speed is not None:
        overrides['speed_mps'] = args.speed
    if args.span is not None:
        overrides['path_span_m'] = args.span
    
    # MQTT parameters
    if args.mqtt_host is not None:
        overrides['mqtt_host'] = args.mqtt_host
    if args.mqtt_port is not None:
        overrides['mqtt_port'] = args.mqtt_port
    if args.mqtt_topic is not None:
        overrides['mqtt_topic'] = args.mqtt_topic
    if args.mqtt_qos is not None:
        overrides['mqtt_qos'] = args.mqtt_qos
    
    # Noise parameters
    if args.pixel_noise is not None:
        overrides['pixel_centroid_sigma_px'] = args.pixel_noise
    if args.bbox_noise is not None:
        overrides['bbox_size_sigma_px'] = args.bbox_noise
    if args.miss_rate is not None:
        overrides['miss_rate_small'] = args.miss_rate
    if args.false_positive_rate is not None:
        overrides['false_positive_rate'] = args.false_positive_rate
    
    # Testing options
    if args.offline:
        overrides['offline_mode'] = True
    if args.seed is not None:
        overrides['deterministic_seed'] = args.seed
    
    return overrides


def load_configuration(config_path: Optional[str], overrides: Dict[str, Any]) -> SimulatorConfig:
    """
    Load configuration from file and apply overrides.
    
    Args:
        config_path: Path to configuration file (optional)
        overrides: Configuration parameter overrides
        
    Returns:
        Loaded and validated SimulatorConfig
        
    Raises:
        SystemExit: If configuration loading fails
    """
    try:
        # Load base configuration
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
                sys.exit(1)
            
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = SimulatorConfig.from_yaml(config_path)
            elif config_path.suffix.lower() == '.json':
                config = SimulatorConfig.from_json(config_path)
            else:
                print(f"Error: Unsupported configuration file format: {config_path.suffix}", file=sys.stderr)
                print("Supported formats: .yaml, .yml, .json", file=sys.stderr)
                sys.exit(1)
        else:
            config = SimulatorConfig()
        
        # Apply overrides
        if overrides:
            config_dict = config.to_dict()
            config_dict.update(overrides)
            config = SimulatorConfig.from_dict(config_dict)
        
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)


def list_example_configurations() -> None:
    """List available example configurations."""
    examples_dir = Path(__file__).parent.parent / "examples"
    
    print("Available example configurations:")
    print()
    
    if not examples_dir.exists():
        print("  No examples directory found")
        return
    
    config_files = list(examples_dir.glob("*.yaml")) + list(examples_dir.glob("*.yml")) + list(examples_dir.glob("*.json"))
    
    if not config_files:
        print("  No example configuration files found")
        return
    
    for config_file in sorted(config_files):
        print(f"  {config_file.name}")
        
        # Try to read description from file
        try:
            with open(config_file, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            
            # Look for description in comments
            description = None
            for line in first_lines:
                if line.startswith('#') and ('example' in line.lower() or 'config' in line.lower()):
                    description = line.lstrip('#').strip()
                    break
            
            if description:
                print(f"    {description}")
        except:
            pass
        
        print()


def validate_configuration(config: SimulatorConfig) -> bool:
    """
    Validate configuration and print results.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        config.validate()
        print("Configuration validation: PASSED")
        return True
    except Exception as e:
        print(f"Configuration validation: FAILED")
        print(f"Error: {e}")
        return False


def print_configuration(config: SimulatorConfig) -> None:
    """
    Print the effective configuration in a readable format.
    
    Args:
        config: Configuration to print
    """
    print("Effective Configuration:")
    print("=" * 50)
    
    config_dict = config.to_dict()
    
    # Group parameters for better readability
    groups = {
        "Camera Intrinsics": [
            'image_width_px', 'image_height_px', 'focal_length_mm', 
            'sensor_height_mm', 'vertical_fov_deg', 'principal_point_px'
        ],
        "Camera Position": [
            'camera_lat_deg', 'camera_lon_deg', 'camera_alt_m',
            'camera_yaw_deg', 'camera_pitch_deg'
        ],
        "Simulation Parameters": [
            'drone_size_m', 'num_drones', 'path_altitude_agl_m',
            'path_span_m', 'speed_mps', 'max_lateral_accel_mps2'
        ],
        "Timing": [
            'duration_s', 'fps'
        ],
        "Noise and Realism": [
            'pixel_centroid_sigma_px', 'bbox_size_sigma_px', 'confidence_noise',
            'miss_rate_small', 'false_positive_rate', 'processing_latency_ms_mean',
            'processing_latency_ms_jitter'
        ],
        "MQTT Settings": [
            'mqtt_host', 'mqtt_port', 'mqtt_topic', 'mqtt_qos', 'retain', 'client_id'
        ],
        "Testing Options": [
            'deterministic_seed', 'offline_mode'
        ]
    }
    
    for group_name, param_names in groups.items():
        print(f"\n{group_name}:")
        for param_name in param_names:
            if param_name in config_dict:
                value = config_dict[param_name]
                if value is not None:
                    print(f"  {param_name}: {value}")


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle utility commands first
    if args.list_examples:
        list_example_configurations()
        return 0
    
    # Setup logging
    setup_logging(args.verbose, args.quiet, args.log_file)
    from .logging_config import SimulatorLogger
    logger = SimulatorLogger.get_logger(__name__)
    
    try:
        # Build configuration overrides
        overrides = build_config_overrides(args)
        
        # Load configuration
        config = load_configuration(args.config, overrides)
        
        # Handle configuration validation
        if args.validate_config:
            return 0 if validate_configuration(config) else 1
        
        # Handle configuration printing
        if args.print_config:
            print_configuration(config)
            return 0
        
        # Log configuration summary
        logger.info(f"Starting simulation: {config.num_drones} drones, "
                   f"{config.duration_s}s at {config.fps} FPS")
        if config.offline_mode:
            logger.info("Running in offline mode (printing JSON)")
        else:
            logger.info(f"Publishing to MQTT: {config.mqtt_host}:{config.mqtt_port}/{config.mqtt_topic}")
        
        # Create and run simulator
        simulator = DroneSimulator(config)
        results = simulator.run()
        
        # Log final results
        if not args.quiet:
            print("\nSimulation Results:")
            print("=" * 50)
            print(f"Frames processed: {results['simulation']['frames_processed']}")
            print(f"Detections generated: {results['detections']['total_generated']}")
            print(f"Messages published: {results['publishing']['messages_published']}")
            print(f"Publish success rate: {results['publishing']['success_rate']:.1%}")
            
            if 'timing' in results:
                timing = results['timing']
                print(f"Actual FPS: {timing['actual_fps']:.2f}")
                print(f"Timing accuracy: ±{timing['timing_error_ms']:.1f}ms")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())