"""
Main simulation orchestrator for the drone detection simulator.

This module provides the DroneSimulator class that coordinates all components
and implements the main simulation loop with proper initialization and cleanup.
"""

import logging
import signal
import sys
import threading
import time
from typing import Optional, Dict, Any, List
import numpy as np

from .config import SimulatorConfig
from .camera import CameraModel
from .motion import MotionGenerator
from .detection import DetectionGenerator
from .noise import NoiseModel
from .message_builder import DetectionMessageBuilder
from .mqtt_publisher import MQTTPublisher
from .timing import SimulationLoop, FrameInfo, format_timestamp_utc
from .error_handling import (
    SimulationError, ConfigurationError, safe_execute, 
    error_context, create_error_summary, ErrorRecovery
)
from .logging_config import SimulatorLogger, log_exception
from .performance import (
    create_performance_context, cleanup_performance_context,
    PerformanceMonitor, PerformanceOptimizer
)


logger = SimulatorLogger.get_logger(__name__)


class DroneSimulator:
    """
    Main simulation orchestrator that coordinates all components.
    
    This class manages the complete simulation lifecycle including:
    - Component initialization and configuration
    - Main simulation loop execution
    - Graceful shutdown and cleanup
    - Error handling and logging
    - Offline mode support for testing
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize the drone simulator with all components.
        
        Args:
            config: Simulator configuration containing all parameters
            
        Raises:
            ConfigurationError: If configuration is invalid
            SimulationError: If component initialization fails
        """
        try:
            # Validate configuration
            config.validate()
            self.config = config
            
            # Initialize state
            self.running = False
            self.shutdown_event = threading.Event()
            self.initialization_errors: List[Exception] = []
            self.runtime_errors: List[Exception] = []
            
            # Initialize random number generator for deterministic behavior
            if config.deterministic_seed is not None:
                self.rng = np.random.default_rng(config.deterministic_seed)
                logger.info(f"Using deterministic seed: {config.deterministic_seed}")
            else:
                self.rng = np.random.default_rng()
                logger.info("Using random seed")
            
            # Statistics tracking (initialize before components)
            self.stats = {
                'frames_processed': 0,
                'detections_generated': 0,
                'messages_published': 0,
                'publish_failures': 0,
                'false_positives_generated': 0,
                'missed_detections': 0,
                'frame_errors': 0,
                'initialization_time_s': 0,
                'total_runtime_s': 0
            }
            
            # Performance monitoring context
            self.performance_context = None
            
            # Initialize components with error handling
            self._initialize_components()
            
            # Setup signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            logger.info("Drone simulator initialized successfully")
            
        except Exception as e:
            log_exception(logger, e, "Failed to initialize drone simulator")
            if isinstance(e, (ConfigurationError, SimulationError)):
                raise
            raise SimulationError(f"Simulator initialization failed: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize all simulation components with comprehensive error handling."""
        initialization_start = time.time()
        
        with error_context("Component initialization", logger):
            logger.info("Initializing simulation components...")
            
            # Initialize camera model
            try:
                with error_context("Camera model initialization", logger):
                    self.camera_model = CameraModel(self.config)
                    focal_length = self.camera_model.get_focal_length_px()
                    logger.info(f"Camera model initialized with focal length: {focal_length:.1f} px")
            except Exception as e:
                error = SimulationError(f"Camera model initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            # Initialize motion generator
            try:
                with error_context("Motion generator initialization", logger):
                    self.motion_generator = MotionGenerator(self.config)
                    motion_stats = self.motion_generator.get_path_statistics()
                    logger.info(f"Motion generator initialized for {motion_stats['num_drones']} drones")
                    logger.info(f"Path span: {motion_stats.get('east_range_m', 0):.1f}m, "
                               f"Max acceleration: {motion_stats.get('max_lateral_acceleration_mps2', 0):.2f} m/s²")
            except Exception as e:
                error = SimulationError(f"Motion generator initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            # Initialize detection generator
            try:
                with error_context("Detection generator initialization", logger):
                    self.detection_generator = DetectionGenerator(self.config, self.camera_model)
                    logger.info("Detection generator initialized")
            except Exception as e:
                error = SimulationError(f"Detection generator initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            # Initialize noise model
            try:
                with error_context("Noise model initialization", logger):
                    self.noise_model = NoiseModel(self.config, self.rng)
                    logger.info("Noise model initialized")
            except Exception as e:
                error = SimulationError(f"Noise model initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            # Initialize message builder
            try:
                with error_context("Message builder initialization", logger):
                    self.message_builder = DetectionMessageBuilder(self.config, self.camera_model, self.rng)
                    logger.info("Message builder initialized")
            except Exception as e:
                error = SimulationError(f"Message builder initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            # Initialize MQTT publisher
            try:
                with error_context("MQTT publisher initialization", logger):
                    self.mqtt_publisher = MQTTPublisher(self.config, self.rng)
                    if not self.config.offline_mode:
                        logger.info(f"MQTT publisher initialized for {self.config.mqtt_host}:{self.config.mqtt_port}")
                    else:
                        logger.info("MQTT publisher initialized in offline mode")
            except Exception as e:
                error = SimulationError(f"MQTT publisher initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            # Initialize simulation loop
            try:
                with error_context("Simulation loop initialization", logger):
                    self.simulation_loop = SimulationLoop(self.config, self.rng)
                    total_frames = int(self.config.duration_s * self.config.fps)
                    logger.info(f"Simulation loop initialized: {self.config.duration_s}s at {self.config.fps} FPS "
                               f"({total_frames} frames)")
            except Exception as e:
                error = SimulationError(f"Simulation loop initialization failed: {e}")
                self.initialization_errors.append(error)
                raise error
            
            initialization_time = time.time() - initialization_start
            self.stats['initialization_time_s'] = initialization_time
            logger.info(f"All components initialized successfully in {initialization_time:.2f}s")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            signal_names = {
                signal.SIGINT: "SIGINT (Ctrl+C)",
                signal.SIGTERM: "SIGTERM"
            }
            signal_name = signal_names.get(signum, f"signal {signum}")
            logger.info(f"Received {signal_name}, initiating graceful shutdown...")
            self.shutdown()
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            logger.debug("Signal handlers configured for graceful shutdown")
        except Exception as e:
            logger.warning(f"Failed to setup signal handlers: {e}")
            # Don't fail initialization if signal handlers can't be set up
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete simulation with proper initialization and cleanup.
        
        Returns:
            Dictionary containing simulation statistics and timing information
            
        Raises:
            SimulationError: If simulation fails
        """
        runtime_start = time.time()
        
        try:
            with error_context("Simulation execution", logger):
                logger.info("Starting drone detection simulation...")
                
                # Validate simulator state
                if not self._validate_simulator_state():
                    raise SimulationError("Simulator is not in a valid state to run")
                
                # Connect to MQTT broker if not in offline mode
                if not self.config.offline_mode:
                    try:
                        with error_context("MQTT connection", logger):
                            if not self.mqtt_publisher.connect():
                                logger.error("Failed to connect to MQTT broker")
                                return self._get_final_statistics()
                    except Exception as e:
                        logger.error(f"MQTT connection failed: {e}")
                        if not self._should_continue_without_mqtt():
                            raise SimulationError(f"Cannot continue without MQTT connection: {e}")
                
                # Reset statistics
                self._reset_statistics()
                
                # Initialize performance monitoring
                self.performance_context = create_performance_context(
                    self.config, enable_optimizations=True
                )
                logger.info("Performance monitoring and optimizations enabled")
                
                # Run the simulation loop with error recovery
                self.running = True
                timing_stats = self._run_simulation_loop()
                
                runtime_duration = time.time() - runtime_start
                self.stats['total_runtime_s'] = runtime_duration
                
                logger.info(f"Simulation completed successfully in {runtime_duration:.2f}s")
                
                # Get performance results and cleanup
                performance_results = None
                if self.performance_context:
                    performance_results = cleanup_performance_context(self.performance_context)
                    self.performance_context = None
                
                # Combine timing stats with simulation stats
                final_stats = self._get_final_statistics()
                final_stats['timing'] = timing_stats
                
                # Add performance results
                if performance_results:
                    final_stats['performance'] = performance_results
                    
                    # Log performance summary
                    perf_summary = performance_results['performance_summary']
                    timing_val = performance_results['timing_validation']
                    
                    logger.info(f"Performance Summary:")
                    logger.info(f"  Actual FPS: {timing_val.get('actual_fps', 0):.2f} "
                               f"(target: {timing_val.get('target_fps', 0):.2f})")
                    logger.info(f"  FPS Error: {timing_val.get('fps_error_percent', 0):.1f}%")
                    logger.info(f"  Timing Valid: {timing_val.get('valid', False)}")
                    
                    if 'memory' in perf_summary:
                        memory = perf_summary['memory']
                        logger.info(f"  Peak Memory: {memory.get('peak_memory_mb', 0):.1f} MB")
                        logger.info(f"  Memory Growth: {memory.get('memory_growth_mb', 0):.1f} MB")
                    
                    # Log optimization recommendations
                    recommendations = performance_results.get('optimization_recommendations', [])
                    if recommendations:
                        logger.info("Performance Recommendations:")
                        for rec in recommendations:
                            logger.info(f"  - {rec}")
                
                # Log error summary if there were runtime errors
                if self.runtime_errors:
                    error_summary = create_error_summary(self.runtime_errors)
                    logger.warning(f"Simulation completed with {error_summary['total_errors']} runtime errors")
                    final_stats['errors'] = error_summary
                
                return final_stats
                
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            return self._get_final_statistics()
        except Exception as e:
            log_exception(logger, e, "Simulation execution failed")
            if isinstance(e, SimulationError):
                raise
            raise SimulationError(f"Simulation failed: {e}")
        finally:
            self._cleanup()
    
    def _validate_simulator_state(self) -> bool:
        """
        Validate that the simulator is in a valid state to run.
        
        Returns:
            True if simulator state is valid, False otherwise
        """
        try:
            # Check if all required components are initialized
            required_components = [
                'camera_model', 'motion_generator', 'detection_generator',
                'noise_model', 'message_builder', 'mqtt_publisher', 'simulation_loop'
            ]
            
            for component in required_components:
                if not hasattr(self, component):
                    logger.error(f"Missing required component: {component}")
                    return False
            
            # Check MQTT publisher health if not in offline mode
            if not self.config.offline_mode:
                if not self.mqtt_publisher.is_healthy():
                    logger.warning("MQTT publisher is not in a healthy state")
                    # Don't fail validation, but log the warning
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating simulator state: {e}")
            return False
    
    def _should_continue_without_mqtt(self) -> bool:
        """
        Determine if simulation should continue without MQTT connection.
        
        Returns:
            True if simulation should continue, False otherwise
        """
        # For now, don't continue without MQTT unless in offline mode
        return self.config.offline_mode
    
    def _run_simulation_loop(self) -> Dict[str, Any]:
        """
        Run the simulation loop with error recovery.
        
        Returns:
            Timing statistics from the simulation loop
        """
        try:
            return self.simulation_loop.run(self._process_frame)
        except Exception as e:
            log_exception(logger, e, "Simulation loop execution failed")
            raise SimulationError(f"Simulation loop failed: {e}")
    
    def _process_frame(self, frame_info: FrameInfo) -> None:
        """
        Process a single simulation frame with comprehensive error handling.
        
        Args:
            frame_info: Information about the current frame
        """
        # Record frame start for performance monitoring
        performance_monitor = None
        if self.performance_context:
            performance_monitor = self.performance_context['monitor']
            frame_start_time = performance_monitor.record_frame_start()
        else:
            frame_start_time = time.time()
        
        try:
            with error_context(f"Frame {frame_info.frame_id} processing", logger, reraise=False):
                # Get drone positions for this frame
                world_positions = safe_execute(
                    lambda: self.motion_generator.get_positions_at_time(frame_info.simulation_time_s),
                    default_return=[],
                    logger=logger,
                    context=f"motion generation for frame {frame_info.frame_id}"
                )
                
                if not world_positions:
                    logger.warning(f"No drone positions available for frame {frame_info.frame_id}")
                    return
                
                # Generate clean detections
                clean_detections = safe_execute(
                    lambda: self.detection_generator.generate_detections(world_positions),
                    default_return=[],
                    logger=logger,
                    context=f"detection generation for frame {frame_info.frame_id}"
                )
                
                # Apply noise and detector behavior
                final_detections = []
                for i, detection in enumerate(clean_detections):
                    try:
                        # Check if detection should be missed
                        if self.noise_model.should_miss_detection(detection):
                            self.stats['missed_detections'] += 1
                            continue
                        
                        # Apply noise to detection
                        noisy_detection = self.noise_model.apply_detection_noise(detection)
                        
                        # Clip detection to image boundaries
                        clipped_detection = self.detection_generator.clip_detection_to_image(noisy_detection)
                        final_detections.append(clipped_detection)
                        
                    except Exception as e:
                        logger.warning(f"Error processing detection {i} in frame {frame_info.frame_id}: {e}")
                        self.runtime_errors.append(e)
                        continue
                
                # Add false positives
                try:
                    false_positive = self.noise_model.generate_false_positive()
                    if false_positive is not None:
                        final_detections.append(false_positive)
                        self.stats['false_positives_generated'] += 1
                except Exception as e:
                    logger.warning(f"Error generating false positive for frame {frame_info.frame_id}: {e}")
                    self.runtime_errors.append(e)
                
                # Build detection message
                detection_message = safe_execute(
                    lambda: self.message_builder.build_detection_message(
                        detections=final_detections,
                        timestamp_utc=frame_info.timestamp_utc
                    ),
                    default_return=None,
                    logger=logger,
                    context=f"message building for frame {frame_info.frame_id}"
                )
                
                if detection_message is None:
                    logger.error(f"Failed to build detection message for frame {frame_info.frame_id}")
                    self.stats['frame_errors'] += 1
                    return
                
                # Update processing latency from frame info
                detection_message['edge']['processing_latency_ms'] = round(frame_info.processing_latency_ms, 1)
                detection_message['frame_id'] = frame_info.frame_id
                
                # Publish message
                publish_success = safe_execute(
                    lambda: self.mqtt_publisher.publish_detection(detection_message),
                    default_return=False,
                    logger=logger,
                    context=f"message publishing for frame {frame_info.frame_id}"
                )
                
                if publish_success:
                    self.stats['messages_published'] += 1
                else:
                    self.stats['publish_failures'] += 1
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.stats['detections_generated'] += len(final_detections)
                
                # Log progress periodically with performance info
                if frame_info.frame_id % max(1, int(self.config.fps * 2)) == 0:  # Every 2 seconds
                    progress = self.simulation_loop.get_progress()
                    frame_time = (time.time() - frame_start_time) * 1000  # Convert to ms
                    
                    # Add performance info to progress log
                    perf_info = ""
                    if performance_monitor:
                        current_metrics = performance_monitor.get_current_metrics()
                        if current_metrics.actual_fps > 0:
                            perf_info = f", {current_metrics.actual_fps:.1f} FPS"
                        if current_metrics.memory_usage_mb > 0:
                            perf_info += f", {current_metrics.memory_usage_mb:.0f} MB"
                    
                    logger.info(f"Progress: {progress*100:.1f}% (Frame {frame_info.frame_id}, "
                               f"{len(final_detections)} detections, {frame_time:.1f}ms{perf_info})")
                
                # Periodic memory optimization for long-running simulations
                if (self.performance_context and 
                    frame_info.frame_id % max(1, int(self.config.fps * 30)) == 0):  # Every 30 seconds
                    optimizer = self.performance_context['optimizer']
                    current_memory = performance_monitor.get_current_metrics().memory_usage_mb
                    if optimizer.should_optimize_memory(current_memory):
                        optimization_result = optimizer.optimize_memory_usage()
                        if optimization_result['memory_freed_mb'] > 10:  # Log if significant
                            logger.info(f"Memory optimization freed {optimization_result['memory_freed_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Critical error processing frame {frame_info.frame_id}: {e}")
            self.runtime_errors.append(e)
            self.stats['frame_errors'] += 1
            # Continue with next frame rather than stopping simulation
        finally:
            # Record frame end for performance monitoring
            if performance_monitor:
                performance_monitor.record_frame_end(frame_start_time)
    
    def _reset_statistics(self) -> None:
        """Reset simulation statistics."""
        self.stats = {
            'frames_processed': 0,
            'detections_generated': 0,
            'messages_published': 0,
            'publish_failures': 0,
            'false_positives_generated': 0,
            'missed_detections': 0
        }
        self.message_builder.reset_frame_counter()
    
    def _get_final_statistics(self) -> Dict[str, Any]:
        """
        Get final simulation statistics.
        
        Returns:
            Dictionary containing comprehensive simulation statistics
        """
        # Get component statistics
        motion_stats = self.motion_generator.get_path_statistics()
        noise_stats = self.noise_model.get_noise_statistics()
        
        # Calculate derived statistics
        total_frames = int(self.config.duration_s * self.config.fps)
        success_rate = (self.stats['messages_published'] / 
                       max(1, self.stats['messages_published'] + self.stats['publish_failures']))
        
        avg_detections_per_frame = (self.stats['detections_generated'] / 
                                   max(1, self.stats['frames_processed']))
        
        return {
            'simulation': {
                'duration_s': self.config.duration_s,
                'fps': self.config.fps,
                'total_frames_expected': total_frames,
                'frames_processed': self.stats['frames_processed'],
                'completion_rate': self.stats['frames_processed'] / max(1, total_frames)
            },
            'detections': {
                'total_generated': self.stats['detections_generated'],
                'false_positives': self.stats['false_positives_generated'],
                'missed_detections': self.stats['missed_detections'],
                'avg_per_frame': avg_detections_per_frame
            },
            'publishing': {
                'messages_published': self.stats['messages_published'],
                'publish_failures': self.stats['publish_failures'],
                'success_rate': success_rate,
                'offline_mode': self.config.offline_mode
            },
            'motion': motion_stats,
            'noise': noise_stats,
            'configuration': {
                'num_drones': self.config.num_drones,
                'deterministic_seed': self.config.deterministic_seed,
                'mqtt_topic': self.config.mqtt_topic
            }
        }
    
    def shutdown(self) -> None:
        """Request graceful shutdown of the simulation."""
        logger.info("Shutting down simulation...")
        self.running = False
        self.shutdown_event.set()
        
        if hasattr(self, 'simulation_loop'):
            self.simulation_loop.shutdown()
    
    def _cleanup(self) -> None:
        """Perform cleanup operations with comprehensive error handling."""
        cleanup_errors = []
        
        try:
            logger.info("Starting cleanup operations...")
            
            # Stop simulation loop
            if hasattr(self, 'simulation_loop'):
                try:
                    self.simulation_loop.shutdown()
                    logger.debug("Simulation loop stopped")
                except Exception as e:
                    cleanup_errors.append(e)
                    logger.error(f"Error stopping simulation loop: {e}")
            
            # Cleanup performance monitoring
            if self.performance_context:
                try:
                    cleanup_performance_context(self.performance_context)
                    self.performance_context = None
                    logger.debug("Performance monitoring cleaned up")
                except Exception as e:
                    cleanup_errors.append(e)
                    logger.error(f"Error cleaning up performance monitoring: {e}")
            
            # Disconnect MQTT publisher
            if hasattr(self, 'mqtt_publisher'):
                try:
                    self.mqtt_publisher.disconnect()
                    logger.debug("MQTT publisher disconnected")
                except Exception as e:
                    cleanup_errors.append(e)
                    logger.error(f"Error disconnecting MQTT publisher: {e}")
            
            # Log final statistics
            try:
                if hasattr(self, 'mqtt_publisher'):
                    mqtt_stats = self.mqtt_publisher.get_statistics()
                    logger.info(f"MQTT Statistics: {mqtt_stats['publish_successes']} successful publishes, "
                               f"{mqtt_stats['publish_failures']} failures")
            except Exception as e:
                cleanup_errors.append(e)
                logger.error(f"Error logging final statistics: {e}")
            
            # Set running flag to false
            self.running = False
            
            if cleanup_errors:
                logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors")
                error_summary = create_error_summary(cleanup_errors)
                logger.debug(f"Cleanup error summary: {error_summary}")
            else:
                logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Critical error during cleanup: {e}")
            # Ensure running flag is set to false even if cleanup fails
            self.running = False
    
    def get_progress(self) -> float:
        """
        Get current simulation progress.
        
        Returns:
            Progress as fraction between 0.0 and 1.0
        """
        if hasattr(self, 'simulation_loop'):
            return self.simulation_loop.get_progress()
        return 0.0
    
    def is_running(self) -> bool:
        """
        Check if simulation is currently running.
        
        Returns:
            True if simulation is running, False otherwise
        """
        return self.running and not self.shutdown_event.is_set()
    
    def get_current_statistics(self) -> Dict[str, Any]:
        """
        Get current simulation statistics (while running).
        
        Returns:
            Dictionary containing current statistics
        """
        return self.stats.copy()


def main(config_path: Optional[str] = None, **config_overrides) -> Dict[str, Any]:
    """
    Main entry point for running the drone detection simulator.
    
    Args:
        config_path: Optional path to configuration file (YAML or JSON)
        **config_overrides: Configuration parameters to override
    
    Returns:
        Dictionary containing simulation statistics
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        # Load configuration
        if config_path:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = SimulatorConfig.from_yaml(config_path)
            elif config_path.endswith('.json'):
                config = SimulatorConfig.from_json(config_path)
            else:
                raise ValueError("Configuration file must be .yaml, .yml, or .json")
        else:
            config = SimulatorConfig()
        
        # Apply configuration overrides
        if config_overrides:
            config_dict = config.to_dict()
            config_dict.update(config_overrides)
            config = SimulatorConfig.from_dict(config_dict)
        
        logger.info(f"Configuration loaded: {config.num_drones} drones, "
                   f"{config.duration_s}s at {config.fps} FPS")
        
        # Create and run simulator
        simulator = DroneSimulator(config)
        results = simulator.run()
        
        # Log final results
        logger.info("Simulation Results:")
        logger.info(f"  Frames processed: {results['simulation']['frames_processed']}")
        logger.info(f"  Detections generated: {results['detections']['total_generated']}")
        logger.info(f"  Messages published: {results['publishing']['messages_published']}")
        logger.info(f"  Publish success rate: {results['publishing']['success_rate']:.1%}")
        
        if 'timing' in results:
            timing = results['timing']
            logger.info(f"  Actual FPS: {timing['actual_fps']:.2f}")
            logger.info(f"  Timing error: {timing['timing_error_ms']:.1f}ms")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    # Use the comprehensive CLI when run directly
    from .cli import main as cli_main
    import sys
    sys.exit(cli_main())