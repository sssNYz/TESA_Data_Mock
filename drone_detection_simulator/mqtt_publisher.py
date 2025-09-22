"""
MQTT publishing system for drone detection simulator.
"""

import json
import logging
import time
import signal
import threading
from typing import Dict, Optional, Tuple
import paho.mqtt.client as mqtt
import numpy as np

from .config import SimulatorConfig
from .error_handling import (
    MQTTError, NetworkError, retry_on_exception, 
    safe_execute, error_context, ErrorRecovery
)
from .logging_config import SimulatorLogger


logger = SimulatorLogger.get_logger(__name__)


class MQTTPublisher:
    """
    MQTT publisher with connection management and retry logic.
    
    Handles publishing detection messages with configurable QoS, retain flags,
    and graceful connection failure handling.
    """
    
    def __init__(self, config: SimulatorConfig, rng: Optional[np.random.Generator] = None):
        """
        Initialize MQTT publisher.
        
        Args:
            config: Simulator configuration containing MQTT settings
            rng: Random number generator for latency simulation
        """
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.connection_attempts = 0
        self.max_connection_attempts = 5
        self.retry_delay_s = 1.0
        self.shutdown_requested = False
        self.connection_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'connection_failures': 0,
            'publish_attempts': 0,
            'publish_successes': 0,
            'publish_failures': 0,
            'disconnections': 0
        }
        
        if not config.offline_mode:
            try:
                self._setup_mqtt_client()
                logger.info(f"MQTT publisher initialized for {config.mqtt_host}:{config.mqtt_port}")
            except Exception as e:
                logger.error(f"Failed to initialize MQTT publisher: {e}")
                raise MQTTError(f"MQTT publisher initialization failed: {e}")
    
    def _setup_mqtt_client(self) -> None:
        """Setup MQTT client with connection callbacks."""
        with error_context("MQTT client setup", logger):
            # Create client with unique ID if not specified
            client_id = self.config.client_id
            if not client_id:
                client_id = f"drone_sim_{int(time.time())}"
            
            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
                
                # Set up callbacks
                self.client.on_connect = self._on_connect
                self.client.on_disconnect = self._on_disconnect
                self.client.on_publish = self._on_publish
                self.client.on_log = self._on_log
                
                # Configure client options
                self.client.reconnect_delay_set(min_delay=1, max_delay=120)
                
                logger.debug(f"MQTT client created with ID: {client_id}")
                
            except Exception as e:
                raise MQTTError(f"Failed to create MQTT client: {e}", error_code="CLIENT_SETUP_ERROR")
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback for when the client receives a CONNACK response from the server."""
        with self.connection_lock:
            if rc == 0:
                self.connected = True
                self.connection_attempts = 0
                self.stats['successful_connections'] += 1
                logger.info(f"Connected to MQTT broker at {self.config.mqtt_host}:{self.config.mqtt_port}")
            else:
                self.connected = False
                self.stats['connection_failures'] += 1
                error_messages = {
                    1: "Connection refused - incorrect protocol version",
                    2: "Connection refused - invalid client identifier",
                    3: "Connection refused - server unavailable",
                    4: "Connection refused - bad username or password",
                    5: "Connection refused - not authorised"
                }
                error_msg = error_messages.get(rc, f"Unknown error code {rc}")
                logger.error(f"Failed to connect to MQTT broker: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc, properties=None):
        """Callback for when the client disconnects from the broker."""
        with self.connection_lock:
            self.connected = False
            self.stats['disconnections'] += 1
            
            if rc != 0:
                logger.warning(f"Unexpected disconnection from MQTT broker, return code {rc}")
                if not self.shutdown_requested:
                    logger.info("Will attempt to reconnect...")
            else:
                logger.info("Disconnected from MQTT broker")
    
    def _on_publish(self, client, userdata, mid, properties=None):
        """Callback for when a message is published."""
        self.stats['publish_successes'] += 1
        logger.debug(f"Message published with message ID: {mid}")
    
    def _on_log(self, client, userdata, level, buf):
        """Callback for MQTT client logging."""
        # Only log MQTT client messages at debug level to avoid noise
        if level == mqtt.MQTT_LOG_ERR:
            logger.error(f"MQTT client error: {buf}")
        elif level == mqtt.MQTT_LOG_WARNING:
            logger.warning(f"MQTT client warning: {buf}")
        else:
            logger.debug(f"MQTT client: {buf}")
    
    @retry_on_exception(
        max_attempts=5,
        delay_seconds=1.0,
        backoff_multiplier=2.0,
        exceptions=(ConnectionError, OSError, MQTTError)
    )
    def connect(self) -> bool:
        """
        Connect to MQTT broker with retry logic.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            MQTTError: If connection fails after all retries
        """
        if self.config.offline_mode or not self.client:
            return True
        
        with self.connection_lock:
            if self.connected:
                return True
            
            if self.shutdown_requested:
                logger.info("Connection aborted due to shutdown request")
                return False
        
        try:
            with error_context("MQTT broker connection", logger):
                self.stats['connection_attempts'] += 1
                self.connection_attempts += 1
                
                logger.info(f"Attempting to connect to MQTT broker at {self.config.mqtt_host}:{self.config.mqtt_port}")
                
                # Validate connection parameters
                if not self.config.mqtt_host or not self.config.mqtt_host.strip():
                    raise MQTTError("MQTT host cannot be empty", error_code="INVALID_HOST")
                
                if not (1 <= self.config.mqtt_port <= 65535):
                    raise MQTTError(f"Invalid MQTT port: {self.config.mqtt_port}", error_code="INVALID_PORT")
                
                # Attempt connection
                self.client.connect(self.config.mqtt_host, self.config.mqtt_port, 60)
                self.client.loop_start()
                
                # Wait for connection to establish
                timeout = 10.0
                start_time = time.time()
                while not self.connected and (time.time() - start_time) < timeout and not self.shutdown_requested:
                    time.sleep(0.1)
                
                if self.shutdown_requested:
                    logger.info("Connection aborted due to shutdown request")
                    return False
                
                if self.connected:
                    logger.info("Successfully connected to MQTT broker")
                    return True
                else:
                    raise MQTTError("Connection timeout", error_code="CONNECTION_TIMEOUT")
                    
        except Exception as e:
            if isinstance(e, MQTTError):
                raise
            raise MQTTError(f"Connection failed: {e}", error_code="CONNECTION_ERROR")
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker gracefully."""
        with error_context("MQTT disconnect", logger, reraise=False):
            self.shutdown_requested = True
            
            if self.client:
                try:
                    if self.connected:
                        logger.info("Disconnecting from MQTT broker...")
                        self.client.loop_stop()
                        self.client.disconnect()
                        
                        # Wait briefly for clean disconnect
                        timeout = 2.0
                        start_time = time.time()
                        while self.connected and (time.time() - start_time) < timeout:
                            time.sleep(0.1)
                        
                        if self.connected:
                            logger.warning("Disconnect timeout, forcing connection close")
                    
                    logger.info("MQTT client disconnected")
                    
                except Exception as e:
                    logger.error(f"Error during MQTT disconnect: {e}")
                finally:
                    self.connected = False
    
    def _simulate_network_effects(self) -> Tuple[float, bool]:
        """
        Simulate network latency and packet drops.
        
        Returns:
            Tuple of (latency_ms, should_drop_packet)
        """
        # Simulate processing latency with jitter
        latency_ms = self.rng.normal(
            self.config.processing_latency_ms_mean,
            self.config.processing_latency_ms_jitter
        )
        latency_ms = max(0, latency_ms)  # Ensure non-negative
        
        # For now, we don't simulate packet drops at the publisher level
        # This would typically be handled at the network/transport level
        should_drop = False
        
        return latency_ms, should_drop
    
    def publish_detection(self, detection_json: Dict) -> bool:
        """
        Publish detection message with simulated latency and error handling.
        
        Args:
            detection_json: Detection message to publish
            
        Returns:
            True if publish successful, False otherwise
        """
        if self.shutdown_requested:
            logger.debug("Publish aborted due to shutdown request")
            return False
        
        self.stats['publish_attempts'] += 1
        
        try:
            with error_context("MQTT message publish", logger, reraise=False):
                # Simulate network effects
                latency_ms, should_drop = self._simulate_network_effects()
                
                if should_drop:
                    logger.debug("Simulating packet drop")
                    self.stats['publish_failures'] += 1
                    return False
                
                # Update processing latency in the message
                if "edge" in detection_json:
                    detection_json["edge"]["processing_latency_ms"] = round(latency_ms, 1)
                
                # Convert to JSON string with error handling
                try:
                    message_json = json.dumps(detection_json, separators=(',', ':'))
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to serialize detection message to JSON: {e}")
                    self.stats['publish_failures'] += 1
                    return False
                
                # Validate message size
                message_size = len(message_json.encode('utf-8'))
                if message_size > 256 * 1024:  # 256KB limit
                    logger.warning(f"Message size ({message_size} bytes) exceeds recommended limit")
                
                # Handle offline mode
                if self.config.offline_mode:
                    print(message_json)
                    return True
                
                # Ensure we're connected
                if not self.connected:
                    try:
                        if not self.connect():
                            logger.error("Cannot publish: failed to connect to MQTT broker")
                            self.stats['publish_failures'] += 1
                            return False
                    except MQTTError as e:
                        logger.error(f"Connection failed during publish: {e}")
                        self.stats['publish_failures'] += 1
                        return False
                
                # Validate topic
                if not self.config.mqtt_topic or not self.config.mqtt_topic.strip():
                    logger.error("Cannot publish: MQTT topic is empty")
                    self.stats['publish_failures'] += 1
                    return False
                
                # Publish message
                result = self.client.publish(
                    topic=self.config.mqtt_topic,
                    payload=message_json,
                    qos=self.config.mqtt_qos,
                    retain=self.config.retain
                )
                
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    logger.debug(f"Published detection message to {self.config.mqtt_topic} "
                               f"(size: {message_size} bytes, QoS: {self.config.mqtt_qos})")
                    return True
                else:
                    error_messages = {
                        mqtt.MQTT_ERR_NO_CONN: "Not connected to broker",
                        mqtt.MQTT_ERR_QUEUE_SIZE: "Message queue is full",
                        mqtt.MQTT_ERR_PAYLOAD_SIZE: "Payload too large",
                        mqtt.MQTT_ERR_MALFORMED_UTF8: "Topic contains malformed UTF-8",
                        mqtt.MQTT_ERR_INVAL: "Invalid input parameters"
                    }
                    error_msg = error_messages.get(result.rc, f"Unknown error code {result.rc}")
                    logger.error(f"Failed to publish message: {error_msg}")
                    self.stats['publish_failures'] += 1
                    return False
                    
        except Exception as e:
            logger.error(f"Unexpected error publishing detection message: {e}")
            self.stats['publish_failures'] += 1
            return False
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get publisher statistics.
        
        Returns:
            Dictionary containing connection and publish statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset publisher statistics."""
        self.stats = {
            'connection_attempts': 0,
            'successful_connections': 0,
            'connection_failures': 0,
            'publish_attempts': 0,
            'publish_successes': 0,
            'publish_failures': 0,
            'disconnections': 0
        }
    
    def is_healthy(self) -> bool:
        """
        Check if the publisher is in a healthy state.
        
        Returns:
            True if publisher is healthy, False otherwise
        """
        if self.config.offline_mode:
            return True
        
        if not self.client:
            return False
        
        if self.shutdown_requested:
            return False
        
        # Check connection health
        if not self.connected:
            return False
        
        # Check error rates
        total_attempts = self.stats['publish_attempts']
        if total_attempts > 10:  # Only check after some attempts
            failure_rate = self.stats['publish_failures'] / total_attempts
            if failure_rate > 0.5:  # More than 50% failure rate
                logger.warning(f"High publish failure rate: {failure_rate:.1%}")
                return False
        
        return True
    
    def __enter__(self):
        """Context manager entry."""
        if not self.config.offline_mode:
            try:
                if not self.connect():
                    raise MQTTError("Failed to connect to MQTT broker in context manager")
            except Exception as e:
                logger.error(f"Context manager entry failed: {e}")
                raise
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with error handling."""
        try:
            self.disconnect()
        except Exception as e:
            logger.error(f"Error during context manager exit: {e}")
            # Don't suppress the original exception
            return False