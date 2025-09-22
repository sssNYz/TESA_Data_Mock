"""
Unit tests for MQTT publisher.
"""

import json
import time
import pytest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import paho.mqtt.client as mqtt

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.mqtt_publisher import MQTTPublisher


class TestMQTTPublisher:
    """Test cases for MQTTPublisher class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SimulatorConfig(
            mqtt_host="test.broker.com",
            mqtt_port=1883,
            mqtt_topic="test/topic",
            mqtt_qos=1,
            retain=True,
            client_id="test_client",
            offline_mode=False,
            processing_latency_ms_mean=50.0,
            processing_latency_ms_jitter=10.0
        )
        
        self.rng = np.random.default_rng(42)  # Fixed seed for reproducible tests
        
        self.sample_detection = {
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
    
    @patch('paho.mqtt.client.Client')
    def test_mqtt_publisher_initialization(self, mock_client_class):
        """Test MQTT publisher initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(self.config, self.rng)
        
        # Verify client creation
        mock_client_class.assert_called_once_with(mqtt.CallbackAPIVersion.VERSION2, client_id="test_client")
        
        # Verify callbacks are set
        assert mock_client.on_connect is not None
        assert mock_client.on_disconnect is not None
        assert mock_client.on_publish is not None
        
        # Verify initial state
        assert publisher.config == self.config
        assert publisher.rng == self.rng
        assert not publisher.connected
        assert publisher.connection_attempts == 0
    
    def test_offline_mode_initialization(self):
        """Test initialization in offline mode."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config)
        
        assert publisher.client is None
        assert not publisher.connected
    
    @patch('paho.mqtt.client.Client')
    def test_client_id_generation(self, mock_client_class):
        """Test automatic client ID generation when not specified."""
        config = SimulatorConfig(client_id="", offline_mode=False)
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        with patch('time.time', return_value=1234567890):
            publisher = MQTTPublisher(config)
        
        mock_client_class.assert_called_once_with(mqtt.CallbackAPIVersion.VERSION2, client_id="drone_sim_1234567890")
    
    @patch('paho.mqtt.client.Client')
    def test_connection_success(self, mock_client_class):
        """Test successful MQTT connection."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(self.config, self.rng)
        
        # Mock successful connection
        def mock_connect(*args):
            # Simulate successful connection callback
            publisher._on_connect(mock_client, None, None, 0)
        
        mock_client.connect.side_effect = mock_connect
        
        result = publisher.connect()
        
        assert result is True
        assert publisher.connected is True
        # connection_attempts is reset to 0 on successful connection
        assert publisher.connection_attempts == 0
        mock_client.connect.assert_called_once_with("test.broker.com", 1883, 60)
        mock_client.loop_start.assert_called_once()
    
    @patch('paho.mqtt.client.Client')
    def test_connection_failure_with_retry(self, mock_client_class):
        """Test connection failure with retry logic."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.max_connection_attempts = 2  # Reduce for faster test
        
        # Mock connection failure
        mock_client.connect.side_effect = Exception("Connection failed")
        
        with patch('time.sleep'):  # Speed up test by mocking sleep
            result = publisher.connect()
        
        assert result is False
        assert not publisher.connected
        assert publisher.connection_attempts == 2
        assert mock_client.connect.call_count == 2
    
    @patch('paho.mqtt.client.Client')
    def test_connection_timeout(self, mock_client_class):
        """Test connection timeout handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.max_connection_attempts = 1
        
        # Mock connection that doesn't call callback (timeout scenario)
        mock_client.connect.return_value = None
        
        with patch('time.sleep'):
            result = publisher.connect()
        
        assert result is False
        assert not publisher.connected
    
    @patch('paho.mqtt.client.Client')
    def test_disconnect(self, mock_client_class):
        """Test graceful disconnection."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = True
        
        publisher.disconnect()
        
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
        assert not publisher.connected
    
    @patch('paho.mqtt.client.Client')
    def test_disconnect_error_handling(self, mock_client_class):
        """Test disconnect error handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.disconnect.side_effect = Exception("Disconnect error")
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = True
        
        # Should not raise exception
        publisher.disconnect()
        assert not publisher.connected
    
    def test_network_effects_simulation(self):
        """Test network latency simulation."""
        publisher = MQTTPublisher(self.config, self.rng)
        
        latency_ms, should_drop = publisher._simulate_network_effects()
        
        # Check that latency is reasonable (within expected range)
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0  # Should be non-negative
        assert isinstance(should_drop, bool)
        assert should_drop is False  # Currently always False
    
    def test_offline_mode_publish(self):
        """Test publishing in offline mode."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        
        with patch('builtins.print') as mock_print:
            result = publisher.publish_detection(self.sample_detection)
        
        assert result is True
        mock_print.assert_called_once()
        
        # Verify JSON was printed
        printed_json = mock_print.call_args[0][0]
        parsed = json.loads(printed_json)
        assert parsed["frame_id"] == 12345
    
    @patch('paho.mqtt.client.Client')
    def test_successful_publish(self, mock_client_class):
        """Test successful message publishing."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful publish
        mock_result = Mock()
        mock_result.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client.publish.return_value = mock_result
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = True  # Simulate connected state
        
        result = publisher.publish_detection(self.sample_detection)
        
        assert result is True
        mock_client.publish.assert_called_once()
        
        # Verify publish parameters
        call_args = mock_client.publish.call_args
        assert call_args[1]['topic'] == "test/topic"
        assert call_args[1]['qos'] == 1
        assert call_args[1]['retain'] is True
        
        # Verify message content
        published_json = call_args[1]['payload']
        parsed = json.loads(published_json)
        assert parsed["frame_id"] == 12345
        assert "processing_latency_ms" in parsed["edge"]
    
    @patch('paho.mqtt.client.Client')
    def test_publish_failure(self, mock_client_class):
        """Test publish failure handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock publish failure
        mock_result = Mock()
        mock_result.rc = mqtt.MQTT_ERR_NO_CONN
        mock_client.publish.return_value = mock_result
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = True
        
        result = publisher.publish_detection(self.sample_detection)
        
        assert result is False
    
    @patch('paho.mqtt.client.Client')
    def test_publish_with_connection_retry(self, mock_client_class):
        """Test publishing with automatic connection retry."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = False  # Start disconnected
        
        # Mock successful connection on retry
        def mock_connect(*args):
            publisher.connected = True
            publisher._on_connect(mock_client, None, None, 0)
        
        mock_client.connect.side_effect = mock_connect
        
        # Mock successful publish
        mock_result = Mock()
        mock_result.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client.publish.return_value = mock_result
        
        result = publisher.publish_detection(self.sample_detection)
        
        assert result is True
        mock_client.connect.assert_called_once()
        mock_client.publish.assert_called_once()
    
    @patch('paho.mqtt.client.Client')
    def test_publish_connection_failure(self, mock_client_class):
        """Test publish when connection cannot be established."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.connect.side_effect = Exception("Connection failed")
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = False
        publisher.max_connection_attempts = 1
        
        with patch('time.sleep'):
            result = publisher.publish_detection(self.sample_detection)
        
        assert result is False
    
    @patch('paho.mqtt.client.Client')
    def test_publish_exception_handling(self, mock_client_class):
        """Test exception handling during publish."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.publish.side_effect = Exception("Publish error")
        
        publisher = MQTTPublisher(self.config, self.rng)
        publisher.connected = True
        
        result = publisher.publish_detection(self.sample_detection)
        
        assert result is False
    
    def test_context_manager_offline(self):
        """Test context manager in offline mode."""
        config = SimulatorConfig(offline_mode=True)
        
        with MQTTPublisher(config, self.rng) as publisher:
            assert publisher is not None
    
    @patch('paho.mqtt.client.Client')
    def test_context_manager_online(self, mock_client_class):
        """Test context manager with MQTT connection."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Create publisher first to get reference
        publisher = MQTTPublisher(self.config, self.rng)
        
        def mock_connect(*args):
            publisher.connected = True
            publisher._on_connect(mock_client, None, None, 0)
        
        mock_client.connect.side_effect = mock_connect
        
        with publisher:
            assert publisher.connected is True
        
        # Verify disconnect was called
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
    
    def test_callback_on_connect_success(self):
        """Test on_connect callback for successful connection."""
        config = SimulatorConfig(offline_mode=True)  # Avoid actual MQTT setup
        publisher = MQTTPublisher(config, self.rng)
        
        publisher._on_connect(None, None, None, 0)
        
        assert publisher.connected is True
        assert publisher.connection_attempts == 0
    
    def test_callback_on_connect_failure(self):
        """Test on_connect callback for failed connection."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        
        publisher._on_connect(None, None, None, 1)  # Non-zero return code
        
        assert publisher.connected is False
    
    def test_callback_on_disconnect(self):
        """Test on_disconnect callback."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        publisher.connected = True
        
        publisher._on_disconnect(None, None, 0)  # Clean disconnect
        
        assert publisher.connected is False
    
    def test_callback_on_disconnect_unexpected(self):
        """Test on_disconnect callback for unexpected disconnection."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        publisher.connected = True
        
        publisher._on_disconnect(None, None, 1)  # Unexpected disconnect
        
        assert publisher.connected is False
    
    def test_callback_on_publish(self):
        """Test on_publish callback."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        
        # Should not raise exception
        publisher._on_publish(None, None, 123)
    
    def test_latency_update_in_message(self):
        """Test that processing latency is updated in published messages."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        
        detection_copy = self.sample_detection.copy()
        original_latency = detection_copy["edge"]["processing_latency_ms"]
        
        with patch('builtins.print') as mock_print:
            publisher.publish_detection(detection_copy)
        
        # Verify latency was updated
        printed_json = mock_print.call_args[0][0]
        parsed = json.loads(printed_json)
        new_latency = parsed["edge"]["processing_latency_ms"]
        
        # Should be different from original (due to simulation)
        assert new_latency != original_latency
        assert isinstance(new_latency, (int, float))
        assert new_latency >= 0
    
    def test_json_serialization(self):
        """Test JSON serialization of detection messages."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config, self.rng)
        
        with patch('builtins.print') as mock_print:
            publisher.publish_detection(self.sample_detection)
        
        printed_json = mock_print.call_args[0][0]
        
        # Verify it's valid JSON
        parsed = json.loads(printed_json)
        assert parsed["timestamp_utc"] == "2025-09-21T08:23:12.123Z"
        assert parsed["frame_id"] == 12345
        assert len(parsed["detections"]) == 1
        assert parsed["detections"][0]["class"] == "drone"
        
        # Verify compact JSON format (no extra spaces)
        assert ", " not in printed_json  # Should use separators=(',', ':')