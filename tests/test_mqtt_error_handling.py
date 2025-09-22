"""
Tests for MQTT publisher error handling and recovery.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
import paho.mqtt.client as mqtt

from drone_detection_simulator.config import SimulatorConfig
from drone_detection_simulator.mqtt_publisher import MQTTPublisher
from drone_detection_simulator.error_handling import MQTTError, NetworkError


class TestMQTTPublisherErrorHandling:
    """Test error handling in MQTT publisher."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SimulatorConfig(
            mqtt_host="test.broker.com",
            mqtt_port=1883,
            mqtt_topic="test/topic",
            offline_mode=False
        )
    
    @pytest.fixture
    def offline_config(self):
        """Create offline test configuration."""
        return SimulatorConfig(offline_mode=True)
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        config = SimulatorConfig(offline_mode=False)
        
        with patch('paho.mqtt.client.Client', side_effect=Exception("Client creation failed")):
            with pytest.raises(MQTTError, match="MQTT publisher initialization failed"):
                MQTTPublisher(config)
    
    def test_initialization_offline_mode(self):
        """Test initialization in offline mode."""
        config = SimulatorConfig(offline_mode=True)
        publisher = MQTTPublisher(config)
        
        assert publisher.client is None
        assert publisher.config.offline_mode == True
    
    @patch('paho.mqtt.client.Client')
    def test_connection_parameter_validation(self, mock_client_class, config):
        """Test connection parameter validation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        
        # Test empty host
        publisher.config.mqtt_host = ""
        with pytest.raises(MQTTError, match="MQTT host cannot be empty"):
            publisher.connect()
        
        # Test invalid port
        publisher.config.mqtt_host = "valid.host.com"
        publisher.config.mqtt_port = 0
        with pytest.raises(MQTTError, match="Invalid MQTT port"):
            publisher.connect()
        
        publisher.config.mqtt_port = 70000
        with pytest.raises(MQTTError, match="Invalid MQTT port"):
            publisher.connect()
    
    @patch('paho.mqtt.client.Client')
    def test_connection_timeout(self, mock_client_class, config):
        """Test connection timeout handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        
        # Simulate connection timeout (connected never becomes True)
        with pytest.raises(MQTTError, match="Connection timeout"):
            publisher.connect()
    
    @patch('paho.mqtt.client.Client')
    def test_connection_retry_logic(self, mock_client_class, config):
        """Test connection retry logic."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock connect to fail first few times, then succeed
        connect_attempts = []
        def mock_connect(host, port, keepalive):
            connect_attempts.append((host, port))
            if len(connect_attempts) < 3:
                raise ConnectionError("Connection failed")
            # On third attempt, simulate successful connection
            publisher.connected = True
        
        mock_client.connect.side_effect = mock_connect
        
        publisher = MQTTPublisher(config)
        
        # Should succeed after retries
        result = publisher.connect()
        assert result == True
        assert len(connect_attempts) == 3
    
    @patch('paho.mqtt.client.Client')
    def test_graceful_disconnect(self, mock_client_class, config):
        """Test graceful disconnect handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        
        # Test normal disconnect
        publisher.disconnect()
        
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()
        assert publisher.shutdown_requested == True
    
    @patch('paho.mqtt.client.Client')
    def test_disconnect_error_handling(self, mock_client_class, config, caplog):
        """Test disconnect error handling."""
        mock_client = Mock()
        mock_client.disconnect.side_effect = Exception("Disconnect failed")
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        
        # Should not raise exception, but log error
        publisher.disconnect()
        
        assert "Error during MQTT disconnect" in caplog.text
        assert publisher.connected == False
    
    @patch('paho.mqtt.client.Client')
    def test_publish_json_serialization_error(self, mock_client_class, config, caplog):
        """Test publish with JSON serialization error."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        
        # Create detection with non-serializable content
        detection = {"timestamp": object()}  # object() is not JSON serializable
        
        result = publisher.publish_detection(detection)
        
        assert result == False
        assert publisher.stats['publish_failures'] == 1
        assert "Failed to serialize detection message to JSON" in caplog.text
    
    @patch('paho.mqtt.client.Client')
    def test_publish_message_size_warning(self, mock_client_class, config, caplog):
        """Test publish with large message size warning."""
        mock_client = Mock()
        mock_client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        
        # Create large detection message
        large_data = "x" * (300 * 1024)  # 300KB
        detection = {"large_field": large_data}
        
        with caplog.at_level("WARNING"):
            result = publisher.publish_detection(detection)
        
        assert result == True
        assert "Message size" in caplog.text
        assert "exceeds recommended limit" in caplog.text
    
    @patch('paho.mqtt.client.Client')
    def test_publish_empty_topic_error(self, mock_client_class, config, caplog):
        """Test publish with empty topic."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        publisher.config.mqtt_topic = ""
        
        detection = {"test": "data"}
        result = publisher.publish_detection(detection)
        
        assert result == False
        assert publisher.stats['publish_failures'] == 1
        assert "MQTT topic is empty" in caplog.text
    
    @patch('paho.mqtt.client.Client')
    def test_publish_mqtt_error_codes(self, mock_client_class, config, caplog):
        """Test publish with various MQTT error codes."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        
        detection = {"test": "data"}
        
        # Test different error codes
        error_codes = [
            (mqtt.MQTT_ERR_NO_CONN, "Not connected to broker"),
            (mqtt.MQTT_ERR_QUEUE_SIZE, "Message queue is full"),
            (mqtt.MQTT_ERR_PAYLOAD_SIZE, "Payload too large"),
            (mqtt.MQTT_ERR_MALFORMED_UTF8, "Topic contains malformed UTF-8"),
            (mqtt.MQTT_ERR_INVAL, "Invalid input parameters"),
            (999, "Unknown error code 999")  # Unknown error code
        ]
        
        for error_code, expected_message in error_codes:
            mock_client.publish.return_value.rc = error_code
            
            with caplog.at_level("ERROR"):
                result = publisher.publish_detection(detection)
            
            assert result == False
            assert expected_message in caplog.text
            caplog.clear()
    
    @patch('paho.mqtt.client.Client')
    def test_publish_connection_recovery(self, mock_client_class, config):
        """Test publish with connection recovery."""
        mock_client = Mock()
        mock_client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = False  # Start disconnected
        
        # Mock successful reconnection
        def mock_connect():
            publisher.connected = True
            return True
        
        publisher.connect = Mock(side_effect=mock_connect)
        
        detection = {"test": "data"}
        result = publisher.publish_detection(detection)
        
        assert result == True
        publisher.connect.assert_called_once()
    
    @patch('paho.mqtt.client.Client')
    def test_publish_connection_recovery_failure(self, mock_client_class, config, caplog):
        """Test publish when connection recovery fails."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = False
        
        # Mock failed reconnection
        publisher.connect = Mock(return_value=False)
        
        detection = {"test": "data"}
        result = publisher.publish_detection(detection)
        
        assert result == False
        assert publisher.stats['publish_failures'] == 1
        assert "failed to connect to MQTT broker" in caplog.text
    
    @patch('paho.mqtt.client.Client')
    def test_publish_shutdown_requested(self, mock_client_class, config):
        """Test publish when shutdown is requested."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.shutdown_requested = True
        
        detection = {"test": "data"}
        result = publisher.publish_detection(detection)
        
        assert result == False
        # Should not increment failure count for shutdown
        assert publisher.stats['publish_failures'] == 0
    
    def test_publish_offline_mode(self, offline_config, capsys):
        """Test publish in offline mode."""
        publisher = MQTTPublisher(offline_config)
        
        detection = {"test": "data", "timestamp": "2023-01-01T00:00:00Z"}
        result = publisher.publish_detection(detection)
        
        assert result == True
        
        # Check that JSON was printed to stdout
        captured = capsys.readouterr()
        assert '"test":"data"' in captured.out
        assert '"timestamp":"2023-01-01T00:00:00Z"' in captured.out
    
    @patch('paho.mqtt.client.Client')
    def test_statistics_tracking(self, mock_client_class, config):
        """Test statistics tracking."""
        mock_client = Mock()
        mock_client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        publisher.connected = True
        
        detection = {"test": "data"}
        
        # Successful publish
        result = publisher.publish_detection(detection)
        assert result == True
        
        stats = publisher.get_statistics()
        assert stats['publish_attempts'] == 1
        assert stats['publish_successes'] == 1
        assert stats['publish_failures'] == 0
        
        # Failed publish
        mock_client.publish.return_value.rc = mqtt.MQTT_ERR_NO_CONN
        result = publisher.publish_detection(detection)
        assert result == False
        
        stats = publisher.get_statistics()
        assert stats['publish_attempts'] == 2
        assert stats['publish_successes'] == 1
        assert stats['publish_failures'] == 1
    
    @patch('paho.mqtt.client.Client')
    def test_health_check(self, mock_client_class, config):
        """Test health check functionality."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        
        # Offline mode should always be healthy
        publisher.config.offline_mode = True
        assert publisher.is_healthy() == True
        
        # Back to online mode
        publisher.config.offline_mode = False
        
        # No client should be unhealthy
        publisher.client = None
        assert publisher.is_healthy() == False
        
        # Shutdown requested should be unhealthy
        publisher.client = mock_client
        publisher.shutdown_requested = True
        assert publisher.is_healthy() == False
        
        # Not connected should be unhealthy
        publisher.shutdown_requested = False
        publisher.connected = False
        assert publisher.is_healthy() == False
        
        # High failure rate should be unhealthy
        publisher.connected = True
        publisher.stats['publish_attempts'] = 20
        publisher.stats['publish_failures'] = 15  # 75% failure rate
        assert publisher.is_healthy() == False
        
        # Good state should be healthy
        publisher.stats['publish_failures'] = 2  # 10% failure rate
        assert publisher.is_healthy() == True
    
    @patch('paho.mqtt.client.Client')
    def test_context_manager_success(self, mock_client_class, config):
        """Test context manager with successful operation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock successful connection
        def mock_connect():
            publisher.connected = True
            return True
        
        with patch.object(MQTTPublisher, 'connect', side_effect=mock_connect):
            with MQTTPublisher(config) as publisher:
                assert publisher.connected == True
    
    @patch('paho.mqtt.client.Client')
    def test_context_manager_connection_failure(self, mock_client_class, config):
        """Test context manager with connection failure."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        with patch.object(MQTTPublisher, 'connect', return_value=False):
            with pytest.raises(MQTTError, match="Failed to connect to MQTT broker"):
                with MQTTPublisher(config) as publisher:
                    pass
    
    def test_context_manager_offline_mode(self, offline_config):
        """Test context manager in offline mode."""
        with MQTTPublisher(offline_config) as publisher:
            assert publisher.config.offline_mode == True
            # Should work without any connection
    
    @patch('paho.mqtt.client.Client')
    def test_callback_error_handling(self, mock_client_class, config):
        """Test MQTT callback error handling."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        publisher = MQTTPublisher(config)
        
        # Test on_connect callback with different return codes
        publisher._on_connect(mock_client, None, None, 0)  # Success
        assert publisher.connected == True
        assert publisher.stats['successful_connections'] == 1
        
        publisher._on_connect(mock_client, None, None, 1)  # Failure
        assert publisher.connected == False
        assert publisher.stats['connection_failures'] == 1
        
        # Test on_disconnect callback
        publisher.connected = True
        publisher._on_disconnect(mock_client, None, 0)  # Clean disconnect
        assert publisher.connected == False
        
        publisher.connected = True
        publisher._on_disconnect(mock_client, None, 1)  # Unexpected disconnect
        assert publisher.connected == False
        assert publisher.stats['disconnections'] == 2
        
        # Test on_publish callback
        publisher._on_publish(mock_client, None, 12345)
        assert publisher.stats['publish_successes'] == 1


if __name__ == "__main__":
    pytest.main([__file__])