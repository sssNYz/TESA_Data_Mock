"""
Tests for error handling utilities and custom exceptions.
"""

import pytest
import logging
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from drone_detection_simulator.error_handling import (
    SimulatorError, ConfigurationError, CameraError, MotionError,
    DetectionError, MQTTError, SimulationError, NetworkError, ValidationError,
    retry_on_exception, safe_execute, error_context, validate_parameter,
    handle_configuration_error, handle_mqtt_error, handle_network_error,
    ErrorRecovery, create_error_summary
)


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_simulator_error_basic(self):
        """Test basic SimulatorError functionality."""
        error = SimulatorError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}
    
    def test_simulator_error_with_code_and_context(self):
        """Test SimulatorError with error code and context."""
        context = {"param": "value", "number": 42}
        error = SimulatorError("Test error", error_code="TEST_ERROR", context=context)
        
        assert str(error) == "[TEST_ERROR] Test error"
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.context == context
    
    def test_specific_exception_types(self):
        """Test that specific exception types inherit properly."""
        exceptions = [
            ConfigurationError("Config error"),
            CameraError("Camera error"),
            MotionError("Motion error"),
            DetectionError("Detection error"),
            MQTTError("MQTT error"),
            SimulationError("Simulation error"),
            NetworkError("Network error"),
            ValidationError("Validation error")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, SimulatorError)
            assert isinstance(exc, Exception)


class TestRetryDecorator:
    """Test retry_on_exception decorator."""
    
    def test_retry_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        mock_func = Mock(return_value="success")
        
        @retry_on_exception(max_attempts=3)
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_success_after_failures(self):
        """Test successful execution after some failures."""
        mock_func = Mock(side_effect=[ValueError("error1"), ValueError("error2"), "success"])
        
        @retry_on_exception(max_attempts=3, delay_seconds=0.01, exceptions=(ValueError,))
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        mock_func = Mock(side_effect=ValueError("persistent error"))
        
        @retry_on_exception(max_attempts=2, delay_seconds=0.01, exceptions=(ValueError,))
        def test_func():
            return mock_func()
        
        with pytest.raises(ValueError, match="persistent error"):
            test_func()
        
        assert mock_func.call_count == 2
    
    def test_retry_non_retryable_exception(self):
        """Test that non-retryable exceptions are not retried."""
        mock_func = Mock(side_effect=TypeError("type error"))
        
        @retry_on_exception(max_attempts=3, delay_seconds=0.01, exceptions=(ValueError,))
        def test_func():
            return mock_func()
        
        with pytest.raises(TypeError, match="type error"):
            test_func()
        
        assert mock_func.call_count == 1
    
    def test_retry_with_logger(self, caplog):
        """Test retry decorator with logging."""
        mock_func = Mock(side_effect=[ValueError("error"), "success"])
        logger = logging.getLogger("test")
        
        @retry_on_exception(max_attempts=2, delay_seconds=0.01, exceptions=(ValueError,), logger=logger)
        def test_func():
            return mock_func()
        
        with caplog.at_level(logging.WARNING):
            result = test_func()
        
        assert result == "success"
        assert "Attempt 1/2 failed" in caplog.text
        assert "Retrying in 0.0s" in caplog.text


class TestSafeExecute:
    """Test safe_execute utility function."""
    
    def test_safe_execute_success(self):
        """Test successful execution."""
        def test_func():
            return "success"
        
        result = safe_execute(test_func)
        assert result == "success"
    
    def test_safe_execute_with_exception(self):
        """Test execution with exception."""
        def test_func():
            raise ValueError("test error")
        
        result = safe_execute(test_func, default_return="default")
        assert result == "default"
    
    def test_safe_execute_with_logger(self, caplog):
        """Test safe_execute with logging."""
        def test_func():
            raise ValueError("test error")
        
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.ERROR):
            result = safe_execute(test_func, default_return="default", logger=logger, context="test operation")
        
        assert result == "default"
        assert "Error in test operation: test error" in caplog.text
    
    def test_safe_execute_reraise(self):
        """Test safe_execute with reraise option."""
        def test_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError, match="test error"):
            safe_execute(test_func, reraise=True)


class TestErrorContext:
    """Test error_context context manager."""
    
    def test_error_context_success(self, caplog):
        """Test successful operation with error context."""
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.DEBUG):
            with error_context("test operation", logger):
                pass
        
        assert "Starting operation: test operation" in caplog.text
        assert "Completed operation: test operation" in caplog.text
    
    def test_error_context_with_exception(self, caplog):
        """Test error context with exception."""
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError, match="test error"):
                with error_context("test operation", logger):
                    raise ValueError("test error")
        
        assert "Error in operation 'test operation': test error" in caplog.text
    
    def test_error_context_with_cleanup(self):
        """Test error context with cleanup function."""
        cleanup_mock = Mock()
        
        with pytest.raises(ValueError):
            with error_context("test operation", cleanup_func=cleanup_mock):
                raise ValueError("test error")
        
        cleanup_mock.assert_called_once()
    
    def test_error_context_cleanup_error(self, caplog):
        """Test error context when cleanup function fails."""
        def failing_cleanup():
            raise RuntimeError("cleanup error")
        
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(ValueError):
                with error_context("test operation", logger, cleanup_func=failing_cleanup):
                    raise ValueError("test error")
        
        assert "Error during cleanup: cleanup error" in caplog.text


class TestValidateParameter:
    """Test validate_parameter utility function."""
    
    def test_validate_parameter_success(self):
        """Test successful parameter validation."""
        validate_parameter(5, "test_param", lambda x: x > 0, "must be positive")
        # Should not raise any exception
    
    def test_validate_parameter_failure(self):
        """Test parameter validation failure."""
        with pytest.raises(ValidationError, match="Invalid value for parameter 'test_param'"):
            validate_parameter(-5, "test_param", lambda x: x > 0, "must be positive")
    
    def test_validate_parameter_validator_exception(self):
        """Test when validator itself raises an exception."""
        def failing_validator(x):
            raise TypeError("validator error")
        
        with pytest.raises(ValidationError, match="Validation failed for parameter 'test_param'"):
            validate_parameter(5, "test_param", failing_validator, "must be valid")
    
    def test_validate_parameter_custom_exception_type(self):
        """Test validation with custom exception type."""
        with pytest.raises(ConfigurationError):
            validate_parameter(
                -5, "test_param", lambda x: x > 0, "must be positive", 
                error_type=ConfigurationError
            )


class TestErrorHandlers:
    """Test error handler utility functions."""
    
    def test_handle_configuration_error(self, caplog):
        """Test configuration error handler."""
        original_error = ValueError("original error")
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.ERROR):
            result = handle_configuration_error(original_error, "test config", logger)
        
        assert isinstance(result, ConfigurationError)
        assert "Configuration error in test config: original error" in str(result)
        assert "Configuration error in test config" in caplog.text
    
    def test_handle_configuration_error_already_config_error(self):
        """Test handling when error is already a ConfigurationError."""
        original_error = ConfigurationError("already config error")
        result = handle_configuration_error(original_error)
        
        assert result is original_error
    
    def test_handle_mqtt_error(self, caplog):
        """Test MQTT error handler."""
        original_error = ConnectionError("connection failed")
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.ERROR):
            result = handle_mqtt_error(original_error, "publish message", logger)
        
        assert isinstance(result, MQTTError)
        assert "MQTT error during publish message: connection failed" in str(result)
        assert result.error_code == "MQTT_ERROR"
    
    def test_handle_network_error(self, caplog):
        """Test network error handler."""
        original_error = OSError("network error")
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.ERROR):
            result = handle_network_error(original_error, "localhost", 1883, logger)
        
        assert isinstance(result, NetworkError)
        assert "Network error connecting to localhost:1883: network error" in str(result)
        assert result.context["host"] == "localhost"
        assert result.context["port"] == 1883


class TestErrorRecovery:
    """Test ErrorRecovery utility class."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        # Test basic exponential backoff
        assert ErrorRecovery.exponential_backoff(0) == 1.0
        assert ErrorRecovery.exponential_backoff(1) == 2.0
        assert ErrorRecovery.exponential_backoff(2) == 4.0
        assert ErrorRecovery.exponential_backoff(3) == 8.0
        
        # Test with custom parameters
        assert ErrorRecovery.exponential_backoff(0, base_delay=0.5, multiplier=3.0) == 0.5
        assert ErrorRecovery.exponential_backoff(1, base_delay=0.5, multiplier=3.0) == 1.5
        
        # Test max delay limit
        assert ErrorRecovery.exponential_backoff(10, max_delay=5.0) == 5.0
    
    def test_should_retry(self):
        """Test retry decision logic."""
        # Test retryable errors
        assert ErrorRecovery.should_retry(ConnectionError("error"), 0, 3) == True
        assert ErrorRecovery.should_retry(NetworkError("error"), 1, 3) == True
        assert ErrorRecovery.should_retry(MQTTError("error"), 2, 5) == True
        
        # Test non-retryable errors
        assert ErrorRecovery.should_retry(ValueError("error"), 0, 3) == False
        assert ErrorRecovery.should_retry(TypeError("error"), 1, 3) == False
        
        # Test max attempts reached
        assert ErrorRecovery.should_retry(ConnectionError("error"), 2, 3) == False
        assert ErrorRecovery.should_retry(NetworkError("error"), 4, 5) == False


class TestCreateErrorSummary:
    """Test create_error_summary utility function."""
    
    def test_create_error_summary_with_exceptions(self):
        """Test error summary creation with exception objects."""
        errors = [
            ValueError("value error 1"),
            ValueError("value error 2"),
            TypeError("type error"),
            ConnectionError("connection error")
        ]
        
        summary = create_error_summary(errors)
        
        assert summary["total_errors"] == 4
        assert summary["error_counts"]["ValueError"] == 2
        assert summary["error_counts"]["TypeError"] == 1
        assert summary["error_counts"]["ConnectionError"] == 1
        
        assert len(summary["error_details"]) == 4
        assert summary["error_details"][0]["type"] == "ValueError"
        assert summary["error_details"][0]["message"] == "value error 1"
    
    def test_create_error_summary_with_strings(self):
        """Test error summary creation with string messages."""
        errors = ["error message 1", "error message 2"]
        
        summary = create_error_summary(errors)
        
        assert summary["total_errors"] == 2
        assert summary["error_counts"]["Unknown"] == 2
        
        assert len(summary["error_details"]) == 2
        assert summary["error_details"][0]["type"] == "Unknown"
        assert summary["error_details"][0]["message"] == "error message 1"
    
    def test_create_error_summary_empty(self):
        """Test error summary creation with empty error list."""
        summary = create_error_summary([])
        
        assert summary["total_errors"] == 0
        assert summary["error_counts"] == {}
        assert summary["error_details"] == []


if __name__ == "__main__":
    pytest.main([__file__])