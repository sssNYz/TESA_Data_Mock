"""
Error handling utilities and custom exceptions for the drone detection simulator.

This module provides custom exception classes, error recovery mechanisms,
and utilities for graceful error handling throughout the simulator.
"""

import logging
import functools
import time
from typing import Optional, Callable, Any, Type, Union, Dict
from contextlib import contextmanager


class SimulatorError(Exception):
    """Base exception class for simulator-specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(SimulatorError):
    """Exception raised for configuration-related errors."""
    pass


class CameraError(SimulatorError):
    """Exception raised for camera-related errors."""
    pass


class MotionError(SimulatorError):
    """Exception raised for motion generation errors."""
    pass


class DetectionError(SimulatorError):
    """Exception raised for detection generation errors."""
    pass


class MQTTError(SimulatorError):
    """Exception raised for MQTT-related errors."""
    pass


class SimulationError(SimulatorError):
    """Exception raised for general simulation errors."""
    pass


class NetworkError(SimulatorError):
    """Exception raised for network-related errors."""
    pass


class ValidationError(SimulatorError):
    """Exception raised for data validation errors."""
    pass


def retry_on_exception(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """
    Decorator to retry function calls on specific exceptions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exception types to retry on
        logger: Optional logger for retry messages
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            current_delay = delay_seconds
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:  # Don't log on final attempt
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= backoff_multiplier
                    else:
                        if logger:
                            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            # Re-raise the last exception if all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
    context: str = "",
    reraise: bool = False
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_return: Value to return on error
        logger: Optional logger for error messages
        context: Context description for error messages
        reraise: Whether to re-raise exceptions after logging
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        if logger:
            message = f"Error in {context}: {e}" if context else f"Error: {e}"
            logger.error(message)
        
        if reraise:
            raise
        
        return default_return


@contextmanager
def error_context(
    operation: str,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True,
    cleanup_func: Optional[Callable] = None
):
    """
    Context manager for error handling with cleanup.
    
    Args:
        operation: Description of the operation
        logger: Optional logger for error messages
        reraise: Whether to re-raise exceptions
        cleanup_func: Optional cleanup function to call on error
    """
    try:
        if logger:
            logger.debug(f"Starting operation: {operation}")
        yield
        if logger:
            logger.debug(f"Completed operation: {operation}")
    except Exception as e:
        if logger:
            logger.error(f"Error in operation '{operation}': {e}")
        
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as cleanup_error:
                if logger:
                    logger.error(f"Error during cleanup: {cleanup_error}")
        
        if reraise:
            raise


def validate_parameter(
    value: Any,
    param_name: str,
    validator: Callable[[Any], bool],
    error_message: str,
    error_type: Type[Exception] = ValidationError
) -> None:
    """
    Validate a parameter value and raise an exception if invalid.
    
    Args:
        value: Value to validate
        param_name: Name of the parameter
        validator: Function that returns True if value is valid
        error_message: Error message to use if validation fails
        error_type: Type of exception to raise
    """
    try:
        if not validator(value):
            raise error_type(
                f"Invalid value for parameter '{param_name}': {error_message}",
                error_code="VALIDATION_ERROR",
                context={"parameter": param_name, "value": value}
            )
    except Exception as e:
        if isinstance(e, error_type):
            raise
        # If validator itself raises an exception, wrap it
        raise error_type(
            f"Validation failed for parameter '{param_name}': {e}",
            error_code="VALIDATION_ERROR",
            context={"parameter": param_name, "value": value}
        )


def handle_configuration_error(
    error: Exception,
    config_source: str = "configuration",
    logger: Optional[logging.Logger] = None
) -> ConfigurationError:
    """
    Handle and wrap configuration-related errors.
    
    Args:
        error: Original error
        config_source: Source of the configuration (file, dict, etc.)
        logger: Optional logger for error messages
        
    Returns:
        ConfigurationError with appropriate context
    """
    if isinstance(error, ConfigurationError):
        return error
    
    error_message = f"Configuration error in {config_source}: {error}"
    
    if logger:
        logger.error(error_message)
    
    return ConfigurationError(
        error_message,
        error_code="CONFIG_ERROR",
        context={"source": config_source, "original_error": str(error)}
    )


def handle_mqtt_error(
    error: Exception,
    operation: str = "MQTT operation",
    logger: Optional[logging.Logger] = None
) -> MQTTError:
    """
    Handle and wrap MQTT-related errors.
    
    Args:
        error: Original error
        operation: Description of the MQTT operation
        logger: Optional logger for error messages
        
    Returns:
        MQTTError with appropriate context
    """
    if isinstance(error, MQTTError):
        return error
    
    error_message = f"MQTT error during {operation}: {error}"
    
    if logger:
        logger.error(error_message)
    
    return MQTTError(
        error_message,
        error_code="MQTT_ERROR",
        context={"operation": operation, "original_error": str(error)}
    )


def handle_network_error(
    error: Exception,
    host: str = "unknown",
    port: int = 0,
    logger: Optional[logging.Logger] = None
) -> NetworkError:
    """
    Handle and wrap network-related errors.
    
    Args:
        error: Original error
        host: Network host
        port: Network port
        logger: Optional logger for error messages
        
    Returns:
        NetworkError with appropriate context
    """
    if isinstance(error, NetworkError):
        return error
    
    error_message = f"Network error connecting to {host}:{port}: {error}"
    
    if logger:
        logger.error(error_message)
    
    return NetworkError(
        error_message,
        error_code="NETWORK_ERROR",
        context={"host": host, "port": port, "original_error": str(error)}
    )


class ErrorRecovery:
    """
    Utility class for error recovery strategies.
    """
    
    @staticmethod
    def exponential_backoff(
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            multiplier: Backoff multiplier
            
        Returns:
            Delay in seconds
        """
        delay = base_delay * (multiplier ** attempt)
        return min(delay, max_delay)
    
    @staticmethod
    def should_retry(
        error: Exception,
        attempt: int,
        max_attempts: int,
        retryable_errors: tuple = (ConnectionError, TimeoutError, NetworkError, MQTTError)
    ) -> bool:
        """
        Determine if an operation should be retried.
        
        Args:
            error: Exception that occurred
            attempt: Current attempt number (0-based)
            max_attempts: Maximum number of attempts
            retryable_errors: Tuple of retryable error types
            
        Returns:
            True if operation should be retried
        """
        if attempt >= max_attempts - 1:
            return False
        
        return isinstance(error, retryable_errors)


def create_error_summary(errors: list) -> Dict[str, Any]:
    """
    Create a summary of multiple errors.
    
    Args:
        errors: List of exceptions or error messages
        
    Returns:
        Dictionary containing error summary
    """
    error_counts = {}
    error_details = []
    
    for error in errors:
        error_type = type(error).__name__ if isinstance(error, Exception) else "Unknown"
        error_message = str(error)
        
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
        error_details.append({
            "type": error_type,
            "message": error_message
        })
    
    return {
        "total_errors": len(errors),
        "error_counts": error_counts,
        "error_details": error_details
    }