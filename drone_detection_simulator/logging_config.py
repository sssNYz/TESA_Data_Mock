"""
Logging configuration and utilities for the drone detection simulator.

This module provides structured logging setup with configurable levels,
formatters, and handlers for different components of the simulator.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum


class LogLevel(Enum):
    """Enumeration of available log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class SimulatorLogger:
    """
    Centralized logging configuration for the drone detection simulator.
    
    Provides structured logging with configurable levels, formatters,
    and output destinations for different components.
    """
    
    _initialized = False
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def setup_logging(
        cls,
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        console_output: bool = True,
        verbose: bool = False,
        quiet: bool = False,
        component_levels: Optional[Dict[str, LogLevel]] = None
    ) -> None:
        """
        Setup logging configuration for the simulator.
        
        Args:
            level: Default logging level
            log_file: Optional file path for log output
            console_output: Whether to output to console
            verbose: Enable verbose logging (DEBUG level)
            quiet: Suppress all output except errors
            component_levels: Per-component logging levels
        """
        if cls._initialized:
            return
        
        # Determine effective log level
        if quiet:
            effective_level = LogLevel.ERROR
        elif verbose:
            effective_level = LogLevel.DEBUG
        else:
            effective_level = level
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(effective_level.value)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler if requested
        if console_output and not quiet:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(effective_level.value)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add error console handler for quiet mode
        if quiet:
            error_handler = logging.StreamHandler(sys.stderr)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)
        
        # Add file handler if requested
        if log_file:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
                file_handler.setLevel(logging.DEBUG)  # Always debug level for files
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                # Don't fail if we can't create log file, just warn
                logging.warning(f"Failed to create log file {log_file}: {e}")
        
        # Set component-specific levels
        if component_levels:
            for component_name, component_level in component_levels.items():
                component_logger = logging.getLogger(component_name)
                component_logger.setLevel(component_level.value)
        
        # Set specific levels for noisy third-party libraries
        logging.getLogger('paho.mqtt.client').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger instance for a specific component.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def log_configuration(cls, config: Dict[str, Any], logger_name: str = "config") -> None:
        """
        Log configuration parameters in a structured way.
        
        Args:
            config: Configuration dictionary to log
            logger_name: Name of logger to use
        """
        logger = cls.get_logger(logger_name)
        
        logger.info("Configuration loaded:")
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
    
    @classmethod
    def log_statistics(cls, stats: Dict[str, Any], logger_name: str = "stats") -> None:
        """
        Log simulation statistics in a structured way.
        
        Args:
            stats: Statistics dictionary to log
            logger_name: Name of logger to use
        """
        logger = cls.get_logger(logger_name)
        
        logger.info("Simulation Statistics:")
        for category, values in stats.items():
            if isinstance(values, dict):
                logger.info(f"  {category}:")
                for key, value in values.items():
                    if isinstance(value, float):
                        logger.info(f"    {key}: {value:.3f}")
                    else:
                        logger.info(f"    {key}: {value}")
            else:
                if isinstance(values, float):
                    logger.info(f"  {category}: {values:.3f}")
                else:
                    logger.info(f"  {category}: {values}")
    
    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration (mainly for testing)."""
        cls._initialized = False
        cls._loggers.clear()
        
        # Clear all handlers from root logger
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)


def setup_component_logging(
    component_name: str,
    level: LogLevel = LogLevel.INFO
) -> logging.Logger:
    """
    Setup logging for a specific component.
    
    Args:
        component_name: Name of the component
        level: Logging level for this component
        
    Returns:
        Configured logger instance
    """
    logger = SimulatorLogger.get_logger(component_name)
    logger.setLevel(level.value)
    return logger


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    context: str = "",
    include_traceback: bool = True
) -> None:
    """
    Log an exception with context and optional traceback.
    
    Args:
        logger: Logger instance to use
        exception: Exception to log
        context: Additional context information
        include_traceback: Whether to include full traceback
    """
    if context:
        message = f"{context}: {type(exception).__name__}: {exception}"
    else:
        message = f"{type(exception).__name__}: {exception}"
    
    if include_traceback:
        logger.exception(message)
    else:
        logger.error(message)


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log performance metrics for operations.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        additional_metrics: Additional metrics to log
    """
    message = f"Performance: {operation} took {duration_ms:.2f}ms"
    
    if additional_metrics:
        metrics_str = ", ".join([f"{k}={v}" for k, v in additional_metrics.items()])
        message += f" ({metrics_str})"
    
    logger.debug(message)