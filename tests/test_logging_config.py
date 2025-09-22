"""
Tests for logging configuration and utilities.
"""

import pytest
import logging
import tempfile
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from io import StringIO

from drone_detection_simulator.logging_config import (
    LogLevel, SimulatorLogger, setup_component_logging,
    log_exception, log_performance_metrics
)


class TestLogLevel:
    """Test LogLevel enumeration."""
    
    def test_log_level_values(self):
        """Test that LogLevel enum has correct values."""
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR
        assert LogLevel.CRITICAL.value == logging.CRITICAL


class TestSimulatorLogger:
    """Test SimulatorLogger class."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        SimulatorLogger.reset()
    
    def teardown_method(self):
        """Clean up after each test."""
        SimulatorLogger.reset()
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        SimulatorLogger.setup_logging(level=LogLevel.INFO)
        
        logger = SimulatorLogger.get_logger("test")
        assert logger.level <= logging.INFO
        assert SimulatorLogger._initialized == True
    
    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        SimulatorLogger.setup_logging(verbose=True)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_setup_logging_quiet(self):
        """Test quiet logging setup."""
        SimulatorLogger.setup_logging(quiet=True)
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.ERROR
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            SimulatorLogger.setup_logging(
                level=LogLevel.INFO,
                log_file=log_file
            )
            
            logger = SimulatorLogger.get_logger("test")
            logger.info("Test message")
            
            # Check that log file was created and contains message
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content
    
    def test_setup_logging_file_creation_error(self, caplog):
        """Test logging setup when file creation fails."""
        # Try to create log file in non-existent directory without creating parents
        invalid_path = Path("/nonexistent/directory/test.log")
        
        with caplog.at_level(logging.WARNING):
            SimulatorLogger.setup_logging(log_file=invalid_path)
        
        # Should not fail, but should log a warning
        assert "Failed to create log file" in caplog.text
    
    def test_setup_logging_component_levels(self):
        """Test setup with component-specific log levels."""
        component_levels = {
            "test.component1": LogLevel.DEBUG,
            "test.component2": LogLevel.ERROR
        }
        
        SimulatorLogger.setup_logging(
            level=LogLevel.INFO,
            component_levels=component_levels
        )
        
        logger1 = logging.getLogger("test.component1")
        logger2 = logging.getLogger("test.component2")
        
        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR
    
    def test_get_logger_caching(self):
        """Test that get_logger caches logger instances."""
        SimulatorLogger.setup_logging()
        
        logger1 = SimulatorLogger.get_logger("test")
        logger2 = SimulatorLogger.get_logger("test")
        
        assert logger1 is logger2
        assert "test" in SimulatorLogger._loggers
    
    def test_log_configuration(self, caplog):
        """Test configuration logging utility."""
        SimulatorLogger.setup_logging(level=LogLevel.INFO)
        
        config = {
            "simple_param": "value",
            "nested_param": {
                "sub_param1": 42,
                "sub_param2": "nested_value"
            }
        }
        
        with caplog.at_level(logging.INFO):
            SimulatorLogger.log_configuration(config, "test_config")
        
        assert "Configuration loaded:" in caplog.text
        assert "simple_param: value" in caplog.text
        assert "nested_param:" in caplog.text
        assert "sub_param1: 42" in caplog.text
        assert "sub_param2: nested_value" in caplog.text
    
    def test_log_statistics(self, caplog):
        """Test statistics logging utility."""
        SimulatorLogger.setup_logging(level=LogLevel.INFO)
        
        stats = {
            "simple_stat": 100,
            "float_stat": 3.14159,
            "nested_stats": {
                "count": 42,
                "rate": 0.95
            }
        }
        
        with caplog.at_level(logging.INFO):
            SimulatorLogger.log_statistics(stats, "test_stats")
        
        assert "Simulation Statistics:" in caplog.text
        assert "simple_stat: 100" in caplog.text
        assert "float_stat: 3.142" in caplog.text  # Should be rounded to 3 decimal places
        assert "nested_stats:" in caplog.text
        assert "count: 42" in caplog.text
        assert "rate: 0.950" in caplog.text
    
    def test_reset(self):
        """Test logger reset functionality."""
        SimulatorLogger.setup_logging()
        logger = SimulatorLogger.get_logger("test")
        
        assert SimulatorLogger._initialized == True
        assert len(SimulatorLogger._loggers) > 0
        
        SimulatorLogger.reset()
        
        assert SimulatorLogger._initialized == False
        assert len(SimulatorLogger._loggers) == 0
    
    def test_setup_logging_idempotent(self):
        """Test that setup_logging can be called multiple times safely."""
        SimulatorLogger.setup_logging(level=LogLevel.INFO)
        assert SimulatorLogger._initialized == True
        
        # Second call should not change anything
        SimulatorLogger.setup_logging(level=LogLevel.DEBUG)
        assert SimulatorLogger._initialized == True
        
        # Logger level should still be INFO (first call wins)
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO


class TestSetupComponentLogging:
    """Test setup_component_logging utility function."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        SimulatorLogger.reset()
    
    def teardown_method(self):
        """Clean up after each test."""
        SimulatorLogger.reset()
    
    def test_setup_component_logging(self):
        """Test component logging setup."""
        logger = setup_component_logging("test.component", LogLevel.DEBUG)
        
        assert logger.name == "test.component"
        assert logger.level == logging.DEBUG
        assert isinstance(logger, logging.Logger)


class TestLogException:
    """Test log_exception utility function."""
    
    def test_log_exception_with_traceback(self, caplog):
        """Test exception logging with traceback."""
        logger = logging.getLogger("test")
        exception = ValueError("test error")
        
        with caplog.at_level(logging.ERROR):
            log_exception(logger, exception, "test context", include_traceback=True)
        
        assert "test context: ValueError: test error" in caplog.text
    
    def test_log_exception_without_traceback(self, caplog):
        """Test exception logging without traceback."""
        logger = logging.getLogger("test")
        exception = ValueError("test error")
        
        with caplog.at_level(logging.ERROR):
            log_exception(logger, exception, "test context", include_traceback=False)
        
        assert "test context: ValueError: test error" in caplog.text
        # Should not contain traceback information
        assert "Traceback" not in caplog.text
    
    def test_log_exception_no_context(self, caplog):
        """Test exception logging without context."""
        logger = logging.getLogger("test")
        exception = ValueError("test error")
        
        with caplog.at_level(logging.ERROR):
            log_exception(logger, exception, include_traceback=False)
        
        assert "ValueError: test error" in caplog.text
        assert "test context:" not in caplog.text


class TestLogPerformanceMetrics:
    """Test log_performance_metrics utility function."""
    
    def test_log_performance_metrics_basic(self, caplog):
        """Test basic performance metrics logging."""
        logger = logging.getLogger("test")
        
        with caplog.at_level(logging.DEBUG):
            log_performance_metrics(logger, "test_operation", 123.45)
        
        assert "Performance: test_operation took 123.45ms" in caplog.text
    
    def test_log_performance_metrics_with_additional(self, caplog):
        """Test performance metrics logging with additional metrics."""
        logger = logging.getLogger("test")
        additional_metrics = {
            "items_processed": 100,
            "cache_hits": 85,
            "memory_mb": 64.5
        }
        
        with caplog.at_level(logging.DEBUG):
            log_performance_metrics(logger, "batch_process", 500.0, additional_metrics)
        
        log_message = caplog.text
        assert "Performance: batch_process took 500.00ms" in log_message
        assert "items_processed=100" in log_message
        assert "cache_hits=85" in log_message
        assert "memory_mb=64.5" in log_message


class TestLoggingIntegration:
    """Test logging integration scenarios."""
    
    def setup_method(self):
        """Reset logger state before each test."""
        SimulatorLogger.reset()
    
    def teardown_method(self):
        """Clean up after each test."""
        SimulatorLogger.reset()
    
    def test_multiple_loggers_same_component(self):
        """Test that multiple requests for same component return same logger."""
        SimulatorLogger.setup_logging()
        
        logger1 = SimulatorLogger.get_logger("drone_detection_simulator.test")
        logger2 = SimulatorLogger.get_logger("drone_detection_simulator.test")
        
        assert logger1 is logger2
    
    def test_hierarchical_logger_levels(self):
        """Test hierarchical logger level inheritance."""
        component_levels = {
            "drone_detection_simulator": LogLevel.INFO,
            "drone_detection_simulator.mqtt": LogLevel.DEBUG
        }
        
        SimulatorLogger.setup_logging(
            level=LogLevel.WARNING,
            component_levels=component_levels
        )
        
        root_logger = logging.getLogger("drone_detection_simulator")
        mqtt_logger = logging.getLogger("drone_detection_simulator.mqtt")
        other_logger = logging.getLogger("other_module")
        
        assert root_logger.level == logging.INFO
        assert mqtt_logger.level == logging.DEBUG
        # Other logger should inherit from root
        assert other_logger.level <= logging.WARNING
    
    def test_console_and_file_output(self, caplog):
        """Test simultaneous console and file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            SimulatorLogger.setup_logging(
                level=LogLevel.INFO,
                log_file=log_file,
                console_output=True
            )
            
            logger = SimulatorLogger.get_logger("test")
            
            with caplog.at_level(logging.INFO):
                logger.info("Test message")
            
            # Should appear in both console (caplog) and file
            assert "Test message" in caplog.text
            assert log_file.exists()
            file_content = log_file.read_text()
            assert "Test message" in file_content


if __name__ == "__main__":
    pytest.main([__file__])