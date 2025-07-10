"""
Logging configuration for Medical AI Assistant.
Provides structured logging with proper formatting and levels.
"""

import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"medical_ai_{timestamp}.log"
    
    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s | %(message)s"
            },
            "json": {
                "format": '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "file": "%(filename)s", "line": %(lineno)d, "message": "%(message)s"}',
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "detailed",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": str(log_file),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(logs_dir / f"medical_ai_errors_{timestamp}.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "app": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            },
            "openai": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "chromadb": {
                "level": "WARNING",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "httpx": {
                "level": "WARNING",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file"]
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger("app.logging")
    logger.info("🔧 Logging system initialized")
    logger.info(f"📝 Log level: {log_level}")
    logger.info(f"📁 Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"app.{name}")


class MedicalAILogger:
    """
    Custom logger class for Medical AI Assistant with specialized methods.
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
    
    def query_start(self, query: str, user_id: str = None):
        """Log the start of a medical query."""
        self.logger.info(f"🔍 Query started: '{query[:100]}...' | User: {user_id or 'anonymous'}")
    
    def query_complete(self, query: str, response_time: float, ragas_score: float = None):
        """Log the completion of a medical query."""
        ragas_info = f" | RAGAS: {ragas_score:.3f}" if ragas_score else ""
        self.logger.info(f"✅ Query completed: '{query[:50]}...' | Time: {response_time:.2f}s{ragas_info}")
    
    def query_error(self, query: str, error: str):
        """Log a query error."""
        self.logger.error(f"❌ Query failed: '{query[:50]}...' | Error: {error}")
    
    def ragas_evaluation(self, faithfulness: float, relevancy: float, precision: float, recall: float = None):
        """Log RAGAS evaluation results."""
        recall_info = f" | Recall: {recall:.3f}" if recall else ""
        self.logger.info(f"📊 RAGAS: Faithfulness: {faithfulness:.3f} | Relevancy: {relevancy:.3f} | Precision: {precision:.3f}{recall_info}")
    
    def document_processed(self, filename: str, chunks: int, tokens: int, processing_time: float):
        """Log document processing completion."""
        self.logger.info(f"📄 Document processed: {filename} | Chunks: {chunks} | Tokens: {tokens} | Time: {processing_time:.2f}s")
    
    def safety_flag(self, query: str, flag: str):
        """Log safety flag raised."""
        self.logger.warning(f"🚨 Safety flag: {flag} | Query: '{query[:50]}...'")
    
    def performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics."""
        self.logger.info(f"⚡ Performance: {metric_name}: {value:.3f}{unit}")
    
    def system_health(self, component: str, status: str, details: str = None):
        """Log system health status."""
        details_info = f" | {details}" if details else ""
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
        self.logger.info(f"{status_emoji} Health: {component}: {status}{details_info}")


# Convenience function to get a medical AI logger
def get_medical_logger(name: str) -> MedicalAILogger:
    """
    Get a MedicalAILogger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        MedicalAILogger instance
    """
    return MedicalAILogger(name) 