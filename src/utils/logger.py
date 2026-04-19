import logging
import sys

def get_logger(name: str):
    """
    Configures a standardized logger for the project.
    Outputs to stdout so it can be captured by Docker, Cloud Run, or Prefect.
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Format: Timestamp - Module - Level - Message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger