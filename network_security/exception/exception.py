import sys
import logging

# Create a logger instance
logger = logging.getLogger(__name__)

class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_message = error_message
        
        _, _, exc_tb = sys.exc_info()
        self.lineno = exc_tb.tb_lineno if exc_tb else None
        
        # Log the error if needed
        logger.error(f"NetworkSecurityException: {error_message}")