from loguru import logger
import sys

def setup_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    logger.add(
        "logs/system.log",
        rotation="10 MB",
        level="DEBUG"
    )
