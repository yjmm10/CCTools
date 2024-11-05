from loguru import logger

# Configure logger
def setup_logger(log_file=None):
    logger.remove()  # Remove default handler
    if log_file:
        logger.add(log_file, rotation="10 MB")  # Log to file
    else:
        logger.add(lambda msg: print(msg, end=""))  # Log to console

# Call this function at the beginning of your script
# setup_logger("log.txt")  # Uncomment to log to file
setup_logger()  # Log to console