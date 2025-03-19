import logging
import os
from datetime import datetime

# Generate log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the log directory
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)  # Ensure the directory exists

# Define full log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s -%(levelname)s - %(message)s",
    level=logging.INFO,  # Set logging level
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Example usage



if __name__ == "__main__":
    logging.info("Logging is successfully set up!")
