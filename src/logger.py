import logger
import os
from datetime import datetime

LOG_FILE_PATH = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
logs_path= os.path.join(os.getcwd(),"logs",LOG_FILE_PATH)
