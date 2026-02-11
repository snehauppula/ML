import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclass import dataclass
@dataclass
class DataTransformationConfig:
    
    
