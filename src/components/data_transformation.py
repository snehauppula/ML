import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #handling missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig):
        self.data_transformation_config = data_transformation_config
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation started")
            numeric_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numeric_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info("Data Transformation started")
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            preprocessor = self.get_data_transformation_object()
            target_column = 'math_score'
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]
            input_feature_train_arr = preprocessor.fit_transform(X_train)
            input_feature_test_arr = preprocessor.transform(X_test)

            train_array = np.c_[input_feature_train_arr, np.array(y_train)]
            test_array = np.c_[input_feature_test_arr, np.array(y_test)]
            save_object( #save as pickle file
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path ) 
        except Exception as e:
            raise CustomException(e, sys)

    
    
