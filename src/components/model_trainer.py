import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig():
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "Support Vector Machine": SVR(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],   
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate':[0.01, 0.05, 0.1, 0.2],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "Linear Regression":{},
                "XGBoost":{
                    'learning_rate':[0.01, 0.05, 0.1, 0.2],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "CatBoost":{
                    'learning_rate':[0.01, 0.05, 0.1, 0.2],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                },
                "Support Vector Machine":{
                    'kernel':['linear','rbf','poly'],
                    'C':[0.1, 1, 10, 100]
                },
                "K-Nearest Neighbors":{
                    'n_neighbors':[3, 5, 7, 9],
                    'weights':['uniform','distance'],
                    'p':[1,2]
                },
                "AdaBoost":{
                    'learning_rate':[0.01, 0.05, 0.1, 0.2],
                    'n_estimators':[8, 16, 32, 64, 128, 256]
                }
            }   
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise Exception("No model found that can give accuracy above 60%")
            logging.info(f"Best model found, {best_model_name} with accuracy: {best_model_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted_y = best_model.predict(X_test)
            best_model_r2_score = r2_score(y_test, predicted_y)
            return best_model_r2_score 

        except Exception as e:
            raise CustomException(e, sys)
    

    
