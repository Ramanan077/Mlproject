import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor

)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object
from src.utils import get_best_model
 
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):

        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "AdaBoostRegressor": AdaBoostRegressor()
            }

            model_report: dict = get_best_model(X_train, y_train, X_test, y_test, models=models)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)



           
