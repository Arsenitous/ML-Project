import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
 
# pyrefly: ignore [missing-import]
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Model Training Initiated')
            logging.info('Splitting data into training and testing data')
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
        
            models = {
                'Linear Regression' : LinearRegression(),
                'Random Forest' : RandomForestRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'AdaBoost Classifier' : AdaBoostRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'K Neighbors Classifier' : KNeighborsRegressor(),
                'XGBoost Classifier' : XGBRegressor(),
                'CatBoost Classifier' : CatBoostRegressor(verbose=False)
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            ## To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
    
            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)

            logging.info("Best Model Found")
            logging.info(f'Best Model Score: {best_model_score}')
            logging.info(f'Best Model Name: {best_model_name}')

            logging.info('Model Training Completed')
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info('Model Training Completed')
            logging.info('Evaluating the best model')
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)