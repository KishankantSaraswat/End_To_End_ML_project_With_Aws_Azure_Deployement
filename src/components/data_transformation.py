import sys 
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation."""
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    '''
    this class is responsible for transforming the data
    1. Numerical columns: impute missing values with median and scale the values
    2. Categorical columns: impute missing values with most frequent and one hot encode the values
    3. The transformed data is saved as a pickle file
    4. The preprocessor object is saved as a pickle file
    5. The preprocessor object is returned

    '''
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):  # Fixed method name
        try:
            numerical_column = ["writing_score", "reading_score"]  # Fixed column name
            categorical_column = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))  # Prevents error with sparse matrices
                ]
            )

            logging.info("Created the numerical and categorical pipelines")
            logging.info(f"Categorical columns: {categorical_column}")
            logging.info(f"Numerical columns: {numerical_column}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_column),
                    ("cat_pipeline", cat_pipeline, categorical_column)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Initiating data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read the train and test datasets")

            logging.info("Obtainig preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()  # Fixed method name
            # logging.info("Fitting the preprocessor object")
            # preprocessor_obj.fit(train_df)

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]  # Fixed column name
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframes and testing dataframes"
            )
            # logging.info("Obtained the input and target feature dataframes")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            '''difference between fit_transform and transform
            is that fit_transform is used when we want to fit the model on the data and then transform the data
            while transform is used when we want to transform the data without fitting the model on the data

            '''
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            '''why we do np.c_[input_feature_train_arr,np.array(target_feature_train_df)] 
            use of np.c_ is to stack the input feature array and target feature array horizontally'''

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,  # Fixed attribute reference
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
