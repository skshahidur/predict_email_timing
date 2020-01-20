import os
import sys
import pandas as pd
import pickle
import pymysql
import xgboost as xgb
from pairfinance.helper_functions.utils import transform_probability, check_input_data_type, convert_to_dataframe, \
    fetch_record_from_sql
import pairfinance.data_processing as data_processing
import pairfinance.helper_functions as helper_functions
import numpy as np

sys.modules['data_processing'] = data_processing
sys.modules['helper_functions'] = helper_functions

np.random.seed(123)

model_file_path = os.path.join(os.path.dirname(__file__), 'model_files_v0.pkl')
with open(model_file_path, 'rb') as f:
    selected_columns, numerical_columns, low_cardinal_columns_before, low_cardinal_columns_after, \
    dtype_mapping_output, train_pipeline, xgb_model = pickle.load(f)

def predictor(record_id: str):
    """This method will take a row in json format and return the model result on the user"""

    try:
        df = convert_to_dataframe(record_id)
        check_input_data_type(df, dtype_mapping_output)
        df_transform = train_pipeline.transform(df)
        y_pred = float(xgb_model.predict(df_transform)[0])
        return {'time': y_pred}
    except:
        return {'time': 'Error'}




