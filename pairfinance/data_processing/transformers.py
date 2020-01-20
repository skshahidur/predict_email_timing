import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import time


# from transformers import ColumnSelector, ReplaceMissingValue, RetrieveFeatureName, TypeCast, CovBooltoInt, ConvertCategorytoBooleanHigh, ConvertCategorytoBooleanLow, ReverttoDataFrame, FeatureEngineer, OutlierCapping


class ColumnSelector(BaseEstimator, TransformerMixin):
    """This class will help in selection of the columns of interest from the provided pandas df"""

    def __init__(self, columns):
        self.feature_names = columns

    def fit(self, X):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.feature_names]
        except KeyError:
            cols_error = list(set(self.feature_names) - set(X.columns))
            raise KeyError("The df does not include the following columns {}".format(cols_error))

    def get_feature_names(self):
        return self.feature_names


class ReplaceMissingValue(BaseEstimator, TransformerMixin):
    """This class will help in replacing the missing values with the already provided values"""

    def __init__(self, to_replace, replacement=np.nan):
        self.to_replace = to_replace
        self.replacement = replacement

    def fit(self, X):
        return self

    def transform(self, X):
        X = X.applymap(lambda val: self.replacement if val in self.to_replace else val)
        return X


class RetrieveFeatureName(BaseEstimator, TransformerMixin):
    """This class will help in retrieving the column names. Though not necessary step for modeling but preffered for further investigation on the data processing"""

    def __init__(self):
        self.feature_names = None

    def fit(self, X):
        return self

    def transform(self, X):
        self.feature_names = X.columns.tolist()
        return X

    def get_feature_names(self):
        return self.feature_names


class TypeCast(BaseEstimator, TransformerMixin):
    """This class will cast the type of the output i.e. casting the output in float or integers etc"""

    def __init__(self, dtype):
        self.dtype = dtype
        self.feature_names = None

    def fit(self, X):
        return self

    def transform(self, X):
        return X.astype(self.dtype)

    def get_feature_names(self):
        return self.feature_names


class ConvertCategorytoBooleanLow(BaseEstimator, TransformerMixin):
    """This transformer will convert the low cardinal categorical variables to dummy one-hot variables"""

    def __init__(self, low_card_columns_before, low_card_columns_after):
        self.low_card_columns_before = low_card_columns_before
        self.low_card_columns_after = low_card_columns_after

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.low_card_columns_before] = X[self.low_card_columns_before].fillna('unk')
        self.low_card_new_data = pd.get_dummies(X[self.low_card_columns_before].astype('str'))
        missing_columns = [i for i in self.low_card_columns_after if i not in self.low_card_new_data.columns.tolist()]

        if len(missing_columns)>0:
            rest_df = pd.DataFrame(np.zeros((X.shape[0], len(missing_columns))), columns=missing_columns)
            rest_df.index = self.low_card_new_data.index
            self.low_card_new_data = pd.concat([self.low_card_new_data, rest_df], axis=1, ignore_index=False)

        self.low_card_new_data = self.low_card_new_data[self.low_card_columns_after]
        return self.low_card_new_data

    def get_feature_names(self):
        return self.low_card_new_data.columns.tolist()


class ConvertCategorytoBooleanHigh(BaseEstimator, TransformerMixin):
    """This transformer will convert the high cardinal categorical variables to dummy one-hot variables based on the fraud rates in the category group"""

    def __init__(self, high_card_columns_before, high_card_columns_after, dictionary_mapping):
        self.high_card_columns_before = high_card_columns_before
        self.high_card_columns_after = high_card_columns_after
        self.mapping = dictionary_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[self.high_card_columns_before] = X[self.high_card_columns_before].apply(lambda x: x.map(self.mapping), axis=0)
        X[self.high_card_columns_before] = X[self.high_card_columns_before].fillna('unk')
        self.high_card_new_data = pd.get_dummies(X[self.high_card_columns_before])
        missing_columns = [i for i in self.high_card_columns_after if i not in self.high_card_new_data.columns.tolist()]

        if len(missing_columns)>0:
            rest_df = pd.DataFrame(np.zeros((X.shape[0], len(missing_columns))), columns=missing_columns)
            rest_df.index = self.high_card_new_data.index
            self.high_card_new_data = pd.concat([self.high_card_new_data, rest_df], axis=1, ignore_index=False)

        self.high_card_new_data = self.high_card_new_data[self.high_card_columns_after]
        return self.high_card_new_data

    def get_feature_names(self):
        return self.high_card_new_data.columns.tolist() 


class ReverttoDataFrame(BaseEstimator, TransformerMixin):
    """This transformer will convert the numpy array to pd.DataFrame. This is important coz the sklearn transformers convert the df to numpy arrays"""

    def __init__(self, num_columns):
        self.num_columns = num_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        temp_df = pd.DataFrame(X, columns=self.num_columns)
        temp_df = temp_df.fillna(0)
        return temp_df

    def get_feature_names(self):
        return self.num_columns 


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """ This transformer is dedicated toward creating the new features for the modeling"""

    def __init__(self, new_features):
        self.new_features = new_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['age_user'] = pd.to_datetime(time.time(), unit='s').year - pd.to_datetime(X.BIRTH_DATE).dt.year
        X['age_platform'] = (pd.to_datetime(X.CREATED_DATE) - pd.to_datetime(X.CREATED_DATE_USER))/np.timedelta64(1, 'D')
        self.day_of_week_data = pd.get_dummies(pd.to_datetime(X.CREATED_DATE).dt.day_name())
        self.day_of_week_data.index = X.index
        X['hour_of_day'] = pd.to_datetime(X.CREATED_DATE).dt.hour
        self.temp_df = pd.concat([X[['age_user','age_platform','hour_of_day']],self.day_of_week_data], axis=1)
        X['spend_deviation'] = np.where(np.isnan(X['mean_spending']), X['AMOUNT_GBP'], X['AMOUNT_GBP'] - X['mean_spending'])
        missing_columns = [i for i in self.new_features if i not in self.temp_df.columns.tolist()]

        if len(missing_columns)>0:
            rest_df = pd.DataFrame(np.zeros((X.shape[0], len(missing_columns))), columns=missing_columns)
            rest_df.index = self.temp_df.index
            self.temp_df = pd.concat([self.temp_df, rest_df], axis=1, ignore_index=False)
        return self.temp_df[self.new_features]

    def get_feature_names(self):
        return self.new_features


class OutlierCapping(BaseEstimator, TransformerMixin):
    """This transformer caps the outlier based quantile calculations"""

    def __init__(self, lower_limit=0.05, upper_limit=0.95, variables=None):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.variables = variables

    def fit(self, X, y=None):
        self.left_tail_caps = pd.DataFrame(np.quantile(X, self.lower_limit, axis=0).reshape(1, X.shape[1]), columns = self.variables)
        self.right_tail_caps = pd.DataFrame(np.quantile(X, self.upper_limit, axis=0).reshape(1, X.shape[1]), columns = self.variables)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns = self.variables)
        for i in self.variables:
            X.loc[:,i] = np.where(X.loc[:,i]<self.left_tail_caps.loc[:,i][0], self.left_tail_caps.loc[:,i][0], X.loc[:,i])
            X.loc[:,i] = np.where(X.loc[:,i]>self.right_tail_caps.loc[:,i][0], self.right_tail_caps.loc[:,i][0], X.loc[:,i])
        return X.values

    def get_feature_names(self):
        return self.variables


class ConvertBooleantoIntegers(BaseEstimator, TransformerMixin):
    """
    This transformer will convert the boolean columns to integers.
    """

    def __init__(self, boolean_columns):
        self.boolean_columns = boolean_columns

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X = X*1
        # X = X.apply(lambda x: x.map({'True':1, 'False':0, 'true':1, 'false':0, 'TRUE':1, 'FALSE':0, True:1, False:0, 1:1, 0:0, '1':1, '0':0, -1:-1, '-1':-1}))
        return X

    def get_feature_names(self):
        return self.boolean_columns























