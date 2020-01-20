import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import RobustScaler

from pairfinance.data_processing.transformers import ColumnSelector, ReplaceMissingValue, RetrieveFeatureName, TypeCast, ConvertCategorytoBooleanHigh, ConvertCategorytoBooleanLow, ReverttoDataFrame, FeatureEngineer, OutlierCapping, ConvertBooleantoIntegers


def get_preprocess_pipeline(selected_columns: list,
                            numerical_columns: list,
                            strategy: str,
                            # boolean_columns: list,
                            low_card_columns_before: list,
                            low_card_columns_after: list
                            # high_card_columns_before: list,
                            # high_card_columns_after: list,
                            # high_card_dictionary: dict
                            # feature_engineer: list,
                            # new_features: list
                            ):
    """
    This pipeline will take data points as pandas df and will return a numpy array after all the processing of the
    dataset 
    """

    return make_pipeline(ColumnSelector(columns=selected_columns),
                         FeatureUnion(transformer_list=[
                             ('numerical_columns', make_pipeline(ColumnSelector(columns=numerical_columns),
                                                                 ReplaceMissingValue(
                                                                     to_replace=['NA', 'na', 'nan', 'NaN', 'Nan',
                                                                                 np.nan, np.inf, 'missing', 'NULL',
                                                                                 'null', 'Null', None, 'None', 'none',
                                                                                 ' ', ''],
                                                                     replacement=np.nan
                                                                 ),
                                                                 SimpleImputer(missing_values=np.nan,
                                                                               strategy=strategy),
                                                                 # OutlierCapping(lower_limit=0.01, upper_limit=0.99, variables=numerical_columns),
                                                                 RobustScaler(quantile_range=(0.01, 99.0)),
                                                                 ReverttoDataFrame(numerical_columns),
                                                                 RetrieveFeatureName(),
                                                                 TypeCast('float32')
                                                                 )
                              ),
                             ('low_card_columns', make_pipeline(ColumnSelector(columns=low_card_columns_before),
                                                                ReplaceMissingValue(
                                                                    to_replace=['NA', 'na', 'nan', 'NaN', 'Nan', np.nan,
                                                                                np.inf, 'missing', 'NULL', 'null',
                                                                                'Null', None, 'None', 'none', ' ', ''],
                                                                    replacement='unk'
                                                                ),
                                                                ConvertCategorytoBooleanLow(low_card_columns_before,
                                                                                            low_card_columns_after),
                                                                RetrieveFeatureName(),
                                                                TypeCast('int16')
                                                                )
                              )
                             # ('high_card_columns', make_pipeline(ColumnSelector(columns=high_card_columns_before),
                             #                                     ReplaceMissingValue(
                             #                                         to_replace=['NA', 'na', 'nan', 'NaN', 'Nan',
                             #                                                     np.nan, np.inf, 'missing', 'NULL',
                             #                                                     'null', 'Null', None, 'None', 'none',
                             #                                                     ' ', ''],
                             #                                         replacement='unk'
                             #                                     ),
                             #                                     ConvertCategorytoBooleanHigh(high_card_columns_before,
                             #                                                                  high_card_columns_after,
                             #                                                                  high_card_dictionary),
                             #                                     RetrieveFeatureName(),
                             #                                     TypeCast('int16')
                             #                                     )
                             #  ),
                             # ('boolean_columns', make_pipeline(ColumnSelector(columns=boolean_columns),
                             #                                     ReplaceMissingValue(
                             #                                         to_replace=['NA', 'na', 'nan', 'NaN', 'Nan',
                             #                                                     np.nan, np.inf, 'missing', 'NULL',
                             #                                                     'null', 'Null', None, 'None', 'none',
                             #                                                     ' ', ''],
                             #                                         replacement=-1
                             #                                     ),
                             #                                     ConvertBooleantoIntegers(boolean_columns),
                             #                                     RetrieveFeatureName(),
                             #                                     TypeCast('int16')
                             #                                     )
                             #  )
                             # ('new_features', make_pipeline(ColumnSelector(columns=feature_engineer),
                             #                                ReplaceMissingValue(
                             #                                    to_replace=['NA', 'na', 'nan', 'NaN', 'Nan', np.nan,
                             #                                                np.inf, 'missing', 'NULL', 'null', 'Null',
                             #                                                None, 'None', 'none', ' ', ''],
                             #                                    replacement=np.nan
                             #                                ),
                             #                                FeatureEngineer(new_features),
                             #                                RetrieveFeatureName(),
                             #                                TypeCast('float32')
                             #                                )
                             #  )
                         ]
                         )
                         )
