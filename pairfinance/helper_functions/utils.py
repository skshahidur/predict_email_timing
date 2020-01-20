import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_matplotlib_support
from sklearn.base import is_classifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as sklearn_auc
import xgboost as xgb
import pymysql
from requests.exceptions import ConnectionError
import sys


def plot_stacked_bar(data, series_labels, category_labels=None,
                     show_values=False, value_format="{}", y_label=None,
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i], color=colors[i]))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w / 2, bar.get_y() + h / 2,
                         value_format.format(h), ha="center",
                         va="bottom")


from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_local(y_true, y_pred, classes,
                                normalize=False,
                                title=None,
                                cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


from itertools import product
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_matplotlib_support
from sklearn.base import is_classifier


class ConfusionMatrixDisplay:
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None, ax=None):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='vertical'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2f' for a normalized matrix, and
            'd' for a unnormalized matrix.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """
        check_matplotlib_support("ConfusionMatrixDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = '.2g'

            # print text with appropriate color depending on background
            thresh = (cm.max() - cm.min()) / 2.
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(j, i,
                                           format(cm[i, j], values_format),
                                           ha="center", va="center",
                                           color=color)

        fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=self.display_labels,
               yticklabels=self.display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(estimator, X, y_true, labels=None,
                          sample_weight=None, normalize=None,
                          display_labels=None, include_values=True,
                          xticks_rotation='horizontal',
                          values_format=None,
                          cmap='viridis', ax=None):
    """Plot Confusion Matrix.
    Read more in the :ref:`User Guide <confusion_matrix>`.
    Parameters
    ----------
    estimator : estimator instance
        Trained classifier.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y : array-like of shape (n_samples,)
        Target values.
    labels : array-like of shape (n_classes,), default=None
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If `None` is given, those that appear at
        least once in `y_true` or `y_pred` are used in sorted order.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    display_labels : array-like of shape (n_classes,), default=None
        Target names used for plotting. By default, `labels` will be used if
        it is defined, otherwise the unique labels of `y_true` and `y_pred`
        will be used.
    include_values : bool, default=True
        Includes values in confusion matrix.
    xticks_rotation : {'vertical', 'horizontal'} or float, \
                        default='vertical'
        Rotation of xtick labels.
    values_format : str, default=None
        Format specification for values in confusion matrix. If `None`,
        the format specification is '.2f' for a normalized matrix, and
        'd' for a unnormalized matrix.
    cmap : str or matplotlib Colormap, default='viridis'
        Colormap recognized by matplotlib.
    ax : matplotlib Axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is
        created.
    Returns
    -------
    display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
    """
    check_matplotlib_support("plot_confusion_matrix")

    if not is_classifier(estimator):
        raise ValueError("plot_confusion_matrix only supports classifiers")

    if normalize not in {'true', 'pred', 'all', None}:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    y_pred = estimator.predict(X)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight,
                          labels=labels)

    if display_labels is None:
        if labels is None:
            display_labels = estimator.classes_
        else:
            display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    return disp.plot(include_values=include_values,
                     cmap=cmap, ax=ax, xticks_rotation=xticks_rotation)


def top_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """This method will produce the top x predictions based on predicted probability"""

    total_population = df.shape[0]
    total_frauds = df[df.is_fraudster == 1].shape[0]
    df_output = pd.DataFrame()
    for i, j in enumerate([100, 200, 500, 1000, 2000]):
        df_temp = df.sort_values("pred_prob", ascending=False).head(j).copy()
        df_output.loc[i, 'total_count'] = j
        df_output.loc[i, 'frauds'] = np.sum(df_temp.is_fraudster)
        df_output.loc[i, 'max_prob'] = np.amax(df_temp.pred_prob)
        df_output.loc[i, 'min_prob'] = np.amin(df_temp.pred_prob)
        df_output.loc[i, 'percent_frauds'] = 100 * (df_output.loc[i, 'frauds'] / total_frauds)
        df_output.loc[i, 'percent_population'] = 100 * (j / total_population)

    return df_output


def transform_probability(x):
    """This method will convert the predicted to log functions. This is sometimes preferred for further calculations"""

    return -28.8539008178 * np.log(x / (1 - x + 0.000001)) + 487.122876205


def score_distribution(df: pd.DataFrame, target_variable: str, predicted_score: str):
    """This method will plot the distribution of the positive and negative classes"""

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.kdeplot(df[df[target_variable] == 1][predicted_score], shade=True, color='r')
    sns.kdeplot(df[df[target_variable] == 0][predicted_score], shade=True, color='g')
    ax.legend(['Fraud', 'Non-fraud'])
    ax.set_xlabel("Predicted Score", fontsize=15)
    ax.set_ylabel("Predicted Density", fontsize=15)
    fig.suptitle("Score Distribution", fontsize=15)
    # return fig


def df_for_output(target_variable: np.ndarray, predicted_proba: np.ndarray) -> pd.DataFrame:
    """This method will produce the df to be used for scoreband summary and distribution plots"""

    output_df = pd.DataFrame()
    output_df['target_variable'] = target_variable
    output_df['predicted_proba'] = predicted_proba
    output_df['scaled_proba'] = transform_probability(predicted_proba)
    output_df['count'] = 1

    return output_df


def band_function(x):
    """This method will use the probability scores to assing them a number"""

    if x < 100:
        return 0
    elif 100 <= x < 200:
        return 1
    elif 200 <= x < 300:
        return 2
    elif 300 <= x < 400:
        return 3
    elif 400 <= x < 500:
        return 4
    elif 500 <= x < 600:
        return 5
    elif 600 <= x < 700:
        return 6
    elif 700 <= x < 800:
        return 7
    elif 800 <= x < 900:
        return 8
    elif x >= 900:
        return 9


def scoreband_analysis(df: pd.DataFrame):
    """This method will produce the summary of the scores and its characteristics at each band level"""

    total_frauds = np.sum(df.target_variable)
    total_count = df.shape[0]
    band_mapping = {0: '<100', 1: '100-200', 2: '200-300', 3: '300-400', 4: '400-500', 5: '500-600', 6: '600-700',
                    7: '700-800', 8: '800-900', 9: '900+'}
    df['score_bucket'] = df['scaled_proba'].apply(lambda x: band_function(x))
    df['score_band'] = df['score_bucket'].map(band_mapping)
    rename_columns = {'target_variable': 'Band true fraud volume',
                      'predicted_proba': 'Avg. fraud prob',
                      'count': 'Band total volume'
                      }
    score_band_summary = df[['score_bucket', 'score_band', 'target_variable', 'count', 'predicted_proba']].groupby(
        ['score_bucket', 'score_band']).agg({'target_variable': 'sum',
                                             'predicted_proba': 'mean',
                                             'count': 'sum'
                                             }).rename(columns=rename_columns).reset_index()
    score_band_summary['Band true fraud volume'] = score_band_summary['Band true fraud volume'].astype('int32')
    score_band_summary['cumulative_target'] = score_band_summary['Band true fraud volume'].cumsum()
    score_band_summary['cumulative_count'] = score_band_summary['Band total volume'].cumsum()
    score_band_summary['band_precision'] = score_band_summary['Band true fraud volume'] / score_band_summary[
        'Band total volume']
    score_band_summary['recall'] = score_band_summary['cumulative_target'] / total_frauds
    score_band_summary['volume_share'] = score_band_summary['cumulative_count'] / total_count

    return score_band_summary[
        ['score_bucket', 'score_band', 'Band true fraud volume', 'Band total volume', 'Avg. fraud prob',
         'cumulative_target', 'cumulative_count', 'band_precision', 'recall', 'volume_share']]


def calibrate_probability(df: pd.DataFrame, target: str, pred_prob: str):
    """This method will take the """

    df['scaled_prob'] = transform_probability(df['pred_prob'])
    lr_calibration = LogisticRegression()
    lr_calibration.fit(df['scaled_prob'].values.reshape(-1, 1), df['is_fraudster'].values)
    # df['prob_calibrated'] = 
    df['prob_calibrated'] = lr_calibration.predict_proba(df['scaled_prob'].values.reshape(-1, 1))[:, 1]
    return df, lr_calibration


def hyperparameter_tuning(train_data: pd.DataFrame,
                          train_data_target: pd.DataFrame,
                          val_data: pd.DataFrame,
                          val_data_target: pd.DataFrame,
                          test1_data: pd.DataFrame,
                          test1_data_target: pd.DataFrame,
                          test2_data: pd.DataFrame,
                          test2_data_target: pd.DataFrame,
                          n_iteration: int,
                          target: str,
                          weight_balance: np.ndarray
                          ):
    """
    This method will come handy for choosing hyper-parameters through a randomized search. The method provides the result on validation dataset as well as on the test datasets so that the job becomes 
    easy for us to select the best performing model on test datasets.
    """

    max_depth_dist = np.random.randint(3, 8, n_iteration)
    learning_rate_dist = np.random.uniform(0.01, 0.5, n_iteration)
    n_estimators_dist = np.random.randint(30, 120, n_iteration)
    gamma_dist = np.random.randint(1, 5, n_iteration)
    min_child_weight_dist = np.random.uniform(0.5, 4, n_iteration)
    subsample_dist = np.random.uniform(0.4, 1, n_iteration)
    colsample_bytree_dist = np.random.uniform(0.6, 1, n_iteration)
    reg_lambda_dist = np.random.uniform(1, 6, n_iteration)

    train_pr_auc = list()
    val_pr_auc = list()
    test1_pr_auc = list()
    test2_pr_auc = list()

    for i in range(n_iteration):
        print("Iteration {} running ...".format(i))
        xgb_model = xgb.XGBClassifier(max_depth=max_depth_dist[i],
                                      learning_rate=learning_rate_dist[i],
                                      n_estimators=n_estimators_dist[i],
                                      verbosity=1,
                                      objective='binary:logistic',
                                      booster='gbtree',
                                      tree_method='auto',
                                      n_jobs=3,
                                      gamma=gamma_dist[i],
                                      min_child_weight=min_child_weight_dist[i],
                                      max_delta_step=0,
                                      subsample=subsample_dist[i],
                                      colsample_bytree=colsample_bytree_dist[i],
                                      colsample_bylevel=1,
                                      colsample_bynode=1,
                                      reg_alpha=0,
                                      reg_lambda=reg_lambda_dist[i],
                                      scale_pos_weight=1,
                                      base_score=0.5,
                                      random_state=123
                                      )

        xgb_model.fit(train_data.values, train_data_target[target].values,
                    eval_set=[(train_data.values, train_data_target[target].values),
                                (val_data.values, val_data_target[target].values)],
                    sample_weight = weight_balance,
                    verbose=False)

        # PR AUC value for train datset
        xgb_predict = xgb_model.predict_proba(train_data.values)[:, 1]
        precision, recall, thresholds = precision_recall_curve(train_data_target[target].values, xgb_predict)
        auc = sklearn_auc(recall, precision)
        train_pr_auc.append(auc)

        # PR AUC value for val datset
        xgb_predict = xgb_model.predict_proba(val_data.values)[:, 1]
        precision, recall, thresholds = precision_recall_curve(val_data_target[target].values, xgb_predict)
        auc = sklearn_auc(recall, precision)
        val_pr_auc.append(auc)

        # PR AUC value for test1 datset
        xgb_predict = xgb_model.predict_proba(test1_data.values)[:, 1]
        precision, recall, thresholds = precision_recall_curve(test1_data_target[target].values, xgb_predict)
        auc = sklearn_auc(recall, precision)
        test1_pr_auc.append(auc)

        # PR AUC value for val datset
        xgb_predict = xgb_model.predict_proba(test2_data.values)[:, 1]
        precision, recall, thresholds = precision_recall_curve(test2_data_target[target].values, xgb_predict)
        auc = sklearn_auc(recall, precision)
        test2_pr_auc.append(auc)

        print("Iteration {} completed.".format(i))

    df = pd.DataFrame({'train_pr_auc': train_pr_auc,
                       'val_pr_auc': val_pr_auc,
                       'test1_pr_auc': test1_pr_auc,
                       'test2_pr_auc': test2_pr_auc
                       })
    return df, max_depth_dist, learning_rate_dist, n_estimators_dist, gamma_dist, min_child_weight_dist, subsample_dist, colsample_bytree_dist, reg_lambda_dist


def check_input_data_type(df: pd.DataFrame, mapping: dict):
    """This method will check the data types of the input data. If there's a mismatch, it'll raise a TyepError."""

    # bool_value = True
    try:
        for i in mapping.keys():
            # bool_value = bool_value & np.where(df.loc[:,i].values[0] in ['NA', 'na', 'nan', 'NaN', 'Nan', np.nan,
            # np.inf, 'missing', 'NULL', 'null', 'Null', None, 'None', 'none', ' ', ''], True, isinstance(df.loc[:,
            # i].values[0], mapping[i]))
            df.loc[:, i] = df.loc[:, i].astype(mapping[i])
    except TypeError:
        pass


#     bool_value = bool_value & isinstance(df['USER_ID'], str)
#     try:
#         bool_value = bool_value & isinstance(df['CREATED_DATE'], str)
#         pd.to_datetime(df['CREATED_DATE'])
#     except TypeError:
#         bool_value = bool_value & False
#     bool_value = bool_value & isinstance(df['TYPE'], str)
#     bool_value = bool_value & isinstance(df['STATE'], str)
#     bool_value = bool_value & isinstance(df['AMOUNT_GBP'], float)
#     bool_value = bool_value & isinstance(df['CURRENCY'], str)
#     try:
#         bool_value = bool_value & isinstance(df['CREATED_DATE_USER'], str)
#         pd.to_datetime(df['CREATED_DATE_USER'])
#     except TypeError:
#         bool_value = bool_value & False
#     bool_value = bool_value & isinstance(df['COUNTRY'], str)
#     try:
#         bool_value = bool_value & isinstance(df['BIRTH_DATE'], str)
#         pd.to_datetime(df['BIRTH_DATE'])
#     except TypeError:
#         bool_value = bool_value & False
#
#     if not bool_value:
#         return {'decision': 'TypeError'} # means there's a mismatch
#         sys.exit()
#     #     # raise TypeError


def dtype_mapping(numeric_columns: list,
                  boolean_columns: list,
                  string_columns: list
                  ) -> dict:
    """This method will create a dictionary of data type mapping which will be used for production data validation"""

    dtype_dictionary = dict()
    for i in numeric_columns:
        dtype_dictionary[i] = float
    for i in boolean_columns:
        dtype_dictionary[i] = bool
    for i in string_columns:
        dtype_dictionary[i] = str

    return dtype_dictionary


def convert_to_dataframe(transaction):
    """This method will convert incoming json into a pd.DataFrame"""

    return pd.DataFrame.from_dict([transaction])


def fetch_record_from_sql(record_id: str, connection):
    """This method will fetch the record from sql for an ID"""

    try:
        with connection.cursor() as cursor:
            sql = """select a.*, b.`CREATED_DATE` as `CREATED_DATE_USER`, b.`COUNTRY`, b.`BIRTH_DATE`, b.`premium`, c.`mean_spending`, c.`new_user`
                    from transactions a
                    join user_profile_new b
                    on a.`USER_ID`=b.`ID`
                    and a.ID='{0}'
                    join (SELECT *
                    FROM user_spending_new x
                    WHERE x.CREATED_DATE <= (select CREATED_DATE from transactions where ID='{1}')
                    and USER_ID=(select USER_ID from transactions where ID='{2}')
                    ORDER BY x.CREATED_DATE DESC
                    limit 1) c
                    on a.`USER_ID`=c.`USER_ID`
                    and a.ID='{3}'
                    where a.ID='{4}'
                    ;""".format(record_id, record_id, record_id, record_id, record_id)
            cursor.execute(sql)
            result = cursor.fetchone()
            return result
    except ConnectionError:
        return 1


def timing_of_reply(x):

    """
    This method will take the input TO timing and will produce a more generalized format
    """
    x_int = int(x.split(':')[0])
    if x_int >=0 and x_int <6:
        return '0 to 6am'
    elif x_int >=6 and x_int <12:
        return '6am to 12pm'
    elif x_int >=12 and x_int <18:
        return '12pm to 18pm'
    elif x_int >=18 and x_int <23:
        return '18pm to 23pm'




