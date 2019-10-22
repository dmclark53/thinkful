import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn import metrics

'''
Global Variables
'''
thresholds = {
    'tukey': 1.5
}

'''
Data Exploration
'''

def find_variable_types(df):
    # Separate variables into numerical and string
    df_described = df.describe(include='all').T

    # If the mean can be calculated, then this must be a numerical variable
    numerical_vars = df_described.loc[df_described['mean'].notnull(), :].index.values.tolist()

    # If there is no mean value, then this will be a categorical variable
    string_vars = df_described.loc[df_described['mean'].isnull(), :].index.values.tolist()
    return numerical_vars, string_vars


def stack_dataframe(df, column_list):
    df_stacked = df.loc[:, column_list].stack().copy()
    df_stacked.index = df_stacked.index.droplevel()
    df_stacked = df_stacked.reset_index().copy()
    df_stacked = df_stacked.rename(columns={'index': 'label', 0: 'value'})
    df_stacked = df_stacked.sort_values(by='label')
    df_stacked.index = np.arange(len(df_stacked))
    return df_stacked


def create_boxplots(df, column_list, max_boxes=None, log_scale=False):

    offset_dict = {}

    if log_scale:
        for column in column_list:
            min_value = df[column].min()
            offset_dict[column] = np.min([0, min_value])
            df[column] = df[column] - offset_dict[column] + 1
            # df[column] = df[column].apply(lambda x: x - offset_dict[column] + 1)

    if len(column_list) > 1:
        if max_boxes:
            max_boxes = np.min([max_boxes, len(column_list)])
            num_plots = len(column_list) // max_boxes
            remainder_boxes = len(column_list) % max_boxes
            for i in range(0, num_plots * max_boxes, max_boxes):
                f, ax = plt.subplots(figsize=(14, 6))
                df_stacked = stack_dataframe(df, column_list[i:i + max_boxes])
                sns.boxplot(x='label', y='value', data=df_stacked)
                if log_scale:
                    ax.set_yscale('log')
                    ax.set_ylabel('log (value)')
                plt.show()
            if remainder_boxes > 0:
                df_stacked = stack_dataframe(df, column_list[i + max_boxes:])
                f, ax = plt.subplots(figsize=(14, 6))
                sns.boxplot(x='label', y='value', data=df_stacked)
                if log_scale:
                    ax.set_yscale('log')
                    ax.set_ylabel('log (value)')
                plt.show()
        else:
            df_stacked = stack_dataframe(df, column_list)
            f, ax = plt.subplots(figsize=(14, 6))
            sns.boxplot(x='label', y='value', data=df_stacked)
            if log_scale:
                ax.set_yscale('log')
                ax.set_ylabel('log (value)')
            plt.show()
    else:
        f, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(y=column_list[0], data=df)
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('log (value)')
        plt.show()

    if log_scale:
        for column in column_list:
            df[column] = df[column] + offset_dict[column] - 1
            # df[column] = df[column].apply(lambda x: x + offset_dict[column] - 1)


'''
Outliers
'''

def apply_tukey(df, column, thresh=1.5):
    q75, q25 = np.percentile(df[column], [75, 25])
    iqr = q75 - q25
    min_value = q25 - thresh*iqr
    max_value = q75 + thresh*iqr
    df[f'{column}_tk'] = df[f'{column}'].apply(lambda x: np.max([np.min([x, max_value]), min_value]))


def correct_outliers(df, column_list):
    for column in column_list:
        # Tukey
        apply_tukey(df, column, thresh=thresholds['tukey'])


'''
Data Transformation
'''


def normalize(df, column_list):
    for column in column_list:
        max_value = df[column].max()
        min_value = df[column].min()
        df[f'{column}_nm'] = (df[column] - min_value) / (max_value - min_value)

transformation_functions = {
    'normalize': normalize
}


def apply_transform(df, column_list, method):
    transformation_functions[method](df, column_list)

'''
Feature Selection
'''

def find_correlated_features(df, thresh=0.8):
    df_corr_abs = df.corr().abs()
    upper_triangle = df_corr_abs.where(np.triu(np.ones(df_corr_abs.shape), k=1).astype(np.bool))
    correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > thresh)]
    print(f'Correlated Columns (r > {thresh}):')
    for column in correlated_columns:
        print(f'* {column}')
    print(f'{len(correlated_columns)} correlated features will be removed.')
    return correlated_columns

'''
Modeling
'''

def fit_kmeans(X, y, num_clusters, evaluate_model=False):
    y_pred = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X)
    if evaluate_model:
        display_clusters(X, y, 'K-Means')
        compute_crosstab(y, y_pred)
        ari = metrics.adjusted_rand_score(y, y_pred)
        print(f'ARI: {ari:0.3f}')


def fit_mean_shift(X, y, evaluate_model=False):
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    y_pred = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters_ = len(np.unique(y_pred))
    if evaluate_model:
        display_clusters(X, y_pred, 'Mean Shift')
        print(f'There are {n_clusters_} estimated clusters.')
        compute_crosstab(y, y_pred)
        ari = metrics.adjusted_rand_score(y, y_pred)
        print(f'ARI: {ari:0.3f}')


def fit_spectral_clustering(X, y, num_clusters, evaluate_model=False):
    sc = SpectralClustering(n_clusters=num_clusters)
    sc.fit(X)
    y_pred=sc.fit_predict(X)
    if evaluate_model:
        display_clusters(X, y, 'Spectral Clustering')
        compute_crosstab(y, y_pred)
        ari = metrics.adjusted_rand_score(y, y_pred)
        print(f'ARI: {ari:0.3f}')


'''
Model Evaluation
'''

def display_clusters(X, y, model_type):
    num_clusters = len(np.unique(y))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(f'{model_type}: Displaying {num_clusters} Clusters')
    plt.show()


def compute_crosstab(y, y_pred):
    df_crosstab = pd.crosstab(y, y_pred)
    df_crosstab.columns.name = 'actual'
    df_crosstab.index.name = 'predicted'
    print('Contingency Table')
    print(df_crosstab)
