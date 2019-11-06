import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
import seaborn as sns
import scipy.stats as stats
from sklearn import metrics
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

# Global variables
boxcox_lambdas = {}


def separate_variable_types(df):

    numeric_vars = df.select_dtypes(include=np.number).columns.tolist()
    string_vars = df.select_dtypes(include='object').columns.tolist()
    return numeric_vars, string_vars


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


thresholds = {
    'tukey': 1.5
}


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


def apply_log_transform(df, column_list):
    for column in column_list:
        min_value = df[column].min()
        offset = np.min([0, min_value])
        df[f'{column}_log'] = np.log(df[column] - offset + 1)


def apply_boxcox_transform(df, column_list):
    for column in column_list:
        min_value = df[column].min()
        offset = np.min([0, min_value])
        boxcox_transform, max_log = boxcox(df[column] - offset + 1)

        # Keep track of lambda value to use when inverse transforming
        boxcox_lambdas[f'{column}_bc'] = max_log

        df[f'{column}_bc'] = boxcox_transform


def apply_normalization(df, column_list):
    sub_df = df.loc[:, column_list].copy()
    df_norm = (sub_df - sub_df.mean()) / (sub_df.max() - sub_df.min())
    columns_map = {f'{x}':f'{x}_nm' for x in column_list}
    df_norm.rename(columns=columns_map, inplace=True)
    normalized_columns = df_norm.columns.tolist()
    for column in normalized_columns:
        df[column] = df_norm[column].values


transformation_functions = {
    'boxcox': apply_boxcox_transform,
    'log': apply_log_transform,
    'normalize': apply_normalization
}


def apply_transform(df, column_list, method):
    transformation_functions[method](df, column_list)


def perform_pca(df, features, num_components):
    X = df.loc[:, features]
    pca = PCA(n_components=num_components)
    pca_components = pca.fit_transform(X)
    variance_explained = pca.explained_variance_ratio_
    print('Variance Explained:')
    for component in range(num_components):
        print(f' * component {component + 1}: {variance_explained[component]:0.1%}')
    print(f' * total: {np.sum(variance_explained):0.1%}')

    # Update DataFrame with PCA components
    # Also, update the features list
    for component in range(num_components):
        df[f'pca_{component}'] = pca_components[:, component]
        features.append(f'pca_{component}')


def ttest_feature(df, feature):
    grouped = df.loc[:, [feature, 'class']].groupby('class')
    for label, group in grouped:
        sns.kdeplot(group[feature].values, label=f'{label}')
    plt.legend()
    plt.title(f'Feature: {feature}')
    plt.show()

    class_0 = df.loc[df['class'] == 0.0, feature]
    class_1 = df.loc[df['class'] == 1.0, feature]
    tstat, pval = stats.ttest_ind(class_0, class_1)
    print(f'T-Test Results for {feature}:')
    print(f' * t-stat: {tstat:0.3f}')
    print(f' * p-value: {pval:0.3e}')


'''
Modeling
'''

def fit_kmeans(X, y, num_clusters, evaluate_model=False):
    # X = X.values.reshape(-1, 1)
    y_pred = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(X)
    if evaluate_model:
        display_clusters(X, y_pred, 'K-Means')
        compute_crosstab(y, y_pred)
        ari = metrics.adjusted_rand_score(y, y_pred)
        print(f'ARI: {ari:0.3f}')


def fit_mean_shift(X, y, evaluate_model=False):
    # X = X.values.reshape(-1, 1)
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
    # X = X.values.reshape(-1, 1)
    sc = SpectralClustering(n_clusters=num_clusters,
                            eigen_solver='arpack',
                            eigen_tol=0.0,
                            affinity='rbf',
                            assign_labels='kmeans',
                            gamma=1.0,
                            degree=5)
    sc.fit(X)
    y_pred=sc.fit_predict(X)
    if evaluate_model:
        display_clusters(X, y_pred, 'Spectral Clustering')
        compute_crosstab(y, y_pred)
        ari = metrics.adjusted_rand_score(y, y_pred)
        print(f'ARI: {ari:0.3f}')


def fit_ward(X, y, num_clusters, evaluate_model=False):

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    sc = AgglomerativeClustering(n_clusters=num_clusters,
                                 linkage='ward',
                                 connectivity=connectivity)
    sc.fit(X)
    y_pred=sc.fit_predict(X)
    if evaluate_model:
        display_clusters(X, y_pred, 'Ward')
        compute_crosstab(y, y_pred)
        ari = metrics.adjusted_rand_score(y, y_pred)
        print(f'ARI: {ari:0.3f}')


def fit_affinity_model(X):
    af = AffinityPropagation().fit(X)
    print('Done')

    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    labels = af.labels_

    print(f'There are {n_clusters_} estimated clusters.')

'''
Model Evaluation
'''


def display_clusters(X, y, model_type):
    num_clusters = len(np.unique(y))
    df = X.copy()
    df['y_pred'] = y
    sns.pairplot(df, vars=X.columns.tolist(), hue='y_pred')
    # columns = X.columns.tolist()
    # sns.scatterplot(x=columns[0], y=columns[1], hue='y_pred', data=df)
    # plt.title(f'{model_type}: Displaying {num_clusters} Clusters')
    plt.show()


def compute_crosstab(y, y_pred):
    df_crosstab = pd.crosstab(y, y_pred)
    df_crosstab.columns.name = 'actual'
    df_crosstab.index.name = 'predicted'
    print('Contingency Table')
    print(df_crosstab)