import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bartlett
from scipy.stats import boxcox
from scipy.stats import chi2_contingency
from scipy.special import inv_boxcox
from scipy.stats import levene
from scipy.stats import normaltest
from sklearn import ensemble
from sklearn.feature_selection import f_regression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf


'''
Global Variables
'''
thresholds = {
    'tukey': 1.5
}


boxcox_lambdas = {}


gbc_params = {'n_estimators': 500,
              'max_depth': 2,
              'loss': 'exponential'}


linear_model_stats = {
    'model_type': [],
    'train_stats': {
        'r-squared': [],
        'features': [],
        'coeffs': [],
        'F-stat': [],
        'pval': []
    },
    'cv_stats': {
        'mean_r-squared': []
    },
    'test_stats': {
        'r-squared': [],
        'mse': [],
        'rmse': [],
        'mae': [],
        'mape': []
    },
    'test_data': {
        'y_test': [],
        'y_pred': []
    }
}

binary_classifier_stats = {
    'train_stats' : {
        'sensitivity': [],
        'specificitity': [],
        'type_1_error': [],
        'type_2_error': [],
        'score': []
    },
    'test_stats': {
        'sensitivity': [],
        'specificitity': [],
        'type_1_error': [],
        'type_2_error': [],
        'score': []
    }
}

'''
Data Cleaning
'''


def find_variable_types(df):
    # Separate variables into continous and categorical
    df_described = df.describe(include='all').T

    # If the mean can be calculated, then this must be a continous variable
    cont_variables = df_described.loc[df_described['mean'].notnull(), :].index.values.tolist()

    # If there is no mean value, then this will be a categorical variable
    cat_variables = df_described.loc[df_described['mean'].isnull(), :].index.values.tolist()
    return cont_variables, cat_variables


def find_na_columns(df, display_fractions=False):
    na_fractions = (df.isnull().sum()/df.isnull().count())
    if display_fractions:
        print('Variables with missing values and their fraction of missing values:')
        print(na_fractions[na_fractions != 0])
    else:
        return na_fractions[na_fractions != 0]


def print_unique_categories(df, cat_variables):
    for cat in cat_variables:
        print(f'{cat}: {df[cat].unique()}')


def find_category_counts(df, cat_variables):

    cat_counts = {'category': [],
                  'count': []}

    if len(cat_variables) == 0:
        print('No categorical variables found.')
        df_cat_counts = pd.DataFrame(cat_counts)
    else:
        for cat in cat_variables:
            cat_counts['category'].append(cat)
            cat_counts['count'].append(len(df[cat].unique()))
        df_cat_counts = pd.DataFrame(cat_counts)
        df_cat_counts.sort_values(by='count', ascending=False, inplace=True)
        df_cat_counts.index = np.arange(len(df_cat_counts))
    return df_cat_counts


def remove_high_category_counts(df, cat_variables, thresh=5):

    df_cat_counts = find_category_counts(df, cat_variables)
    accepted_cats = df_cat_counts.loc[df_cat_counts['count'] <= thresh, 'category'].tolist()

    cats_to_remove = [x for x in cat_variables if x not in accepted_cats]
    old_column_count = len(df.columns)
    df.drop(cats_to_remove, axis=1, inplace=True)
    new_column_count = len(df.columns)

    print(f'{len(cats_to_remove)} categorical variables with at least {thresh} categories will be removed.')
    print(f'The dataset went from {old_column_count} to {new_column_count} columns.')


def fill_missing_categories(df, cat_variables):
    for cat in cat_variables:
        df[cat].fillna(value='None', inplace=True)
    return df.copy()


def stack_dataframe(df, column_list):
    df_stacked = df.loc[:, column_list].stack().copy()
    df_stacked.index = df_stacked.index.droplevel()
    df_stacked = df_stacked.reset_index().copy()
    df_stacked = df_stacked.rename(columns={'index': 'label', 0: 'value'})
    df_stacked = df_stacked.sort_values(by='label')
    df_stacked.index = np.arange(len(df_stacked))
    return df_stacked


def look_for_outliers(df, column_list, max_boxes=None, log_scale=False):

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


def apply_boxcox_transform(df, columns):
    for column in columns:
        min_value = df[column].min()
        offset = np.min([0, min_value])
        boxcox_transform, max_log = boxcox(df[column] - offset + 1)

        # Keep track of lambda value to use when inverse transforming
        boxcox_lambdas[f'{column}_bc'] = max_log

        df[f'{column}_bc'] = boxcox_transform
    return df


def apply_tukey(df, column, thresh=1.5):
    q75, q25 = np.percentile(df[column], [75, 25])
    iqr = q75 - q25
    min_value = q25 - thresh*iqr
    max_value = q75 + thresh*iqr
    df[f'{column}_tk'] = df[column].apply(lambda x: np.min([x, max_value]))
    df[f'{column}_tk'] = df[f'{column}_tk'].apply(lambda x: np.max([x, min_value]))
    return df


def correct_outliers(df, column_list):
    for column in column_list:
        # Tukey
        df = apply_tukey(df, column, thresh=thresholds['tukey'])

    return df


def check_for_normality(df, column_list):
    print('Normal test results:')
    for column in column_list:
        normal_stat, pval = normaltest(df[column])
        print(f'* {column}: normal_stat = {normal_stat:0.3f}, p-value = {pval:0.4f}.')


def standardize(df, column_list):
    standardized_columns = [f'{x}_sd' for x in column_list]
    data = df.loc[:, column_list].values
    scaler = StandardScaler()
    scaler.fit(data)
    df_standardized = pd.DataFrame(scaler.transform(data), columns=standardized_columns, index=df.index)
    df = pd.concat([df, df_standardized], axis=1).copy()
    return df


def dummyify(df, column_list):
    dummy_list = []
    for column in column_list:
        df_dummy = pd.get_dummies(df[column], drop_first=True)
        dummy_columns = df_dummy.columns.tolist()
        new_dummy_columns = [f'{column}_{x}_oh' for x in dummy_columns]
        df_dummy.columns = new_dummy_columns
        dummy_list.append(df_dummy)
    dummy_list.append(df)
    return pd.concat(dummy_list, axis=1)


def invert_transformations(df, column, data):
    split_column = column.split('_')
    while split_column:
        extension = split_column.pop()
        if extension == 'bc':
            joined_column = '_'.join(split_column)
            bc_column = f'{joined_column}_{extension}'
            max_log = boxcox_lambdas[bc_column]
            data = inv_boxcox(data, max_log) - 1 + np.min([0, df[joined_column].min()])

        if extension == 'sd':
            joined_column = '_'.join(split_column)
            df_described = df.describe().T.loc[:, ['mean', 'std']]
            mean_value = df_described.loc[joined_column, 'mean']
            std_value = df_described.loc[joined_column, 'std']
            data = data * std_value + mean_value

        if extension == 'tk':
            break

    return data

def transform_test(df, column):

    initial_mean = df[column].mean()
    initial_std = df[column].std()

    # 1. Boxcox transform
    boxcox_transformed, max_log = boxcox(df[column].values)

    # 2. Standardize
    mean_value = boxcox_transformed.mean()
    std_value = boxcox_transformed.std()
    standardized = (boxcox_transformed - mean_value) / std_value

    # 3. Inverse standardize
    inverse_standardized = (standardized * std_value) + mean_value

    # 4. Inverse Boxcox transform
    inversed_boxcox = inv_boxcox(inverse_standardized, max_log)
    # inversed_boxcox = (inversed_boxcox + initial_mean) * initial_std

    print('Test Results:')
    print(f'max_log = {max_log}')
    print(f' * Intial Array: mean = {df[column].values.mean()}, std = {df[column].values.std()}')
    print(f' * Boxcox Transformed: mean = {boxcox_transformed.mean()}, std = {boxcox_transformed.std()}')
    print(f' * Standardized: mean = {standardized.mean()}, std = {standardized.std()}')
    print(f' * Inverse Standardized: mean = {inverse_standardized.mean()}, std = {inverse_standardized.std()}')
    print(f' * Inverse Boxcox: mean = {inversed_boxcox.mean()}, std = {inversed_boxcox.std()}')


'''
Feature Selection
'''

def select_cat_variables(df, target_var, cat_list, alpha=0.05):
    y = df[target_var]
    cat_features = []
    print(f'Feature Chi-squared Statistics: (p-value < {alpha})')
    for cat in cat_list:
        chi2, p, dof, expected = chi2_contingency(pd.crosstab(y, df[cat]))
        if p < alpha:
            print(f'* {cat}: chi2 = {chi2:0.3f}, p-value = {p:0.3e}, dof = {dof}')
            cat_features.append(cat)
    print(f'{len(df.columns) - len(cat_features)} categorical features will be removed.')
    return cat_features


def ftest_feature_selection(alpha=0.05):

    df_ftest = pd.DataFrame({'feature': linear_model_stats['train_stats']['features'],
                             'F': linear_model_stats['train_stats']['F-stat'],
                             'pval': linear_model_stats['train_stats']['pval'],
                             'coeffs': linear_model_stats['train_stats']['coeffs']})
    df_ftest = df_ftest[df_ftest['pval'] < alpha]
    return df_ftest


def find_correlated_features(df, thresh=0.8):
    df_corr_abs = df.corr().abs()
    upper_triangle = df_corr_abs.where(np.triu(np.ones(df_corr_abs.shape), k=1).astype(np.bool))
    correlated_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > thresh)]
    if len(correlated_columns) > 0:
        print(f'Correlated Columns (r > {thresh}):')
        for column in correlated_columns:
            print(f'* {column}')
        print(f'{len(correlated_columns)} correlated features will be removed.')
    else:
        print(f'No columns found with r > {thresh}.')
    return correlated_columns


'''
Modeling
'''


def run_linear_model(model_class, X_train, X_test, y_train, y_test, num_folds=None, alpha=None, l1_ratio=None,
                     print_results=False):

    if model_class.__module__.split('.')[-1] == 'base':
        linear_model_stats['model_type'] = 'OLS'
        model = model_class()
    elif model_class.__module__.split('.')[-1] == 'ridge':
        linear_model_stats['model_type'] = 'ridge'
        model = model_class(alpha=alpha)
    elif (model_class.__module__.split('.')[-1] == 'coordinate_descent') and (l1_ratio is None):
        linear_model_stats['model_type'] = 'lasso'
        model = model_class(alpha=alpha)
    elif (model_class.__module__.split('.')[-1] == 'coordinate_descent') and (l1_ratio is not None):
        linear_model_stats['model_type'] = 'elastic_net'
        model = model_class(alpha=alpha, l1_ratio=l1_ratio)
    else:
        print(f'Unrecognized model class: {model_class}')
        return None

    # Train model
    model.fit(X_train, y_train)
    F, pval = f_regression(X_train, y_train)
    linear_model_stats['train_stats']['r-squared'] = [model.score(X_train, y_train)]
    linear_model_stats['train_stats']['features'] = X_train.columns.tolist()
    linear_model_stats['train_stats']['coeffs'] = model.coef_
    linear_model_stats['train_stats']['F-stat'] = F
    linear_model_stats['train_stats']['pval'] = pval

    # Perform cross-validation
    if num_folds:
        cv_scores = cross_val_score(model, X_train, y_train, cv=num_folds)
        linear_model_stats['cv_stats']['mean_r-squared'] = [np.mean(cv_scores)]

    # Evaluate predictions on test set
    y_pred = model.predict(X_test)

    # Save model datasets
    linear_model_stats['test_data']['y_test'] = y_test
    linear_model_stats['test_data']['y_pred'] = y_pred

    # Save model statistics
    linear_model_stats['test_stats']['r-squared'] = [model.score(X_test, y_test)]
    linear_model_stats['test_stats']['mse'] = [mse(y_test, y_pred)]
    linear_model_stats['test_stats']['rmse'] = [rmse(y_test, y_pred)]
    linear_model_stats['test_stats']['mae'] = [mae(y_test, y_pred)]
    linear_model_stats['test_stats']['mape'] = [mape(y_test, y_pred)]

    if print_results:
        print_prediction_metrics(y_test, y_pred)


def run_knn_regressor(X_train, X_test, y_train, y_test, k=5, weights='uniform', print_results=False):

    knn = KNeighborsRegressor(n_neighbors=k, weights=weights)

    # Train the model
    knn.fit(X_train, y_train)
    F, pval = f_regression(X_train, y_train)
    linear_model_stats['train_stats']['r-squared'] = [knn.score(X_train, y_train)]
    linear_model_stats['train_stats']['features'] = X_train.columns.tolist()
    linear_model_stats['train_stats']['F-stat'] = F
    linear_model_stats['train_stats']['pval'] = pval

    # Evaluate predictions on test set
    y_pred = knn.predict(X_test)

    # Save model datasets
    linear_model_stats['test_data']['y_test'] = y_test
    linear_model_stats['test_data']['y_pred'] = y_pred

    # Save model statistics
    linear_model_stats['test_stats']['r-squared'] = [knn.score(X_test, y_test)]
    linear_model_stats['test_stats']['mse'] = [mse(y_test, y_pred)]
    linear_model_stats['test_stats']['rmse'] = [rmse(y_test, y_pred)]
    linear_model_stats['test_stats']['mae'] = [mae(y_test, y_pred)]
    linear_model_stats['test_stats']['mape'] = [mape(y_test, y_pred)]

    if print_results:
        print_prediction_metrics(y_test, y_pred)


def run_gbc_classifier(X_train, X_test, y_train, y_test, print_results=False):

    gbc = ensemble.GradientBoostingClassifier(**gbc_params)
    gbc_fit = gbc.fit(X_train, y_train)

    # Evaluate training set
    train_predictions = gbc.predict(X_train)
    binary_classifier_stats['train_stats']['score'] = gbc_fit.score(X_train, y_train)
    compute_binary_classifier_stats(y_train, train_predictions, 'train')

    if print_results:
        print_binary_classification_stats('train')

    # Evaluate test set
    test_predictions = gbc.predict(X_test)
    binary_classifier_stats['test_stats']['score'] = gbc_fit.score(X_test, y_test)
    compute_binary_classifier_stats(y_test, test_predictions, 'test')

    if print_results:
        print_binary_classification_stats('test')

    # Save feature importances
    feature_importance = gbc.feature_importances_

    if print_results:
        plot_feature_importance(feature_importance, X_train.columns)


def plot_feature_importance(feature_importance, columns):

    scaled_importances = 100.0 * (feature_importance / feature_importance.max())
    sorted_indices = np.argsort(scaled_importances)
    pos = np.arange(sorted_indices.shape[0]) + 0.5

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.barh(pos, scaled_importances[sorted_indices], align='center')
    plt.yticks(pos, columns[sorted_indices])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def grid_search(model_class, X, y, alpha_list, l1_ratio_list=None, num_folds=None):

    store_alphas = []
    store_l1_ratios = []
    store_mean_cv_scores = []

    df = pd.DataFrame({
        'alpha': [],
        'mean_cv_score': []
    })

    if l1_ratio_list is None:

        for alpha in alpha_list:
            store_alphas.append(alpha)
            run_linear_model(model_class, X, y, num_folds=num_folds, alpha=alpha)
            store_mean_cv_scores.append(linear_model_stats['cv_stats']['mean_r-squared'][0])

        df['alpha'] = store_alphas
        df['mean_cv_score'] = store_mean_cv_scores

    else:

        for alpha in alpha_list:
            for l1_ratio in l1_ratio_list:
                store_alphas.append(alpha)
                store_l1_ratios.append(l1_ratio)
                run_linear_model(model_class, X, y, num_folds=num_folds, alpha=alpha, l1_ratio=l1_ratio)
                store_mean_cv_scores.append(linear_model_stats['cv_stats']['mean_r-squared'][0])

        df['alpha'] = store_alphas
        df['mean_cv_score'] = store_mean_cv_scores
        df['l1_ratio'] = store_l1_ratios

    return df


'''
Model Evaluation
'''

def check_feature_linearity(df, feature_list, y_pred):
    for feature in feature_list:
        plt.scatter(df[feature], y_pred)
        plt.title(f'Inspect Linearity for Feature {feature}')
        plt.ylabel('target prediction')
        plt.xlabel(f'{feature}')
        plt.show()


def check_homoscedasticity(y_pred, errors):
    plt.scatter(y_pred, errors)
    plt.title('Errors vs. Predictions')
    plt.xlabel('predictions')
    plt.ylabel('errors')
    plt.show()

    bartlett_stats = bartlett(y_pred, errors)
    levene_stats = levene(y_pred, errors)

    print(f'The Bartlett test is {bartlett_stats[0]}, with a p-value of {bartlett_stats[1]}.')
    print(f'The Levine test is {levene_stats[0]}, with a p-value of {levene_stats[1]}.')


def check_error_autocorrelation(errors):
    error_autocorr = acf(errors, fft=False)

    plt.plot(error_autocorr[1:])
    plt.title('Error Autocorrelation Values')
    plt.xlabel('lag')
    plt.ylabel('autocorrelation')
    plt.show()

    print(f'Max autocorrelation: {error_autocorr[1:].max():0.3f}')
    print(f'Min autocorrelation: {error_autocorr[1:].min():0.3f}')


def compare_scores():
    print('Comparing Scores:')
    print(f" * train: {linear_model_stats['train_stats']['r-squared'][0]:0.3f}")
    print(f" * test: {linear_model_stats['test_stats']['r-squared'][0]:0.3f}")


'''
Prediction Evaluation
'''


def mae(y, y_pred):
    return np.sum(np.abs(y - y_pred)) / len(y)


def mse(y, y_pred):
    return np.sum((y - y_pred)**2) / len(y)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))


def mape(y, y_pred):
    return np.sum(np.abs((y - y_pred) / y_pred)) / len(y)


def print_prediction_metrics(y, y_pred):
    print('Prediction Metrics:')
    print(f"* R-squared: {linear_model_stats['test_stats']['r-squared'][0]:0.3f}")
    print(f'* MAE = {mae(y, y_pred):0.3f}')
    print(f'* MSE = {mse(y, y_pred):0.3f}')
    print(f'* RMSE = {rmse(y, y_pred):0.3f}')
    print(f'* MAPE = {mape(y, y_pred):0.3%}')


def plot_predictions(df, column):

    y_test_transformed = linear_model_stats['test_data']['y_test']
    y_pred_transformed = linear_model_stats['test_data']['y_pred']

    y_test = invert_transformations(df, column, y_test_transformed)
    y_pred = invert_transformations(df, column, y_pred_transformed)

    fig, ax = plt.subplots(figsize=(14, 6))

    formatter = ticker.FormatStrFormatter('$%1.2f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.scatter(y_test, y_pred)
    plt.title('Housing Sale Price: Predictions vs. Actual Value')
    plt.xlabel('Actual Sale Price')
    plt.ylabel('Predicted Sale Price')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_visible(True)
        tick.label2.set_visible(False)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_visible(True)
        tick.label2.set_visible(False)

    plt.show()


def compute_binary_classifier_stats(y, y_pred, dataset_type):

    # Create crosstab table
    df_crosstab = pd.crosstab(y, y_pred, margins=True)

    # Compute counts for each region in confusion matrix
    true_positive_count = df_crosstab.loc[0, 0]
    true_negative_count = df_crosstab.loc[1, 1]
    false_negative_count = df_crosstab.loc[1, 0]
    false_positive_count = df_crosstab.loc[0, 1]

    # Compute type 1 error
    type_1_error = false_positive_count / df_crosstab.loc['All', 'All']
    binary_classifier_stats[f'{dataset_type}_stats']['type_1_error'] = type_1_error

    # Compute type 2 error
    type_2_error = false_negative_count / df_crosstab.loc['All', 'All']
    binary_classifier_stats[f'{dataset_type}_stats']['type_2_error'] = type_2_error

    # Compute sensitivity
    total_positives = true_negative_count + false_positive_count
    sensitivity = (true_negative_count / total_positives)
    binary_classifier_stats[f'{dataset_type}_stats']['sensitivity'] = sensitivity

    # Compute specificity
    total_negatives = true_positive_count + false_negative_count
    specificity = (true_positive_count / total_negatives)
    binary_classifier_stats[f'{dataset_type}_stats']['specificity'] = specificity


def print_binary_classification_stats(dataset_type):

    # Metrics
    score = binary_classifier_stats[f'{dataset_type}_stats']['score']
    type_1_error = binary_classifier_stats[f'{dataset_type}_stats']['type_1_error']
    type_2_error = binary_classifier_stats[f'{dataset_type}_stats']['type_2_error']
    sensitivity = binary_classifier_stats[f'{dataset_type}_stats']['sensitivity']
    specificity = binary_classifier_stats[f'{dataset_type}_stats']['specificity']

    print(f'{dataset_type.title()} Metrics:')
    print(f' * score: {score:0.3%}')
    print(f' * type 1 error: {type_1_error:0.3%}')
    print(f' * type 2 error: {type_2_error:0.3%}')
    print(f' * sensitivity: {sensitivity:0.3%}')
    print(f' * specificity: {specificity:0.3%}')
