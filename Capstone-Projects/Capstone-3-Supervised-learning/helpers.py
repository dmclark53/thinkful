import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bartlett
from scipy.stats import boxcox
from scipy.stats import chi2_contingency
from scipy.stats import levene
from scipy.stats import normaltest
from sklearn.feature_selection import f_regression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from statsmodels.tsa.stattools import acf


'''
Global Variables
'''
thresholds = {
    'tukey': 1.5
}


boxcox_lambdas = {}


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
    'classes': [],
    'features': [],
    'train_stats' : {
        'sensitivity': [],
        'specificitity': [],
        'type_1_error': [],
        'type_2_error': [],
        'score': [],
        'coeffs': []
    },
    'test_stats': {
        'sensitivity': [],
        'specificitity': [],
        'type_1_error': [],
        'type_2_error': [],
        'score': [],
        'f1_scores': [],
        'coeffs': [],
        'y_probs': []
    },
    'test_data': {
        'y_pred': []
    }
}

'''
Missing Values
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


'''
Data Exploration
'''


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
Transformative functions
'''


def standardize(df, column_list):
    for column in column_list:
        std_column = df[column].std()
        mean_column = df[column].mean()
        df[f'{column}_sd'] = (df[column] - mean_column) / std_column


def normalize(df, column_list):
    for column in column_list:
        max_value = df[column].max()
        min_value = df[column].min()
        df[f'{column}_nm'] = (df[column] - min_value) / (max_value - min_value)


def apply_boxcox_transform(df, column_list):
    for column in column_list:
        min_value = df[column].min()
        offset = np.min([0, min_value])
        boxcox_transform, max_log = boxcox(df[column] - offset + 1)

        # Keep track of lambda value to use when inverse transforming
        boxcox_lambdas[f'{column}_bc'] = max_log

        df[f'{column}_bc'] = boxcox_transform


def apply_log_transform(df, column_list):
    for column in column_list:
        min_value = df[column].min()
        offset = np.min([0, min_value])
        df[f'{column}_log'] = np.log(df[column] - offset + 1)


transformation_functions = {
    'boxcox': apply_boxcox_transform,
    'log': apply_log_transform,
    'standardize': standardize,
    'normalize': normalize
}


def apply_transform(df, column_list, method):
    transformation_functions[method](df, column_list)


'''
Categorical variables
'''


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


'''
Outlier functions
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
Data Evaluation
'''

def check_for_normality(df, column_list):
    print('Normal test results:')
    for column in column_list:
        normal_stat, pval = normaltest(df[column])
        print(f'* {column}: normal_stat = {normal_stat:0.3f}, p-value = {pval:0.4f}.')


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
    print(f'{len(cat_list) - len(cat_features)} categorical features will be removed.')
    return cat_features


'''
Modeling
'''


def run_linear_regressor(model_class, X_train, X_test, y_train, y_test, num_folds=None, alpha=None, l1_ratio=None,
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
        print_regression_model_results()


def run_svc_classifier(X_train, X_test, y_train, y_test):  # , print_results=False, coeff_thresh=0):

    svc = SVC(kernel='linear', probability=True)
    svc_fit = svc.fit(X_train, y_train)
    binary_classifier_stats['features'] = X_test.columns.tolist()
    binary_classifier_stats['classes'] = np.unique(y_test)
    binary_classifier_stats['train_stats']['coeffs'] = svc_fit.coef_[0]

    # Evaluate training set
    train_predictions = svc.predict(X_train)
    binary_classifier_stats['train_stats']['score'] = svc_fit.score(X_train, y_train)
    compute_binary_classifier_stats(y_train, train_predictions, 'train')

    # Evaluate test set
    y_pred = svc.predict(X_test)
    binary_classifier_stats['test_data']['y_pred'] = y_pred
    binary_classifier_stats['test_data']['y_test'] = y_test

    binary_classifier_stats['test_stats']['score'] = svc_fit.score(X_test, y_test)
    binary_classifier_stats['test_stats']['f1_scores'] = f1_score(y_test, y_pred, average=None)
    y_probs = svc.predict_proba(X_test)
    positive_probs = y_probs[:, 1]
    binary_classifier_stats['test_stats']['y_probs'] = positive_probs
    compute_binary_classifier_stats(y_test, y_pred, 'test')


'''
Linear Model Evaluation
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


def mae(y, y_pred):
    return np.sum(np.abs(y - y_pred)) / len(y)


def mse(y, y_pred):
    return np.sum((y - y_pred)**2) / len(y)


def rmse(y, y_pred):
    return np.sqrt(mse(y, y_pred))


def mape(y, y_pred):
    return np.sum(np.abs((y - y_pred) / y_pred)) / len(y)


def print_regression_model_results():
    print('Cross-validation Metrics:')
    print(f"* Mean R-squared: {linear_model_stats['cv_stats']['mean_r-squared'][0]:0.3f}")
    print('Training Metrics:')
    print(f"* R-squared: {linear_model_stats['train_stats']['r-squared'][0]:0.3f}")
    print('Test Metrics:')
    print(f"* R-squared: {linear_model_stats['test_stats']['r-squared'][0]:0.3f}")
    print(f"* MAE = {linear_model_stats['test_stats']['mae'][0]:0.3f}")
    print(f"* MSE = {linear_model_stats['test_stats']['mse'][0]:0.3f}")
    print(f"* RMSE = {linear_model_stats['test_stats']['rmse'][0]:0.3f}")
    print(f"* MAPE = {linear_model_stats['test_stats']['mape'][0]:0.3%}")


'''
Binary Classification Model Evaluation 
'''
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
    if dataset_type == 'test':
        classes = binary_classifier_stats['classes']
        f1_scores = binary_classifier_stats['test_stats']['f1_scores']
        for c, f in zip(classes, f1_scores):
            print(f' * class {c} F1 score: {f:0.3f}')


def plot_confussion_matrix(y_true, y_pred):
    cm_array = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm_array, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title('Confussion Matrix')
    ax.yaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0.5))
    sns.heatmap(df_cm, cmap='Blues', annot=True, fmt='g', ax=ax)
    ax.set(yticks=[0, 2],
           xticks=[0, 1])
    plt.show()


def plot_feature_importance(features, coeffs, coeff_thresh=0):
    df_fi = pd.DataFrame({'feature': features, 'coef': coeffs})
    df_fi_sorted = df_fi[(df_fi['coef'] > coeff_thresh) |
                         (df_fi['coef'] < -1.0*coeff_thresh)].sort_values(by='coef', ascending=False)
    f, ax = plt.subplots(figsize=(6, 8))
    df_fi_sorted.plot.barh(x='feature', ax=ax)
    plt.title(f'Features w/ Coefficents > |{coeff_thresh}|')
    plt.show()


def plot_roc_curve():
    # Compute ROC curve and area under curve
    y_test = binary_classifier_stats['test_data']['y_test']
    y_probs = binary_classifier_stats['test_stats']['y_probs']
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_probs)
    plt.plot(false_positive_rate, true_positive_rate, marker='.')
    plt.title('ROC Curve')
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Postive Rate')
    plt.show()


def evaluate_binary_classifier_model():
    print_binary_classification_stats('train')
    print_binary_classification_stats('test')

