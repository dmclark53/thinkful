import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import boxcox
from scipy.stats import chi2_contingency
from scipy.stats import normaltest
from sklearn.preprocessing import StandardScaler


# Global Variables
thresholds = {
    'tukey': 1.5
}

# Data Cleaning
def find_variable_types(df):
    df_described = df.describe(include='all').T
    cont_variables = df_described.loc[df_described['mean'].notnull(), :].index.values.tolist()
    cat_variables = df_described.loc[df_described['mean'].isnull(), :].index.values.tolist()
    return cont_variables, cat_variables


def print_unique_categories(df, cat_variables):
    for cat in cat_variables:
        print(f'{cat}: {df[cat].unique()}')


def fill_missing_categories(df, cat_variables):
    for cat in cat_variables:
        df[cat].fillna(value='None', inplace=True)
    return df.copy()


def find_na_columns(df, display_fractions=False):
    na_fractions = (df.isnull().sum()/df.isnull().count())*100
    if display_fractions:
        print(na_fractions[na_fractions != 0])
    else:
        return na_fractions[na_fractions != 0]


def stack_dataframe(df, column_list):
    df_stacked = df.loc[:, column_list].stack().copy()
    df_stacked.index = df_stacked.index.droplevel()
    df_stacked = df_stacked.reset_index().copy()
    df_stacked = df_stacked.rename(columns={'index': 'label', 0: 'value'})
    df_stacked = df_stacked.sort_values(by='label')
    df_stacked.index = np.arange(len(df_stacked))
    return df_stacked


def look_for_outliers(df, column_list, max_boxes=None):
    if len(column_list) > 1:
        if max_boxes:
            max_boxes = np.min([max_boxes, len(column_list)])
            num_plots = len(column_list) // max_boxes
            remainder_boxes = len(column_list) % max_boxes
            for i in range(0, num_plots * max_boxes, max_boxes):
                df_stacked = stack_dataframe(df, column_list[i:i + max_boxes])
                sns.catplot(x='label', y='value', data=df_stacked, height=6, aspect=12 / 6, kind='box')
                plt.show()
            if remainder_boxes > 0:
                df_stacked = stack_dataframe(df, column_list[i + max_boxes:])
                sns.catplot(x='label', y='value', data=df_stacked, height=6, aspect=12 / 6, kind='box')
                plt.show()
        else:
            df_stacked = stack_dataframe(df, column_list)
            sns.catplot(x='label', y='value', data=df_stacked, height=6, aspect=12 / 6, kind='box')
            plt.show()
    else:
        sns.catplot(y=column_list[0], data=df, height=6, aspect=12 / 6, kind='box')
        plt.show()


def apply_boxcox_transform(df, columns):
    for column in columns:
        min_value = df[column].min()
        offset = np.min([0, min_value])
        boxcox_transform, _ = boxcox(df[column] - offset + 1)
        df[f'{column}_bc'] = boxcox_transform + offset -1
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
    for column in column_list:
        mean_value = df[column].mean()
        std_value = df[column].std()
        df[f'{column}_sd'] = (df[column] - mean_value) / std_value
    return df


def select_cat_variables(df, target_var, cat_list, alpha=0.05):
    y = df[target_var].astype(str)
    cat_features = []
    print('Chi-squared Statistics:')
    for cat in cat_list:
        chi2, p, dof, expected = chi2_contingency(pd.crosstab(y, df[cat]))
        if p < alpha:
            # print(f'* {cat}: chi2 = {chi2}, p-value = {p}, dof = {dof}, expected = {expected}')
            print(f'* {cat}: chi2 = {chi2}, p-value = {p}, dof = {dof}')
            cat_features.append(cat)
    return cat_features


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
