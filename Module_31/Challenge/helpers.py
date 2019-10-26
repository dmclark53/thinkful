import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier

'''
Global Variables
'''

rfc_params = {
}

mlp_params = {

}

model_stats_dict = {
    'model_type': [],
    'train_stats': {
        'score': []
    },
    'cv_stats': {
        'mean_score': [],
        'std_score': []
    },
    'test_stats': {
        'score': []
    },
    'test_data': {
        'y_test': [],
        'y_pred': []
    }
}


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


def run_random_forest_model(X_train, X_test, y_train, y_test, num_folds=None, evaluate_model=False):

    model_stats_dict['model_type'] = 'Random Forest'

    rfc = ensemble.RandomForestClassifier(**rfc_params)
    cv_scores = cross_val_score(rfc, X_train, y_train, cv=num_folds)

    model_stats_dict['cv_stats']['mean_score'] = [np.mean(cv_scores)]
    model_stats_dict['cv_stats']['std_score'] = [np.std(cv_scores)]

    rfc_fit = rfc.fit(X_train, y_train)
    training_score = rfc_fit.score(X_train, y_train)
    model_stats_dict['train_stats']['score'] = training_score

    test_score = rfc_fit.score(X_test, y_test)
    model_stats_dict['test_stats']['score'] = test_score

    if evaluate_model:
        do_model_evaluation()


def run_mlp_model(X_train, X_test, y_train, y_test, num_folds=None, evaluate_model=False):

    model_stats_dict['model_type'] = 'MLP'

    mlp = MLPClassifier(**mlp_params)
    cv_scores = cross_val_score(mlp, X_train, y_train, cv=num_folds)

    model_stats_dict['cv_stats']['mean_score'] = [np.mean(cv_scores)]
    model_stats_dict['cv_stats']['std_score'] = [np.std(cv_scores)]

    rfc_fit = mlp.fit(X_train, y_train)
    training_score = rfc_fit.score(X_train, y_train)
    model_stats_dict['train_stats']['score'] = training_score

    test_score = rfc_fit.score(X_test, y_test)
    model_stats_dict['test_stats']['score'] = test_score

    if evaluate_model:
        do_model_evaluation()



'''
Model Evalution
'''


def do_model_evaluation():
    print(f"{model_stats_dict['model_type']} Model")
    print(f" * training score: {model_stats_dict['train_stats']['score']:0.3f}")
    print(f" * mean CV score: {model_stats_dict['cv_stats']['mean_score'][0]:0.3f}")
    print(f" * std CV score: {model_stats_dict['cv_stats']['std_score'][0]:0.3f}")
    print(f" * test score: {model_stats_dict['test_stats']['score']:0.3f}")
