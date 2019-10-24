from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier


def run_mlp(X_train, X_test, y_train, y_test, hidden_layer_sizes):

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    mlp.fit(X_train, y_train)

    # cv_scores = cross_val_score(mlp, X_train, y_train, cv=5)
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test, y_test)

    # print(f'Cross-validation scores: {cv_scores}')
    print(f'Training score: {train_score:0.3f}')
    print(f'Test score: {test_score:0.3f}')
