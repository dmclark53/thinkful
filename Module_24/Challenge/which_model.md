# Which Model?

## Scenarios

1. Predict the running times of prospective Olympic sprinters using data from the last 20 Olympics.
    * This is a linear regression problem. The goal is to predict a continuous variable, running time, given a set of
    features from this dataset of Olympic runners.

2. You have more features (columns) than rows in your dataset.
    * Perform feature engineering
        * Remove correlated continuous variables
        * Use a Chi-squared test to remove categorical variables that aren't related to the target variable.
        * Perform PCA to reduce the dimensionality of the data.

3. Identify the most important characteristic predicting likelihood of being jailed before age 20.
    * A gradient boosting algorithm would be ideal for this problem. The output from the model fit can include
     information on which features are most important in the prediction.

4. Implement a filter to “highlight” emails that might be important to the recipient.
    * This is a binary classification problem to predict important and non-important emails. A good algorithm choice
     would be a Naive Bayes model, which performs well for this kind of problem.

5. You have 1000+ features.
    * I would use the same answer as question 2, above, since the problem here is way too many features.
    * Perform feature engineering
        * Remove correlated continuous variables
        * Use a Chi-squared test to remove categorical variables that aren't related to the target variable.
        * Perform PCA to reduce the dimensionality of the data.

6. Predict whether someone who adds items to their cart on a website will purchase the items.
    * This is a Bayesian likelihood problem. The goal here is to:
        1. Predict the likelihood of a purchase given the items in the cart.
        2. Update the cart with new items.
        3. Redo likelihood model fit and see if the prediction has changed.

7. Your dataset dimensions are 982400 x 500.
    * Despite ~500 features, there are many more row than columns. Therefore, reducing the dimensionality of the
    dataset is not necessarily that important. A bigger issue is the dataset size. To solve this, I would store the
    dataset in a SQL database, which will make it quicker to access. If I reduce the number of features, I could
    query for only those columns of interest in the dataset.

8. Identify faces in an image.
    * This a multi-class classification problem. While out of scope for this section, a neural network such as a
     convolutional neural network would be ideal for this kind of problem.

9. Predict which of three flavors of ice cream will be most popular with boys vs girls.
    * Here, the goal is to make predictions for group combinations. A clustering algorithm, like K-nearest neighbors
    , might perform well here.