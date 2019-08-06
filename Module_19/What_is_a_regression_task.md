# What is a regression problem?

1. Let's assume that you have World Bank data on financial, economic and social indicators for several countries. You 
want to measure the factors that affect the level of development in these countries. To this end, you decide to use per 
capita income as a proxy for the development level, which is defined as the national income divided by the population. 
You want to use some features in your dataset to predict per capita income. Is this task a classification or a 
regression task? Why?
**Answer:**
This is a regression problem. The problem is to predict per capita income. This is a function of national income and
population, which will create a continuous set of numbers. Therefore, this is a regression problem.

2. Which of the following arguments are false and why?
* OLS is a special type of linear regression models.
    * **FALSE**  OLS is used to optimize a linear regression. It itself is not a model.
* Regression models become useless if they don’t meet the assumptions of linear regression.
    * **TRUE**
    * **Correction:** This should be **FALSE**. A regression problem can be adjusted to meet the assumptions of a
    regression problem.
* Estimation and prediction are the same thing in the context of linear regression models.
    * **TRUE**
    * **Correction** This should be **FALSE**. Estimation is used to estimate the regression coefficients. Prediction
    is used to make predictions on the outcome.
* Linear regression is only one kind of regression model. Regression problems can also be solved with other kind of
 models like Support Vector Machines or Random Forests.
    * **TRUE**  The objective of a regression problem is to create a function using the features to predict the 
    target variable. The function does not necessarily have to be linear.

3. Assume that your project manager wants you to discover which free services your company offers make your customers 
buy more of your paid services. Formulate this task as a regression problem and write down the potential outcome and 
features that you’d like to work on.
* features: 
    * **number of users**
    * **paid service cost**
    * **sign-up rate for free service**
    * sign-up date
    * **free service**
    * **paid service**
* outcome:
    * sign-up rate for paid service