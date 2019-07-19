# Exploratory Data Analysis

1. What is the goal of EDA (exploratory data analysis)?

**Answer:**
The goal of EDA is to clean up a dataset so that it is scientifically useful, visualize a dataset to understand its
contents, and extract/engineer features from the dataset to address the hypothesis of the data science problem.

2. Suppose that you are given a dataset of customer product reviews for an e-commerce company. Each review is scored as 
a Likert-style survey item where 1 indicates a negative sentiment about the product and a 5 is positive. These reviews 
are collected on the company's website.

    a. What problems do you expect to find in the raw data? 
    
    **Answer:**
    * Empty values for unanswered questions.
    * Biases towards all fives or all ones.
    * Review problems:
        * String formatting
        * Misspellings
        * Unrecognizable characters
        * Unnecessary words (e.g the, and, then, etc.)
    
    b. If your task is to build features that give information about customer sentiments, how would you approach this 
    task and what kind of methods would you apply to accomplish it?
    
    **Answer:**
    1. Visualize distribution of Likert scores.
    2. Visualize distribution of review word counts by Likert score.
    3. Extract common words from each review. Keep track of their usage count.
    
    c. Try to identify some potentially useful features that you might derive from the raw data. How would you derive 
    them and how would you assess the usefulness of those features?
    
    **Answer:**
    * Common words in reviews for positive/negative sentiment.
        1. Group data by Likert score.
        2. Search for common words in reviews for each group.
        3. Tag each Likert group by common keywords.
        4. Determine how common each word is to each group. The more common words would be the more useful features.