# methods.py
import numpy as np
import pandas as pd

# import constants as con


def search_keywords(explanation, keyword):
    explanation_list = explanation.split(' ')
    in_explanation = [keyword in x.lower() for x in explanation_list]
    if np.sum(in_explanation) > 0:
        return 1
    else:
        return 0


def tag_keywords(df, keywords):
    for keyword in keywords:
        df[keyword] = df.apply(lambda x: search_keywords(x['explanation'], keyword), axis=1)


def gather_keywords(df, keywords):
    total = 0
    all_gathered_keywords = []
    for keyword in keywords:
        if 1 in df[keyword].value_counts().index:
            num_tags = df[keyword].value_counts()[1]
            gathered_keyword = keyword
            if gathered_keyword == 'galax':
                gathered_keyword = 'galaxy'
            gathered_keywords = [gathered_keyword.title() for _ in range(num_tags)]
            all_gathered_keywords += gathered_keywords
            total += num_tags
    keyword_counts = pd.Series(all_gathered_keywords)
    return keyword_counts
