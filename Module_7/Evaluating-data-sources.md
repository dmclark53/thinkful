# Evaluating Data Sources

## Drill 1

**Data Source:** Amsterdam availability data scraped from AirBnB on December
24th. **Question:** What are the popular neighborhoods in Amsterdam?

### Solution:
There are two sources of bias here. One is restricting the analysis to only data
from AirBnB. The other is choosing a date right before Christmas. People's
travel preferences could be different right before a holiday.  Therefore, to
correct these biases, I would reframe the analysis.

**Data Source:** Amsterdam availability data scraped from AirBnB, Bookings,
Expedia, and Hotels.com on March 15th, June 15th, September 15th, and December
15th. **Question:** What are the popular neighborhoods in Amsterdam?

## Drill 2

**Data Source:** Mental health services use on September 12, 2001 in San
Francisco, CA and New York City, NY. **Question:** How do patterns of mental
health service use vary between cities?

### Solution

Here, the data source only considers two cities when the question is interested
in how health services vary across cities in general. This bias towards only
considering two cities can be addressed by reframing the question.

**Data Source:** Mental health services use on September 12, 2001 in San
Francisco, CA and New York City, NY. **Question:** How do patterns of mental
health service use vary between San Francisco and New York.

**Correction:** Choose a date that is not right after a major disaster.

## Drill 3

**Data Source:** [Armenian Pub Survey](https://www.kaggle.com/erikhambardzumyan/pubs)
**Question:** What are the most common reasons Armenians visit local pubs?

### Solution

After examining the data, I noticed that most of the people who filled out the
survey were young. Therefore, the dataset is biased towards young people. This
bias could be address in two ways. One would be to expand the survey coverage to
pubs that featured older people. Another option would be to refrain the question
to ask: What are the most common reasons **students** in Armenia visit local
pubs.
