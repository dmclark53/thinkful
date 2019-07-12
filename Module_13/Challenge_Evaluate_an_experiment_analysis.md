# Challenge: Evaluate an experiment analysis

## 1.0
### Scenario:
The Sith Lords are concerned that their recruiting slogan, "Give In to Your Anger," isn't very effective. Darth Vader
develops an alternative slogan, "Together We Can Rule the Galaxy." They compare the slogans on two groups of 50 
captured droids each. In one group, Emperor Palpatine delivers the "Anger" slogan. In the other, Darth Vader presents 
the "Together" slogan. 20 droids convert to the Dark Side after hearing Palpatine's slogan, while only 5 droids 
convert after hearing Vader's. The Sith's data scientist concludes that "Anger" is a more effective slogan and should 
continue to be used.

### Flaws with fixes:
1. Bias towards presenter.
* Fix: Droids should be presented each slogan anonymously or by the same presenter.
 
 
## 2.0
### Scenario:
In the past, the Jedi have had difficulty with public relations. They send two envoys, Jar Jar Binks and Mace Windu, 
to four friendly and four unfriendly planets respectively, with the goal of promoting favorable feelings toward the 
Jedi. Upon their return, the envoys learn that Jar Jar was much more effective than Windu: Over 75% of the people 
surveyed said their attitudes had become more favorable after speaking with Jar Jar, while only 65% said their 
attitudes had become more favorable after speaking with Windu. This makes Windu angry, because he is sure that he had
a better success rate than Jar Jar on every planet. The Jedi choose Jar Jar to be their representative in the future.
 
### Flaws with fixes:
1. Samples not random
* Fix: Jar Jar and Mace should each visit an equal number of friendly and unfriendly planets.

## 3.0
### Scenario:
A company with work sites in five different countries has sent you data on employee satisfaction rates for workers in 
Human Resources and workers in Information Technology. Most HR workers are concentrated in three of the countries, 
while IT workers are equally distributed across worksites. The company requests a report on satisfaction for each job 
type. You calculate average job satisfaction for HR and for IT and present the report.

### Flaw with fixes:
1. Samples not random
* Fix: Only perform analysis on the three countries that have both IT and HR workers.

## 4.0
### Scenario:
When people install the Happy Days Fitness Tracker app, they are asked to "opt in" to a data collection scheme where 
their level of physical activity data is automatically sent to the company for product research purposes. During your 
interview with the company, they tell you that the app is very effective because after installing the app, the data 
show that people's activity levels rise steadily.

### Flaw with fixes:
1. Inaccurate Sample
    * They are only collecting data on activite people, so it is not surprising that with the app are more active.
* Fix: Redo the sample. Survey people who use the app and people who do not. Test whether there is a difference in
activity levels for each group.

## 5.0
### Scenario:
To prevent cheating, a teacher writes three versions of a test. She stacks the three versions together, first all 
copies of Version A, then all copies of Version B, then all copies of Version C. As students arrive for the exam, each 
student takes a test. When grading the test, the teacher finds that students who took Version B scored higher than 
students who took either Version A or Version C. She concludes from this that Version B is easier, and discards it.

1. The sample isn't random.
* Fix: Randomize the three groups of tests and then distribute them to the students. Also, distribute them at the 
same time to each of the students.