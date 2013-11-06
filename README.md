weatherTweets
=============

<h1>UPDATE</h1>
<h3>While this repo was a fun meander through Naive Bayes and Hidden Markov Models, the results have been completely usurped by my second attempt at this challenge (RMSE improved by about 40%). Please refer to "github.com/EricChiang/cloudy-tweets" for the improved solution.</h3>


<h4>Machine Learning solution for Kaggle.com competition "Partlmy Sunny with a Chance of Hashtags"</h4>

<h5>The challenge:</h5>
Given a training set of tweets, can we extract weather information?

<h5>The Machine Learning:</h5>
Use of a customized Bag-of-Words Naive Bayes algorithm. Modifing similar algorithms for spam filters, the feature space of a given tweet is simply the set of words which comprise it, similar to an email. However, for five of the target fields in-class confidence intervals (between 0 and 1) are offered as input, and required output rather than a discrete classification (such as ham or spam). 
