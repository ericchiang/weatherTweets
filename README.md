weatherTweets
=============

<h1>UPDATE</h1>
<h1>Please refer to my "<a herf="github.com/EricChiang/cloudy-tweets">cloudy-tweets</a>" repo for the improved solution.</h1>


<h4>Machine Learning solution for Kaggle.com competition "Partlmy Sunny with a Chance of Hashtags"</h4>

<h5>The challenge:</h5>
Given a training set of tweets, can we extract weather information?

<h5>The Machine Learning:</h5>
Use of a customized Bag-of-Words Naive Bayes algorithm. Modifing similar algorithms for spam filters, the feature space of a given tweet is simply the set of words which comprise it, similar to an email. However, for five of the target fields in-class confidence intervals (between 0 and 1) are offered as input, and required output rather than a discrete classification (such as ham or spam). 
