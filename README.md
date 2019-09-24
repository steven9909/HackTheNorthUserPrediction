# HackTheNorthUserPrediction

Inspired from http://www.sfu.ca/~jiaxit/resources/wsdm18caser.pdf

This is a Hack the North project (2019) that uses CNN to predict the next User's action.It converts each businesses inside yelp business dataset into a vector with a specified length.
The first element of the vector is the business rating and the subsequent elements are the categories of each restaraunt that has been assigned to a numerical value. If there are not enough cateogries, a default value of 0 is assigned.

This is using a CNN to perform a regression task. Given a sequence of vectors, it tries to predict the next business vector that the user will want to visit.

There are two problems with this. One, the layout of the network was not deep enough for it to actually converge given the large amount of data that we parsed thorugh the yelp dataset. Another type of CNN network might have achieved better results.
Second, the data that we generated from the yelp dataset were not statistically relevant since we generated the sequence of business vectors based on the dates that the users have reviewed. For instance, if user A reviewed restraunt A, restaraunt B, business C respectively, then the sequence of vectors that would be generated is (A, B, C). This is not relevant to where the user would most likely visit next, but rather where the user will most likely review next. The breadth of the problem due to the dataset that the CNN has to solve becomes even wider.
