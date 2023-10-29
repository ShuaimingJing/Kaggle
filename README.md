# Kaggle
In this project, I used two files of data which are account.csv and subscriptions.csv. 

First, I dropped all the irrelevant features from account.csv, and only left two columns which are account.id (target) and no.donations.lifetime (feature). Since there are no nah values for these two columns so I didn't use dropna. Then I combined train and test sets with the result.The second data file I used is subscriptions.csv. Since there are a lot of duplicate account.id, I used groupby function to find the aggregate by calculating the sums of numerical values and the most frequent values of categorial value. Then I get dummies for useful features and drop the irrelevant features, and convert the result of dummies into 0/1, and filled all nah values with 0. Then I combined the result with the combined train and test sets I got from the last process.

Finally, I defined x_train by dropping 'account.id', which is the index, and 'label', which is the target from the combined trained dataframe. Defined y_train by only the 'label' column. Defined x_test by dropping 'account.id' from the combined test dataframe. Then define the model to be logisticregression (tried some other models but this one performs the best) and get the result of y_test.
Append the result into the test data and save it as csv file.
