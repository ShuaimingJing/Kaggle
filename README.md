# Kaggle
In this project, I used two files of data which are account.csv and subscriptions.csv. 

The first function in the class imports required csv files, which are two data files mentioned above and train/test sets. The second function is a static method, which converts the True/False into 1/0. This will be used after getting dummies of categorial values since the model I used (logistic regression) only allws numerical values as input. Then the data_processing function is the main part of the code. It is created to merge the data files by using keyword 'account.id', and avoid duplicates by calculating the sums of numerical values and most frequent values of categorial values of subscriptions.csv and then get dummies for useful features and drop the irrelevant features. The function model_running helps to define the model and required inputs, which used to run the model. Finally the function save_predictions save the resultas a single csv files with required columns.

The last part of the code is simply using the functions inside the class.
