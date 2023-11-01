import pandas as pd
from sklearn.linear_model import LogisticRegression

class PredictionsFromData:

    # import required csv files
    def __init__(self, train_path, subscriptions_path, account_path, test_path):
        self.train = pd.read_csv(train_path)
        self.subscriptions = pd.read_csv(subscriptions_path)
        self.account = pd.read_csv(account_path, encoding='latin1')
        self.test = pd.read_csv(test_path)

    # convert True/False values into 1/0
    @staticmethod
    def replace_boolean_with_binary(df):
        boolean_columns = [col for col in df.columns if df[col].dtype == bool]
        replacement_dict = {True: 1, False: 0}
        for col in boolean_columns:
            df[col] = df[col].replace(replacement_dict)

    def data_processing(self):
        # rename the column in order for convenience of merging
        self.test.rename(columns={'ID': 'account.id'}, inplace=True)
        acc_dropped = self.account[['account.id', 'no.donations.lifetime']]
        self.train = self.train.merge(acc_dropped, on='account.id', how='left')
        self.test = self.test.merge(acc_dropped, on='account.id', how='left')
        
        # avoid duplicates by calculating the sums of numerical values and most frequent values of categorial values of subscriptions.csv
        aggregated_subscriptions = self.subscriptions.groupby('account.id').agg({
            'season': lambda x: x.mode()[0] if not x.mode().empty else None,
            'package': lambda x: x.mode()[0] if not x.mode().empty else None,
            'no.seats': 'sum',
            'section': lambda x: x.mode()[0] if not x.mode().empty else None,
            'location': lambda x: x.mode()[0] if not x.mode().empty else None,
            'price.level': 'sum',
            'subscription_tier': lambda x: x.mode()[0] if not x.mode().empty else None,
            'multiple.subs': lambda x: x.mode()[0] if not x.mode().empty else None,
        }).reset_index()
       
        # get dummies for useful features and drop the irrelevant features
        dummy_subscriptions = pd.get_dummies(aggregated_subscriptions, columns=['location', 'section', 'season'])
        clean_subscriptions = dummy_subscriptions.drop(['multiple.subs', 'package', 'price.level'], axis=1)
        self.replace_boolean_with_binary(clean_subscriptions)

        # merge the result dataframe with train and test sets, then fill the nah values with 0
        self.train = self.train.merge(clean_subscriptions, on='account.id', how='left').fillna(0)
        self.test = self.test.merge(clean_subscriptions, on='account.id', how='left').fillna(0)

    # Define the model, required indices and train it
    def model_running(self):
        
        # define the inputs for the model
        x_train = self.train.drop(columns=['account.id', 'label'])
        y_train = self.train['label']
        x_test = self.test.drop(columns=['account.id'])

        # run the model
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_test = model.predict_proba(x_test)
        result = [item[1] for item in y_test]

        # append the result in the test file and change the names with requirements
        self.test['predicted'] = result
        self.test.rename(columns={'account.id': 'ID'}, inplace=True)

    # Save it to csv file
    def save_predictions(self, file_path):
        self.test[['ID', 'predicted']].to_csv(file_path, index=False)

# Usage
predictor = PredictionsFromData(
    '/Users/shuai/Desktop/AIPI 520/Kaggle Competitiom/for_students/train.csv',
    '/Users/shuai/Desktop/AIPI 520/Kaggle Competitiom/for_students/subscriptions.csv',
    '/Users/shuai/Desktop/AIPI 520/Kaggle Competitiom/for_students/account.csv',
    '/Users/shuai/Desktop/AIPI 520/Kaggle Competitiom/for_students/test.csv'
)
predictor.data_processing()
predictor.model_running()

# Save the result as csv file
predictor.save_predictions('test_with_predictions.csv')