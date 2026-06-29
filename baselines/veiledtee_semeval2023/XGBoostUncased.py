import xgboost
import pandas as pd
from save_results import export_results
from Embed import load
import os


class XGBoostUncased:
    def __init__(self, parameters: dict, test_frame: pd.DataFrame, training_embeddings=None, training_file: str = '', testing_embeddings=None, testing_file: str = '', tf_idf_threshold: int = 1000):
        if training_embeddings is None:
            training_embeddings = {}
        self.parameters = parameters
        self.training_embeddings = training_embeddings
        self.training_file = training_file
        self.testing_embeddings = testing_embeddings
        self.testing_file = testing_file
        self.test_frame = test_frame
        self.tf_idf_threshold = tf_idf_threshold

    def run(self, test_frame: pd.DataFrame, labels: pd.DataFrame):
        training_labels = labels
        test_premise = [text for text in test_frame['Premise']]  # get string representation of test data
        tf_idf = pd.read_csv('tf_idf_final.csv', index_col='Unnamed: 0')
        # load data and args
        training_data, training_args = load(self.training_embeddings, self.training_file)
        test_data, test_args = load(self.testing_embeddings, self.testing_file)

        output = []
        # Split the data into features and labels
        for category in training_labels.columns[1:]:
            sorted_tf_idf = tf_idf.sort_values(by=[category])
            to_compare = list(sorted_tf_idf.index[:self.tf_idf_threshold])
            # Convert the data into DMatrix format
            dmatrix = xgboost.DMatrix(data=training_data, label=training_labels[category])

            # Train the XGBoost model
            bst = xgboost.train(self.parameters, dmatrix)
            # Make predictions
            prediction = bst.predict(xgboost.DMatrix(test_data))
            round_pred = [round(pred) for pred in prediction]  # round prediction to 1 or 0
            for i in range(len(test_premise)):  # employ TF-IDF Values
                if any(x in test_premise[i] for x in to_compare):
                    round_pred[i] = 1
            # Save prediction to dataframe
            output.append(round_pred)
        if not os.path.exists('output'):
            os.makedirs('output')
        export_results(training_labels.columns, test_args, output, 'output/xgboost_uncased_final.tsv')
