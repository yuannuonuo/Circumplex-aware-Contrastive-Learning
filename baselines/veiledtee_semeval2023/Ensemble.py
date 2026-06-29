import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from Embed import load
from save_results import export_results


class Ensemble:
    def __init__(self, parameters, test_frame: pd.DataFrame, training_embeddings=None, training_file: str = '', testing_embeddings=None, testing_file: str = '', uncased: bool = True):
        self.parameters = parameters
        self.uncased = uncased
        self.parameters = parameters
        self.training_embeddings = training_embeddings
        self.training_file = training_file
        self.testing_embeddings = testing_embeddings
        self.testing_file = testing_file
        self.test_frame = test_frame

    def model(self) -> VotingClassifier:
        clf1 = LogisticRegression(multi_class=self.parameters['multi_class'], max_iter=self.parameters['max_iter'], n_jobs=-1)
        clf2 = RandomForestClassifier(n_estimators=self.parameters['n_estimators'], max_depth=self.parameters['max_depth'], n_jobs=-1)
        clf3 = GaussianNB()
        return VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting=self.parameters['voting'])

    def run(self, labels: pd.DataFrame):
        voter = self.model()
        if not os.path.exists('output'):
            os.makedirs('output')
        train_labels = labels
        if self.uncased:
            # load data and args
            training_data, training_args = load(self.training_embeddings, self.training_file)
            test_data, test_args = load(self.testing_embeddings, self.testing_file)
            # build results dataframe
            results_frame = pd.DataFrame(columns=train_labels.columns)
            results_frame[train_labels.columns[0]] = test_args

            output = []
            for category in train_labels.columns[1:]:
                category_labels = list(train_labels[category].values)
                voter = voter.fit(training_data, category_labels)
                prediction = list(voter.predict(test_data))
                output.append(prediction)
            export_results(train_labels.columns, test_args, output, 'output/ensemble_uncased_final.tsv')
        else:
            training_data, training_args = load(self.training_embeddings, self.training_file)
            test_data, test_args = load(self.testing_embeddings, self.testing_file)

            output = []
            for i in range(len(train_labels.columns[1:])):
                category = train_labels.columns[1:][i]
                voter = voter.fit(training_data, train_labels[category])
                prediction = list(voter.predict(test_data))
                output.append(prediction)
            export_results(list(train_labels.columns), test_args, output, 'output/ensemble_sentence_final.tsv')
