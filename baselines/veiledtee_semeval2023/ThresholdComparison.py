import numpy as np
import pandas as pd
from scipy import spatial
from sentence_transformers import SentenceTransformer

from Embed import load
from save_results import export_results
import os

class_desc = {
    "Self-direction: thought": "It is good to have own ideas and interests. Be creative: arguments towards more creativity or imagination. Be curious: arguments towards more curiosity, discoveries, or general interestingness. Have freedom of thought: arguments toward people figuring things out on their own, towards less censorship, or towards less influence on thoughts",
    'Self-direction: action': "It is good to determine one's own actions. Be choosing own goals: arguments towards allowing people to choose what is best for them, to decide on their life, and to follow their dreams. Be independent: arguments towards allowing people to plan on their own and not ask for consent. Have freedom of action: arguments towards allowing people to be self-determined and able to do what they want. Have privacy: arguments towards allowing for private spaces, time alone, and less surveillance, or towards more control on what to disclose and to whom",
    'Stimulation': "It is good to experience excitement, novelty, and change. Have an exciting life: arguments towards allowing people to experience foreign places and special activities or having perspective-changing experiences. Have a varied life: arguments towards allowing people to engage in many activities and change parts of their life or towards promoting local clubs (sports, ...). Be daring: arguments towards more risk-taking.",
    'Hedonism': "It is good to experience pleasure and sensual gratification. Have pleasure: arguments towards making life enjoyable or providing leisure, opportunities to have fun, and sensual gratification.",
    'Achievement': "It is good to be successful in accordance with social norms. Be ambitious: arguments towards allowing for ambitions and climbing up the social ladder. Have success: arguments towards allowing for success and recognizing achievements. Be capable: arguments towards acquiring competence in certain tasks, being more effective, and showing competence in solving tasks. Be intellectual: arguments towards acquiring high cognitive skills, being more reflective, and showing intelligence. Be courageous: arguments towards being more courageous and having people stand up for their beliefs.",
    'Power: dominance': "It is good to be in positions of control over others. Have influence: arguments towards having more people to ask for a favor, more influence, and more ways to control events. Have the right to command: arguments towards allowing the right people to take command, putting experts in charge, and clearer hierarchies of command, or towards fostering leadership.",
    'Power: resources': "It is good to have material possessions and social resources. Have wealth: arguments towards allowing people to gain wealth and material possession, show their wealth, and exercise control through wealth, or towards financial prosperity.",
    'Face': "It is good to maintain one's public image. Have social recognition: arguments towards allowing people to gain respect and social recognition or avoid humiliation. Have a good reputation: arguments towards allowing people to build up their reputation, protect their public image, and spread reputation.",
    'Security: personal': "It is good to have a secure immediate environment. Have a sense of belonging: arguments towards allowing people to establish, join, and stay in groups, show their group membership, and show that they care for each other, or towards fostering a sense of belonging. Have good health: arguments towards avoiding diseases, preserving health, or having physiological and mental well-being. Have no debts: arguments towards avoiding indebtedness and having people return favors. Be neat and tidy: arguments towards being more clean, neat, or orderly. Have a comfortable life: arguments towards providing subsistence income, having no financial worries, and having a prosperous life, or towards resulting in a higher general happiness.",
    'Security: societal': "It is good to have a secure and stable wider society. Have a safe country: arguments towards a state that can better act on crimes, and defend or care for its citizens, or towards a stronger state in general. Have a stable society: arguments towards accepting or maintaining the existing social structure or towards preventing chaos and disorder at a societal level.",
    'Tradition': "It is good to maintain cultural, family, or religious traditions. Be respecting traditions: arguments towards allowing to follow one's family's customs, honoring traditional practices, maintaining traditional values and ways of thinking, or promoting the preservation of customs. Be holding religious faith: arguments towards allowing the customs of a religion and to devote one's life to their faith, or towards promoting piety and the spreading of one's religion.",
    'Conformity: rules': "It is good to comply with rules, laws, and formal obligations. Be compliant: arguments towards abiding to laws or rules and promoting to meet one's obligations or recognizing people who do. Be self-disciplined: arguments towards fostering to exercise restraint, follow rules even when no-one is watching, and to set rules for oneself. Be behaving properly: arguments towards avoiding to violate informal rules or social conventions or towards fostering good manners.",
    'Conformity: interpersonal': "It is good to avoid upsetting or harming others. Be polite: arguments towards avoiding to upset other people, taking others into account, and being less annoying for others. Be honoring elders: arguments towards following one's parents or showing faith and respect towards one's elders.",
    'Humility': "It is good to recognize one's own insignificance in the larger scheme of things. Be humble: arguments towards demoting arrogance, bragging, and thinking one deserves more than other people, or towards emphasizing the successful group over single persons and giving back to society. Have life accepted as is: arguments towards accepting one's fate, submitting to life's circumstances, and being satisfied with what one has.",
    'Benevolence: caring': "It is good to work for the welfare of one's group's members. Be helpful: arguments towards helping the people in one's group and promoting to work for the welfare of others in one group. Be honest: arguments towards being more honest and recognizing people for their honesty. Be forgiving: arguments towards allowing people to forgive each other, giving people a second chance, and being merciful, or towards providing paths to redemption. Have the own family secured: arguments towards allowing people to have, protect, and care for their family. Be loving: arguments towards fostering close relationships and placing the well-being of others above the own, or towards allowing to show affection, compassion, and sympathy.",
    'Benevolence: dependability': "It is good to be a reliable and trustworthy member of one's group. Be responsible: arguments towards clear responsibilities, fostering confidence, and promoting reliability. Have loyalty towards friends: arguments towards being a dependable, trustworthy, and loyal friend, or towards allowing to give friends a full backing.",
    'Universalism: concern': "It is good to strive for equality, justice, and protection for all people. Have equality: arguments towards fostering people of a lower social status, helping poorer regions of the world, providing all people with equal opportunities in life, and resulting in a world where success is less determined by birth. Be just: arguments towards allowing justice to be 'blind' to irrelevant aspects of a case, promoting fairness in competitions, protecting the weak and vulnerable in society, and resulting a world were people are less discriminated based on race, gender, and so on, or towards fostering a general sense for justice. Have a world at peace: arguments towards nations ceasing fire, avoiding conflicts, and ending wars, or promoting to see peace as fragile and precious or to care for all of humanity.",
    'Universalism: nature': "It is good to preserve the natural environment. Be protecting the environment: arguments towards avoiding pollution, fostering to care for nature, or promoting programs to restore nature. Have harmony with nature: arguments towards avoiding chemicals and genetically modified organisms (especially in nutrition), or towards treating animals and plants like having souls, promoting a life in harmony with nature, and resulting in more people reflecting the consequences of their actions towards the environment. Have a world of beauty: arguments towards allowing people to experience art and stand in awe of nature, or towards promoting the beauty of nature and the fine arts.",
    'Universalism: tolerance': "It is good to accept and try to understand those who are different from oneself. Be broadminded: arguments towards allowing for discussion between groups, clearing up with prejudices, listening to people who are different from oneself, and promoting to life within a different group for some time, or towards promoting tolerance between all kinds of people and groups in general. Have the wisdom to accept others: arguments towards allowing people to accept disagreements and people even when one disagrees with them, to promote a mature understanding of different opinions, or to decrease partisanship or fanaticism.",
    'Universalism: objectivity': "It is good to search for the truth and think in a rational and unbiased way. Be logical: arguments towards going for the numbers instead of gut feeling, towards a rational, focused, and consistent way of thinking, towards a rational analysis of circumstances, or towards promoting the scientific method. Have an objective view: arguments towards fostering to seek the truth, to take on a neutral perspective, to form an unbiased opinion, and to weigh all pros and cons, or towards providing people with the means to make informed decisions.",
}


class ThresholdComparison:
    def __init__(self, training_embeddings=None, training_file: str = '', testing_embeddings=None, testing_file: str = '', uncased: bool = True):
        self.uncased = uncased
        self.training_embeddings = training_embeddings
        self.training_file = training_file
        self.testing_embeddings = testing_embeddings
        self.testing_file = testing_file
        self.threshold = 0.2
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def create_description_embeddings(self, dict_of_descriptions=None, description_filter: str = 'second'):
        """
        Using the filter paramter, creates a dictionary of the embedding for each class presented by the ```dict_of_descriptions``` parameter
        :param dict_of_descriptions: A dictionary containing the name of the class as a key, and it's description in english as the value
        :param description_filter: Can be "split", "first", or "second".
            split - splits the descriptions on the '. ' character, replaces ":" with ",", and then encodes the edited sentence.
            first - splits the descriptions on the '. ' character, and removes any characters after ":", if it exists. If ":" is absent from the sentence, the unmodified sentence is encoded.
            second - splits the descriptions on the '. ' character, and removes any characters before ":", if it exists. If ":" is absent from the sentence, the unmodified sentence is encoded.
        :return: A dictionary with the average embedding of each class as the value, and the name of the class as the key
        """
        if dict_of_descriptions is None:
            dict_of_descriptions = class_desc
        embeddings = {}
        for k, v in dict_of_descriptions.items():
            model_embeddings = []
            split_premise = v.split('. ')
            for i in range(len(split_premise)):
                # if i != 0:  # skip "it is good..." statement
                sentence = split_premise[i]
                sentence = sentence.lower()
                if description_filter.lower() == 'split':
                    sentence = sentence.replace(':', ',').strip()
                elif description_filter.lower() == 'first' and ":" in sentence:
                    colon_index = sentence.index(":")
                    sentence = sentence[:colon_index].strip()
                elif description_filter.lower() == 'second' and ":" in sentence:
                    colon_index = sentence.index(":")
                    sentence = sentence[colon_index + 1:].strip()
                sentence = sentence.replace('.', '')
                model_embeddings.append(self.model.encode(sentence))
            embeddings[k] = np.mean(model_embeddings, axis=0)
        return embeddings

    def label_premise(self, premise_embeddings, class_embedding):
        """
        Groups premises together based on a threshold of how similar they are to the class we are currently looking at
        :param premise_embeddings: A list of embeddings
        :param class_embedding: The embedding of the class we wish to test
        :return:
        """
        class_predictions = []
        for premise in premise_embeddings:
            if 1 - spatial.distance.cosine(premise, class_embedding) > self.threshold:  # check similarity
                class_predictions.append(1)
            else:
                class_predictions.append(0)
        return class_predictions

    def predict(self, premise_embeddings, embeddings, training_labels):
        """
        Calculate precision, recall, and f1 score for all classes
        """
        output = []
        for category in list(training_labels.columns)[1:]:
            predicted_labels = self.label_premise(premise_embeddings, embeddings[category])
            output.append(predicted_labels)
        return output

    def run(self, labels: pd.DataFrame):
        # retrieve embeddings for records we are classifying
        train_labels = labels
        test_embeddings, test_args = load(self.testing_embeddings, self.testing_file)
        class_embeddings = self.create_description_embeddings()
        test_predictions = self.predict(test_embeddings, class_embeddings, train_labels)  # predict and save
        if not os.path.exists('output'):
            os.makedirs('output')
        export_results(list(train_labels.columns), test_args, test_predictions, f'output/unsupervised_final.tsv')

