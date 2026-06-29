import os
import json

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, logging
from sentence_transformers import SentenceTransformer


def load(embeddings: dict = None, filename: str = '') -> tuple[list[list[float]], list[str]]:
    """
    Takes embeddings from dictionary or a file and adds them to the XGBoost models attributes
    :param embeddings: A dictionary containing embeddings for arguments. The keys are argument IDs and the values are the embeddings.
    :param filename: The filename to get embeddings from. The keys are argument IDs and the values are the embeddings.
    :return tuple: The embeddings and argument IDs, in order, are returned as a tuple of shape [Embeddings, Argument IDs].
    """
    data = []
    arguments = []
    if embeddings is not None and embeddings != {}:
        for k, v in embeddings.items():
            data.append(v)
            arguments.append(k)
    elif filename != '':
        with open(filename) as json_file:
            for k, v in json.load(json_file).items():
                arguments.append(k)
                data.append(v)
    else:
        raise ValueError("Either embedding or filename must be defined.")
    return data, arguments


class Embed:
    def __init__(self, data: pd.DataFrame, filename: str, uncased: bool = True) -> None:
        self.data = data
        self.filename = filename
        self.uncased = uncased

    def saveEmbeddings(self) -> dict:
        """
        Uses uncased or sentence BERT to generate an embedding for the [CLS] token of every premise. Saves each arguments' embedding to a JSON file for use later.
        :return: A dictionary with Argument IDs as Keys and the re[restive embeddings as Values. This is the same dictionary saved to the JSON file.
        """
        to_save = {}
        if os.path.exists(f"JSON/{self.filename}.json"):  # get embeddings if they exist already
            with open(f"JSON/{self.filename}.json") as json_file:
                to_save = json.load(json_file)
        elif self.uncased:
            # uncased embeddings
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # load model
            logging.set_verbosity_error()
            for arg_id, text in zip(list(self.data["Argument ID"]), list(self.data["Premise"])):  # loop through
                # Add the special tokens.
                marked_text = "[CLS] " + text + " [SEP]"

                # Split the sentence into tokens.
                tokenized_text = tokenizer.tokenize(marked_text)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                segments_ids = [1] * len(tokenized_text)

                # convert input to torch tensors
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segments_ids])

                model = BertModel.from_pretrained(
                    "bert-base-uncased",
                    output_hidden_states=True,  # Whether the model returns all hidden-states.
                )

                model.eval()

                # Run the text through BERT, and collect all hidden states produced from all 12 layers.
                with torch.no_grad():
                    outputs = model(tokens_tensor, segments_tensors)
                    hidden_states = outputs[2]
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings = token_embeddings.permute(1, 0, 2)
                # Stores the token vectors, with shape [22 x 768]
                token_vecs_sum = []
                # `token_embeddings` is a [22 x 12 x 768] tensor.

                # For each token in the sentence...
                for token in token_embeddings:
                    # `token` is a [12 x 768] tensor
                    # Sum the vectors from the last four layers.
                    sum_vec = torch.sum(token[-4:], dim=0)

                    # Use `sum_vec` to represent `token`.
                    token_vecs_sum.append(sum_vec)

                for i, token_str in enumerate(tokenized_text):
                    if token_str == "[CLS]":
                        to_save[f"{arg_id}"] = [float(j) for j in token_vecs_sum[i]]
            if not os.path.exists('JSON'):
                os.makedirs('JSON')
            with open(
                    f"JSON/{self.filename}.json", "w"
            ) as filename:
                json.dump(to_save, filename)
            print(f"Uncased embeddings saved to: JSON/{self.filename.lower()}.json")
        else:
            # sentence embeddings
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # get model
            for arg_id, text in zip(list(self.data["Argument ID"]), list(self.data["Premise"])):  # loop through data
                to_save[f"{arg_id}"] = list(model.encode(text).astype(float))  # encode each premise and save to dict
            if not os.path.exists('JSON'):  # make JSON folder is necessary
                os.makedirs('JSON')
            with open(  # save embeddings
                    f"JSON/{self.filename}.json", "w"
            ) as filename:
                json.dump(to_save, filename)
            print(f"Sentence embeddings saved to: JSON/{self.filename.lower()}.json")
        return to_save
