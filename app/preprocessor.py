import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torchtext.data.utils import get_tokenizer
from nltk.stem import WordNetLemmatizer
import pickle
import string


class Preprocessor:
    """A class for managing preprocessing text data and encoded the text.
    """
    def __init__(self):
        with open("./saved_objects/vocab.pkl", "rb") as outfile:
            vocab = pickle.load(outfile)
        self.word_map = {x: vocab[x] + 1 for x in vocab.keys()}
        self.tokenizer = get_tokenizer("basic_english")
        self.lemmatizer = WordNetLemmatizer()

    def _vocab(self, text_tokens):
        """
        take text tokens and map it to numeric data

        Parameters
        ----------
        text_tokens : list
            contain list of the string

        Returns
        -------
        list
            numeric representation of word
        """
        encoded_text = []
        for x in text_tokens:
            if x in self.word_map:
                encoded_text.append(self.word_map[x])
            else:
                encoded_text.append(0)

        return encoded_text

    def _encoder(self, text, max_word=100):
        """
        take sentence text, remove punctuation, lemmatize it and
        encoded it to maximum word length

        Parameters
        ----------
        text : str
            use text preprocessing and encoded it
        max_word : int, optional
            maximum length of encoded text, by default 100

        Returns
        -------
        list
            encoded text and length of word in the sentence
        """
        X = [0] * max_word
        text_tokens = self.tokenizer(text.translate(str.maketrans('', '', string.punctuation)))
        text_tokens = [self.lemmatizer.lemmatize(x) for x in text_tokens]
        encoded = self._vocab(text_tokens)
        length = min(max_word, len(encoded))
        X[:length] = encoded[:length]
        return [X, length]

    def get_vocab_length(self):
        return len(self.word_map)

    def text_encode(self, df, max_word):
        """
        take dataframe to encoded the text to same encoded lenght
        and put it in new column.

        Parameters
        ----------
        df : dataframe
            ready text data for the model
        max_word : int
            maximum word lenght of the encoded text. 

        Returns
        -------
        dataframe
            return dataframe with encoded text
        """
        df["pattern encoded"] = df["pattern"].apply(
            lambda x: self._encoder(x, max_word)
        )
        return df

    def label_encoding(self, df):
        """
        label encoding for class column and save label encoder object

        Parameters
        ----------
        df : dataframe
            take dataframe which intent column is to be label encoded

        Returns
        -------
        dataframe
            return label encoded dataframe
        """
        label_encoding = LabelEncoder()
        label_encoding.fit(df["intent"])
        df["intent"] = label_encoding.transform(df["intent"])
        with open("./saved_objects/label_encoding.pkl", "wb") as handle:
            pickle.dump(label_encoding, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return df
