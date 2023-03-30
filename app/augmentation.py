import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw



stopword = list(stopwords.words("english"))
aug = naw.RandomWordAug(action="swap", stopwords=stopword, aug_min=1, aug_max=5)

def data_aug(df):
    for x in df["intent"].unique():
        new_pattern = []
        df_n = df[df["intent"] == x].reset_index(drop=True)
        if len(df_n) <= 5:
            n = 25
        elif len(df_n) > 5 and len(df_n) < 10:
            n = 15
        elif len(df_n) < 20:
            n = 5
        else:
            n = 1
        for i in np.random.randint(0, len(df_n), n):
            text = df_n.iloc[i]["pattern"]
            augmented_text = aug.augment(text)
            new_pattern.extend(augmented_text)

        new = pd.DataFrame({"pattern": new_pattern, "intent": x})
        df = shuffle(pd.concat([df, new]).reset_index(drop=True), random_state=100)

    return df
