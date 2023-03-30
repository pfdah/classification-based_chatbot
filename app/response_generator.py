import json
import random
from better_profanity import profanity
import pandas as pd
import pickle
import torch
from dataloader import get_dataloder
import torch.nn.functional as F


torch.manual_seed(100)


model = torch.load("./models/model.pt")
model.to(torch.device('cpu'))
model.eval()


def generate_response(input_string):
    """generates the model response from the given input string and returns it

    Parameters
    ----------
    input_string : string
        the user input query

    Returns
    -------
    string
        the appropriate response
    """
    if profanity.contains_profanity(input_string):
        return "Your query contains profanity. I'm sorry, I cannot respond to that."

    # Convert the input string to pandas dataframe, so preprocessing will be easier
    dataframe = pd.DataFrame({"pattern": [input_string], "intent": [0]})
    
    # Open the preprocessor file and perform preprocessing
    with open("./saved_objects/preprocessor.pkl", "rb") as handle:
        preprocessor = pickle.load(handle)
    dataframe = preprocessor.text_encode(dataframe, 1024)
    
    # Break into X and Y and get dataloader item
    x = dataframe["pattern encoded"]
    y = dataframe["intent"]
    datas = get_dataloder(x, y)
    x_input, _, input_len = next(iter(datas))
    
    y_hat = F.softmax(model(x_input, input_len), dim=1)
    val, prediction = torch.max(y_hat, 1)
    
    if val.item() > 0.4:
        # Load label_encoder pickle and reverse map the intent predicted
        with open("./saved_objects/label_encoding.pkl", "rb") as handle:
            label_encoder = pickle.load(handle)
        intent = label_encoder.inverse_transform(prediction)
        
        # Map the intent into JSON and return a random response
        file = open("./dataset/data.json", "r", encoding="utf8")
        json_content = json.load(file)

        content = json_content["intents"]
        for item in content:
            if item["tag"] == intent:
                respone_list = item["responses"]
                return str(respone_list[random.randrange(len(respone_list))])
            else:
                continue
    return "I could not understand your query."
