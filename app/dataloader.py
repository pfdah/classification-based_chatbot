import torch
from torch.utils.data import DataLoader
from dataset_template import ChatBotDataset

torch.manual_seed(100)


def get_dataloder(X, Y, batch_size=128, shuffle=True):
    """
    this function takes X and Y and creates a Dataloader object of given batch size

    Parameters
    ----------
    X : series or array or list
        the input for the data loader, in this case: 'patterns'  
    Y : series or array or list
        the class for the given X, in this case: 'intents'
    batch_size : int, optional
        the batch size for the dataloader, by default 128
    shuffle : bool, optional
        the variable to indicate if data will be shuffled or not, by default True

    Returns
    -------
    Dataloader
        object of given batch size, shuffled or not according to given shuffle argument
        
    """
    dataset = ChatBotDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
