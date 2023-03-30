import torch
from torch.utils.data import Dataset


class ChatBotDataset(Dataset):
    """Chat Bot Dataset."""
    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : list
            encoded text list
        y : list
            label encoded class for respective text
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.tensor(self.X[index][0], dtype=torch.int32),
            torch.tensor(self.y[index], dtype=torch.int64),
            torch.tensor(self.X[index][1], dtype=torch.int64),
        )
