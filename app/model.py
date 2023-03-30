import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.optim import Adam

from tqdm import tqdm

torch.manual_seed(100)


class Classifier(nn.Module):
    def __init__(
        self, hidden_dim=75, n_layers=1, n_classes=2, bidirectional=False
    ):
        """
        classifier model which contain
        Embedding layer, LSTM layer and Fully connected layer.

        Parameters
        ----------
        hidden_dim : int, optional
            The number of features in the hidden state for LSTM, by default 75
        n_layers : int, optional
            number of recurrent layers for LSTM, by default 1
        n_classes : int, optional
            number of neurons for the final layers, by default 2
        bidirectional : bool, optional
            If True, becomes a bidirectional LSTM, by default False
        """
        super().__init__()
        # word_vectors_for_training= np.insert(
        #     vector,
        #     0,
        #     np.random.uniform(model.wv.vectors.min(), model.wv.vectors.max(), vector.shape[1]),
        #     axis = 0
        # )
        with open('./saved_objects/vectors.npy', 'rb') as f:
            vector = np.load(f)
        word_vectors_for_training = np.insert(vector, 0, np.zeros(vector.shape[1]), axis=0)
        word_vectors_for_training = torch.FloatTensor(word_vectors_for_training)
        self.D = 1
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim

        if bidirectional:
            self.D = 2

        num_embedding, embed_len = word_vectors_for_training.shape
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embedding, embedding_dim=embed_len
        )
        self.embedding_layer.load_state_dict({"weight": word_vectors_for_training})
        self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embed_len,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.linear = nn.Linear(hidden_dim, n_classes)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, X, length):
        X = self.embedding_layer(X)

        x_pack = pack_padded_sequence(X, length, batch_first=True, enforce_sorted=False)
        output, (hidden, carry) = self.lstm(x_pack)
        return self.linear(hidden[-1])


class ModelOperations:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def training(self, train_loader, val_loader, epoch=10, learning_rate=1e-2):
        """
        training the model and valided the model

        Parameters
        ----------
        train_loader : DataLoader
            train dataloader contain input data, labels and length of word tokens.
        val_loader : DataLoader
            validation dataloader contain input data, labels and length of word tokens.
        epoch : int, optional
            number loop to train the model, by default 10
        learning_rate : float, optional
            learning rate, by default 1e-2

        Returns
        -------
        tuple
            list of tuple which contain accuracy and loss of training and validation and the best epoch index.  
        """
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        training_loss = []
        training_accuracy = []
        val_loss = []
        val_accuracy = []
        min_loss = 100000
        best_epoch = 0

        for i in range(epoch):
            train_losses = []
            train_accuracy = []
            for X, Y, length in train_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                optimizer.zero_grad()
                Y_preds = F.softmax(self.model(X, length), dim=1)
                loss = loss_fn(Y_preds, Y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                _, prediction = torch.max(Y_preds, 1)
                train_accuracy.append(
                    ((Y == prediction).sum() / len(prediction)).item()
                )

            with torch.no_grad():
                data_loop = tqdm(val_loader, leave=True)
                losses = []
                accuracy = []
                for X, y, length in data_loop:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    data_loop.set_description(f"Epoch {i+1}")
                    y_hat = F.softmax(self.model(X, length), dim=1)
                    loss = F.cross_entropy(y_hat, y)
                    losses.append(loss.item())
                    _, prediction = torch.max(y_hat, 1)
                    accuracy.append(((y == prediction).sum() / len(prediction)).item())
                    data_loop.set_postfix(
                        train_loss=sum(train_losses) / len(train_losses),
                        train_accracy=round(
                            sum(train_accuracy) / len(train_accuracy), 4
                        ),
                        val_loss=sum(losses) / len(losses),
                        val_accuracy=f"{round(sum(accuracy)/len(accuracy), 4)}",
                    )

            training_loss.append(sum(train_losses) / len(train_losses))
            training_accuracy.append(
                round(sum(train_accuracy) / len(train_accuracy), 4)
            )

            val_loss.append(round(sum(losses) / len(losses), 4))
            val_accuracy.append(round(sum(accuracy) / len(accuracy), 4))
            if min_loss > val_loss[-1]:
                min_loss = val_loss[-1]
                best_epoch = i
                self.save_model("./models/model.pt")
        return (training_accuracy, training_loss, val_accuracy, val_loss, best_epoch)

    def pred(self, data_loader):
        self.model.eval()
        intent = []
        trues = []
        model = self.model.to(torch.device("cpu"))
        for X, y, length in data_loader:
            y_hat = F.softmax(model(X, length), dim=1)
            _, prediction = torch.max(y_hat, 1)
            intent.extend(prediction.tolist())
            trues.extend(y.tolist())
        return intent, trues

    def save_model(self, path):
        torch.save(self.model, path)

