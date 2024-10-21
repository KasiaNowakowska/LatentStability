import torch
import torch.nn as nn

from torch.utils.data import Dataset


class SequentialTimeSeriesDataset(Dataset):
    def __init__(self, data, batch_size, window_size):
        self.data = data
        self.batch_size = batch_size
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        # Get a batch of consecutive timesteps
        input_batch = self.data[idx : idx + self.window_size]
        # Get the corresponding label batch (shifted by one time step)
        label_batch = self.data[idx + 1 : idx + self.window_size + 1]
        return input_batch, label_batch



class torchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, window_size, device):
        super().__init__()
        self.device = device
        self.num_layers = 1
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.lstm = nn.LSTM(
            input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True
        ).to(device)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.input_size).to(device)

    def forward(self, x):
        x, _ = self.lstm(x.float())
        x = self.linear(x)
        return x

    def closed_loop_step(self, x, h, c):
        out, (h0, c0) = self.lstm(x, (h, c))
        # Pass LSTM output through the linear layer to get predictions
        pred = self.linear(out.squeeze(1))
        return pred.unsqueeze(1)

    def closed_loop_prediction(self, x, prediction_length):
        batch_size, seq_len, _ = x.size()
        x = x.to(self.device)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)  # Initial hidden state
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)  # Initial cell state

        output_seq = []
        for t in range(seq_len):
            # Forward pass through LSTM
            out, (h0, c0) = self.lstm(x[:, t : t + 1, :], (h0, c0))

            # Pass LSTM output through the linear layer to get predictions
            pred = self.linear(out.squeeze(1))
            output_seq.append(pred.unsqueeze(1))

        for t in range(prediction_length):
            # Forward pass through LSTM
            out, (h0, c0) = self.lstm(pred.unsqueeze(1), (h0, c0))

            # Pass LSTM output through the linear layer to get predictions
            pred = self.linear(out.squeeze(1))
            output_seq.append(pred.unsqueeze(1))

        # Concatenate the predictions along the sequence dimension
        output_seq = torch.cat(output_seq, dim=1)
        return output_seq
