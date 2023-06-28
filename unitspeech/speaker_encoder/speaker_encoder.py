""" from https://github.com/HarryVolek/PyTorch_Speaker_Verification """
import torch
import torch.nn as nn


class SpeechEmbedder(nn.Module):
    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(80, 768, num_layers=2, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(768, 256)

    def forward(self, x, x_length=None):
        # x : [N * M, frames, 80]
        if x_length is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, x_length.cpu().numpy(), batch_first=True, enforce_sorted=False)
            self.LSTM_stack.flatten_parameters()
            _, (x, _) = self.LSTM_stack(x.float())
            x = x[-1]
        else:
            x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
            # only use last frame
            x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x

