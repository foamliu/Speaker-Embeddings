import torch
import torch.nn as nn

import config as hp


class SpeechEmbedder(nn.Module):

    def __init__(self):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.n_mels, hp.hidden_size, num_layers=hp.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.hidden_size, hp.embedding_dim)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
