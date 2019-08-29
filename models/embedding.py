import torch
import torch.nn as nn


class SpeakerEmbedding(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder):
        super(SpeakerEmbedding, self).__init__()
        self.encoder = encoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        embedding = encoder_padded_outputs[:, -1, :]
        # print('embedding.size(): ' + str(embedding.size()))

        return embedding
