from typing import Text
import torch.nn as nn
from zsl_kg.example_encoders.text_encoder import TextEncoder


class DescEncoder(nn.Module):
    def __init__(self, text_encoder: TextEncoder, description_dict: dict):
        """Description-based zero-shot learning for fine
        grained entity typing
        https://aclanthology.org/N19-1087.pdf

        Args:
            text_encoder (TextEncoder): bilstm with attention
            description_dict (dict): description dict with tokens
                as the key.
        """
        super(DescEncoder, self).__init__()
        self.text_encoder = text_encoder
        self.description_dict = description_dict
        self.output_dim = text_encoder.hidden_dim * 2

    def forward(self, label_idx):
        desc_tensor = self.text_encoder(self.description_dict)
        return desc_tensor[label_idx, :]
