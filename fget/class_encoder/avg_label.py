import torch
import torch.nn as nn


class AvgLabel(nn.Module):
    def __init__(
        self,
        features,
    ):
        """OTyper based on
        https://users.cs.northwestern.edu/~ddowney/publications/zheng_aaai_2018.pdf

        Args:
            features (nn.Embedding): embedding of the class names
        """
        super(AvgLabel, self).__init__()
        self.features = features
        self.output_dim = 300

    def forward(self, label_idx):
        return self.features(label_idx)
