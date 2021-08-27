import torch
import torch.nn as nn
from allennlp.models import Model


class BiLinearModel(Model):
    def __init__(self, vocab, example_encoder, label_encoder, options=None):
        super().__init__(vocab)
        self.example_encoder = example_encoder
        self.label_encoder = label_encoder

        # additional information
        self.options = options

        self.mention_dim = self.example_encoder.output_dim
        self.label_dim = self.label_encoder.output_dim

        self.text_joint = nn.Linear(
            self.mention_dim, options["joint_dim"], bias=False
        )
        self.label_joint = nn.Linear(
            self.label_dim, options["joint_dim"], bias=False
        )

        if self.options["dataset"] == "ontonotes":
            self.other = nn.Parameter(torch.empty(1, options["joint_dim"]))
            nn.init.xavier_uniform_(self.other)
            self.register_parameter("other", self.other)
        else:
            self.other = None

    def forward(self, batch, label_idx, kg=None):

        mention_tokens = batch["mention_tokens"]
        left_tokens = batch["left_tokens"]
        right_tokens = batch["right_tokens"]

        # get the encoder out
        mention_rep = self.example_encoder(
            mention_tokens, left_tokens, right_tokens
        )

        if self.options["dataset"] == "ontonotes":
            label_idx = label_idx[1:]

        if kg is None:
            label_rep = self.label_encoder(label_idx)
        else:
            label_rep = self.label_encoder(label_idx, kg)

        mention_joint = self.text_joint(mention_rep)
        label_joint = self.label_joint(label_rep)

        if self.other is not None:
            label_joint = torch.cat((self.other, label_joint), dim=0)

        logits = torch.matmul(mention_joint, label_joint.t())

        return logits
