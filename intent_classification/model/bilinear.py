import os

import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class BiLinearModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        example_encoder: object,
        class_encoder: object,
        options: dict = None,
    ):
        super().__init__(vocab)
        self.example_encoder = example_encoder
        self.class_encoder = class_encoder
        self.options = options

        self.text_joint = nn.Linear(
            self.example_encoder.output_dim,
            options["joint_dim"],
        )
        self.class_joint = nn.Linear(
            self.class_encoder.output_dim,
            options["joint_dim"],
        )

    def forward(self, batch, node_idx, kg=None):
        encoder_out = self.example_encoder(batch)
        text_rep = self.text_joint(encoder_out)

        # get label representation
        if kg is None:
            class_out = self.class_encoder(node_idx)
        else:
            class_out = self.class_encoder(node_idx, kg)
        class_rep = self.class_joint(class_out)

        logits = torch.matmul(text_rep, class_rep.t())

        return logits


# class BiLinear(Model):
#     def __init__(self, vocab, text_encoder, label_encoder, options):
#         super().__init__(vocab)
#         self.text_encoder = text_encoder
#         self.label_encoder = label_encoder

#         self.options = options

#         # additional information
#         self.seen_classes = options["seen_classes"]
#         self.device = options["device"]
#         self.cuda_device = options["cuda_device"]
#         self.dataset = options["dataset"]
#         self.label_encoder_type = options["label_encoder_type"]
#         self.arch = "bilinear"

#         # TODO: add a parameter for this
#         self.total_classes = 7

#         self.joint_dim = options["joint"]

#         self.text_joint = nn.Linear(64, self.joint_dim, bias=False)
#         self.label_joint = nn.Linear(
#             self.label_encoder.label_dim, self.joint_dim, bias=False
#         )

#         self.unseen_classes = options["unseen_classes"]
#         self.dev_classes = options["dev_classes"]

#         self.all_classes = self.seen_classes + self.dev_classes + self.unseen_classes

#         self.loss_function = nn.CrossEntropyLoss()
#         self.accuracy = CategoricalAccuracy()
#         self.log_softmax = nn.LogSoftmax()

#     def forward(self, sentence, labels=None, train=False, dev=False):
#         # get the encoder out
#         encoder_out = self.text_encoder(sentence)
#         text_rep = self.text_joint(encoder_out)

#         # get label representation
#         label_rep = self.label_encoder()
#         label_rep = self.label_joint(label_rep)

#         logits = torch.matmul(text_rep, label_rep.t())
#         output = {"logits": logits}

#         # additional compute
#         batch_size = sentence["tokens"].size(0)

#         unseen_mask = torch.zeros(batch_size, len(self.all_classes)).to(self.device)
#         unseen_mask[:, self.unseen_classes] = 1
#         unseen_probs = masked_softmax(logits, unseen_mask, dim=1)
#         output["unseen_probs"] = unseen_probs

#         if labels is not None:
#             # Computing log loss
#             if train:
#                 logits = logits[:, self.seen_classes]
#             if dev:
#                 logits = logits[:, self.dev_classes]

#             output["loss"] = self.loss_function(logits, labels)

#         return output
