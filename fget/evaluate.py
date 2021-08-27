import os

import torch
from allennlp.common.tqdm import Tqdm
from allennlp.nn import util as nn_util

from utils.eval import (
    get_true_and_prediction,
    loose_macro,
    loose_micro,
    strict,
)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def eval_model(
    model, dataset, iterator, label_idx, graph, seen_idx=None, unseen_idx=None
):
    model.eval()

    generator_tqdm = Tqdm.tqdm(
        iterator(dataset, num_epochs=1),
        total=iterator.get_num_batches(dataset),
    )

    one_hot = torch.Tensor().to(model.options["device"])
    prob_preds = torch.Tensor().to(model.options["device"])

    with torch.no_grad():
        for batch in generator_tqdm:
            batch = nn_util.move_to_device(batch, model.options["cuda_device"])
            logits = model(batch, label_idx, graph)
            prob = torch.sigmoid(logits)

            one_hot = torch.cat([one_hot, batch["labels"].float()], dim=0)
            prob_preds = torch.cat([prob_preds, prob], dim=0)

    result_col = ["overall", "seen", "unseen"]
    labels = [list(range(len(label_idx))), seen_idx, unseen_idx]

    result = {}

    for i in range(3):
        from IPython import embed

        one_hot_list = one_hot[:, labels[i]].cpu().numpy().tolist()
        prob_list = prob_preds[:, labels[i]].cpu().numpy().tolist()
        true_and_pred = get_true_and_prediction(prob_list, one_hot_list)

        # remove true and pred that are empty
        clean_true_pred = []
        for k in range(len(true_and_pred)):
            if true_and_pred[k][0]:
                clean_true_pred.append(true_and_pred[k])

        strict_acc = strict(clean_true_pred)[2]
        loose_micro_acc = loose_micro(clean_true_pred)[2]
        loose_macro_acc = loose_macro(clean_true_pred)[2]

        result[result_col[i]] = {
            "strict": strict_acc,
            "loose_micro": loose_micro_acc,
            "loose_macro": loose_macro_acc,
        }

        print(
            "{}: strict={:.4f}, "
            "loose micro={:.4f}, "
            "loose macro={:.4f}".format(
                result_col[i].upper(),
                strict_acc,
                loose_micro_acc,
                loose_macro_acc,
            )
        )

    return result
