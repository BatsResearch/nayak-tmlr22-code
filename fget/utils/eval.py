## Code obtained from https://github.com/shimaokasonse/NFGEC
import torch


def get_true_and_prediction(scores, true_labels):
    # obtained from shimaoka sonse
    true_and_prediction = []
    assert len(scores) == len(true_labels)
    for score, true_label in zip(scores, true_labels):
        assert len(score) == len(true_label)
        predicted_tag = []
        true_tag = []
        for label_id, label_score in enumerate(list(true_label)):
            if label_score > 0:
                true_tag.append(label_id)
        lid, ls = max(enumerate(list(score)), key=lambda x: x[1])
        predicted_tag.append(lid)
        for label_id, label_score in enumerate(list(score)):
            if label_score > 0.5:
                if label_id != lid:
                    predicted_tag.append(label_id)

        true_and_prediction.append((true_tag, predicted_tag))

    return true_and_prediction


def f1(p, r):
    if r == 0.0:
        return 0.0
    return 2 * p * r / float(p + r)


def strict(true_and_prediction, count=False):
    num_entities = len(true_and_prediction)
    correct_num = 0.0
    for true_labels, predicted_labels in true_and_prediction:
        correct_num += set(true_labels) == set(predicted_labels)
    precision = recall = correct_num / num_entities
    if count:
        return precision, recall, f1(precision, recall), correct_num
    return precision, recall, f1(precision, recall)


def loose_macro(true_and_prediction):
    num_entities = len(true_and_prediction)
    p = 0.0
    r = 0.0
    for true_labels, predicted_labels in true_and_prediction:
        if len(predicted_labels) > 0:
            p += len(
                set(predicted_labels).intersection(set(true_labels))
            ) / float(len(predicted_labels))
        if len(true_labels):
            r += len(
                set(predicted_labels).intersection(set(true_labels))
            ) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1(precision, recall)


def loose_micro(true_and_prediction):
    num_predicted_labels = 0.0
    num_true_labels = 0.0
    num_correct_labels = 0.0
    for true_labels, predicted_labels in true_and_prediction:
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(
            set(predicted_labels).intersection(set(true_labels))
        )
    precision = num_correct_labels / num_predicted_labels
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)


if __name__ == "__main__":
    # test the eval function for seen and unseen classes
    seen_y = torch.Tensor(
        [[0.9, 0.3, 0.2, 0.45], [0.1, 0.3, 0.25, 0.1], [0.1, 0.2, 0.4, 0.5]]
    )
    unseen_y = torch.Tensor([[0.6, 0.45], [0.1, 0.4], [0.65, 0.55]])

    prob_preds = torch.cat([seen_y, unseen_y], dim=1)
    one_hot = torch.Tensor(
        [[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1]]
    )

    #
    test_len = [3, 3, 2]
    test_acc = [1 / 3, 2 / 3, 1]

    labels = [list(range(6)), list(range(4)), [4, 5]]

    result = {}

    for i in range(3):
        one_hot_list = one_hot[:, labels[i]].cpu().numpy().tolist()
        prob_list = prob_preds[:, labels[i]].cpu().numpy().tolist()
        true_and_pred = get_true_and_prediction(prob_list, one_hot_list)

        # remove true and pred that are empty
        clean_true_pred = []
        for k in range(len(true_and_pred)):
            if true_and_pred[k][0]:
                clean_true_pred.append(true_and_pred[k])

        strict_acc = strict(clean_true_pred)[2]

        print(true_and_pred)
        print(
            "expected value {:.4f} function value {:.4f}".format(
                test_acc[i], strict_acc
            )
        )

        assert test_acc[i] == strict_acc
        assert len(clean_true_pred) == test_len[i]
