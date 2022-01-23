import re
import numpy as np

import torch
import torch.nn.functional as F


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def compute_embeddings(nodes, glove_path):
    """The function is used to compute the initial node embeddings
    for all the nodes in the graph.

    Args:
        graph_path (str): path to the conceptnet subgraph directory
        glove_path (str): path to the glove file
    """
    # get words from the nodes
    print("extract individual words from concepts")
    words = set()
    all_concepts = []
    for index, node in nodes.iterrows():
        concept_words = get_individual_words(node["uri"])
        all_concepts.append(concept_words)
        for w in concept_words:
            words.add(w)

    #
    word_to_idx = dict([(word, idx + 1) for idx, word in enumerate(words)])
    word_to_idx["<PAD>"] = 0
    idx_to_word = dict([(idx, word) for word, idx in word_to_idx.items()])

    # load glove 840
    print("loading glove from file")
    glove = load_embeddings(glove_path)

    # get the word embedding
    embedding_matrix = torch.zeros(len(word_to_idx), 300)
    for idx, word in idx_to_word.items():
        if word in glove:
            embedding_matrix[idx] = torch.Tensor(glove[word])

    #
    print("padding concepts")
    max_length = max([len(concept_words) for concept_words in all_concepts])
    padded_concepts = []
    for concept_words in all_concepts:
        concept_idx = [word_to_idx[word] for word in concept_words]
        concept_idx += [0] * (max_length - len(concept_idx))
        padded_concepts.append(concept_idx)

    # add the word embeddings of indivual words
    print("adding the word embeddings and l2 norm-> conceptnet embeddings")
    concept_embs = torch.zeros((0, 300))
    padded_concepts = torch.tensor(padded_concepts)
    for pc in chunks(padded_concepts, 100000):
        concept_words = embedding_matrix[pc]
        embs = torch.sum(concept_words, dim=1)
        embs = F.normalize(embs)
        concept_embs = torch.cat((concept_embs, embs), dim=0)

    return concept_embs


def load_embeddings(file_path):
    """file to load glove"""
    embeddings = {}
    with open(file_path) as fp:
        for line in fp:
            fields = line.rstrip().split(" ")
            vector = np.asarray(fields[1:], dtype="float32")
            embeddings[fields[0]] = vector

    return embeddings


def get_individual_words(concept):
    """extracts the individual words from a concept"""
    clean_concepts = re.sub(r"\/c\/[a-z]{2}\/|\/.*", "", concept)
    return clean_concepts.strip().split("_")
