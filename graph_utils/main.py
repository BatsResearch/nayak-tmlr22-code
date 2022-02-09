import os
from graph import compute_embeddings
import argparse

from imagenet_syns import IMAGENET_SYNS
from query import query_conceptnet

import pandas as pd
from allennlp.common.params import Params
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG

DATABASE_PATH = "data/scads.spring2021.sqlite3"
GRAPH_PATH = "data/example_graph"
GLOVE_PATH = "data/glove.840B.300d.txt"


def graph_setup(syns, database_path, glove_path, self_loop=False):

    nodes, edges, relations = query_conceptnet(syns, database_path)

    # remap edges starting from 0
    idx_to_node_idx = dict(nodes["id"])
    node_id_to_idx = dict([(n, i) for i, n in idx_to_node_idx.items()])

    # post processing (mapping ed)
    mapped_edges = []
    for i, row in edges.iterrows():
        start_id = node_id_to_idx[row["start_id"]]
        end_id = node_id_to_idx[row["end_id"]]
        mapped_edges.append((start_id, int(row["relation_id"]), end_id))
    mapped_edges = pd.DataFrame(
        mapped_edges, columns=["start_id", "relation_id", "end_id"]
    )

    if self_loop:
        self_edges = []
        for node_id in range(len(nodes["id"])):
            self_edges.append([node_id, 50, node_id])
        self_edges = pd.DataFrame(
            self_edges, columns=["start_id", "relation_id", "end_id"]
        )

        mapped_edges = pd.concat([mapped_edges, self_edges])

    features = compute_embeddings(nodes, glove_path)

    params = Params({"bidirectional": True})

    kg = ConceptNetKG(
        nodes["uri"].tolist(),
        features,
        mapped_edges,
        relations["uri"].tolist(),
        params,
    )
    kg.run_random_walk()

    kg.save_to_disk("data/example_graph")

    return kg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self_loop", action="store_true", help="add self loop in the graph"
    )
    args = parser.parse_args()

    kg = graph_setup(IMAGENET_SYNS, DATABASE_PATH, GLOVE_PATH, self_loop=args.self_loop)

    print("and done!")
