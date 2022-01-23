import os
from graph import (
    compute_embeddings,
)

from imagenet_syns import IMAGENET_SYNS
from query import query_conceptnet

from allennlp.common.params import Params
from zsl_kg.knowledge_graph.conceptnet import ConceptNetKG


def graph_setup(syns, database_path, glove_path):

    nodes, edges, relations = query_conceptnet(syns, database_path)

    features = compute_embeddings(nodes, glove_path)

    params = Params({"bidirectional": True})

    kg = ConceptNetKG(
        nodes["uri"].tolist(),
        features,
        edges.values.tolist(),
        relations["uri"].tolist(),
        params,
    )
    kg.run_random_walk()

    kg.save_to_disk("data/example_graph")


DATABASE_PATH = "data/scads.spring2021.sqlite3"
GRAPH_PATH = "data/example_graph"
GLOVE_PATH = "data/glove.840B.300d.txt"

graph_setup(IMAGENET_SYNS, DATABASE_PATH, GLOVE_PATH)

print("and done!")
