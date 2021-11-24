import os
from conceptnet import query_conceptnet
from graph import (
    post_process_graph,
    compute_union_graph,
    compute_embeddings,
    compute_mapping,
)
from random_walk import graph_random_walk

from imagenet_syns import IMAGENET_SYNS
from idx_to_concept import IDX_TO_CONCEPT_IMGNET


def graph_setup(syns, graph_path, database_path, glove_path, id_to_concept):

    query_conceptnet(graph_path, syns, database_path)

    # post process graph
    post_process_graph(graph_path)

    # take the union of the graph
    compute_union_graph(graph_path)

    # run random walk on the graph
    graph_random_walk(graph_path, k=20, n=10)

    # compute embeddings for the nodes
    compute_embeddings(graph_path, glove_path)

    # compute mapping
    compute_mapping(id_to_concept, graph_path)

    print("completed graph related processing!")


print("example graph")
os.makedirs("data/example_graph")

DATABASE_PATH = "data/conceptnet.db"
GRAPH_PATH = "data/example_graph"
GLOVE_PATH = "data/glove.840B.300d.txt"

graph_setup(
    IMAGENET_SYNS, GRAPH_PATH, DATABASE_PATH, GLOVE_PATH, IDX_TO_CONCEPT_IMGNET
)

print("and done!")
