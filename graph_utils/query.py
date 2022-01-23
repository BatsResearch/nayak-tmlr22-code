import sqlite3
import itertools
import pandas as pd


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def query_conceptnet(concept_syns, database_path, n=2):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    edges = []
    nodes = []
    relations = []

    # get all the concepts including syns for the concepts
    # example {"/c/en/cat": ["/c/en/cat/n"]}
    concepts = []
    for concept, syns in concept_syns.items():
        concepts.append(concept)
        concepts += syns

    # label_node_ids = set()
    node_ids = set()
    syn_id_to_node_id = {}

    print("querying the concepts")
    for concept, syns in concept_syns.items():
        x = cursor.execute(
            'SELECT * from nodes where conceptnet_id="' + concept + '" LIMIT 1'
        ).fetchall()

        # if concept is not found
        if not x:
            raise Exception(concept + " not found")

        main_node_id, node_uri = x[0]
        node_ids.add(main_node_id)

        for _syn in syns:
            x = cursor.execute(
                'SELECT * from nodes where conceptnet_id="'
                + _syn
                + '" LIMIT 1'
            ).fetchall()

            for node_id, _ in x:
                syn_id_to_node_id[node_id] = main_node_id
                node_ids.add(node_id)

    # query the edges table
    hops = []
    hops.append(list(node_ids))
    edges_list = []

    print("querying the edges")
    for i in range(n):
        print("Hop ", i)
        new_nodes = set()
        for batch_nodes in chunks(hops[i], 5000):
            for x in ["start_node", "end_node"]:
                query_string = "select start_node, end_node, relation_type, weight from edges where"
                for node_id in batch_nodes:
                    query_string += " " + x + "=" + str(node_id)
                    query_string += " or"

                query_string = query_string[:-3]
                neigh_concepts = cursor.execute(query_string).fetchall()
                edges_list.extend(neigh_concepts)
                all_concepts = set((itertools.chain.from_iterable(hops)))
                for start_id, end_id, relation_id, weight in neigh_concepts:
                    if start_id not in all_concepts:
                        new_nodes.add(start_id)
                    if end_id not in all_concepts:
                        new_nodes.add(end_id)

        hops.append(list(new_nodes))

    # get all concepts
    all_concepts = list(set((itertools.chain.from_iterable(hops))))
    nodes = []
    i = 0
    for batch_concepts in chunks(all_concepts, 5000):
        i += 1
        query_string = "select * from nodes where"
        for c in batch_concepts:
            query_string += ' id="' + str(c) + '"'
            query_string += " or"
        query_string = query_string[:-3]

        x = cursor.execute(query_string).fetchall()
        nodes.extend(x)

    # replace edges list with the syn ids
    edges = []
    for start_id, end_id, relation_id, _ in edges_list:

        if start_id in syn_id_to_node_id:
            start_id = syn_id_to_node_id[start_id]

        if end_id in syn_id_to_node_id:
            end_id = syn_id_to_node_id[end_id]

        edges.append((start_id, relation_id, end_id))

    relations = cursor.execute(
        "SELECT id, type, is_directed from relations"
    ).fetchall()

    nodes = pd.DataFrame(nodes, columns=["id", "uri"])

    edges = pd.DataFrame(edges, columns=["start_id", "relation_id", "end_id"])

    relations = pd.DataFrame(relations, columns=["id", "uri", "directed"])

    return nodes, edges, relations
