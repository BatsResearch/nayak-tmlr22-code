## Graph preprocessing

### Example
We include the scripts to query ImageNet graph, post-process, compute union
of the grpah, get initial node features, and run
random walk on the graph.

## setup

```
mkdir data
cd data
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
wget -nc https://storage.googleapis.com/taglets-public/scads.spring2021.sqlite3
```

## knowledge graph construction

```
python main.py
```

You may need to edit the paths for `DATABASE_PATH`, `GRAPH_PATH`, and `GLOVE_PATH` in the `main.py`.

### ConceptNet Schema
We query three tables from the conceptnet database.
We assume the tables have the following schema:
```
CREATE TABLE nodes (
	id INTEGER NOT NULL,
	conceptnet_id VARCHAR,
	PRIMARY KEY (id)
);
CREATE TABLE relations (
	id INTEGER NOT NULL,
	type VARCHAR,
	is_directed BOOLEAN,
	PRIMARY KEY (id),
	CHECK (is_directed IN (0, 1))
);
CREATE TABLE edges (
	id INTEGER NOT NULL,
	relation_type INTEGER,
	weight FLOAT,
	start_node INTEGER,
	end_node INTEGER,
	PRIMARY KEY (id),
	FOREIGN KEY(relation_type) REFERENCES relations (id),
	FOREIGN KEY(start_node) REFERENCES nodes (id),
	FOREIGN KEY(end_node) REFERENCES nodes (id)
);
```

