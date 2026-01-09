# src/ontology.py
import yaml

with open("ontology.yaml") as f:
    ONTOLOGY = yaml.safe_load(f)

L1_LIST = ONTOLOGY["L1"]
L2_MAP = ONTOLOGY["L2"]
