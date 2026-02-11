# data_processing.py
import gzip
import os
import pandas as pd
import torch

# Load entity and relation mappings
def load_mappings(entity_to_id_path, relation_to_id_path):
    """Load entity and relation mapping files and return mapping dictionaries."""
    if not os.path.exists(entity_to_id_path) or not os.path.exists(relation_to_id_path):
        raise FileNotFoundError(f"Mapping file not found: {entity_to_id_path} or {relation_to_id_path}")

    print(f"Loading entity mapping file: {entity_to_id_path}")
    print(f"Loading relation mapping file: {relation_to_id_path}")
    try:
        entity_to_id = pd.read_csv(entity_to_id_path, sep='\t', compression='gzip')
        relation_to_id = pd.read_csv(relation_to_id_path, sep='\t', compression='gzip')
        entity_to_id.set_index('label', inplace=True)
        relation_to_id.set_index('label', inplace=True)
        print(f"Successfully loaded entity and relation mappings: num_entities = {len(entity_to_id)}, num_relations = {len(relation_to_id)}")
    except Exception as e:
        raise ValueError(f"Error loading mapping files: {e}")
    return entity_to_id, relation_to_id

# ---------------- File operation module ----------------
def read_triples(file_path):
    """Read triples file and return a list of triples, skipping the header line."""
    triples = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Reading triples file: {file_path}")
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            # skip the first header line
            next(f)
            for line in f:
                subject, predicate, obj = line.strip().split('\t')
                triples.append((subject, predicate, obj))
        print(f"Successfully read {len(triples)} triples")
    except Exception as e:
        raise ValueError(f"Error reading triples file: {e}")
    return triples

def load_entity_embeddings(entity_embeddings_path):
    """Load entity embeddings"""
    entity_embeddings = {}
    with open(entity_embeddings_path, 'rb') as f:
        entity_embeddings = torch.load(f)
    return entity_embeddings


# ---------------- Embedding save module ----------------
def save_features(features, data_ids, file_path):
    """Save embedding features to a file."""
    features_dict = {node_id: feat for node_id, feat in zip(data_ids, features)}
    torch.save(features_dict, file_path)
    print(f"Features saved to: {file_path}")
    return features_dict


# ---------------- Graph save module ----------------
def save_graph(features, file_path):
    """Save graph structure to a file."""
    torch.save(features, file_path)
    print(f"Graph saved to: {file_path}")