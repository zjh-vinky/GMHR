import json
from collections import defaultdict

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from Reassign_ids import id_to_name


# Build co-occurrence graphs
def build_cooccurrence_graph(triples, entity_to_id, entity_embeddings):
    """Construct symptom-symptom, efficacy-efficacy, and herb-herb co-occurrence graphs from triples and attach features."""
    symptom_symptom_graph = nx.Graph()
    efficacy_efficacy_graph = nx.Graph()
    herb_herb_graph = nx.Graph()

    symptom_pairs = defaultdict(int)
    efficacy_pairs = defaultdict(int)
    herb_pairs = defaultdict(int)
    prescription_to_symptoms = defaultdict(list)
    prescription_to_efficacies = defaultdict(list)
    prescription_to_herbs = defaultdict(list)
    node_types = {}

    # Step 1: classify symptoms, efficacies, and herbs by relation and record node types
    for subj, pred, obj in triples:
        if pred == '4':  # prescription -> symptom relation
            prescription_to_symptoms[subj].append(obj)
            node_types[obj] = 'symptom'
        elif pred == '3':  # prescription -> efficacy relation
            prescription_to_efficacies[subj].append(obj)
            node_types[obj] = 'efficacy'
        elif pred == '2':  # prescription -> herb relation
            prescription_to_herbs[subj].append(obj)
            node_types[obj] = 'herb'

    # Step 2: count co-occurrence pairs for symptom-symptom, efficacy-efficacy, herb-herb
    for prescription, symptoms in prescription_to_symptoms.items():  # symptom-symptom
        for i in range(len(symptoms)):
            for j in range(i + 1, len(symptoms)):
                symptom_pairs[(symptoms[i], symptoms[j])] += 1
    for prescription, efficacies in prescription_to_efficacies.items():  # efficacy-efficacy
        for i in range(len(efficacies)):
            for j in range(i + 1, len(efficacies)):
                efficacy_pairs[(efficacies[i], efficacies[j])] += 1
    for prescription, herbs in prescription_to_herbs.items():  # herb-herb
        for i in range(len(herbs)):
            for j in range(i + 1, len(herbs)):
                herb_pairs[(herbs[i], herbs[j])] += 1

    print(f"Successfully extracted prescription info, symptom-symptom pairs: {len(symptom_pairs)}, efficacy-efficacy pairs: {len(efficacy_pairs)}, herb-herb pairs: {len(herb_pairs)}")

    # Step 3: build graphs and add edges
    for (node1, node2), weight in symptom_pairs.items():
        node1_name = id_to_name(node1, entity_to_id)
        node2_name = id_to_name(node2, entity_to_id)
        symptom_symptom_graph.add_edge(node1_name, node2_name, weight=weight)
    for (node1, node2), weight in efficacy_pairs.items():
        node1_name = id_to_name(node1, entity_to_id)
        node2_name = id_to_name(node2, entity_to_id)
        efficacy_efficacy_graph.add_edge(node1_name, node2_name, weight=weight)

    for (node1, node2), weight in herb_pairs.items():
        node1_name = id_to_name(node1, entity_to_id)
        node2_name = id_to_name(node2, entity_to_id)

        herb_herb_graph.add_node(node1_name, category=node_types[node1])
        herb_herb_graph.add_node(node2_name, category=node_types[node2])
        herb_herb_graph.add_edge(node1_name, node2_name, weight=weight)

    print("Successfully constructed node pairs and edges:")
    print(f"Symptom-Symptom graph edges: {len(symptom_symptom_graph.edges)}, Efficacy-Efficacy graph edges: {len(efficacy_efficacy_graph.edges)}, Herb-Herb graph edges: {len(herb_herb_graph.edges)}")

    # Step 4: attach embedding vectors to each node
    for graph in [symptom_symptom_graph, efficacy_efficacy_graph, herb_herb_graph]:
        for node in graph.nodes:
            entity_id = entity_to_id['id'].get(node)
            if entity_id is not None:
                entity_vector = entity_embeddings[0]._embeddings.weight.data[entity_id].cpu().detach().numpy()
                graph.nodes[node]['feature'] = entity_vector

    return symptom_symptom_graph, efficacy_efficacy_graph, herb_herb_graph



def build_symptom_efficacy_herb_graph(triples, entity_to_id, symptom_features_dict, efficacy_features_dict, herb_dict, entity_embeddings):
    """Construct a heterogeneous symptom-efficacy-herb co-occurrence graph and attach features."""
    print("Starting to build co-occurrence graph...")
    symptom_efficacy_herb_graph = nx.Graph()

    symptom_efficacy_pairs = defaultdict(int)
    symptom_herb_pairs = defaultdict(int)
    efficacy_herb_pairs = defaultdict(int)
    prescription_to_symptoms = defaultdict(list)
    prescription_to_efficacies = defaultdict(list)
    prescription_to_herbs = defaultdict(list)
    node_types = {}
    node_types_name = {}

    # Step 1: classify symptoms, efficacies, and herbs by relation and record node types
    for subj, pred, obj in triples:
        if pred == '4':  # prescription -> symptom
            prescription_to_symptoms[subj].append(obj)
            node_types[obj] = 'symptom'
            node_types_name[entity_to_id.index[entity_to_id['id'] == int(obj)][0]] = 'symptom'
        elif pred == '3':  # prescription -> efficacy
            prescription_to_efficacies[subj].append(obj)
            node_types[obj] = 'efficacy'
            node_types_name[entity_to_id.index[entity_to_id['id'] == int(obj)][0]] = 'efficacy'
        elif pred == "2":  # prescription -> herb
            prescription_to_herbs[subj].append(obj)
            node_types[obj] = 'herb'
            node_types_name[entity_to_id.index[entity_to_id['id'] == int(obj)][0]] = 'herb'

    # Step 2: count symptom-efficacy, symptom-herb, and efficacy-herb co-occurrence pairs
    for prescription, symptoms in prescription_to_symptoms.items():    # symptom-efficacy
        if prescription in prescription_to_efficacies:
            for symptom in symptoms:
                for efficacy in prescription_to_efficacies[prescription]:
                    symptom_efficacy_pairs[(symptom, efficacy)] += 1
                    node_types[symptom] = 'symptom'
                    node_types[efficacy] = 'efficacy'

    for prescription, symptoms in prescription_to_symptoms.items():    # symptom-herb
        if prescription in prescription_to_herbs:
            for symptom in symptoms:
                for herb in prescription_to_herbs[prescription]:
                    symptom_herb_pairs[(symptom, herb)] += 1
                    node_types[symptom] = 'symptom'
                    node_types[herb] = 'herb'

    for prescription, efficacies in prescription_to_efficacies.items():    # efficacy-herb
        if prescription in prescription_to_herbs:
            for efficacy in efficacies:
                for herb in prescription_to_herbs[prescription]:
                    efficacy_herb_pairs[(efficacy, herb)] += 1
                    node_types[efficacy] = 'efficacy'
                    node_types[herb] = 'herb'

    print(f"Successfully extracted prescription info, symptom-efficacy pairs: {len(symptom_efficacy_pairs)}, symptom-herb pairs: {len(symptom_herb_pairs)}, efficacy-herb pairs: {len(efficacy_herb_pairs)}")

    # Step 3: build graph and add edges
    for (node1, node2), weight in symptom_efficacy_pairs.items():
        node1_name = id_to_name(node1, entity_to_id)
        node2_name = id_to_name(node2, entity_to_id)
        symptom_efficacy_herb_graph.add_node(node1_name, category=node_types[node1])
        symptom_efficacy_herb_graph.add_node(node2_name, category=node_types[node2])
        symptom_efficacy_herb_graph.add_edge(node1_name, node2_name, weight=weight)

    for (node1, node2), weight in symptom_herb_pairs.items():
        node1_name = id_to_name(node1, entity_to_id)
        node2_name = id_to_name(node2, entity_to_id)
        symptom_efficacy_herb_graph.add_node(node1_name, category=node_types[node1])
        symptom_efficacy_herb_graph.add_node(node2_name, category=node_types[node2])
        symptom_efficacy_herb_graph.add_edge(node1_name, node2_name, weight=weight)

    for (node1, node2), weight in efficacy_herb_pairs.items():
        node1_name = id_to_name(node1, entity_to_id)
        node2_name = id_to_name(node2, entity_to_id)
        symptom_efficacy_herb_graph.add_node(node1_name, category=node_types[node1])
        symptom_efficacy_herb_graph.add_node(node2_name, category=node_types[node2])
        symptom_efficacy_herb_graph.add_edge(node1_name, node2_name, weight=weight)

    # Add prescription-related edges with weight 1
    for prescription, symptoms in prescription_to_symptoms.items():
        for symptom in symptoms:
            prescription_name = id_to_name(prescription, entity_to_id)
            symptom_name = id_to_name(symptom, entity_to_id)
            symptom_efficacy_herb_graph.add_node(prescription_name)
            if not symptom_efficacy_herb_graph.has_node(symptom_name):
                symptom_efficacy_herb_graph.add_node(symptom_name)
            symptom_efficacy_herb_graph.add_edge(prescription_name, symptom_name, weight=1)

    for prescription, efficacies in prescription_to_efficacies.items():
        for efficacy in efficacies:
            prescription_name = id_to_name(prescription, entity_to_id)
            efficacy_name = id_to_name(efficacy, entity_to_id)
            symptom_efficacy_herb_graph.add_node(prescription_name)
            if not symptom_efficacy_herb_graph.has_node(efficacy_name):
                symptom_efficacy_herb_graph.add_node(efficacy_name)
            symptom_efficacy_herb_graph.add_edge(prescription_name, efficacy_name, weight=1)

    for prescription, herbs in prescription_to_herbs.items():
        for herb in herbs:
            prescription_name = id_to_name(prescription, entity_to_id)
            herb_name = id_to_name(herb, entity_to_id)
            symptom_efficacy_herb_graph.add_node(prescription_name)
            if not symptom_efficacy_herb_graph.has_node(herb_name):
                symptom_efficacy_herb_graph.add_node(herb_name)
            symptom_efficacy_herb_graph.add_edge(prescription_name, herb_name, weight=1)

    print("Successfully constructed node pairs and edges:")
    print(f"Symptom-Efficacy-Herb heterogeneous graph edges: {len(symptom_efficacy_herb_graph.edges)}")

    # Step 4: attach embedding vectors to each node
    for graph in [symptom_efficacy_herb_graph]:
        for node in graph.nodes:
            entity_id = entity_to_id['id'].get(node)
            if entity_id is not None:
                if node in symptom_features_dict:
                    entity_vector = symptom_features_dict[node].cpu().detach().numpy()
                    graph.nodes[node]['feature'] = entity_vector
                elif node in efficacy_features_dict:
                    entity_vector = efficacy_features_dict[node].cpu().detach().numpy()
                    graph.nodes[node]['feature'] = entity_vector
                elif node in herb_dict:
                    entity_vector = herb_dict[node].cpu().detach().numpy()
                    graph.nodes[node]['feature'] = entity_vector
                else:
                    entity_vector = entity_embeddings[0]._embeddings.weight.data[entity_id].cpu().detach().numpy()
                    graph.nodes[node]['feature'] = entity_vector

    return symptom_efficacy_herb_graph


# Create PyG Data object from networkx graph
def create_data_graph(graph, new_entity_to_id):
    """Convert a networkx graph to a PyG Data object and ensure edge node indices are valid."""
    # Get graph edges
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Get edge weights with default value handling
    edge_attr = []
    for edge in graph.edges:
        weight = graph[edge[0]][edge[1]].get('weight', 1.0)  # default to 1.0
        edge_attr.append(weight)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Remove self-loops
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # Assertions: ensure edge node indices are within node count range
    num_nodes = len(new_entity_to_id)
    assert torch.max(edge_index) < num_nodes, "Edge node index exceeds number of nodes in the graph!"
    assert torch.min(edge_index) >= 0, "Edge node index is negative!"

    # Get node features
    node_features = []
    for node in graph.nodes:
        if node in new_entity_to_id.values():
            if 'feature' in graph.nodes[node]:
                node_features.append(graph.nodes[node]['feature'])
            else:
                print(f"Warning: node {node} lacks features, using default feature vector!")
                node_features.append([0.0] * 128)  # default feature vector (zero vector)

    node_features = torch.tensor(node_features, dtype=torch.float)

    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return data