import json

import networkx as nx


def id_to_name(node, entity_to_id):
    return entity_to_id.index[entity_to_id['id'] == int(node)][0]


# Reassign node IDs in graphs
def reassign_ids(symptom_symptom_graph, efficacy_efficacy_graph, herb_herb_graph, entity_to_id):
    symptom_to_id = {}
    efficacy_to_id = {}
    herb_to_id = {}

    # Reassign IDs for symptom-symptom graph
    symptom_id = 0
    for node in symptom_symptom_graph.nodes:
        if node not in symptom_to_id:
            symptom_to_id[node] = symptom_id
            symptom_id += 1

    # Reassign IDs for efficacy-efficacy graph
    efficacy_id = 0
    for node in efficacy_efficacy_graph.nodes:
        if node not in efficacy_to_id:
            efficacy_to_id[node] = efficacy_id
            efficacy_id += 1

    # Reassign IDs for herb-herb graph
    herb_id = 0
    for node in herb_herb_graph.nodes:
        if node not in herb_to_id:
            herb_to_id[node] = herb_id
            herb_id += 1

    # Update node IDs in each graph
    symptom_symptom_graph = update_graph_node_ids(symptom_symptom_graph, symptom_to_id)
    efficacy_efficacy_graph = update_graph_node_ids(efficacy_efficacy_graph, efficacy_to_id)
    herb_herb_graph = update_graph_node_ids(herb_herb_graph, herb_to_id)

    # Build new id-to-name mappings based on TransD entity_to_id
    symptom_new_id_to_name, efficacy_new_id_to_name, herb_new_id_to_name = create_new_id_to_name_mapping_from_transd(
        entity_to_id, symptom_to_id, efficacy_to_id, herb_to_id
    )

    return (
        symptom_symptom_graph,
        efficacy_efficacy_graph,
        herb_herb_graph,
        symptom_to_id,
        efficacy_to_id,
        herb_to_id,
        symptom_new_id_to_name,
        efficacy_new_id_to_name,
        herb_new_id_to_name,
    )


# Reassign node IDs for heterogeneous symptom-efficacy-herb graph
def reassign_ids_symptom_efficacy_herb_graph(symptom_efficacy_herb_graph, entity_to_id):
    symptom_efficacy_herb_to_id = {}

    # Reassign IDs for heterogeneous graph
    symptom_efficacy_herb_id = 0
    for node in symptom_efficacy_herb_graph.nodes:
        if node not in symptom_efficacy_herb_to_id:
            symptom_efficacy_herb_to_id[node] = symptom_efficacy_herb_id
            symptom_efficacy_herb_id += 1

    # Update node IDs in the graph
    symptom_efficacy_herb_graph = update_graph_node_ids(symptom_efficacy_herb_graph, symptom_efficacy_herb_to_id)

    # Build new id-to-name mapping
    symptom_efficacy_herb_new_id_to_name = create_new_id_to_name_mapping_from_transd_symptom_efficacy(
        entity_to_id, symptom_efficacy_herb_to_id
    )

    return symptom_efficacy_herb_graph, symptom_efficacy_herb_to_id, symptom_efficacy_herb_new_id_to_name


def update_graph_node_ids(graph, entity_to_id_mapping):
    """
    Reassign node IDs in a graph according to a provided ID mapping dictionary.

    Parameters:
    - graph: NetworkX graph to update
    - entity_to_id_mapping: dict mapping original node identifiers (names) to new numeric IDs

    Returns:
    - updated_graph: the graph with node identifiers replaced by the new IDs
    """
    updated_graph = nx.Graph()

    # Update node IDs and features
    for node in graph.nodes:
        if 'feature' in graph.nodes[node]:
            feature = graph.nodes[node]['feature']
            new_node_id = entity_to_id_mapping[node]  # get new ID
            if 'category' in graph.nodes[node]:
                category = graph.nodes[node]['category']
                updated_graph.add_node(new_node_id, feature=feature, category=category)
            else:
                updated_graph.add_node(new_node_id, feature=feature)

    # Update edges to use new node IDs
    for node1, node2, data in graph.edges(data=True):
        new_node1 = entity_to_id_mapping[node1]
        new_node2 = entity_to_id_mapping[node2]
        updated_graph.add_edge(new_node1, new_node2, **data)

    return updated_graph


def create_new_id_to_name_mapping_from_transd(entity_to_id, symptom_to_id, efficacy_to_id, symptom_herb_to_id):
    """
    Construct new mappings from graph-assigned IDs to original names using TransD entity_to_id.

    Parameters:
    - entity_to_id: DataFrame from TransD mapping (index: name, column 'id' is original numeric id)
    - symptom_to_id: mapping from symptom name to graph node ID
    - efficacy_to_id: mapping from efficacy name to graph node ID
    - symptom_herb_to_id: mapping from symptom/herb name to graph node ID

    Returns:
    - symptom_new_id_to_name, efficacy_new_id_to_name, symptom_herb_new_id_to_name: dictionaries mapping new graph IDs to names
    """
    symptom_new_id_to_name = {}
    efficacy_new_id_to_name = {}
    symptom_herb_new_id_to_name = {}

    # Reverse mappings: graph ID -> name
    reversed_symptom_to_id = {v: k for k, v in symptom_to_id.items()}
    reversed_efficacy_to_id = {v: k for k, v in efficacy_to_id.items()}
    reversed_symptom_herb_to_id = {v: k for k, v in symptom_herb_to_id.items()}

    # Add symptom mappings
    for new_id, name in reversed_symptom_to_id.items():
        if entity_to_id.loc[name, 'id'] in entity_to_id['id'].values:
            symptom_new_id_to_name[new_id] = name

    # Add efficacy mappings
    for new_id, name in reversed_efficacy_to_id.items():
        if entity_to_id.loc[name, 'id'] in entity_to_id['id'].values:
            efficacy_new_id_to_name[new_id] = name

    # Add symptom/herb mappings
    for new_id, name in reversed_symptom_herb_to_id.items():
        if entity_to_id.loc[name, 'id'] in entity_to_id['id'].values:
            symptom_herb_new_id_to_name[new_id] = name

    # Save mapping to JSON
    with open('mapping/symptom_herb_new_id_to_name.json', 'w', encoding='utf-8') as f:
        json.dump(symptom_herb_new_id_to_name, f, ensure_ascii=False, indent=4)

    return symptom_new_id_to_name, efficacy_new_id_to_name, symptom_herb_new_id_to_name


def create_new_id_to_name_mapping_from_transd_symptom_efficacy(entity_to_id, symptom_efficacy_to_id):
    """
    Construct new ID-to-name mapping for symptom-efficacy graph from TransD entity_to_id.

    Parameters:
    - entity_to_id: TransD entity-to-id mapping DataFrame
    - symptom_efficacy_to_id: mapping from symptom/efficacy name to graph node ID

    Returns:
    - symptom_efficacy_new_id_to_name: dict mapping new graph IDs to names
    """
    symptom_efficacy_new_id_to_name = {}

    # Reverse mapping: graph ID -> name
    reversed_symptom_efficacy_to_id = {v: k for k, v in symptom_efficacy_to_id.items()}

    # Add mappings
    for new_id, name in reversed_symptom_efficacy_to_id.items():
        if entity_to_id.loc[name, 'id'] in entity_to_id['id'].values:
            symptom_efficacy_new_id_to_name[new_id] = name

    # Save mapping to JSON
    with open('mapping/symptom_efficacy_new_id_to_name.json', 'w', encoding='utf-8') as f:
        json.dump(symptom_efficacy_new_id_to_name, f, ensure_ascii=False, indent=4)

    return symptom_efficacy_new_id_to_name