# main.py
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from Data_processing import load_mappings, read_triples, load_entity_embeddings, save_features, save_graph
from Build_graph import build_cooccurrence_graph, create_data_graph, build_symptom_efficacy_herb_graph
from Reassign_ids import reassign_ids, reassign_ids_symptom_efficacy_herb_graph
from Model import AdvancedGAT, train_gat
import torch

def main():
    """Main program: load data, build graphs, and train the GAT model."""
    path = 'trained_model_TransD_Chinese/training_triples/'
    entity_to_id_path = path + 'entity_to_id.tsv.gz'
    relation_to_id_path = path + 'relation_to_id.tsv.gz'
    triples_path = path + 'numeric_triples.tsv.gz'
    entity_embeddings_path = 'trained_model_TransD_Chinese/entity_embeddings.pkl'

    entity_to_id, relation_to_id = load_mappings(entity_to_id_path, relation_to_id_path)
    triples = read_triples(triples_path)
    entity_embeddings = load_entity_embeddings(entity_embeddings_path)

    # Build co-occurrence graphs
    print("Start building co-occurrence graphs...")
    symptom_symptom_graph, efficacy_efficacy_graph, herb_herb_graph = build_cooccurrence_graph(triples, entity_to_id, entity_embeddings)

    # Save PyTorch graphs and their features as binary files
    save_graph(symptom_symptom_graph, "models/symptom_symptom_graph.pt")
    save_graph(efficacy_efficacy_graph, "models/efficacy_efficacy_graph.pt")
    save_graph(herb_herb_graph, "models/herb_herb_graph.pt")

    # Reassign IDs
    print("Start reassigning IDs...")
    symptom_symptom_graph, efficacy_efficacy_graph, herb_herb_graph, symptom_to_id, efficacy_to_id, herb_to_id, symptom_new_id_to_name, efficacy_new_id_to_name, herb_new_id_to_name = reassign_ids(symptom_symptom_graph, efficacy_efficacy_graph, herb_herb_graph, entity_to_id)

    # Create PyG Data objects
    print("Start creating PyG data graphs...")
    symptom_data = create_data_graph(symptom_symptom_graph, symptom_to_id)
    efficacy_data = create_data_graph(efficacy_efficacy_graph, efficacy_to_id)
    herb_data = create_data_graph(herb_herb_graph, herb_to_id)


    # Initialize TCM_PR model
    in_channels = symptom_data.x.size(1)  # node feature dimension
    out_channels = 256  # TCM_PR output dimension
    final_out_dim = 128  # set to match target data dimension
    hidden_channels = 512  # hidden layer dimension
    num_layers = 2  # number of TCM_PR layers
    heads = 4  # heads per layer
    dropout = 0.2  # dropout probability

    # Initialize model
    model = AdvancedGAT(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, final_out_dim=final_out_dim, heads=heads, num_layers=num_layers, dropout=dropout)

    # Check GPU support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Use Adam optimizer
    optimizer1 = Adam(model.parameters(), lr=0.0001, weight_decay=7e-4)
    optimizer2 = Adam(model.parameters(), lr=0.0001, weight_decay=7e-4)
    optimizer3 = Adam(model.parameters(), lr=0.0001, weight_decay=7e-4)
    optimizer4 = Adam(model.parameters(), lr=0.0001, weight_decay=7e-4)

    # Use StepLR learning rate schedulers
    scheduler1 = StepLR(optimizer1, step_size=20, gamma=0.9)
    scheduler2 = StepLR(optimizer2, step_size=20, gamma=0.9)
    scheduler3 = StepLR(optimizer3, step_size=20, gamma=0.9)
    scheduler4 = StepLR(optimizer4, step_size=20, gamma=0.9)

    # Train TCM_PR model
    print("Start training TCM_PR model...")
    train_gat(model, symptom_data, optimizer1, scheduler=scheduler1, epochs=100, device=device, patience=10)
    train_gat(model, efficacy_data, optimizer2, scheduler=scheduler2, epochs=100, device=device, patience=10)
    train_gat(model, herb_data, optimizer3, scheduler=scheduler3, epochs=100, device=device, patience=10)


    # Extract graph features
    with torch.no_grad():
        symptom_features = model(symptom_data)
        efficacy_features = model(efficacy_data)
        herb_features = model(herb_data)
        print("symptom_features", symptom_features)
        print("efficacy_features", efficacy_features)
        print("herb_features", herb_features)


    # Save features to files
    symptom_features_dict = save_features(symptom_features, symptom_to_id, "models/symptom_features.pt")
    efficacy_features_dict = save_features(efficacy_features, efficacy_to_id, "models/efficacy_features.pt")
    herb_features_dict = save_features(herb_features, herb_to_id, "models/herb_features.pt")

    # Heterogeneous graph part
    symptom_efficacy_herb_graph = build_symptom_efficacy_herb_graph(triples, entity_to_id, symptom_features_dict, efficacy_features_dict, herb_features_dict, entity_embeddings)
    symptom_efficacy_herb_graph, symptom_efficacy_herb_to_id, symptom_efficacy_herb_new_id_to_name = reassign_ids_symptom_efficacy_herb_graph(symptom_efficacy_herb_graph, entity_to_id)

    save_graph(symptom_efficacy_herb_graph, "models/symptom_efficacy_herb_graph.pt")

    symptom_efficacy_herb_data = create_data_graph(symptom_efficacy_herb_graph, symptom_efficacy_herb_to_id)

    train_gat(model, symptom_efficacy_herb_data, optimizer4, scheduler=scheduler4, epochs=100, device=device, patience=10)

    with torch.no_grad():
        symptom_efficacy_herb_features = model(symptom_efficacy_herb_data)
        print("symptom_efficacy_herb_features", symptom_efficacy_herb_features)

    # Save heterogeneous graph features to file
    save_features(symptom_efficacy_herb_features, symptom_efficacy_herb_to_id, "models/symptom_efficacy_herb_features.pt")

if __name__ == "__main__":
    main()
