import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from responses import target
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- 1. Data loading and preprocessing -------------------------
# Load entity and relation mappings
def load_mappings(entity_to_id_path, relation_to_id_path):
    """Load entity and relation mapping files and return mapping DataFrames."""
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
            # skip first header line
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

def data_processing(triples, entity_to_id, entity_embeddings, symptom_efficacy_features):
    """
    Process triples to build dictionaries for prescriptions, symptoms, efficacies, herbs,
    and produce formatted dataset entries.
    """
    prescription_dict = {}
    prescription_vector = {}
    symptom_dict = {}
    efficacy_dict = {}
    herb_dict = {}
    herb_index = 0  # start index for reindexed herbs
    original_to_new_herb_mapping = {}  # map original herb id to new index
    formatted_data = []  # store input features and targets

    for prescription, relation, obj in triples:
        if relation == '4':  # symptom
            if obj not in symptom_dict:
                if str(entity_to_id.index[entity_to_id['id'] == int(obj)][0]) in symptom_efficacy_features.keys():
                    symptom_dict_gat = symptom_efficacy_features[
                        str(entity_to_id.index[entity_to_id['id'] == int(obj)][0])].to(device)
                else:
                    symptom_dict_gat = torch.zeros(128)
                symptom_dict[obj] = torch.cat(
                    [symptom_dict_gat, entity_embeddings[0]._embeddings.weight.data[int(obj)].to(device)])
            prescription_dict.setdefault(prescription, {"symptoms": [], "efficacies": [], "herbs": []})[
                "symptoms"].append(obj)
        elif relation == '3':  # efficacy
            if obj not in efficacy_dict:
                if str(entity_to_id.index[entity_to_id['id'] == int(obj)][0]) in symptom_efficacy_features.keys():
                    efficacy_dict_gat = symptom_efficacy_features[
                        str(entity_to_id.index[entity_to_id['id'] == int(obj)][0])].to(device)
                else:
                    efficacy_dict_gat = torch.zeros(128)
                efficacy_dict[obj] = torch.cat(
                    [efficacy_dict_gat, entity_embeddings[0]._embeddings.weight.data[int(obj)].to(device)])
            prescription_dict.setdefault(prescription, {"symptoms": [], "efficacies": [], "herbs": []})[
                "efficacies"].append(obj)
        elif relation == '2':  # herb
            if obj not in herb_dict:
                if str(entity_to_id.index[entity_to_id['id'] == int(obj)][0]) in symptom_efficacy_features.keys():
                    herb_dict_gat = symptom_efficacy_features[str(entity_to_id.index[entity_to_id['id'] == int(obj)][0])].to(device)
                else:
                    herb_dict_gat = torch.zeros(128)
                herb_dict[obj] = torch.cat([herb_dict_gat, entity_embeddings[0]._embeddings.weight.data[int(obj)].to(device)])
                original_to_new_herb_mapping[obj] = herb_index  # save mapping from original id to new id
                herb_index += 1  # increment herb index
            prescription_dict.setdefault(prescription, {"symptoms": [], "efficacies": [], "herbs": []})["herbs"].append(obj)

    # Format into inputs and labels
    for prescription, features in prescription_dict.items():
        # aggregate symptoms and efficacies
        symptoms = torch.stack([symptom_dict[s] for s in features["symptoms"]]).mean(dim=0)
        efficacies = torch.stack([efficacy_dict[e] for e in features["efficacies"]]).mean(dim=0)
        input_vector = torch.cat([symptoms])

        # main task label: prescription embedding
        prescription_emb = entity_embeddings[0]._embeddings.weight.data[int(prescription)].to(device)
        prescription_emb_gat = symptom_efficacy_features[str(entity_to_id.index[entity_to_id['id'] == int(prescription)][0])].to(device)
        prescription_embs = torch.cat([prescription_emb, prescription_emb_gat])
        prescription_vector[prescription] = prescription_embs

        # auxiliary task 1 label: herb distribution
        # need to create a new index mapping for herbs
        herb_labels = torch.zeros(len(herb_dict), dtype=torch.long)  # initialize as zeros (long)
        for h in features["herbs"]:
            new_herb_idx = original_to_new_herb_mapping[h]  # get new herb index
            herb_labels[new_herb_idx] = 1  # mark using new herb index

        # auxiliary task 2 label: herb-prescription consistency
        herb_mean_embedding = torch.stack([herb_dict[h] for h in features["herbs"]]).mean(dim=0)

        # append formatted data
        formatted_data.append((input_vector, prescription_embs, herb_labels, herb_mean_embedding))

    return formatted_data, symptom_dict, efficacy_dict, prescription_dict, prescription_vector


# ------------------------- 2. Dataset definition -------------------------
class PrescriptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_vector, prescription_embedding, herb_labels, herb_mean_embedding = self.data[idx]
        return input_vector, prescription_embedding, herb_labels, herb_mean_embedding


# ------------------------- 3. Model definitions -------------------------

# Cosine similarity loss
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, outputs, labels):
        cos_sim = F.cosine_similarity(outputs, labels, dim=1)
        loss = 1 - cos_sim.mean()
        return loss

# Modified MLP model
class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prescription_output_dim, herb_output_dim, dropout_rate=0.3):
        super(MLPModel, self).__init__()

        # increased layers and width
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # adjusted residual connection
        self.fc_res = nn.Linear(hidden_dim, hidden_dim // 2)

        # output layers
        self.fc_prescription = nn.Linear(hidden_dim // 2, prescription_output_dim)
        self.fc_herb = nn.Linear(hidden_dim // 2, herb_output_dim)

        # activation and Dropout
        self.relu = nn.ReLU()  # changed activation
        self.dropout = nn.Dropout(dropout_rate)

        # weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # forward through first layer
        x1 = self.relu(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)

        # forward through second layer
        x2 = self.relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)

        # forward through third layer
        x3 = self.relu(self.bn3(self.fc3(x2)))
        x3 = self.dropout(x3)

        # residual connection
        x1_res = self.fc_res(x1)
        x_res = x1_res + x3

        # prescription prediction (aux task 1)
        prescription_output = self.fc_prescription(x_res)

        # herb prediction (main task)
        herb_output = self.fc_herb(x_res)

        return prescription_output, herb_output


# ------------------------- 4. Training and evaluation -------------------------

def train_and_evaluate(
    model, train_loader, val_loader, test_loader, optimizer, prescription_criterion, herb_criterion, TCM_path, epochs,
    top_k_values=range(1, 21), patience=5, scheduler_patience=5, Alpha=1.0, Beta=1.0, Gamma=1.0
):
    """
    Train and evaluate the model with multi-task losses, and compute Top-K Precision, Recall, and F1.
    """
    # store train and validation losses per epoch
    train_losses = []
    val_losses = []

    # early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, verbose=True)

    # training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, prescription_embedding, herb_labels, herb_mean_embedding in train_loader:
            features = features.to(device)
            prescription_embedding = prescription_embedding.to(device)
            herb_labels = herb_labels.to(device)
            herb_labels = herb_labels.float()
            herb_mean_embedding = herb_mean_embedding.to(device)

            optimizer.zero_grad()

            # forward pass
            prescription_output, herb_output = model(features)

            # compute task losses
            loss_herb = herb_criterion(herb_output, herb_labels)  # main task loss
            loss_prescription = prescription_criterion(prescription_output, prescription_embedding)  # aux task 1 loss
            loss_align = prescription_criterion(prescription_embedding, herb_mean_embedding)  # aux task 2 loss

            # total loss
            total_loss = Alpha * loss_herb + Beta * loss_prescription + Gamma * loss_align

            # backward pass
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, prescription_embedding, herb_labels, herb_mean_embedding in val_loader:
                features = features.to(device)
                prescription_embedding = prescription_embedding.to(device)
                herb_labels = herb_labels.to(device)
                herb_labels = herb_labels.float()
                herb_mean_embedding = herb_mean_embedding.to(device)

                prescription_output, herb_output = model(features)

                # compute task losses
                loss_herb = herb_criterion(herb_output, herb_labels)  # main task loss
                loss_prescription = prescription_criterion(prescription_output, prescription_embedding)  # aux task 1 loss
                loss_align = prescription_criterion(prescription_embedding, herb_mean_embedding)  # aux task 2 loss

                # total loss
                total_loss = Alpha * loss_herb + Beta * loss_prescription + Gamma * loss_align
                val_loss += total_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        scheduler.step(val_loss)

    # plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # evaluation on test set
    model.eval()

    test_predictions = []  # predicted herbs
    all_true_labels = []  # true herb label lists

    with torch.no_grad():
        for features, _, herb_labels, _ in test_loader:
            features = features.to(device)
            herb_labels = herb_labels.to(device)

            # collect true labels per sample
            batch_true_labels = []
            for i in range(herb_labels.size(0)):  # iterate over samples
                sample_indices = torch.nonzero(herb_labels[i] == 1).squeeze()
                batch_true_labels.append(sample_indices)

            # model herb predictions
            _, herb_output = model(features)

            # get predictions per sample
            batch_predictions = []
            for i in range(herb_labels.size(0)):  # iterate over samples
                # get top-k herb recommendations
                _, top_k_preds = torch.topk(torch.sigmoid(herb_output[i]), k=max(top_k_values), dim=-1)
                batch_predictions.append(top_k_preds)

            # append batch results
            all_true_labels.extend(batch_true_labels)
            test_predictions.extend(batch_predictions)


    # compute global Precision@K, Recall@K, F1@K
    results = []

    for k in top_k_values:
        total_predicted_size = 0  # sum of predicted set sizes
        total_intersection_size = 0  # sum of intersections between predictions and true labels
        total_relevant_size = 0  # sum of true label sizes

        for prediction, true_label in zip(test_predictions, all_true_labels):
            predicted_herbs = prediction[:k].tolist()  # take top k predicted herbs
            relevant_herbs = true_label.tolist()

            # ensure both are iterable lists
            if isinstance(predicted_herbs, int):  # if single int
                predicted_herbs = [predicted_herbs]
            if isinstance(relevant_herbs, int):
                relevant_herbs = [relevant_herbs]

            # compute |R(i) ∩ T(i)|
            intersection_size = len(set(predicted_herbs) & set(relevant_herbs))
            total_intersection_size += intersection_size

            # predicted set size |R(i)|
            total_predicted_size += k

            # true label set size |T(i)|
            total_relevant_size += len(relevant_herbs)

        # Precision@K
        precision_at_k = total_intersection_size / total_predicted_size if total_predicted_size != 0 else 0

        # Recall@K
        recall_at_k = total_intersection_size / total_relevant_size if total_relevant_size != 0 else 0

        # F1@K
        if precision_at_k + recall_at_k > 0:
            f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        else:
            f1_at_k = 0

        # print metrics for each K
        print(f"K = {k}, Precision@{k}: {precision_at_k:.4f}, "
              f"Recall@{k}: {recall_at_k:.4f}, "
              f"F1@{k}: {f1_at_k:.4f}")

        # record results
        results.append([k, precision_at_k, recall_at_k, f1_at_k])

    # save to Excel
    df = pd.DataFrame(results, columns=["K", "Precision", "Recall", "F1"])
    df.to_excel(f"result/{TCM_path}_evaluation_results.xlsx", index=False)

    print(f"Evaluation results saved to 'result/{TCM_path}_evaluation_results.xlsx'.")

    return results


def set_seed(seed=2025):
    """
    Set random seed to ensure experiment reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU seed
    torch.cuda.manual_seed_all(seed)  # GPU seed (if available)
    torch.backends.cudnn.deterministic = True  # ensure deterministic conv results
    torch.backends.cudnn.benchmark = False  # may cause instability in some cases

# ------------------------- Main entry -------------------------
def main_GMHR(lr= 5e-5, weight_decay=3e-3,dropout_rate=0.2):
    # set random seed
    set_seed(2025)

    # load data, model and other settings
    TCM_path = "../TCM_PR"
    entity_to_id, relation_to_id = load_mappings(
        TCM_path + '/trained_model_TransD_Chinese/training_triples/entity_to_id.tsv.gz',
        TCM_path + '/trained_model_TransD_Chinese/training_triples/relation_to_id.tsv.gz')
    entity_embeddings = load_entity_embeddings(TCM_path + '/trained_model_TransD_Chinese/entity_embeddings.pkl')
    symptom_efficacy_features = load_entity_embeddings(
        TCM_path + '/models/symptom_efficacy_herb_features.pt')
    triples = read_triples(TCM_path + '/trained_model_TransD_Chinese/training_triples/numeric_triples.tsv.gz')

    formatted_data, symptom_dict, efficacy_dict, prescription_dict, prescription_vector = data_processing(
        triples, entity_to_id, entity_embeddings, symptom_efficacy_features
    )

    train_size = int(0.7 * len(formatted_data))
    val_size = int(0.1 * len(formatted_data))
    test_size = len(formatted_data) - train_size - val_size
    train_data, val_data, test_data = random_split(formatted_data, [train_size, val_size, test_size])

    train_loader = DataLoader(PrescriptionDataset(train_data), batch_size=64, shuffle=True)
    val_loader = DataLoader(PrescriptionDataset(val_data), batch_size=64, shuffle=False)
    test_loader = DataLoader(PrescriptionDataset(test_data), batch_size=64, shuffle=False)

    input_dim = 256
    hidden_dim = 64
    prescription_output_dim = 256
    herb_output_dim = 1018
    model = MLPModel(input_dim, hidden_dim, prescription_output_dim, herb_output_dim, dropout_rate=dropout_rate).to(device)

    prescription_criterion = CosineSimilarityLoss()
    herb_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr= lr, weight_decay=weight_decay)

    results = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, prescription_criterion, herb_criterion, TCM_path=TCM_path, epochs=100, top_k_values=range(1, 21), patience=10, Alpha=0.5, Beta=0.4, Gamma=0.1)

    return results

if __name__ == "__main__":

    main_GMHR(lr= 7e-5, weight_decay=5e-3,dropout_rate=0.2)

