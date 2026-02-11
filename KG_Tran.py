# -*- coding: utf-8 -*-

import numpy as np
import torch
import logging
from pykeen.evaluation import RankBasedEvaluator
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransD

# Configure logging
logging.basicConfig(
    filename='training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Read triples file (triples.txt)
triples = []
with open('triples_TCM_Lung.txt', 'r', encoding='utf-8') as file:
    for line in file:
        subject, predicate, obj = line.strip().split('\t')
        triples.append((subject, predicate, obj))

# Log number of triples
logger.info(f"Number of triples: {len(triples)}")

# Convert list to NumPy array
triples = np.array(triples)

# Create TriplesFactory using from_labeled_triples
triples_factory = TriplesFactory.from_labeled_triples(triples=triples)

# Split data into training and testing sets
train_factory, test_factory = triples_factory.split([0.8, 0.2], random_state=42)

# Run embedding model training pipeline
result = pipeline(
    training=triples_factory,
    testing=test_factory,
    model=TransD,
    optimizer='Adam',
    optimizer_kwargs=dict(
        lr=0.0001,  # set learning rate
    ),
    model_kwargs=dict(
        embedding_dim=128
    ),
    training_kwargs=dict(
        num_epochs=100,  # number of training epochs
        batch_size=128,  # batch size
    ),
    training_loop='slcwa',  # use SLCWA training loop
    negative_sampler='basic',
    random_seed=42,
)

# Log training completion
logger.info("Model training completed.")

# Get embedding results
entity_embeddings = result.model.entity_representations
relation_embeddings = result.model.relation_representations

# Save trained model
model_save_path = 'trained_model_TransD_Chinese_PR'
result.save_to_directory(model_save_path)
logger.info(f"Model saved to {model_save_path}")

# Save embeddings to files
torch.save(entity_embeddings, f'{model_save_path}/entity_embeddings.pkl')
torch.save(relation_embeddings, f'{model_save_path}/relation_embeddings.pkl')
torch.save(result.model, f'{model_save_path}/trained_model_full.pth')

# Evaluation function
def evaluate_embeddings(triples_factory, entity_embeddings_path, relation_embeddings_path):
    # Load embeddings
    entity_embeddings = torch.load(entity_embeddings_path)
    relation_embeddings = torch.load(relation_embeddings_path)

    # Create evaluator
    evaluator = RankBasedEvaluator()

    # Evaluate embeddings
    results = evaluator.evaluate(
        model=result.model,  # use the trained model
        mapped_triples=triples_factory.mapped_triples,
        batch_size=128,  # batch size
        device='gpu',
    )

    # Output evaluation metrics to log
    print("Evaluation metrics:")
    print(f"Hits@1: {results.get_metric('hits_at_1')}")
    print(f"Hits@3: {results.get_metric('hits_at_3')}")
    print(f"Hits@10: {results.get_metric('hits_at_10')}")
    print(f"Mean Rank (MR): {results.get_metric('mean_rank')}")
    print(f"Mean Reciprocal Rank (MRR): {results.get_metric('mean_reciprocal_rank')}")
    logger.info("Evaluation metrics:")
    logger.info(f"Hits@1: {results.get_metric('hits_at_1')}")
    logger.info(f"Hits@3: {results.get_metric('hits_at_3')}")
    logger.info(f"Hits@10: {results.get_metric('hits_at_10')}")
    logger.info(f"Mean Rank (MR): {results.get_metric('mean_rank')}")
    logger.info(f"Mean Reciprocal Rank (MRR): {results.get_metric('mean_reciprocal_rank')}")

# Call evaluation function
evaluate_embeddings(
    triples_factory=triples_factory,
    entity_embeddings_path=f'{model_save_path}/entity_embeddings.pkl',
    relation_embeddings_path=f'{model_save_path}/relation_embeddings.pkl'
)

# Log evaluation completion
logger.info("Evaluation completed and logged.")