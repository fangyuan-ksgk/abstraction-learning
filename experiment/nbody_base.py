# Base Script for Transformer (SSl)

from model import GATConfig, GAT
from dataset.nbody_dataset import NBodyDataset

import torch
import wandb

from dataclasses import asdict


# Configuration
gat_config = GATConfig(K=3, 
                       L=1, 
                       n_embd=128, 
                       n_head=4, 
                       n_layer=4, 
                       device="cpu", 
                       _compile=False)

gat = GAT(gat_config)

ssl_config = {
    "gat_config": asdict(gat_config),
    "dataset_path": "dataset/nbody/2body_2k.bin",
    "num_iterations": 2000, 
    "temperature": 1.0, 
    "max_length": 1024,
    "learning_rate": 1e-3
}

dataset = NBodyDataset.from_file(ssl_config["dataset_path"])
dataset.update_abstract_params(L=gat_config.L, K=gat_config.K)

# wandb init 
wandb.init(
    project="abstraction-learning", 
    name=f"GAT-nbody-base",
    config=ssl_config
)

from search import compute_ssl_loss, get_batch


optimizer = torch.optim.Adam(gat.parameters(), lr=ssl_config["learning_rate"])

for iteration in range(ssl_config["num_iterations"]): 

    print(f"\nIteration {iteration+1}/{ssl_config['num_iterations']}")

    gat.train() 

    batch_data = get_batch(dataset.sequences, dataset.lengths, ssl_config["max_length"], gat.L, gat.K)

    ppt = gat(batch_data)
    ssl_loss = compute_ssl_loss(batch_data, ppt)

    optimizer.zero_grad()
    ssl_loss.backward()
    optimizer.step()

    print(f"Iteration {iteration+1}/{ssl_config['num_iterations']}, ssl_loss: {ssl_loss.item():.4f}")

    wandb.log({"train/ssl_loss": ssl_loss.item()}, step=iteration)







