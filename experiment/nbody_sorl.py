# SoRL ver.3
# --------------------------------
from model import GATConfig, GAT
from dataset.nbody_dataset import NBodyDataset

from dataclasses import asdict
from search import SORLConfig 
import wandb


# Configuration
gat_config = GATConfig(K=3, L=2, n_embd=128, n_head=4, n_layer=4, device="cpu", _compile=False,
                       vocab_size_list=[17, 8])
        
# (TBD). we'd need a t_search curriculum gadget

config = SORLConfig(gat_config=gat_config,
           n_generations=2, 
           temperature=1.0, 
           num_iterations=10, 
           joint_steps=20, 
           max_length=2048, 
           learning_rate=1e-3,
           t_search=None,
           dataset_name="2body_2k", 
           dataset_path="dataset/nbody/2body_2k.bin")

# initialize model
gat = GAT(gat_config)

# load dataset
dataset = NBodyDataset.from_file(config.dataset_path)

# wandb init 
wandb.init(
    project="abstraction-learning", 
    name=f"SoRL-GRPO-per-token-reward-nbody",
    config=asdict(config)
)

import copy 
import wandb
import torch
from search import generate_rollout_data, compute_grpo_loss, compute_gspo_loss, compute_ssl_loss, get_batch, eval_search_improvement, compute_emb_mbe, eval_entropy_ppl_deviation

from search import sorl_search, compute_abs_ssl_loss, compute_ssl_loss, get_batch, observe_abstraction


n = config.n_generations # number of generations per sample
temperature = config.temperature 
num_iterations = config.num_iterations 
joint_steps = config.joint_steps
max_length = config.max_length
t_search = config.t_search

global_step = 0 


for iteration in range(num_iterations): 

    optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3)
    gat.train() 

    batch_data = get_batch(dataset.sequences, dataset.lengths, max_length, gat.L, gat.K)

    with torch.no_grad(): 
        repeat_batch = sorl_search(gat, batch_data, n, temperature, t_search)

        info_str = observe_abstraction(repeat_batch, gat, t_search)
        print(info_str)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    for joint_iter in range(joint_steps): 

        ppt = gat(repeat_batch)

        ssl_loss = compute_ssl_loss(repeat_batch, ppt)
        abs_loss = compute_abs_ssl_loss(repeat_batch, ppt, level=1)

        loss = abs_loss + ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clear cache after each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        improve_ppl_percentage = eval_search_improvement(gat, batch_data)

        wandb.log({
            "train/loss": loss.item(), 
            "train/ssl_loss": ssl_loss.item(), 
            "train/abs_loss": abs_loss.item(),

            "train/improve_ppl_percentage": improve_ppl_percentage.item(), 
            "progress/iteration": iteration, 
            "progress/joint_iter": joint_iter, 
        }, step=global_step)

        global_step += 1

        del loss, abs_loss, ssl_loss, ppt

ckpt_path=f"experiment/nbody/{wandb.run.name}.pt"
wandb.run.config["ckpt_path"] = ckpt_path
gat.save_checkpoint(ckpt_path)