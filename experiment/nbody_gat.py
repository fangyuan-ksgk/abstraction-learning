from model import GATConfig, GAT
from dataset.nbody_dataset import NBodyDataset

from dataclasses import asdict
from search import SORLConfig 
import wandb

# Configuration
gat_config = GATConfig(K=3, 
                       L=2, 
                       n_embd=128, 
                       n_head=4, 
                       n_layer=4, 
                       device="cpu", 
                       _compile=False)

config = SORLConfig(gat_config=gat_config,
           n_generations=4, 
           temperature=1.0, 
           num_iterations=20, 
           num_steps=10, 
           grpo_steps=10, 
           max_length=1024, 
           learning_rate=1e-3,
           dataset_name="2body_2k", 
           dataset_path="dataset/nbody/2body_2k.bin")

# initialize model
gat = GAT(gat_config)

# load dataset
dataset = NBodyDataset.from_file(config.dataset_path)

# wandb init 
wandb.init(
    project="abstraction-learning", 
    name=f"GAT-nbody",
    config=asdict(config)
)


import copy 
import wandb
import torch
from search import generate_rollout_data, compute_grpo_loss, compute_ssl_loss, get_batch

n = config.n_generations # number of generations per sample
temperature = config.temperature 
num_iterations = config.num_iterations 
num_steps = config.num_steps
grpo_steps = config.grpo_steps
max_length = config.max_length

global_step = 0 

for iteration in range(num_iterations): 
    print(f"\nIteration {iteration+1}/{num_iterations}")

    ref_model = copy.deepcopy(gat)
    ref_model.eval() 
    for param in ref_model.parameters(): 
        param.requires_grad = False
    print("Reference model created")

    optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3)
    gat.train() 

    for step in range(num_steps): 
        print(f"\nStep {step+1}/{num_steps}")
        batch_data = get_batch(dataset.sequences, dataset.lengths, max_length, gat.L, gat.K)

        with torch.no_grad(): 
            repeat_batch, old_log_probs, ref_log_probs = generate_rollout_data(gat, ref_model, 
                                                                                batch_data, n, temperature)
            print("\n\n------------------------ \n Example abstraction: \n", repeat_batch.tokens[repeat_batch.levels > 0][:15])
            # Clear cache after generating rollouts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
       
        for grpo_iter in range(grpo_steps): 
            print(f"GRPO inner loop {grpo_iter+1}/{grpo_steps}")
            ppt = gat(repeat_batch)
            grpo_loss = compute_grpo_loss(repeat_batch, ppt, old_log_probs, ref_log_probs)
            ssl_loss = compute_ssl_loss(repeat_batch, ppt)

            loss = grpo_loss + ssl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clear cache after each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Iteration {iteration+1}/{num_iterations}, Step {step+1}/{num_steps}, "
                      f"GRPO iter {grpo_iter+1}/{grpo_steps}, loss: {loss.item():.4f}, grpo_loss: {grpo_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}")

            wandb.log({
                "train/loss": loss.item(), 
                "train/grpo_loss": grpo_loss.item(), 
                "train/ssl_loss": ssl_loss.item(),

                "progress/iteration": iteration, 
                "progress/step": step, 
                "progress/grpo_iter": grpo_iter, 
            }, step=global_step)

            global_step += 1
            del loss









