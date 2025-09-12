# SoRL ver.3
# --------------------------------
from model import GATConfig, GAT
# from dataset.nbody import NBodyDataset
from dataset.arithmetic import ArithmeticDataset

from dataclasses import asdict
from search import SORLConfig 
import wandb


# Configuration
gat_config = GATConfig(K=3, L=2, n_embd=128, n_head=4, n_layer=4, device="cpu", _compile=False,
                       vocab_size_list=[17, 8])
        
# (TBD). we'd need a t_search curriculum gadget

config = SORLConfig(gat_config=gat_config,
           n_generations=3, 
           temperature=1.0, 
           num_iterations=2000, 
           joint_steps=10, 
           context_length=1024, 
           learning_rate=1e-3,
           t_curriculum=True,
           log_interval=11,
           use_v2=True,
           switch_abs_ppl_threshold=0.02,

           dataset_name="100K-123", 
           dataset_path="dataset/multiplication/100K-123.bin",
           id_validate_dataset_name="2k-123",
           id_validate_dataset_path="dataset/multiplication/2k-123.bin",
           ood_validate_dataset_name="2k-123",
           ood_validate_dataset_path="dataset/multiplication/2k-123.bin")


# initialize model
gat = GAT(gat_config)

# load dataset
dataset = ArithmeticDataset.from_file(config.dataset_path)
id_val_dataset = ArithmeticDataset.from_file(config.id_validate_dataset_path)
ood_val_dataset = ArithmeticDataset.from_file(config.ood_validate_dataset_path)

# wandb init 
wandb.init(
    project="abstraction-learning2", 
    name=f"SoRL-v3-bidirecional-anneal-100-0.1-unidirectional",
    # name="Transformer-baseline",
    config=asdict(config)
)

import copy 
import wandb
import torch
from search import compute_curriculum_t_increment, compute_abs_ssl_loss, compute_ssl_loss, get_batch, observe_abstraction, eval_search_improvement, sorl_search, sorl_search_v2, curriculum_iter
from search import eval_ppl_with_search, eval_generate_ppl
from search import PhaseScheduler

n = config.n_generations # number of generations per sample
temperature = config.temperature 
num_iterations = config.num_iterations 
context_length = config.context_length
scheduler = PhaseScheduler(init_joint_steps=config.joint_steps, 
                           init_abs_switch_ppl_threshold=config.switch_abs_ppl_threshold)

if config.t_curriculum: 
    t_search = 0
    t_delta, t_max = compute_curriculum_t_increment(num_iterations=num_iterations, context_length=context_length, K=gat.K, max_ts=max(dataset.lengths), num_loops=config.num_loops)
else: 
    t_search, t_delta, t_max = context_length, 0, 0

global_step = 0 

while global_step < num_iterations: 

    optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3)
    gat.train() 

    # include curriculum loops
    t_search = min(t_search + t_delta, t_max)
    # t_search = curriculum_iter(t_search, t_delta, t_max)

    batch_data = get_batch(dataset.sequences, dataset.lengths, context_length // n, gat.L, gat.K)

    with torch.no_grad(): 
        repeat_batch, switch_ratio = sorl_search_v2(gat, batch_data, n, temperature, t_search, switch_abs_ppl_threshold=scheduler.abs_switch_ppl_threshold)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    for joint_iter in range(scheduler.joint_steps): 

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

        if global_step % config.log_interval == 0:

            val_data = get_batch(id_val_dataset.sequences, id_val_dataset.lengths, context_length, gat.L, gat.K)
            
            with torch.no_grad(): 
                improve_ppl_train = eval_search_improvement(gat, batch_data, t_search=t_search)
                print(f"\nImprove ppl percentage (train): {improve_ppl_train:.4f}")

                improve_ppl_val = eval_search_improvement(gat, val_data, t_search=t_search)
                print(f"\nImprove ppl percentage (val): {improve_ppl_val:.4f}")
            
                if t_search == t_max and config.t_curriculum: 
                    traj_ppl_val = eval_ppl_with_search(val_data, gat, dataset.answer_token_id, n=6, temperature=1.0)
                    print(f"Traj ppl (val): {traj_ppl_val.mean().item():.4f}\n")
                elif not config.t_curriculum: 
                    traj_ppl_val = eval_generate_ppl(gat, val_data, n=1, temperature=0.0, t_search=t_search).mean()
                    print(f"Traj ppl (val): {traj_ppl_val.item():.4f}\n")
                else: 
                    traj_ppl_val = torch.tensor([0.0])
 
                info_str = observe_abstraction(val_data, gat, t_search)
                print(info_str)
                wandb.log({"val/info_str": wandb.Table(columns=["info"], data=[[info_str]])}, step=global_step)

            
            wandb.log({
                "train/loss": loss.item(), 
                "train/ssl_loss": ssl_loss.item(), 
                "train/abs_loss": abs_loss.item(),

                "train/improve_ppl_percentage": improve_ppl_train.item(), 
                "train/abstraction_switch_ratio": switch_ratio, # how often greedy sampled abstraction is rejected for other abstraction
                "val(in-domain)/improve_ppl_percentage": improve_ppl_val.item(), 
                "val(in-domain)/traj_ppl": traj_ppl_val.mean().item(), 

                "progress/iteration": global_step, 
                "progress/joint_iter": joint_iter, 
                "progress/t_search": t_search,
                "progress/joint_steps": scheduler.joint_steps,
                "progress/abs_switch_ppl_threshold": scheduler.abs_switch_ppl_threshold,
            }, step=global_step)

        print(f"Iteration {global_step+1}/{num_iterations} "
                    f"- inner step {joint_iter+1}/{scheduler.joint_steps}, loss: {loss.item():.4f}, abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}, t_search: {t_search}")
        

        global_step += 1

        del loss, abs_loss, ssl_loss, ppt
   

ckpt_path=f"experiment/nbody/{wandb.run.name}.pt"
wandb.run.config["ckpt_path"] = ckpt_path
gat.save_checkpoint(ckpt_path)