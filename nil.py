# NIL algorithm
# Within each generation, we inherit prev generation's abstraction, and continue training 
# Reset -> Weak-supervision -> Train -> Record 
# ------------------------------------------------------------------------------------------------
from dataset.base import BaseHierDataset
from search import get_hier_batch_with_index, pad_abstract_tokens, get_hier_batch, compute_ssl_loss, compute_abs_ssl_loss, evaluate_gat
from search import compute_curriculum_t_increment, get_batch, eval_search_improvement
from sorl import sorl_search, pad_abstract_tokens

from dataset.base import BaseDataset
from search import SORLConfig
from model import GAT
import torch
import wandb

def annotate_abstraction(record_dataset: BaseHierDataset, gat: GAT, context_length: int = 1024, temperature: float = 0.0): 

    next_idx = 0
    total_seq = len(record_dataset.sequences)

    while next_idx < total_seq: 
        start_idx = next_idx

        # (1). Loop through each sequence (require modified 'get_batch' that returns end index, and start from argumented index)
        batch_data = get_hier_batch_with_index(record_dataset.sequences, record_dataset.lengths, context_length, gat.L, gat.K, 
                                            start_idx=start_idx, device=gat.device)
        assert (batch_data.levels == 1).sum() == 0, "HierSeq to be annotated should not contain any abstract tokens!"
        next_idx = batch_data.indices[-1] + 1 


        # (2). GAT inference
        pad_abstract_tokens(batch_data)
        batch_data = gat.generate(batch_data, parallel=True, temperature=temperature)

        # (3). Insert abstract tokens back into HierDataset
        annotated_hier_seqs, _ = batch_data.to_hierarchical_data() # timestamps is not used without sparsity for now
        record_dataset.sequences[start_idx:next_idx] = annotated_hier_seqs

    return record_dataset

# Weak-supervision involves direct supervision on abstraction-labeled trajectory dataset
# ------------------------------------------------------------------------------------------------
def supervise_gat(record_dataset: BaseHierDataset, gat: GAT, num_iterations: int, context_length: int, start_step: int = 0, wandb_log_prefix: str = None): 

    optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3)
    gat.train() 

    for i in range(num_iterations):
        global_step = start_step + i
        batch_data = get_hier_batch(record_dataset.sequences, record_dataset.lengths, context_length, gat.L, gat.K, device=gat.device)

        ppt = gat(batch_data)

        ssl_loss = compute_ssl_loss(batch_data, ppt)
        abs_loss = compute_abs_ssl_loss(batch_data, ppt, level=1)

        loss = abs_loss + ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if wandb_log_prefix:
            wandb.log({
                f"{wandb_log_prefix}/loss": loss.item(),
                f"{wandb_log_prefix}/abs_loss": abs_loss.item(),
                f"{wandb_log_prefix}/ssl_loss": ssl_loss.item(),
            }, step=global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Iteration {i+1}/{num_iterations}, loss: {loss.item():.4f}, abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}")

        del loss, abs_loss, ssl_loss

    return gat, start_step + num_iterations

# Self organizing reinforcement learning (SoRL)
# ------------------------------------------------------------------------------------------------
def sorl_gat(dataset: BaseDataset, id_val_dataset: BaseDataset, ood_val_dataset: BaseDataset, gat: GAT, config: SORLConfig, start_step: int = 0, wandb_log_prefix: str = None): 
    
    if config.t_curriculum: 
        t_search = 0
        t_delta, t_max = compute_curriculum_t_increment(num_iterations=config.num_iterations, context_length=config.context_length, K=gat.K, max_ts=max(dataset.lengths))
    else: 
        t_search, t_delta, t_max = config.context_length, 0, 0
    
    optimizer = torch.optim.Adam(gat.parameters(), lr=config.learning_rate)

    for i in range(config.num_iterations):
        global_step = start_step + i
        gat.train() 

        batch_data = get_batch(dataset.sequences, dataset.lengths, config.context_length // config.n_generations, gat.L, gat.K, device=gat.device)

        with torch.no_grad(): 
    
            repeat_batch, switch_ratio, rollout_advantages = sorl_search(gat, batch_data, config.n_generations, config.temperature, t_search)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        

        ppt = gat(repeat_batch)

        ssl_loss = compute_ssl_loss(repeat_batch, ppt)
        abs_loss = compute_abs_ssl_loss(repeat_batch, ppt, level=1)

        loss = abs_loss + ssl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if global_step % config.log_interval == 0 and global_step > 0 and wandb_log_prefix:

            # Validation needs to be more rigorous : more samples
            gat.eval()

            with torch.no_grad(): 
                improve_ppl_train = eval_search_improvement(gat, batch_data, t_search=t_search)
                improve_ppl_val, traj_ppl_val, info_str = evaluate_gat(gat, id_val_dataset, config, t_search, t_max)

            wandb.log({f"{wandb_log_prefix}/val/info_str": wandb.Table(columns=["info"], data=[[info_str]])}, step=global_step)
            
            wandb.log({
                f"{wandb_log_prefix}/train/loss": loss.item(), 
                f"{wandb_log_prefix}/train/ssl_loss": ssl_loss.item(), 
                f"{wandb_log_prefix}/train/abs_loss": abs_loss.item(),

                f"{wandb_log_prefix}/train/improve_ppl_percentage": improve_ppl_train.item(), 
                f"{wandb_log_prefix}/train/abstraction_switch_ratio": switch_ratio, # how often greedy sampled abstraction is rejected for other abstraction
                f"{wandb_log_prefix}/train/mean_rollout_advantages": rollout_advantages.mean().item(), # average advantage over greedy choice
                f"{wandb_log_prefix}/train/max_rollout_advantages": rollout_advantages.max().item(), # max advantage over greedy choice
                f"{wandb_log_prefix}/val(in-domain)/improve_ppl_percentage": improve_ppl_val.item(), 

                f"{wandb_log_prefix}/progress/iteration": global_step, 
                f"{wandb_log_prefix}/progress/t_search": t_search,
            }, step=global_step)

            del val_data
            gat.train()

            if traj_ppl_val > 0.0: 
                wandb.log({f"{wandb_log_prefix}/val(in-domain)/traj_ppl": traj_ppl_val.mean().item()}, step=global_step)
   
        print(f"Iteration {i+1}/{config.num_iterations} "
                    f"- loss: {loss.item():.4f}, abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}, t_search: {t_search}")
        
        t_search = min(t_search + t_delta, t_max)

        del loss, abs_loss, ssl_loss, ppt

    return gat, start_step + config.num_iterations