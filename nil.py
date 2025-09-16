# NIL algorithm
# Within each generation, we inherit prev generation's abstraction, and continue training 
# Reset -> Weak-supervision -> Train -> Record 
# ------------------------------------------------------------------------------------------------
from dataset.base import BaseHierDataset
from search import get_hier_batch_with_index, pad_abstract_tokens, get_hier_batch, compute_ssl_loss, compute_abs_ssl_loss
from search import compute_curriculum_t_increment, sorl_search_v2, get_batch, eval_search_improvement, eval_ppl_with_search, eval_generate_ppl, observe_abstraction
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
                                            start_idx=start_idx)
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
def supervise_gat(record_dataset: BaseHierDataset, gat: GAT, num_iterations: int, context_length: int, wandb_log_prefix: str = None): 

    global_step = 0 

    optimizer = torch.optim.Adam(gat.parameters(), lr=1e-3)
    gat.train() 

    while global_step < num_iterations: 

        batch_data = get_hier_batch(record_dataset.sequences, record_dataset.lengths, context_length, gat.L, gat.K)

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
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Iteration {global_step+1}/{num_iterations}, loss: {loss.item():.4f}, abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}")

        global_step += 1
        del loss, abs_loss, ssl_loss

    return gat

# Self organizing reinforcement learning (SoRL)
# ------------------------------------------------------------------------------------------------
def sorl_gat(dataset: BaseDataset, id_val_dataset: BaseDataset, ood_val_dataset: BaseDataset, gat: GAT, config: SORLConfig, wandb_log_prefix: str = None): 
    
    if config.t_curriculum: 
        t_search = 0
        t_delta, t_max = compute_curriculum_t_increment(num_iterations=config.num_iterations, context_length=config.context_length, K=gat.K, max_ts=max(dataset.lengths))
    else: 
        t_search, t_delta, t_max = config.context_length, 0, 0

    global_step = 0 
    optimizer = torch.optim.Adam(gat.parameters(), lr=config.learning_rate)

    while global_step < config.num_iterations: 

        gat.train() 

        batch_data = get_batch(dataset.sequences, dataset.lengths, config.context_length // config.n_generations, gat.L, gat.K)

        with torch.no_grad(): 
    
            repeat_batch, switch_ratio, rollout_advantages = sorl_search_v2(gat, batch_data, config.n_generations, config.temperature, t_search)
            
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

        if global_step % config.log_interval == 0 and wandb_log_prefix:

            val_data = get_batch(id_val_dataset.sequences, id_val_dataset.lengths, config.context_length, gat.L, gat.K)
            
            with torch.no_grad(): 
                improve_ppl_train = eval_search_improvement(gat, batch_data, t_search=t_search)

                improve_ppl_val = eval_search_improvement(gat, val_data, t_search=t_search)
            
                if t_search == t_max and config.t_curriculum: 
                    traj_ppl_val = eval_ppl_with_search(val_data, gat, dataset.answer_token_id, n=6, temperature=1.0)
                elif not config.t_curriculum: 
                    traj_ppl_val = eval_generate_ppl(gat, val_data, n=1, temperature=0.0, t_search=t_search).mean()
                else: 
                    traj_ppl_val = torch.tensor([0.0])

                info_str = observe_abstraction(val_data, gat, t_search)
                print(info_str)
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
                f"{wandb_log_prefix}/val(in-domain)/traj_ppl": traj_ppl_val.mean().item(), 

                f"{wandb_log_prefix}/progress/iteration": global_step, 
                f"{wandb_log_prefix}/progress/t_search": t_search,
                f"{wandb_log_prefix}/progress/abs_switch_ppl_threshold": config.switch_abs_ppl_threshold,
            }, step=global_step)

        print(f"Iteration {global_step+1}/{config.num_iterations} "
                    f"- loss: {loss.item():.4f}, abs_loss: {abs_loss.item():.4f}, ssl_loss: {ssl_loss.item():.4f}, t_search: {t_search}")
        
        global_step += 1

        t_search = min(t_search + t_delta, t_max)

        del loss, abs_loss, ssl_loss, ppt