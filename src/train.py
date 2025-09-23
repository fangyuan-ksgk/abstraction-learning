# SoRL & NIL training pipeline
# -------------------------------------------------------------
# (1). data loading (get_batch)
# (2). curriculum on t_search, t_keep
# (3). evaluate search improvement / ppl evaluation gadget
# -------------------------------------------------------------


import wandb, torch
from dataset.base import BaseDataset
from src.gat import GAT
from src.sorl import SORLConfig, sorl_search, compute_loss


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

        level_loss = compute_loss(repeat_batch, gat, ppt)
        ssl_loss, abs_loss = level_loss[0], level_loss[config.l]
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