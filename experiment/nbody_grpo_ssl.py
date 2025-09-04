from model import GATConfig, GAT
from dataset.nbody_dataset import NBodyDataset

from dataclasses import asdict
from search import SORLConfig 
import wandb

# Story 
# ------------------------------------------------------------------------------------------------
# 1. SoRL v1. fails due to complication in buffer that creates too many bugs for me to fix. 
#    this script contains SoRL v2. that adopts GRPO / GSPO (RL loss) for abstraction searching
# 2. SoRL v2. fails due to the incompatibility of RL with abstract learning, RL assumes prior causal
#    structure and search which action works. However, no prior causal structure is assigned to abstraction
#    and searching less important than assigning semantics to abstract tokens. 
# 3. Specifically, when 2 abstraction has mediocre reward, RL views them as equal and select them with same prob, 
#    however, SoRL picks one and learns to improve its semantic to maximize reward. This is also a result of 
#    'internal reward' signal, instead of being external, the internal reward directly comes from model's own
#    perplexity, and is therefore a term that can be directly optimized. 
# ------------------------------------------------------------------------------------------------

# Configuration
gat_config = GATConfig(K=3, L=2, n_embd=128, n_head=4, n_layer=4, device="cpu", _compile=False,
                       vocab_size_list=[17, 8])
        

config = SORLConfig(gat_config=gat_config,
           gspo=False,
           n_generations=8, 
           temperature=1.0, 
           num_iterations=10, 
           num_steps=10, 
           joint_steps=20, # grpo & ssl are optimized jointly
           max_length=1024, 
           learning_rate=1e-3,
           epsilon=0.1,
           beta=0.1,
           per_token_reward=True,
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

n = config.n_generations # number of generations per sample
temperature = config.temperature 
num_iterations = config.num_iterations 
num_steps = config.num_steps
joint_steps = config.joint_steps
max_length = config.max_length
epsilon = config.epsilon
beta = config.beta

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
        # (Idea). at epoch 80, a new batch is used -- marking the start of policy collapse
        # - this reminds me of the curriculum-based sample selection mechanism -- we need to pick easier sample first (!)
        batch_data = get_batch(dataset.sequences, dataset.lengths, max_length, gat.L, gat.K)

        with torch.no_grad(): 
            repeat_batch, old_log_probs, ref_log_probs = generate_rollout_data(gat, ref_model, 
                                                                                batch_data, n, temperature, t_search=config.t_search)
            print("\n\n------------------------ \n Example abstraction: \n", repeat_batch.tokens[repeat_batch.levels > 0][:15])
            # Clear cache after generating rollouts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
       
        for joint_iter in range(joint_steps): 
            print(f"Joint inner loop {joint_iter+1}/{joint_steps}")
            ppt = gat(repeat_batch)
            if config.gspo:
                rl_loss = compute_gspo_loss(repeat_batch, ppt, old_log_probs, ref_log_probs, epsilon, beta)
            else:
                rl_loss = compute_grpo_loss(repeat_batch, ppt, old_log_probs, ref_log_probs, epsilon, beta, per_token_reward=config.per_token_reward)

            ssl_loss = compute_ssl_loss(repeat_batch, ppt)

            # Alternate gadget (1-1 SL & RL)
            if joint_iter % 2 == 0:
                rl_loss *= 0
                loss = ssl_loss
            else: 
                ssl_loss *= 0
                loss = rl_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clear cache after each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            improve_ppl_percentage = eval_search_improvement(gat, batch_data)
            mbe_wte, mbe_proj = compute_emb_mbe(gat, l=1)
            entropy_per_level, ppl_per_level, ppl_deviation_per_level = eval_entropy_ppl_deviation(gat, repeat_batch)

            
            if rl_loss == 0: 
                wandb.log({
                    "train/loss": loss.item(), 
                    "train/ssl_loss": ssl_loss.item(),
                    "train/improve_ppl_percentage": improve_ppl_percentage.item(),
                    "train/mbe_wte": mbe_wte.item(),
                    "train/mbe_proj": mbe_proj.item(),
                    "train/traj_entropy": entropy_per_level[0],
                    "train/abs_1_entropy": entropy_per_level[1],
                    "train/abs_1_reward_deviation": ppl_deviation_per_level[0],

                    "progress/iteration": iteration, 
                    "progress/step": step, 
                    "progress/joint_iter": joint_iter, 
                }, step=global_step)
            elif ssl_loss == 0: 
                wandb.log({
                    "train/loss": loss.item(), 
                    "train/rl_loss": rl_loss.item(), 
                    "train/improve_ppl_percentage": improve_ppl_percentage.item(),
                    "train/mbe_wte": mbe_wte.item(),
                    "train/mbe_proj": mbe_proj.item(),
                    "train/traj_entropy": entropy_per_level[0],
                    "train/abs_1_entropy": entropy_per_level[1],
                    "train/abs_1_reward_deviation": ppl_deviation_per_level[0],

                    "progress/iteration": iteration, 
                    "progress/step": step, 
                    "progress/joint_iter": joint_iter, 
                }, step=global_step)

            global_step += 1

            if rl_loss < 0.0 and joint_iter > 10: 
                continue

            del loss



ckpt_path=f"experiment/nbody/{wandb.run.name}.pt"
wandb.run.config["ckpt_path"] = ckpt_path
gat.save_checkpoint(ckpt_path)