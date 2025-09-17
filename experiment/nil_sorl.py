# SoRL with NIL generations
# --------------------------------

from model import GATConfig, GAT
from dataset.arithmetic import ArithmeticDataset, ArithmeticHierDataset
from nil import annotate_abstraction, supervise_gat, sorl_gat
from dataclasses import asdict
from search import SORLConfig 
import wandb


# Generative Abstraction Transformer (GAT)
# ------------------------------------------------------------------------------------------------
gat_config = GATConfig(K=3, L=2, n_embd=128, n_head=4, n_layer=4, device="cpu", _compile=False,
                       vocab_size_list=[17, 8])
        

# SORL & NIL configs
# ------------------------------------------------------------------------------------------------
config = SORLConfig(gat_config=gat_config,
           n_generations=3, 
           temperature=1.0, 
           num_iterations=2000, 
           joint_steps=1, 
           context_length=1024, 
           learning_rate=1e-3,
           t_curriculum=True,
           log_interval=100,
           use_v2=True,
           nil_weak_iterations=20,
           nil_num_generations=4,

           dataset_name="100K-123", 
           dataset_path="dataset/multiplication/100K-123.bin",
           id_validate_dataset_name="2k-123",
           id_validate_dataset_path="dataset/multiplication/2k-123.bin",
           ood_validate_dataset_name="2k-123",
           ood_validate_dataset_path="dataset/multiplication/2k-123.bin")


# Load (Non-hierarchical) dataset
dataset = ArithmeticDataset.from_file(config.dataset_path)
id_val_dataset = ArithmeticDataset.from_file(config.id_validate_dataset_path)
ood_val_dataset = ArithmeticDataset.from_file(config.ood_validate_dataset_path)

# Wandb init 
# ------------------------------------------------------------------------------------------------
wandb.init(
    project="abstraction-learning2", 
    name=f"SoRL-with-NIL-val10-weak50",
    config=asdict(config)
)

# ------------------------------------------------------------------------------------------------
# Generation Bottleneck Experiment set-ups
# ------------------------------------------------------------------------------------------------

total_step = 0
for gen in range(config.nil_num_generations):
    gen_prefix = f"Generation {gen+1}"

    # (1). Reset / Initialize GAT
    gat = GAT(gat_config)

    # (2). Weak-supervision (if previous generation exists) 
    if gen > 0: 
        gat, total_step = supervise_gat(record_dataset, gat, config.nil_weak_iterations, config.context_length, total_step, wandb_log_prefix=gen_prefix+".weak")
    
    # (3). Train with SoRL 
    gat, total_step = sorl_gat(dataset, id_val_dataset, ood_val_dataset, gat, config, total_step, wandb_log_prefix=gen_prefix+".sorl")

    # (4). Record abstraction 
    record_dataset = ArithmeticHierDataset.from_dataset(dataset)
    record_dataset = annotate_abstraction(record_dataset, gat, config.context_length, config.temperature)