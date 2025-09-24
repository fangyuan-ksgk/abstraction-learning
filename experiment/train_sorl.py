from src.gat import GATConfig, GAT
from src.sorl import SORLConfig
from src.train import self_organizing_reinforcement_learning
from dataset.arithmetic import ArithmeticDataset
import argparse

from dataclasses import asdict
import wandb
import torch

# SoRL config 
sorl_config = SORLConfig(

    n = 3,
    temperature = 0.75,   
    # rollout specific 
    causal_rollout=False, 
    
    l=1,
    steps=1,
    use_rhythmic_placeholders=True,
    use_spike_placeholders=True,
    abstract_budget=5,
    max_t_search=5,

    # curriculum progress rate
    curriculum_ratio=0.6,

    # memory fading
    max_seq_len = 17,
    use_fade_memory=True,
    min_keep=6,

    # dataset specific
    train_dataset_path="dataset/multiplication/100K-123.bin",
    val_dataset_path="dataset/multiplication/2K-123.bin",
    train_batch_size=16,
    val_batch_size=16,
    train_iterations=2000,
    val_iterations=100,

    # optimization
    learning_rate=1e-3, 
    log_interval=100
)

train_dataset = ArithmeticDataset.from_file(sorl_config.train_dataset_path)
val_dataset = ArithmeticDataset.from_file(sorl_config.val_dataset_path)
traj_vocab_size = train_dataset.vocab_size_list[0]

gat_config = GATConfig(K=3, L=2, n_embd=128, n_head=4, n_layer=4, 
                       device="cuda" if torch.cuda.is_available() else "cpu", 
                       _compile=False, # not compatible with search curriculum / varying attention pattern
                       vocab_size_list=[traj_vocab_size, 8], memory_span=sorl_config.max_seq_len)

gat = GAT(gat_config)

gat.to(gat.device)

if torch.cuda.is_available(): 
   gat = torch.compile(gat, dynamic=True)


if __name__ == "__main__":

    # add-in argument parser to modify sorl_config
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--use_spike_placeholders", type=bool, default=False)
    parser.add_argument("--use_rhythmic_placeholders", type=bool, default=True)
    parser.add_argument("--abstract_budget", type=int, default=5)
    parser.add_argument("--max_t_search", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--causal_rollout", type=bool, default=False)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_iterations", type=int, default=2000)
    parser.add_argument("--val_iterations", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--curriculum_ratio", type=float, default=0.6)
    parser.add_argument("--use_fade_memory", type=bool, default=True)
    parser.add_argument("--min_keep", type=int, default=6)
    parser.add_argument("--max_seq_len", type=int, default=17)
    parser.add_argument("--train_dataset_path", type=str, default="dataset/multiplication/100K-123.bin")
    parser.add_argument("--val_dataset_path", type=str, default="dataset/multiplication/2K-123.bin")
    parser.add_argument("--run_name", type=str, default="SoRL-v4-heuristic-t-curriculum")
    args = parser.parse_args()
    
    from dataclasses import asdict, fields

    sorl_fields = {f.name for f in fields(SORLConfig)}
    sorl_args = {k: v for k, v in vars(args).items() if k in sorl_fields}
    sorl_config = SORLConfig(**sorl_args)

    wandb.init(
        project="abstraction-learning2", 
        name=args.run_name,
        config={**asdict(sorl_config), **asdict(gat_config)}
    )

    total_step = 0
    gat, total_step = self_organizing_reinforcement_learning(train_dataset, 
                                                            val_dataset, 
                                                            gat, 
                                                            sorl_config, 
                                                            total_step)