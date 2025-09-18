from model import HierSeq
import torch
from model import GAT
from search import compute_grouped_mean


def sanity_check_same_abs_toks(batch_data: HierSeq): 
    per_sample_abs_ts = []
    for sample_idx in batch_data.indices: 
        sample_abs_ts = batch_data.timestamps[(batch_data.levels>0) & (batch_data.sample_idx == sample_idx)]
        per_sample_abs_ts.append(sample_abs_ts)

    # cross match
    for i in range(len(per_sample_abs_ts)): 
        for j in range(i+1, len(per_sample_abs_ts)): 
            if len(per_sample_abs_ts[i]) != len(per_sample_abs_ts[j]): 
                print(f"Sample {i} and {j} have different number of abstract tokens: {len(per_sample_abs_ts[i])} vs {len(per_sample_abs_ts[j])}")
                shorter_ts = per_sample_abs_ts[i] if len(per_sample_abs_ts[i]) < len(per_sample_abs_ts[j]) else per_sample_abs_ts[j]
                longer_ts = per_sample_abs_ts[j] if len(per_sample_abs_ts[i]) < len(per_sample_abs_ts[j]) else per_sample_abs_ts[i]
                shorter_sample = i if len(per_sample_abs_ts[i]) < len(per_sample_abs_ts[j]) else j
                longer_sample = j if len(per_sample_abs_ts[i]) < len(per_sample_abs_ts[j]) else i
                
                missing_tokens = []
                for ts in longer_ts:
                    if ts not in shorter_ts:
                        missing_tokens.append(ts)
                if missing_tokens:
                    print(f"Sample {shorter_sample} is missing tokens at timestamps: {missing_tokens[0].item()}")


# Useful for poking around, stop when encountered error on HierSeq 
def check_hierseq_complete(batch_data: HierSeq): 
    no_missing = True
    assert batch_data.indices.size(0) == batch_data.batch_size, f"Batch size {batch_data.batch_size} mismatch with indices number {batch_data.indices.size(0)}"
    for sample_idx in batch_data.indices: 
        sample_mask = batch_data.sample_idx == sample_idx

        for level in range(1, batch_data.L): 

            level_mask = (batch_data.levels == level)
            batch_data.timestamps[sample_mask & level_mask]

            et, st = batch_data.timestamps[sample_mask & level_mask][[-1, 0]]
            desired_level_ts = torch.arange(0, et, batch_data.K**level)[1:]
            desired_level_ts = desired_level_ts[desired_level_ts >= st]
            level_ts = batch_data.timestamps[sample_mask & level_mask]

            missing_level_ts = desired_level_ts[torch.isin(desired_level_ts, level_ts, invert=True)]
            if missing_level_ts.numel() > 0: 
                print(f" - sample {sample_idx} level {level} missing abstract token at timestamps: {missing_level_ts.tolist()}")
                no_missing = False
    assert no_missing, " - Missing abstract tokens found"


def check_repeat_hseq(repeat_batch: HierSeq): 
    for orig_idx in repeat_batch.idx_map.values(): 
        for idx1 in repeat_batch.indices: 
            if repeat_batch.idx_map[idx1.item()] != orig_idx: 
                continue
            for idx2 in repeat_batch.indices: 
                if repeat_batch.idx_map[idx2.item()] != orig_idx: 
                    continue
                if idx1 == idx2: 
                    continue
                assert (repeat_batch.tokens[repeat_batch.sample_idx == idx1] == repeat_batch.tokens[repeat_batch.sample_idx == idx2]).all(), f"Tokens at {idx1} and {idx2} are different"
                assert (repeat_batch.levels[repeat_batch.sample_idx == idx1] == repeat_batch.levels[repeat_batch.sample_idx == idx2]).all(), f"Levels at {idx1} and {idx2} are different"
                assert (repeat_batch.timestamps[repeat_batch.sample_idx == idx1] == repeat_batch.timestamps[repeat_batch.sample_idx == idx2]).all(), f"Timestamps at {idx1} and {idx2} are different"


def sanity_check_repeat_hseq(rep_hseq: HierSeq, gat: GAT): 
        
    with torch.no_grad(): 
        ppt_rep = gat(rep_hseq)

    mask = (rep_hseq.timestamps[1:] > 1)

    valid_ppt = ppt_rep[mask]
    valid_idx = rep_hseq.sample_idx[1:][mask]
    ppl_per_sample = compute_grouped_mean(valid_ppt, valid_idx)

    n_unique = torch.unique(rep_hseq.idx_map).size(0)
    n_repeat = len(ppl_per_sample) // n_unique

    assert torch.allclose(ppl_per_sample[:n_unique].repeat(n_repeat), ppl_per_sample, rtol=1e-5, atol=1e-6), "Repeated HierSeq has different ppl per sample"

    print("Sanity check passed for repeated HierSeq: all repeated samples have same avg. ppl")


def inspect_seq(hseq: HierSeq, dataset):
    traj_mask = hseq.levels == 0
    seq_id = hseq.tokens[traj_mask].tolist()
    seq = dataset.tokenizer.decode(seq_id)
    print(seq)


def answer_token_count_sanity_check(hseq: HierSeq, answer_token_id: int): 
    answer_mask = (hseq.tokens == answer_token_id) & (hseq.levels == 0)
    assert answer_mask.sum() > 0, "No answer token found in the batch"
    assert hseq.sample_idx[answer_mask].size(0) == hseq.indices.size(0), f"Number of answer tokens {hseq.sample_idx[answer_mask].size(0)} must match number of samples {hseq.indices.size(0)}"


def sanity_check_repeat_batch(repeat_batch: HierSeq, batch_data: HierSeq):
    assert (repeat_batch.idx_map[repeat_batch.indices] == batch_data.indices).all(), "Repeat batch indices do not match batch data indices"


def print_switch_abstraction_ratio(repeat_batch: HierSeq, argmax_indices: torch.Tensor, rollout_advantages: torch.Tensor): 
    n_unique_indices = torch.unique(repeat_batch.idx_map).size(0)
    n_switch_abstraction = (argmax_indices >= n_unique_indices).sum().item()
    ratio = n_switch_abstraction / n_unique_indices
    avg_advantage = rollout_advantages[rollout_advantages > 0].mean()
    avg_advantage = avg_advantage.item() if not avg_advantage.isnan() else 0.0
    print(f" - number of switched abstraction: {n_switch_abstraction}, ratio: {ratio:.4f}")
    print(f" - average advantage over greedy choice: {avg_advantage:.4f}")
    return ratio

def check_model_device(model):
    for name, param in model.named_parameters(): 
        assert param.device.type == model.device, f"parameter {name} not on same device as model"
    print("All parameters matching module device on", model.device)


def test_generate():

    from gat import GATConfig, GAT 
    from utils import HierSeq
    # 1. Setup a dummy GAT model
    config = GATConfig(
        L=3,
        n_layer=2, # Keep the model small for fast testing
        n_head=2,
        n_embd=32,
        vocab_size_list=[10, 5, 3], # L0, L1, L2
        device='cpu',
        _compile=False
    )
    model = GAT(config)
    model.eval() # Set model to evaluation mode
    level_mask_tokens = model.level_mask_tokens.tolist()

    print("--- 1. Testing Parallel Denoising (_denoise) ---")

    # We expect a level-1 token at ts=2 and a level-2 token at ts=4
    h_data_denoise = [
        ([1, 2, 3, 4, 5], [level_mask_tokens[1]], [level_mask_tokens[2]]), # Tokens for L0, L1, L2
        ([1, 2, 3, 4, 5], [2], [4])  # Timestamps for L0, L1, L2
    ]

    batch_denoise = HierSeq.from_hierarchical_data(
        [(h_data_denoise[0], h_data_denoise[1])], K=2, L=3, device='cpu'
    )

    print("Original sequence with masks:")
    print(batch_denoise.to_hierarchical_data()[0])
    assert (batch_denoise.tokens == model.level_mask_tokens[batch_denoise.levels]).sum() == 2

    # Run denoising
    denoised_batch = model.generate(batch_denoise, parallel=True, temperature=0.0)

    print("\nDenoised sequence:")
    print(denoised_batch.to_hierarchical_data()[0])

    assert (denoised_batch.tokens == model.level_mask_tokens[denoised_batch.levels]).sum() == 0
    # Check if the levels of the new tokens are correct
    assert denoised_batch.levels[denoised_batch.timestamps == 2][-1].item() == 1
    assert denoised_batch.levels[denoised_batch.timestamps == 4][-1].item() == 2
    print("✅ Parallel denoising test passed!")

    print("\n--- 2. Testing Auto-regressive Generation (_generate) ---")

    # Create an initial HierSeq to start generation from
    h_data_generate = [
        ([1, 2], [], []), # Tokens for L0, L1, L2
        ([1, 2], [], [])  # Timestamps for L0, L1, L2
    ]
    batch_generate = HierSeq.from_hierarchical_data(
        [(h_data_generate[0], h_data_generate[1])], K=2, L=3, device='cpu'
    )

    print("Initial sequence for generation:")
    print(batch_generate.to_hierarchical_data()[0])
    initial_len = len(batch_generate.tokens)

    # Generate a few tokens
    num_steps = 3
    for _ in range(num_steps):
        batch_generate = model.generate(batch_generate, parallel=False, temperature=0.0)

    print(f"\nSequence after {num_steps} generation steps:")
    print(batch_generate.to_hierarchical_data()[0])

    # Assertions to verify the logic
    assert len(batch_generate.tokens) == initial_len + num_steps
    # Check if the timestamps are sequential
    assert batch_generate.timestamps[-1].item() > batch_generate.timestamps[-2].item()
    print("✅ Auto-regressive generation test passed!")



def test_compile(): 
    import time
    from gat import GAT, GATConfig
    from utils import HierSeq
    """
    Benchmarks the forward pass of a GAT model with and without torch.compile.
    """
    print("--- Benchmarking torch.compile() ---")
    
    # 1. Setup model configurations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = GATConfig(
        L=3,
        n_layer=6,
        n_head=4,
        n_embd=128,
        vocab_size_list=[128, 64, 32],
        device=device,
        _compile=False  # We will compile manually
    )
    
    # 2. Create two identical models
    eager_model = GAT(config).to(device)
    eager_model.eval()
    
    # Use dynamic=False for best performance with fixed-shape inputs
    compiled_model = torch.compile(GAT(config), dynamic=False).to(device)
    compiled_model.eval()

    # 3. Create a realistic, random batch of data
    # Let's create a batch with 8 samples, each having around 256 tokens
    batch_size = 8
    seq_len_per_sample = 256
    samples = []
    for _ in range(batch_size):
        tokens = torch.randint(0, config.vocab_size_list[0], (seq_len_per_sample,))
        levels = torch.zeros_like(tokens)
        timestamps = torch.arange(seq_len_per_sample)
        samples.append(
            ([tokens.tolist()] + [[] for _ in range(config.L-1)], None)
        )

    batch_data = HierSeq.from_hierarchical_data(samples, K=config.K, L=config.L, device=device)
    print(f"Batch created with {batch_data.tokens.shape[0]} total tokens.")

    # 4. Define benchmark parameters
    num_runs = 50
    warmup_runs = 5

    # --- Benchmarking Eager Model ---
    print("\nRunning eager model benchmark...")
    eager_times = []
    for i in range(num_runs + warmup_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = eager_model(batch_data)
        
        # Synchronize for accurate timing on GPU
        if device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        if i >= warmup_runs:
            eager_times.append(end_time - start_time)

    eager_total_time = sum(eager_times)
    eager_avg_time = eager_total_time / num_runs
    print(f"Eager model: {eager_avg_time * 1000:.2f} ms per run (average of {num_runs} runs)")

    # --- Benchmarking Compiled Model ---
    print("\nRunning compiled model benchmark...")
    
    # Perform a warm-up run for the compiled model
    print("Performing warm-up run for compilation...")
    with torch.no_grad():
        _ = compiled_model(batch_data)
    if device == "cuda":
        torch.cuda.synchronize()
    print("Warm-up complete.")

    compiled_times = []
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = compiled_model(batch_data)

        if device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.time()
        compiled_times.append(end_time - start_time)

    compiled_total_time = sum(compiled_times)
    compiled_avg_time = compiled_total_time / num_runs
    print(f"Compiled model: {compiled_avg_time * 1000:.2f} ms per run (average of {num_runs} runs)")
    
    # --- Results ---
    speedup = eager_avg_time / compiled_avg_time
    print("\n--- Results ---")
    print(f"Speedup factor: {speedup:.2f}x")