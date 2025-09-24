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


def test_drop_traj_tokens():
    """
    Tests the drop_traj_tokens function from sorl.py.
    """
    from sorl import drop_traj_tokens

    print("\n--- Testing drop_traj_tokens ---")
    
    # 1. Setup a dummy batch with mixed levels
    h_data = [
        ([0,1,2,3,4,5,6,7,8,9], [10, 11], [12]), # L0, L1, L2 tokens
        ([0,1,2,3,4,5,6,7,8,9], [4, 8], [0])
    ]
    batch = HierSeq.from_hierarchical_data([(h_data[0], h_data[1])], K=4, L=3, device='cpu')
    
    # 2. Drop tokens, keeping the last 3 level-0 tokens
    diminished_batch = drop_traj_tokens(batch, t_keep=3)
    
    print("Sequence after dropping all but last 3 trajectory tokens:")
    data, ts = diminished_batch.to_hierarchical_data()
    print("Data:", data)
    print("Timestamps:", ts)

    # Assertions
    # Check that all abstract tokens are kept
    assert len(data[0][1]) == 2 and data[0][1] == [10, 11]
    assert len(data[0][2]) == 1 and data[0][2] == [12]
    # Check that only the last 3 level-0 tokens are kept
    assert len(data[0][0]) == 3 and data[0][0] == [7, 8, 9]
    print("✅ Memory diminishing test passed!")


def test_add_rhythmic_placeholders():
    from gat import GATConfig, GAT
    from sorl import add_rhythmic_placeholders
    """
    Tests the add_rhythmic_placeholders function from sorl.py.
    """
    print("\n--- Testing add_rhythmic_placeholders ---")
    
    # 1. Setup a dummy model and data
    config = GATConfig(L=3, K=4, vocab_size_list=[10, 5, 3], device='cpu')
    model = GAT(config)
    
    # Create a simple batch with one sample
    h_data = [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], [], []), # L0 tokens, no abstractions
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], [], [])
    ]
    batch = HierSeq.from_hierarchical_data([(h_data[0], h_data[1])], K=4, L=3, device='cpu')
    
    # 2. Test full padding (t_search=None)
    padded_batch = add_rhythmic_placeholders(batch.clone(), model.level_mask_tokens, t_search=None)
    
    print("Padded sequence (full):")
    _, ts = padded_batch.to_hierarchical_data()
    print(ts)

    # L1 should have placeholders at ts = 4, 8 (since K=4)
    # L2 should have placeholders at ts = 16 (but max ts is 9, so none)
    assert torch.equal(torch.tensor(ts[0][1]), torch.tensor([4, 8]))
    assert len(ts[0][2]) == 0
    print("✅ Full padding test passed!")

    # 3. Test partial padding (t_search=5)
    partial_padded_batch = add_rhythmic_placeholders(batch.clone(), model.level_mask_tokens, t_search=5)
    
    print("\nPadded sequence (t_search=5):")
    _, ts = partial_padded_batch.to_hierarchical_data()
    print(ts)
    
    # L1 should only have a placeholder at ts = 4
    assert torch.equal(torch.tensor(ts[0][1]), torch.tensor([4]))
    print("✅ Partial padding test passed!")


def test_add_spike_placeholders():
    """
    Tests the add_spike_placeholders function from sorl.py, which uses
    proportional budget allocation based on perplexity spikes.
    """
    from gat import GATConfig, GAT
    from utils import HierSeq
    from sorl import add_spike_placeholders

    print("\n--- Testing add_spike_placeholders ---")

    # 1. Setup a dummy model and data
    config = GATConfig(L=2, K=4, vocab_size_list=[20, 5], device='cpu')
    model = GAT(config)
    model.eval()

    # 2. Create a batch with one long sample
    h_data = [
        (list(range(10)), []), # L0 tokens
        (list(range(1, 11)), [])  # L0 timestamps (1-indexed)
    ]
    batch = HierSeq.from_hierarchical_data([(h_data[0], h_data[1])], K=4, L=2, device='cpu')

    # 3. Mock the model's forward pass to return a predictable ppt tensor.
    # The ppt tensor should have N-1 elements for N tokens, representing the
    # loss of predicting token i+1 given tokens 0...i.
    # We create a ppt that results in two clear "spikes" (dip then rise).
    mock_ppt = torch.tensor([5.0, 4.0, 8.0, 4.0, 3.0, 10.0, 5.0, 5.0, 5.0], device='cpu')
    
    original_forward = model.forward
    model.forward = lambda b: mock_ppt

    # 4. Call the function with a budget of 5 tokens for level 1
    budgets = torch.tensor([0, 5]) 
    processed_batch = add_spike_placeholders(model, batch.clone(), budgets)

    # 5. Assertions
    print("Processed batch with spike-based placeholders:")
    data, ts = processed_batch.to_hierarchical_data()
    print("Data (Tokens):", data)
    print("Data (Timestamps):", ts)

    # Expected behavior based on mock_ppt:
    # mock_ppt (len=9)
    # ppt_increase = [-1, 4, -4, -1, 7, -5, 0, 0] (len=8)
    # Spikes (dip then rise) are at indices 1 and 4 of ppt_increase.
    # Spike values are 4 and 7. Total weight = 11. Budget = 5.
    # Allocation for spike 1 (weight 4): floor((4/11)*5) = 1. Remainder: 0.81
    # Allocation for spike 2 (weight 7): floor((7/11)*5) = 3. Remainder: 0.18
    # Remaining budget (5 - 1 - 3 = 1) goes to spike 1 due to larger remainder.
    # Final counts: 2 tokens for spike 1, 3 for spike 2.
    # Spike 1 is at ppt_increase index 1, insertion ts = original_ts[1+1] = 3.
    # Spike 2 is at ppt_increase index 4, insertion ts = original_ts[4+1] = 6.
    
    level1_timestamps = torch.tensor(ts[0][1])
    
    expected_total_tokens = 5
    assert len(level1_timestamps) == expected_total_tokens, \
        f"Expected {expected_total_tokens} L1 tokens, but got {len(level1_timestamps)}"
    print(f"✅ Correct total number of tokens added ({expected_total_tokens}).")

    expected_ts_3_count = 2
    ts_3_count = (level1_timestamps == 3).sum().item()
    assert ts_3_count == expected_ts_3_count, \
        f"Expected {expected_ts_3_count} tokens at ts=3, but got {ts_3_count}"
    print(f"✅ Correct number of tokens at ts=3 ({expected_ts_3_count}).")
    
    expected_ts_6_count = 3
    ts_6_count = (level1_timestamps == 6).sum().item()
    assert ts_6_count == expected_ts_6_count, \
        f"Expected {expected_ts_6_count} tokens at ts=6, but got {ts_6_count}"
    print(f"✅ Correct number of tokens at ts=6 ({expected_ts_6_count}).")

    print("✅ `add_spike_placeholders` test passed!")
    
    # Restore original method
    model.forward = original_forward


def test_generate_rollout_data():
    """
    Tests the consolidated generate_rollout_data function from sorl.py.
    """
    from gat import GATConfig, GAT
    from sorl import generate_rollout_data
    from utils import repeat_hseq # Assuming repeat_hseq is in utils.py
    
    print("\n--- Testing generate_rollout_data ---")
    
    # 1. Setup a dummy model and data
    config = GATConfig(L=2, K=3, vocab_size_list=[10, 5], device='cpu')
    model = GAT(config)
    model.eval()

    h_data = [
        ([1, 2, 3, 4, 5], []), # L0 tokens, no abstractions
        ([1, 2, 3, 4, 5], [])
    ]
    batch = HierSeq.from_hierarchical_data([(h_data[0], h_data[1])], K=3, L=2, device='cpu')

    # 2. Generate rollouts
    n_rollouts = 4
    rollout_batch = generate_rollout_data(model, batch, n=n_rollouts, temperature=0.0)

    # 3. Assertions
    # Check 1: The batch size should be correct
    assert rollout_batch.batch_size == n_rollouts, \
        f"Expected batch size {n_rollouts}, but got {rollout_batch.batch_size}"
    print("✅ Batch size is correct.")

    # Check 2: There should be no MASK_TOK placeholders left
    is_mask = (rollout_batch.tokens == model.level_mask_tokens[rollout_batch.levels])
    assert is_mask.sum() == 0, "MASK_TOK placeholders were not filled."
    print("✅ All placeholders were denoised.")

    # Check 3: Check if the abstract tokens have been added at the correct timestamps
    data, ts = rollout_batch.to_hierarchical_data()
    # For a sequence of length 5 with K=3, L1 tokens should be at ts=3
    for sample_ts in ts:
        assert torch.equal(torch.tensor(sample_ts[1]), torch.tensor([3])), \
            f"Expected L1 tokens at timestamp [3], but got {sample_ts[1]}"
    print("✅ Abstract tokens were added at the correct rhythmic intervals.")
    
    print("✅ `generate_rollout_data` test passed!")


def test_select_best_abstraction():
    """
    Tests the select_best_abstraction function from sorl.py.
    """
    from sorl import select_best_abstraction
    from utils import HierSeq

    print("\n--- Testing select_best_abstraction ---")

    # 1. Setup: 2 original samples, 2 rollouts each. Total batch size = 4.
    # Rollout from idx 1 should be best for sample 0.
    # Rollout from idx 2 should be best for sample 1.
    samples = []
    for _ in range(4):
        # Each sample has 5 level-0 tokens, ts 1-5
        data = [list(range(5)), []] # L0, L1 data
        ts = [list(range(1, 6)), []] # L0, L1 timestamps
        samples.append((data, ts))

    repeat_batch = HierSeq.from_hierarchical_data(samples, K=4, L=2, device='cpu')
    repeat_batch.idx_map = torch.tensor([0, 0, 1, 1])

    # 2. Craft a PPT tensor. The function averages ppl for level-0 tokens at ts > 1.
    # Total tokens = 4 samples * 5 tokens = 20. PPT length = 19.
    ppt = torch.ones(19, dtype=torch.float32) * 5.0

    # For idx 1 (best for sample 0), make its traj ppl low.
    # Trajectory tokens for sample 1 are tokens 6-9, predicted by ppt[5-8]
    ppt[5:9] = 1.0

    # For idx 2 (best for sample 1), make its traj ppl low.
    # Trajectory tokens for sample 2 are tokens 11-14, predicted by ppt[10-13]
    ppt[10:14] = 2.0

    # 3. Execute
    select_batch, _, _ = select_best_abstraction(repeat_batch, ppt, duplicate=False)

    # 4. Assert
    expected_indices = torch.tensor([1, 2])
    assert torch.equal(select_batch.indices.sort().values, expected_indices.sort().values), \
        f"Expected selected indices {expected_indices}, but got {select_batch.indices}"
    print("✅ `select_best_abstraction` test passed!")


def test_sorl_search():
    """
    Tests the main sorl_search function from sorl.py.
    """
    from sorl import sorl_search
    from gat import GAT, GATConfig
    from utils import HierSeq
    from unittest.mock import patch

    print("\n--- Testing sorl_search ---")

    # 1. Setup
    config = GATConfig(L=2, K=3, vocab_size_list=[10, 5], device='cpu')
    model = GAT(config)
    model.eval()
    batch = HierSeq.from_hierarchical_data([([ [1, 2], [] ], [ [1, 2], [] ])], K=3, L=2, device='cpu')

    # 2. Mocking generate_rollout_data and gat.forward
    # R0 = ref, R1, R2 = exploration. R2 will be made the best.
    ref_batch = HierSeq.from_hierarchical_data([([ [1, 2, 3], [] ], [ [1, 2, 3], [] ])], K=3, L=2, device='cpu')
    explore_batch = HierSeq.from_hierarchical_data(
        [
            ([ [1, 2, 4], [] ], [ [1, 2, 3], [] ]), # R1
            ([ [1, 2, 5], [] ], [ [1, 2, 3], [] ])  # R2 -> best
        ], K=3, L=2, device='cpu'
    )
    
    # Mock ppt. Total tokens = 3 (R0) + 3 (R1) + 3 (R2) = 9. PPT length = 8.
    # The selection function only looks at traj tokens (ts > 1), so 2 per sample.
    # We only need to mock the full ppt; the function will apply the mask.
    mock_ppt = torch.tensor([
        5.0, 5.0,  # R0 preds for t_1, t_2
        5.0, 5.0,  # R1 preds for t_1, t_2
        1.0, 1.0,  # R2 preds for t_1, t_2 -> best
        0.0, 0.0   # Dummy values to make length 8
    ], dtype=torch.float)
    
    mock_generate = patch('sorl.generate_rollout_data').start()
    mock_generate.side_effect = [ref_batch, explore_batch]
    
    original_forward = model.forward
    model.forward = lambda b: mock_ppt

    # 3. Execute
    select_batch, _, _ = sorl_search(model, batch, n=3, temperature=1.0)

    # 4. Assertions
    assert select_batch.batch_size == 1
    expected_tokens = torch.tensor([1, 2, 5])
    assert torch.equal(select_batch.tokens, expected_tokens), \
        f"Expected selected tokens {expected_tokens}, but got {select_batch.tokens}"
    print("✅ `sorl_search` test passed!")

    # 5. Cleanup
    patch.stopall()
    model.forward = original_forward




def check_denoise():

    import torch
    from src.regat import GAT, GATConfig
    from src.utils import infer_level
    
    # 1. Setup a minimal model and configuration
    config = GATConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        vocab_size_list=[10, 10, 10], # 3 levels, vocab size 10 each
        L=3,
        K=4,
        device='cpu',
        _compile=False
    )
    model = GAT(config)

    # Vocabulary details from the model
    l1_mask_tok = model.level_mask_tokens[1].item() # Placeholder for level 1
    l2_mask_tok = model.level_mask_tokens[2].item() # Placeholder for level 2

    # 2. Create a sample input sequence `idx`
    # We use `0` as a dummy token value that we expect to be replaced.
    # Placeholders are at:
    # - Batch 0, Pos 2 (level 1) -> The token at Pos 3 should be replaced
    # - Batch 0, Pos 5 (level 2) -> The token at Pos 6 should be replaced
    # - Batch 1, Pos 1 (level 1) -> The token at Pos 2 should be replaced
    # - Batch 1, Pos 7 (level 2) -> Edge case: last token, should be ignored
    original_idx = torch.tensor([
        [1, 5, l1_mask_tok, 0, 2, l2_mask_tok, 0, 3],
        [4, l1_mask_tok, 0, 6, 8, 3, 1, l2_mask_tok]
    ], dtype=torch.long, device=config.device)

    # Clone for modification and later comparison
    test_idx = original_idx.clone()

    # 3. Prepare the inputs for the denoise function
    denoise_mask = torch.isin(test_idx, model.level_mask_tokens)
    placeholder_levels = infer_level(test_idx[denoise_mask], model.vocab_sizes, model.level_mask_tokens[0])

    print("--- Input Data ---")
    print("Original sequence:\n", original_idx)
    print("Placeholder mask (denoise_mask):\n", denoise_mask)
    print("Levels of placeholders:\n", placeholder_levels)
    print("-" * 20)

    # 4. Call the denoise function
    with torch.no_grad():
        denoised_idx = model.denoise(test_idx, denoise_mask, placeholder_levels, temperature=0.0)

    print("\n--- Output Data ---")
    print("Denoised sequence:\n", denoised_idx)
    print("-" * 20)

    # 5. Assertions and Checks
    print("\n--- Verification ---")


    # Check 2: The tokens at positions to be replaced (originally 0) should now be different
    assert denoised_idx[0, 2] != 0, "Test Failed: Token at (0, 3) was not replaced."
    assert denoised_idx[0, 5] != 0, "Test Failed: Token at (0, 6) was not replaced."
    assert denoised_idx[1, 1] != 0, "Test Failed: Token at (1, 2) was not replaced."
    print("✅ Pass: Tokens following placeholders were correctly replaced.")

    # Check 3: The values that were replaced should be valid tokens for their respective levels.
    new_tok_0_2_level = infer_level(denoised_idx[0, 2].unsqueeze(0), model.vocab_sizes, model.level_mask_tokens[0])
    assert new_tok_0_2_level.item() == 1, f"Test Failed: Token at (0,3) has wrong level {new_tok_0_2_level.item()}, expected 1."

    new_tok_0_5_level = infer_level(denoised_idx[0, 5].unsqueeze(0), model.vocab_sizes, model.level_mask_tokens[0])
    assert new_tok_0_5_level.item() == 2, f"Test Failed: Token at (0,6) has wrong level {new_tok_0_5_level.item()}, expected 2."

    new_tok_1_1_level = infer_level(denoised_idx[1, 1].unsqueeze(0), model.vocab_sizes, model.level_mask_tokens[0])
    assert new_tok_1_1_level.item() == 1, f"Test Failed: Token at (1,2) has wrong level {new_tok_1_1_level.item()}, expected 1."
    print("✅ Pass: Replaced tokens belong to the correct vocabulary level.")

    # Check 4: All other tokens (not placeholders, not replacements) should be unchanged
    stable_mask = ~torch.isin(original_idx, model.level_mask_tokens)
    stable_mask[:, [2, 5]] = False
    stable_mask[1, 1] = False

    assert torch.all(denoised_idx[stable_mask] == original_idx[stable_mask]), \
        "Test Failed: Other tokens in the sequence were unexpectedly changed."
    print("✅ Pass: All other tokens remain unchanged.")
    print("\nSanity check complete. All tests passed!")
