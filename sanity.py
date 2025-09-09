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


def print_switch_abstraction_ratio(repeat_batch: HierSeq, argmax_indices: torch.Tensor): 
    n_unique_indices = torch.unique(repeat_batch.idx_map).size(0)
    n_switch_abstraction = (argmax_indices >= n_unique_indices).sum().item()
    ratio = n_switch_abstraction / n_unique_indices
    # print(f" - number of switched abstraction: {n_switch_abstraction}, ratio: {ratio:.4f}")
    return ratio