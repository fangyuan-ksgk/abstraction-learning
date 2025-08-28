from model import HierSeq
import torch


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