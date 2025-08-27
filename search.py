import torch 

from model import GAT 
from utils import remove_pad_tokens, HierSeq
import numpy as np


# Evaluation Gadget
# --------------------------------------------------------------------------------------------------------------------------
def eval_hseq(gat: GAT, batch_data: HierSeq, p_thres=4.16): 
    # remove_pad_tokens(batch_data)
    p_per_sample, critical_ts, cr_per_sample, ppt = gat(batch_data, evaluate=True, p_thres=p_thres) # per-sample avg. perplexity (different weight for each level's avg. perplexity)
    
    return p_per_sample, critical_ts, cr_per_sample, ppt