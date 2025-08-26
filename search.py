import torch 

from model import GAT 
from utils import remove_pad_tokens, HierSeq
import numpy as np


# Evaluation Gadget
# --------------------------------------------------------------------------------------------------------------------------
def eval_hseq(gat: GAT, batch_data: HierSeq, search_mask: np.ndarray, p_thres=4.16): 
    remove_pad_tokens(batch_data)
    p_per_sample, critical_ts, cr_per_sample = gat(batch_data, evaluate=True, p_thres=p_thres) # per-sample avg. perplexity (different weight for each level's avg. perplexity)
    
    # (TBD). Implement the 'search_mask' for buffer object, to indicate 'no meaningful composition exists, learn stuff instead'
    #        here we essentially apply the 'search_mask' to stop the search and focus on optimization
    critical_ts[search_mask] = -1
    
    return p_per_sample, critical_ts, cr_per_sample