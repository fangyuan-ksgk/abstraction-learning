import torch 

from model import GAT 
from utils import remove_pad_tokens, HierSeq



# Evaluation Gadget
# --------------------------------------------------------------------------------------------------------------------------
def eval_hseq(gat: GAT, batch_data: HierSeq, p_thres=4.16): 
    remove_pad_tokens(batch_data)
    gat.eval()
    with torch.no_grad():
        p_per_sample, loss_info, critical_ts, cr_per_sample = gat(batch_data, evaluate=True, p_thres=p_thres) # per-sample avg. perplexity (different weight for each level's avg. perplexity)
    return p_per_sample, loss_info, critical_ts, cr_per_sample
