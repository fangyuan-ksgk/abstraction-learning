# NIL algorithm
# Within each generation, we inherit prev generation's abstraction, and continue training 
# Reset -> Weak-supervision -> Train -> Record 
# ------------------------------------------------------------------------------------------------
from dataset.base import BaseHierDataset
from search import get_hier_batch_with_index, pad_abstract_tokens
from model import GAT


def annotate_abstraction(record_dataset: BaseHierDataset, gat: GAT, context_length: int = 1024, temperature: float = 0.0): 

    context_length = 1024 
    next_idx = 0
    total_seq = len(record_dataset.sequences)
    temperature = 0.0

    while next_idx < total_seq: 
        start_idx = next_idx

        # (1). Loop through each sequence (require modified 'get_batch' that returns end index, and start from argumented index)
        batch_data = get_hier_batch_with_index(record_dataset.sequences, record_dataset.lengths, context_length, gat.L, gat.K, 
                                            start_idx=start_idx)
        assert (batch_data.levels == 1).sum() == 0, "HierSeq to be annotated should not contain any abstract tokens!"
        next_idx = batch_data.indices[-1] + 1 


        # (2). GAT inference
        pad_abstract_tokens(batch_data)
        batch_data = gat.generate(batch_data, parallel=True, temperature=temperature)

        # (3). Insert abstract tokens back into HierDataset
        annotated_hier_seqs, _ = batch_data.to_hierarchical_data() # timestamps is not used without sparsity for now
        record_dataset.sequences[start_idx:next_idx] = annotated_hier_seqs

    return record_dataset

# Weak-supervision involves direct supervision on abstraction-labeled trajectory dataset
# controlling iteration number to avoid overfitting & underfitting


