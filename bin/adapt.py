import torch
import torch.nn as nn
import itertools
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import HierSeq
import torch.nn.functional as F

class SoRLAdapter(nn.Module):
    """
    A wrapper to adapt a pre-trained transformer model for the SoRL framework.
    """
    def __init__(self, model_name: str, abstract_vocab_sizes: list):
        """
        Initializes the adapter.

        Args:
            model_name (str): The name of the pre-trained model from the Hugging Face Hub.
            abstract_vocab_sizes (list): A list of integers representing the number of 
                                         new abstraction tokens for each level.
        """
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.abstract_vocab_sizes = abstract_vocab_sizes
        self.num_abstract_levels = len(abstract_vocab_sizes)
        
        self._expand_vocabulary()

    def _expand_vocabulary(self):
        """
        Adds new tokens for each abstraction level to the tokenizer and resizes the model's embeddings.
        """
        new_tokens = []
        for i, size in enumerate(self.abstract_vocab_sizes):
            level_tokens = [f"[ABS_L{i+1}_{j}]" for j in range(size)]
            new_tokens.extend(level_tokens)
            
        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Calculate the starting ID for the new abstract tokens
        self.original_vocab_size = len(self.tokenizer) - len(new_tokens)
        
        # Create offsets for each abstraction level
        level_offsets = torch.tensor([0] + list(itertools.accumulate(self.abstract_vocab_sizes)))[:-1]
        self.register_buffer('level_offsets', self.original_vocab_size + level_offsets)

    def _flatten_hier_seq(self, batch_data: HierSeq) -> torch.Tensor:
        """
        Converts a HierSeq object into a flat tensor of token IDs for the pre-trained model.
        """
        flat_tokens = batch_data.tokens.clone()
        
        for i in range(self.num_abstract_levels):
            level_mask = (batch_data.levels == i + 1)
            if level_mask.any():
                flat_tokens[level_mask] += self.level_offsets[i]
                
        return flat_tokens.unsqueeze(0) # Add batch dimension

    def forward(self, batch_data: HierSeq):
        """
        Performs a forward pass with a HierSeq object.
        """
        # 1. Flatten the hierarchical input
        input_ids = self._flatten_hier_seq(batch_data)
        
        # 2. Get the model's output
        # Note: We're not using a custom causal mask here for simplicity, but you could add one.
        outputs = self.model(input_ids, labels=input_ids)
        
        # 3. For now, we'll return the standard language modeling loss.
        #    A more advanced implementation would compute a hierarchical loss.
        return outputs.loss

    def generate(self, *args, **kwargs):
        """
        A placeholder for a hierarchical generation method.
        """
        # This would require more complex logic to handle the generation of abstract tokens
        # and the subsequent generation of lower-level tokens.
        print("Hierarchical generation is not yet implemented.")
        return self.model.generate(*args, **kwargs)


if __name__ == '__main__':
    # --- Example Usage ---
    
    # 1. Configuration
    model_name = "gpt2"  # or "llama-2-7b", etc.
    abstract_vocab_sizes = [64, 32] # L1 has 64 tokens, L2 has 32

    # 2. Initialize the adapter
    adapter = SoRLAdapter(model_name, abstract_vocab_sizes)
    
    # 3. Create a dummy HierSeq object
    #    (Replace this with your actual data loading)
    dummy_tokens = torch.tensor([10, 20, 0, 30, 40, 1])
    dummy_levels = torch.tensor([0, 0, 1, 0, 0, 2])
    dummy_timestamps = torch.tensor([0, 1, 1, 2, 3, 3])
    dummy_sample_idx = torch.tensor([0, 0, 0, 0, 0, 0])
    
    batch_data = HierSeq(
        tokens=dummy_tokens,
        levels=dummy_levels,
        timestamps=dummy_timestamps,
        sample_idx=dummy_sample_idx,
        batch_size=1,
        K=2, L=3
    )

    # 4. Perform a forward pass
    loss = adapter(batch_data)
    print(f"Loss: {loss.item()}")
    
    # 5. Check the tokenizer and model vocabulary
    print(f"Original vocab size: {adapter.original_vocab_size}")
    print(f"New vocab size: {len(adapter.tokenizer)}")
    print(f"Model embedding size: {adapter.model.get_input_embeddings().num_embeddings}")
