# Arithmetic Experiments Data Generation 
# ---------------------------------------------------------------------------------
from dataset.base import BaseDataset, BaseHierDataset
from dataclasses import dataclass, field
from typing import Optional
import random
from tqdm import tqdm


# Per-digit tokenizer 
# ------------------------------------------------------------
class DigitTokenizer:
    def __init__(self):
        # Special tokens
        self.special_tokens = {
            "x": 0,
            "=": 1,
            " ": 2
        }
        self.digit_tokens = {str(i): i + 3 for i in range(10)}
        self.vocab = {**self.special_tokens, **self.digit_tokens}
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)
        self.answer_token = "="
        self.answer_token_id = self.vocab["="]
    
    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text[i:].startswith(token):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            
            if not matched:
                raise KeyError(f"Unknown token at position {i}: '{text[i:]}'")
        
        return ids
    
    def decode(self, ids):
        return ''.join(self.inv_vocab[idx] for idx in ids)
    
    def encode_multiplication(self, a, b, c):
        text = f"{a} x {b} ={c}"
        return self.encode(text)
    
    
# Data Generation Utils
# ------------------------------------------------------------
def generate_multiplication_examples(min_digits_a, max_digits_a, 
                                    min_digits_b, max_digits_b, 
                                    num_examples):
    examples = []
    for _ in tqdm(range(num_examples)):
        digits_a = random.randint(min_digits_a, max_digits_a)
        digits_b = random.randint(min_digits_b, max_digits_b)
        a = random.randint(10**(digits_a-1), 10**digits_a - 1)
        b = random.randint(10**(digits_b-1), 10**digits_b - 1)        
        c = a * b
        examples.append((a, b, c))
    return examples


# Dataclass 
# -----------------------------------------------------------------------

@dataclass 
class ArithmeticDataset(BaseDataset): 
    min_digit: int = 1
    max_digit: int = 3
    num_data: int = 10000000
    filepath: str = "dataset/multiplication/sequences.bin"
    vocab_size: Optional[int] = None 

    tokenizer: DigitTokenizer = DigitTokenizer()
    answer_token: str = DigitTokenizer().answer_token
    answer_token_id: int = DigitTokenizer().answer_token_id

    def build(self): 
        examples = generate_multiplication_examples(self.min_digit, self.max_digit, self.min_digit, self.max_digit, self.num_data)
        self.vocab_size = self.tokenizer.vocab_size
        self.sequences = [self.tokenizer.encode_multiplication(a, b, c) for a, b, c in examples]
        self.lengths = [len(seq) for seq in self.sequences]
        self._save()
        return self


@dataclass 
class ArithmeticHierDataset(BaseHierDataset): 
    min_digit: int = 1
    max_digit: int = 3
    num_data: int = 10000000
    filepath: str = "dataset/multiplication/sequences_hier.bin"
    vocab_size: Optional[int] = None 

    tokenizer: DigitTokenizer = DigitTokenizer()
    answer_token: str = DigitTokenizer().answer_token
    answer_token_id: int = DigitTokenizer().answer_token_id

    @classmethod 
    def from_dataset(cls, dataset: ArithmeticDataset): 
        
        hier_seqs = []
        for seq in dataset.sequences: 
            hier_seq = [seq] + [[] for _ in range(1, dataset.L)]
            hier_seqs.append(hier_seq)

        instance = cls(
            filepath=dataset.filepath.replace(".bin", "_hier.bin"),
            num_data=len(hier_seqs),
            K=dataset.K,
            L=dataset.L,
        )
        instance.sequences = hier_seqs
        instance.lengths = dataset.lengths
        return instance       