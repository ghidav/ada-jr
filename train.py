import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae import SaeConfig, SaeTrainer, TrainConfig
from sae.data import chunk_and_tokenize

if __name__ == "__main__":
    MODEL = "EleutherAI/pythia-70m-deduped"
    N_TOKENS = 100_000_000
    seq_len = 1024

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=seq_len).select(
        range(N_TOKENS // seq_len)
    )

    gpt = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map={"": "cuda"},
        torch_dtype=torch.float32,
    )

    cfg = TrainConfig(SaeConfig(expansion_factor=8, k=128), batch_size=16, lr=3e-4)
    trainer = SaeTrainer(cfg, tokenized, gpt)
    trainer.fit()
