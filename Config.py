from tokenizer.tokenizer import tokenizer
import torch

# Config
class Config:
    epochs = 1
    train_batch_size = 16
    val_batch_size = 16
    test_batch_size = 16
    n_embd = 512
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    max_encoder_seq_len = 512
    max_decoder_seq_len = 512
    num_heads = 8
    n_encoder_layers = 6
    n_decoder_layers = 6
    dropout = 0.1
    eval_interval = 20_000 # iters
    max_eval_iters = 1500
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()