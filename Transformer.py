# Transformers Model
import torch
from Config import config

# Multi Head Attention
class Multi_Head_Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = torch.nn.Linear(config.n_embd, config.n_embd)
        self.key = torch.nn.Linear(config.n_embd, config.n_embd)
        self.value = torch.nn.Linear(config.n_embd, config.n_embd)
        self.dropout = torch.nn.Dropout(config.dropout)

        self.output_proj = torch.nn.Linear(config.n_embd, config.n_embd)
    

    def forward(self, query, key, value, attention_mask=None):
        B, Tq, C = query.shape
        Tk = key.shape[1]
        head_dim = C//self.config.num_heads

        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
    
        q = q.view(B, Tq, self.config.num_heads, head_dim).permute(0, 2, 1, 3) # B, N, T, H
        k = k.view(B, Tk, self.config.num_heads, head_dim).permute(0, 2, 3, 1) # B, N, H, T
        v = v.view(B, Tk, self.config.num_heads, head_dim).permute(0, 2, 1, 3) # B, N, T, H

        # print(f"Q → {q.shape}")
        # print(f"K → {k.shape}")
        # print(f"V → {v.shape}")
    
        scaled_dot_product = (q @ k)/(head_dim**0.5) # B, N, Tq, H @ B, N, H, Tk ---> B, N, Tq, Tk
    
        if attention_mask is not None:
            scaled_dot_product += attention_mask
    
        softmax_scores = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        softmax_scores = self.dropout(softmax_scores)
        # print(f"attention scores → {softmax_scores.shape}")
        dot_product_attention = softmax_scores @ v # B, N, Tq, Tk @ B, N, Tk, H ---> B, N, Tq, H

        dot_product_attention = dot_product_attention.permute(0, 2, 1, 3).reshape(B, Tq, C)
        return self.output_proj(dot_product_attention)

# Causal Mask
def causal_mask(x): # we only use this in decoder self attention so we can just have T value and construct the mask
    T = x.shape[1]
    mask = torch.where((1-torch.tril(torch.ones(T, T, device=x.device))==0), 0., float('-inf'))
    return mask

# Padding Mask
def padding_mask(x, pad_token_id):
    B, Tk = x.shape[:2]
    return torch.where(x == pad_token_id, float('-inf'), 0).reshape(B, 1, 1, Tk).to(x.device) # Reshaping to broadcast for B, N, Tq, Tk


# Feed Forward
class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffw = torch.nn.Sequential(torch.nn.Linear(config.n_embd, 4*config.n_embd),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(4*config.n_embd, config.n_embd),)

    def forward(self, x):
        return self.ffw(x)

# Encoder Block
class EncoderBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.residual_drop = torch.nn.Dropout(config.dropout)
        # Multi_Head_Attention(config)
        self.mha = Multi_Head_Attention(config)

        self.ln1 = torch.nn.LayerNorm(config.n_embd)
        self.ln2 = torch.nn.LayerNorm(config.n_embd)
        self.ffw = FeedForward(config)

    def forward(self, x, attention_mask=None): # Here the x will be the result of embedding + positional-embedding layer so the shape is B, Tq, C
        # Multi-head attention with add and norm
        x = x + self.residual_drop(self.mha(x, x, x, attention_mask=attention_mask))
        x = self.ln1(x)

        # Feed forwarding with add and norm
        x = x + self.residual_drop(self.ffw(x))
        x = self.ln2(x)

        return x

# Encoder
class Encoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = torch.nn.Embedding(config.max_encoder_seq_len, config.n_embd)
        self.encoder_blocks = torch.nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layers)])
        self.drop = torch.nn.Dropout(config.dropout)
        
    def forward(self, x, encoder_attention_mask=None):
        T = x.shape[1] # Seq_length
        x = self.drop(self.embedding(x) + self.pos_embedding(torch.arange(T, device=x.device))) # B, T, C
        for block in self.encoder_blocks:
            x = block(x, attention_mask=encoder_attention_mask)

        return x

# Decoder Block
class DecoderBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # masked multi-head attention
        self.masked_mha = Multi_Head_Attention(config)
       
        # multi-head cross attention
        self.mha = Multi_Head_Attention(config)
        
        # layer normalizations
        self.ln1 = torch.nn.LayerNorm(config.n_embd)
        self.ln2 = torch.nn.LayerNorm(config.n_embd)
        self.ln3 = torch.nn.LayerNorm(config.n_embd)

        # Feed Forward layer
        self.ffw = FeedForward(config)

        self.residual_drop = torch.nn.Dropout(config.dropout)

    def forward(self, encoder_output, decoder_input, combined_decoder_mask=None, cross_attention_mask=None):
        # causal_mask is decoder_causal_mask + decoder_padding_mask
        # cross_attention_mask is encoder_padding_mask
        
        # Multi-head attention with add and norm
        x = decoder_input + self.residual_drop(self.masked_mha(decoder_input, decoder_input, decoder_input, attention_mask=combined_decoder_mask))
        x = self.ln1(x)

        # Multi-head cross attention with add and norm
        x = x + self.residual_drop(self.mha(x, encoder_output, encoder_output, attention_mask=cross_attention_mask))
        x = self.ln2(x)
    
        # Feed forwarding with add and norm
        x = x + self.residual_drop(self.ffw(x))
        x = self.ln3(x)

        return x

# Decoder
class Decoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = torch.nn.Embedding(config.max_decoder_seq_len, config.n_embd)
        self.decoder_blocks = torch.nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layers)])
        self.drop = torch.nn.Dropout(config.dropout)

    def forward(self, encoder_output, decoder_input, combined_decoder_mask=None, cross_attention_mask=None):
        T = decoder_input.shape[1] # Seq_length
        x = self.drop(self.embedding(decoder_input) + self.pos_embedding(torch.arange(T, device=decoder_input.device)))

        for block in self.decoder_blocks:
            x = block(encoder_output, x, combined_decoder_mask=combined_decoder_mask, cross_attention_mask=cross_attention_mask)
        return x

# Transformer
class Transformer(torch.nn.Module):
    def __init__(self, config=config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = torch.nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, encoder_input, decoder_input):
        # Encoder block
        encoder_attention_mask = padding_mask(encoder_input, self.config.pad_token_id) 
        encoder_output = self.encoder(encoder_input, encoder_attention_mask=encoder_attention_mask)

        # Decoder block
        # causal_mask is decoder_causal_mask + decoder_padding_mask
        # cross_attention_mask is encoder_padding_mask
        combined_decoder_mask = causal_mask(decoder_input)+padding_mask(decoder_input, self.config.pad_token_id)
        cross_attention_mask = padding_mask(encoder_input, self.config.pad_token_id) # because we need to get what the actual pad_tokens are in encoder input, we won't have these values in encoder_output
        decoder_output = self.decoder(encoder_output, decoder_input, combined_decoder_mask=combined_decoder_mask, cross_attention_mask=cross_attention_mask)

        # Output projection
        linear_output = self.linear(decoder_output)     
        return linear_output
