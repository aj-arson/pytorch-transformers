import torch
from tokenizer.tokenizer import tokenizer
from Transformer import Transformer

special_tokens = tokenizer.get_special_tokens()
special_token_ids = tokenizer.get_special_token_ids()

def translate(input_text, model, tokenizer, max_tokens):
    input_text = special_tokens.get('sos','<s>')+input_text+special_tokens.get('eos','</s>')
    output_text = special_tokens.get('sos','<s>')

    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    output_ids = torch.tensor(tokenizer.encode(output_text)).unsqueeze(0)

    for _ in range(max_tokens):
        with torch.inference_mode():
            next_logits = model(input_ids, output_ids)[:,-1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            if next_token.item() not in special_token_ids.values():
                output_ids = torch.cat((output_ids, next_token), dim=-1)
            if next_token==tokenizer.eos_token_id or len(output_ids)==max_tokens:
                break
    return tokenizer.decode(output_ids.squeeze(0).tolist())

if __name__ == "__main__":
    max_tokens = 16
    input_text = "Hi there!"
    checkpoint_path = None

    model = Transformer()
    if checkpoint_path:
        model.load_checkpoint(checkpoint_path)
    print(translate(input_text, model, tokenizer, max_tokens))
    print(2 not in special_token_ids.values())