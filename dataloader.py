from datasets import load_dataset
from Config import config
from tokenizer.tokenizer import tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

dataset = load_dataset(
    "wmt/wmt14",
    "de-en"
)
special_tokens = tokenizer.get_special_tokens()
special_token_ids = tokenizer.get_special_token_ids()

start_token, end_token, pad_token = special_tokens.get('sos'), special_tokens.get('eos'), special_tokens.get('pad')

class NMT_DATASET(Dataset):
    def __init__(self, dataset, tokenizer, config):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = special_token_ids.get('pad')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # eng to de translation
        model_input_token_ids = [self.tokenizer.bos_token_id]+self.tokenizer.encode(self.dataset[idx]['translation']['en'][:self.config.max_encoder_seq_len-2])+[self.tokenizer.eos_token_id]
        model_output_token_ids = [self.tokenizer.bos_token_id]+self.tokenizer.encode(self.dataset[idx]['translation']['de'][:self.config.max_encoder_seq_len-2])+[self.tokenizer.eos_token_id]

        encoder_input = torch.tensor(model_input_token_ids)
        decoder_input = torch.tensor(model_output_token_ids[:-1])
        decoder_output = torch.tensor(model_output_token_ids[1:])
        
        return encoder_input.to(config.device), decoder_input.to(config.device), decoder_output.to(config.device)

train_dataset = NMT_DATASET(dataset['train'].select(range(2000)), tokenizer, config)
val_dataset = NMT_DATASET(dataset['validation'].select(range(2000)), tokenizer, config)
test_dataset = NMT_DATASET(dataset['test'], tokenizer, config)

def custom_collate_fn(batch):
  # Extract the three components from each sample in the batch
  encoder_inputs = [item[0] for item in batch]
  decoder_inputs = [item[1] for item in batch]
  decoder_outputs = [item[2] for item in batch]

  # Get your padding token ID (change 0 to your actual pad token ID if different)
  pad_id = special_token_ids.get("pad", 0)

  # Pad all three sequences to the maximum length within this specific batch
  encoder_padded = pad_sequence(
      encoder_inputs, batch_first=True, padding_value=pad_id
  )
  decoder_input_padded = pad_sequence(
      decoder_inputs, batch_first=True, padding_value=pad_id
  )
  decoder_output_padded = pad_sequence(
      decoder_outputs, batch_first=True, padding_value=pad_id
  )

  return encoder_padded, decoder_input_padded, decoder_output_padded

train_data_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=False, collate_fn=custom_collate_fn)
val_data_loader = DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False, collate_fn=custom_collate_fn)

def get_data_loaders():
    return {"train":train_data_loader,
            "val":val_data_loader}

# print(len(train_data_loader))
# print(len(val_data_loader))
# if __name__=="__main__":
#     # max_train_seq_len = -1
#     print(dataset['train'])
#     # # for d