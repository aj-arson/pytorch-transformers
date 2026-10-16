import sentencepiece as spe
"""
<pad> - 0
<unk> - 1
<s> - 2
</s> - 3
"""

class Tokenizer:
    def __init__(self, model_path:str):
        self.model_path = model_path if model_path.endswith(".model") else f"{model_path}.model"
        self.loaded_model = spe.SentencePieceProcessor(model_file=self.model_path)
        self.pad_token_id = self.loaded_model.pad_id()
        self.bos_token_id = self.loaded_model.bos_id()
        self.eos_token_id = self.loaded_model.eos_id()
        self.vocab_size = self.loaded_model.get_piece_size()

    def encode(self, text):
        return self.loaded_model.encode(text)

    def decode(self, ids):
            return self.loaded_model.decode(ids)

    def get_special_tokens(self):
         return {"sos":self.loaded_model.id_to_piece(2),
                 "eos":self.loaded_model.id_to_piece(3),
                 "pad":self.loaded_model.id_to_piece(0)}
    
    def get_special_token_ids(self):
             return {"sos":self.bos_token_id,
                     "eos":self.eos_token_id,
                     "pad":self.pad_token_id}


tokenizer = Tokenizer(r"de_en_tokenizer.model")

if __name__ == "__main__":
    print(tokenizer.loaded_model.id_to_piece(0))
    print(tokenizer.loaded_model.id_to_piece(1))
    print(tokenizer.loaded_model.id_to_piece(2))
    print(tokenizer.loaded_model.id_to_piece(3))
    print(tokenizer.vocab_size)
    print(tokenizer.pad_token_id)
    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)