import sentencepiece as spe

class TokenizerTrainer:
    def __init__(self, text_file_path, model_prefix, vocab_size):
        self.text_file_path = text_file_path
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size

    def train(self):
        spe.SentencePieceTrainer.train(input=self.text_file_path,
                                       model_prefix = self.model_prefix,
                                       vocab_size = self.vocab_size,
                                       model_type="bpe",
                                       character_coverage=0.9995,
                                       pad_id=0,
                                       unk_id=1,
                                       bos_id=2,
                                       eos_id=3,
                                       hard_vocab_limit=False,)
        print("Tokenizer is ready!")

if __name__ == "__main__":
    file_path = r"tokenizer\corpus.txt"
    tokenizer_trainer = TokenizerTrainer(file_path, "de_en_tokenizer", 32_000)
    tokenizer_trainer.train()

    