import datasets

if __name__=="__main__":
    corpus_path = r"tokenizer\corpus.txt"
    dataset = datasets.load_dataset( "wmt/wmt14",
        "de-en", cache_dir=r"data\cache", num_proc=4)

    with open(corpus_path, "w", encoding="utf-8") as f:
        for example in dataset['train']:
            translation = example["translation"]
            f.write(translation["en"].replace("\n", " ") + "\n")
            f.write(translation["de"].replace("\n", " ") + "\n")