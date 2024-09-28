class TranslationDataset(Dataset):
    def __init__(self):
        # Load English tokens
        self.en_tokens = self.load_tokens(en_filepath, en_tokenizer, en_vocab, "Building English tokens", 'en')
        # Load Chinese tokens
        self.cn_tokens = self.load_tokens(cn_filepath, cn_tokenizer, cn_vocab, "Building Chinese tokens", 'cn')

    def __getitem__(self, index):
        return self.en_tokens[index], self.cn_tokens[index]

    def __len__(self):
        return row_count

    def load_tokens(self, file_path, tokenizer, vocab, desc, lang):
        """
        Load tokens by converting text sentences into indices.
        :param file_path: File path, e.g., "./dataset/train.en"
        :param tokenizer: Tokenizer function, e.g., en_tokenizer
        :param vocab: Vocabulary object, e.g., en_vocab
        :param desc: Description for progress display, e.g., "Building English tokens"
        :param lang: Language code used to differentiate cache files, e.g., 'en'
        :return: Returns the constructed tokens, e.g., [[6, 8, 93, 12, ...], [62, 891, ...], ...]
        """
        # Define the cache file storage path
        cache_file = work_dir / f"tokens_list.{lang}.pt"
        # If using cache and cache file exists, load it directly
        if use_cache and cache_file.exists():
            print(f"Loading cache file {cache_file}, please wait...")
            return torch.load(cache_file, map_location="cpu")

        # Start building from scratch; define tokens_list to store results
        tokens_list = []
        # Open the file
        with open(file_path, encoding='utf-8') as file:
            # Read line by line
            for line in tqdm(file, desc=desc, total=row_count):
                # Perform tokenization
                tokens = tokenizer(line)
                # Convert tokenized text into indices using the vocabulary
                tokens = vocab(tokens)
                # Append to the result list
                tokens_list.append(tokens)
        # Save the cache file
        if use_cache:
            torch.save(tokens_list, cache_file)

        return tokens_list

def collate_fn(batch):
    """
    Further process the dataset's data and form a batch.
    :param batch: A batch of data, e.g.,
                  [([6, 8, 93, 12, ...], [62, 891, ...]),
                   ...]
    :return: Padded and equal-length data, including src, tgt, tgt_y, n_tokens
             where src is the source sentence (to be translated),
             tgt is the target sentence (translated sentence without the last token),
             tgt_y is the label (translated sentence without the first token, i.e., <bos>),
             n_tokens is the number of tokens in tgt_y, excluding <pad>.
    """
    # Define the index of '<bos>' (beginning of sentence), which is 0 in the vocabulary
    bos_id = torch.tensor([0])
    # Define the index of '<eos>' (end of sentence)
    eos_id = torch.tensor([1])
    # Define the index of '<pad>'
    pad_id = 2

    # Lists to store processed src and tgt
    src_list, tgt_list = [], []

    # Iterate over the sentence pairs
    for (_src, _tgt) in batch:
        """
        _src: English sentence indices, e.g., corresponding to 'I love you'
        _tgt: Chinese sentence indices, e.g., corresponding to '我 爱 你'
        """
        # Concatenate <bos>, sentence indices, and <eos>
        processed_src = torch.cat(
            [
                bos_id,
                torch.tensor(_src, dtype=torch.int64),
                eos_id,
            ],
            dim=0,
        )
        processed_tgt = torch.cat(
            [
                bos_id,
                torch.tensor(_tgt, dtype=torch.int64),
                eos_id,
            ],
            dim=0,
        )

        """
        Pad sentences shorter than max_length and add them to the list.

        For example, if processed_src is [0, 1136, 2468, 1349, 1],
        the second parameter is: (0, max_length - 5),
        the third parameter is: 2 (pad_id).
        The pad function pads 0 times 2 on the left and (max_length - len(processed_src)) times 2 on the right.
        The final result is: [0, 1136, 2468, 1349, 1, 2, 2, 2, ..., 2]
        """
        src_list.append(
            pad(
                processed_src,
                (0, max_length - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_length - len(processed_tgt)),
                value=pad_id,
            )
        )

    # Stack multiple src and tgt sentences together
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)

    # tgt_y is the target sentence without the first token (<bos>)
    tgt_y = tgt[:, 1:]
    # tgt is the target sentence without the last token (<eos>)
    tgt = tgt[:, :-1]

    # Calculate the number of tokens to predict in this batch
    n_tokens = (tgt_y != pad_id).sum()

    # Return the batched results
    return src, tgt, tgt_y, n_tokens

# Create the dataset and data loader
dataset = TranslationDataset()
train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# Get one batch of data
src, tgt, tgt_y, n_tokens = next(iter(train_loader))
src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
