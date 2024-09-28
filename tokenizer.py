import os
from pathlib import Path
import torch
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer
from tqdm import tqdm

# Load the base tokenizer model using the base BERT model.
# 'uncased' means case-insensitive.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def en_tokenizer(line):
    """
    Define the English tokenizer to be used later.
    :param line: An English sentence, e.g., "I'm learning Deep learning."
    :return: Result of subword tokenization, e.g., ['i', "'", 'm', 'learning', 'deep', 'learning', '.']
    """
    # Use BERT for tokenization and get tokens.
    # add_special_tokens=False means not to add special tokens like '<bos>' and '<eos>' in the result.
    return tokenizer.tokenize(line)

def yield_en_tokens():
    """
    Yield one tokenized English sentence at a time to save memory.
    If all sentences are tokenized at once to build the vocabulary,
    it would cause a large amount of text to reside in memory, leading to memory overflow.
    """
    with open(en_filepath, encoding='utf-8') as file:
        print("-------Starting to build English vocabulary-----------")
        for line in tqdm(file, desc="Building English Vocabulary", total=row_count):
            yield en_tokenizer(line)

# Specify the English vocabulary cache file path
en_vocab_file = work_dir / "vocab_en.pt"

# Load the cached vocabulary if it exists and use_cache is True
if use_cache and en_vocab_file.exists():
    en_vocab = torch.load(en_vocab_file, map_location="cpu")
# Otherwise, build the vocabulary from scratch
else:
    # Build the vocabulary
    en_vocab = build_vocab_from_iterator(
        # Pass an iterable of token lists, e.g., [['i', 'am', ...], ['machine', 'learning', ...], ...]
        yield_en_tokens(),
        # Minimum frequency is 2; a word must appear at least twice to be included in the vocabulary
        min_freq=2,
        # Add these special tokens at the beginning of the vocabulary
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    # Set the default index; if a token is not found in the vocabulary during text-to-index conversion,
    # it will be replaced with this index
    en_vocab.set_default_index(en_vocab["<unk>"])
    # Save the cached vocabulary
    if use_cache:
        torch.save(en_vocab, en_vocab_file)

def zh_tokenizer(line):
    """
    Define the Chinese tokenizer
    :param line: A Chinese sentence, e.g., "机器学习" (Machine Learning)
    :return: Tokenization result, e.g., ['机', '器', '学', '习']
    """
    return list(line.strip().replace(" ", ""))

def yield_zh_tokens():
    """
    Yield one tokenized Chinese sentence at a time to save memory.
    """
    with open(cn_filepath, encoding='utf-8') as file:
        for line in tqdm(file, desc="Building Chinese Vocabulary", total=row_count):
            yield zh_tokenizer(line)

# Specify the Chinese vocabulary cache file path
zh_vocab_file = work_dir / "vocab_zh.pt"

if use_cache and zh_vocab_file.exists():
    zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
else:
    zh_vocab = build_vocab_from_iterator(
        yield_zh_tokens(),
        min_freq=1,
        specials=["<s>", "</s>", "<pad>", "<unk>"],
    )
    zh_vocab.set_default_index(zh_vocab["<unk>"])
    torch.save(zh_vocab, zh_vocab_file)
