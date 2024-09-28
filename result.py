def translate(src: str, model):
    """
    Translate an English sentence to Chinese.
    :param src: English sentence, e.g., "I like machine learning."
    :return: Translated sentence, e.g., "我喜欢机器学习"
    """
    # Tokenize the source sentence, convert to indices via the vocabulary, and add <bos> and <eos>
    src_indices = [0] + en_vocab(en_tokenizer(src)) + [1]
    src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
    # Initialize target with <bos>
    tgt = torch.tensor([[0]]).to(device)
    # Predict one word at a time until predicting <eos> or reaching max sentence length
    for i in range(max_length):
        # Perform transformer computation
        out = model(src_tensor, tgt)
        # Prediction result, we only need the last word, so take 'out[:, -1]'
        predict = model.predictor(out[:, -1])
        # Find the index of the maximum value
        y = torch.argmax(predict, dim=1)
        # Concatenate with previous prediction results
        tgt = torch.cat([tgt, y.unsqueeze(0)], dim=1)
        # If the predicted word is <eos>, end the prediction
        if y.item() == 1:
            break
    # Convert the predicted indices back to tokens and join them into a string
    tgt_tokens = cn_vocab.lookup_tokens(tgt.squeeze().tolist())
    translation = ''.join(tgt_tokens).replace("<s>", "").replace("</s>", "")
    return translation
    

with open('cn_test.txt', 'r') as file:
    lines = file.readlines()

lines = [line.strip() for line in lines]
cn_test = lines

# 打开文本文件
with open('en_test.txt', 'r') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]
en_test = lines

import random

# Combine two lists into tuple pairs and randomly sample 100 pairs
random_samples = random.sample(list(zip(en_test, cn_test)), 100)

# Unzip the sampled results into two lists
en_sample, cn_sample = zip(*random_samples)

# Test translation on the first sample (optional)
translate(en_sample[0], model)

def compare_results(en_sample, cn_sample, model):
    for en_text, cn_text in zip(en_sample, cn_sample):
        pred = translate(en_text, model)
        print("Field Name: " + en_text)
        print("Original Annotation: " + cn_text)
        print("Predicted Annotation: " + pred, end='\n\n')
        
compareresult(cn_sample,en_sample,model)
