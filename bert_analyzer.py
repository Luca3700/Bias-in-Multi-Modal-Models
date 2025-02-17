import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
import torch
from collections import defaultdict
from typing import Iterable, Dict, List

"""
code adapted from
https://github.com/keitakurita/contextual_embedding_bias_measure/blob/master/notebooks/bert_expose_bias_with_prior.py
"""

# Configuration class to store settings
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


# Define the configuration parameters
config = Config(
    model_type="bert-base-uncased",
    max_seq_len=128,
)


# Function to flatten a list of lists
def flatten(x: List[List]) -> List:
    return [item for sublist in x for item in sublist]


# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(config.model_type)
model = BertForMaskedLM.from_pretrained(config.model_type)
model.eval()


# Function to tokenize input using the Hugging Face tokenizer
def tokenize(text: str):
    encoded_input = tokenizer(
        text,
        add_special_tokens=True,
        max_length=config.max_seq_len,
        truncation=True,
        return_tensors='pt'
    )
    return encoded_input['input_ids']


# Function to get logits from the model
def get_logits(input_sentence: str) -> torch.Tensor:
    token_ids = tokenize(input_sentence)
    with torch.no_grad():
        out_logits = model(token_ids).logits.squeeze(0)
    return out_logits.cpu().numpy()


# Function to convert token indices to words
def indices_to_words(indices: Iterable[int]) -> List[str]:
    return [tokenizer.decode([x], skip_special_tokens=True) for x in indices]


# Function to calculate softmax probabilities
def softmax(x, axis=0, eps=1e-9):
    e = np.exp(x)
    return e / (e.sum(axis, keepdims=True) + eps)


# Measure the difference between male and female logits for a masked word
male_logits = get_logits("he is very [MASK].")[4, :]
female_logits = get_logits("she is very [MASK].")[4, :]

male_probs = softmax(male_logits)
female_probs = softmax(female_logits)

msk = ((male_probs >= 1e-6) & (female_probs >= 1e-6))
male_probs = male_probs[msk]
female_probs = female_probs[msk]

[(pos + 1, tokenizer.decode([i])) for i, pos in enumerate((male_probs / female_probs).argsort()) if pos < 10]
[(pos + 1, tokenizer.decode([i])) for i, pos in enumerate((female_probs / male_probs).argsort()) if pos < 10]


# Construct measure of bias

input_sentence = "[MASK] is intelligent"


def get_logits_multiple(input_sentence: str, n_calc: int = 10) -> np.ndarray:
    """
    n_calc: Since the logits are non-deterministic,
    computing the logits multiple times might be better
    """
    token_ids = tokenize(input_sentence)
    logits = None
    for _ in range(n_calc):
        with torch.no_grad():
            out_logits = model(token_ids).logits.squeeze(0)
        if logits is None:
            logits = np.zeros(out_logits.shape)
        logits += out_logits.cpu().numpy()
    return logits / n_calc


def get_logit_scores(input_sentence: str, wordSets: List[List[str]]) -> Dict[str, float]:
    out_logits = get_logits_multiple(input_sentence)
    token_ids = tokenize(input_sentence)
    positions = torch.where(token_ids[0] == tokenizer.mask_token_id)[0]
    positions = positions.numpy().tolist()
    scores = {}
    # compute scores
    for wordSet in wordSets:
        assert len(wordSet) == len(positions)
        final_probability = 1
        for i in range(len(wordSet)):
            final_probability *= out_logits[positions[i], tokenizer.convert_tokens_to_ids(wordSet[i])]

        total_str = " ".join(wordSet)
        scores[total_str] = final_probability

    return scores

def get_log_odds(input_sentence: str, wordSets: List[List[str]]):
    scores = get_logit_scores(input_sentence, wordSets)
    scores = np.array(list(scores.values()))
    return scores




"""
code adapted from
https://github.com/keitakurita/contextual_embedding_bias_measure/blob/master/lib/bert_utils.py
"""

class BertPreprocessor:
    def __init__(self, model_type: str, max_seq_len: int):
        self.tokenizer = BertTokenizer.from_pretrained(model_type)
        self.max_seq_len = max_seq_len

    def to_bert_model_input(self, sentence: str):
        inputs = self.tokenizer(sentence, return_tensors="pt", max_length=self.max_seq_len, truncation=True)
        return inputs["input_ids"], inputs["attention_mask"]

    def get_index(self, sentence: str, token: str, last=False) -> int:
        tokens = self.tokenizer.tokenize(sentence)
        if last:
            tokens = tokens[::-1]
        try:
            index = tokens.index(token)
            if last:
                return len(tokens) - index - 1
            return index
        except ValueError:
            return -1

    def token_to_index(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

config = Config(
    model_type="bert-base-uncased",
    max_seq_len=128,
)

processor = BertPreprocessor(config.model_type, config.max_seq_len)

model = BertForMaskedLM.from_pretrained(config.model_type)
model.eval()

def get_logits(sentence: str) -> np.ndarray:
    input_ids, attention_mask = processor.to_bert_model_input(sentence)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits[0, :, :].cpu().numpy()

def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)

def get_mask_fill_logits(sentence: str, words: Iterable[str], use_last_mask=False, apply_softmax=False) -> Dict[str, float]:
    mask_i = processor.get_index(sentence, "[MASK]", last=use_last_mask)
    logits = defaultdict(list)
    out_logits = get_logits(sentence)
    if apply_softmax:
        out_logits = softmax(out_logits)
    return {w: out_logits[mask_i, processor.token_to_index(w)] for w in words}

def bias_score(sentence: str, gender_words: Iterable[str], word: str, gender_comes_first=True) -> Dict[str, float]:
    mw, fw = gender_words
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"),
        gender_words, use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]

    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"),
        gender_words, use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction = subject_fill_prior_logits[mw] - subject_fill_prior_logits[fw]

    try:
        mw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", mw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        fw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", fw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
    except:
        tgt_fill_bias = np.nan

    return {
        "gender_fill_bias": subject_fill_bias,
        "gender_fill_prior_correction": subject_fill_bias_prior_correction,
        "gender_fill_bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
        "target_fill_bias": tgt_fill_bias,
    }




"""
code adapted from
https://github.com/keitakurita/contextual_embedding_bias_measure/blob/master/notebooks/Exposing_Bias_BERT.ipynb
"""

import matplotlib.pyplot as plt

def Txt2List(file):
    ll=[]
    with open(file) as f:
        for line in f:
            ll.append(line.strip().lower())
    return ll

def plot_pie(file, sizes, labels):
    fig = plt.figure()
    plt.pie(sizes, labels=labels,
    autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    fig.savefig(file)
    plt.show()

def list2Bias_norm(plotfile, var_list, abs_str, print_str): #Example of abs_str is "good at ", print_str is "is good at "
    mc=0
    fc=0
    for var in var_list:
        strr = abs_str+ var
        ans = bias_score("GGG is XXX", ["he", "she"], strr)
        score= ans['gender_fill_bias_prior_corrected']

        if score>=0:
            mc+=1
            print("Man ",print_str,  var, " by ", score)

        else:
            fc+=1
            print("Woman ",print_str,  var, " by ", score)



    plot_pie(plotfile, mc, fc)


def list2Bias(plotfile, var_list, abs_str, typology, verbose=True): #Example of abs_str "is good at "

    res1 = [0, 0, 0, 0]
    res2 = [0, 0]
    res3 = [0, 0]

    if verbose:
        print("The margin is computed w.r.t. the second biggest score")

    for var in var_list:

        if typology == 1:
            wordsSet1 = ["white", "man"]
            wordsSet2 = ["white", "woman"]
            wordsSet3 = ["black", "man"]
            wordsSet4 = ["black", "woman"]
            wordSets = [wordsSet1, wordsSet2, wordsSet3, wordsSet4]
            scores = get_log_odds("A [MASK] skin [MASK] %s%s"%(abs_str,var), wordSets)

        elif typology == 2:
            wordSets = [["he"], ["she"]]
            scores = get_log_odds("[MASK] %s%s"%(abs_str,var), wordSets)

        elif typology == 3:
            wordSets = [["white"], ["black"]]
            scores = get_log_odds("A [MASK] skin person %s%s"%(abs_str,var), wordSets)

        else:
            print("Typology not defined")
            return

        index_biggest = np.argmax(scores)

        sorted_arr = np.sort(scores, axis=None)[::-1]
        # Get the biggest and second biggest elements
        biggest = sorted_arr[0]
        second_biggest = sorted_arr[1]
        margin = biggest - second_biggest

        if typology == 1:
            res1[index_biggest] += 1
        elif typology == 2:
            res2[index_biggest] += 1
        elif typology == 3:
            res3[index_biggest] += 1

        if verbose:
            if typology == 1:
                print(f'A {wordSets[index_biggest][0]} skin {wordSets[index_biggest][1]}',abs_str,  var, ' by ', margin)
            elif typology == 2:
                print(f'{wordSets[index_biggest][0]}',abs_str,  var, ' by ', margin)
            elif typology == 3:
                print(f'A {wordSets[index_biggest][0]} skin person',abs_str,  var, ' by ', margin)

    labels = [" ".join(words) for words in wordSets]
    if typology == 1:
        plot_pie(plotfile, res1, labels)
    elif typology == 2:
        plot_pie(plotfile, res2, labels)
    elif typology == 3:
        plot_pie(plotfile, res3, labels)
