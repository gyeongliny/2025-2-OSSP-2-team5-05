import os
import re
import argparse
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

from transformers import AutoTokenizer, AutoModel
import torch


SECTION_TITLES = re.compile(
    r'('
    r'ABDOMEN AND PELVIS|CLINICAL HISTORY|CLINICAL INDICATION|COMPARISON|COMPARISON STUDY DATE'
    r'|EXAM|EXAMINATION|FINDINGS|HISTORY|IMPRESSION|INDICATION'
    r'|MEDICAL CONDITION|PROCEDURE|REASON FOR EXAM|REASON FOR STUDY|REASON FOR THIS EXAMINATION'
    r'|TECHNIQUE'
    r'):|FINAL REPORT',
    re.I | re.M,
)


def pattern_repl(matchobj):
    return " ".rjust(len(matchobj.group(0)))


def find_end(text: str) -> int:
    ends = [len(text)]
    patterns = [
        re.compile(r"BY ELECTRONICALLY SIGNING THIS REPORT", re.I),
        re.compile(r"\n {3,}DR\.", re.I),
        re.compile(r"[ ]{1,}RADLINE ", re.I),
        re.compile(r".*electronically signed on", re.I),
        re.compile(r"M\[0KM\[0KM"),
    ]
    for pattern in patterns:
        matchobj = pattern.search(text)
        if matchobj:
            ends.append(matchobj.start())
    return min(ends)


def split_heading(text: str):
    start = 0
    for matcher in SECTION_TITLES.finditer(text):
        end = matcher.start()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end
        end = matcher.end()
        if end != start:
            section = text[start:end].strip()
            if section:
                yield section

        start = end

    end = len(text)
    if start < end:
        section = text[start:end].strip()
        if section:
            yield section


def clean_text(text: str) -> str:
    text = re.sub(r"\[\*\*.*?\*\*\]", pattern_repl, text)
    text = re.sub(r"_", " ", text)

    start = 0
    end = find_end(text)
    new_text = ""
    new_text = text[start:end]
    if len(text) - end > 0:
        new_text += " " * (len(text) - end)
    return new_text


stemmer = WordNetLemmatizer()


def preprocess_mimic(text: str):
    for sec in split_heading(clean_text(text)):
        for sent in sent_tokenize(sec):
            sent = re.sub(
                r"\[\*\*(.*?)\*\*\]|[_\,\d\*:~=\.\-\+\\/\"\'^&]+", " ", sent
            )
            text = " ".join(
                [stemmer.lemmatize(word) for word in word_tokenize(sent)]
            )
            yield text.lower()


def getText(t):
    if not isinstance(t, str):
        if pd.isna(t):
            t = ""
        else:
            t = str(t)
    return "\n".join(list(preprocess_mimic(t)))


def clean_docs(args):
    print("1. Creating vocab on whole corpus...")
    word_freq = Counter()

    for split in ["test", "train"]:
        partition = split + "_note"
        note_dir = os.path.join(args.raw_path, args.task, partition)
        if not os.path.isdir(note_dir):
            print(f"[WARN] note_dir not found: {note_dir}")
            continue

        patients = [
            f for f in os.listdir(note_dir)
            if "episode" in f and f.endswith(".csv")
        ]

        for patient in tqdm(patients, desc=f"Iterating {split}"):
            p_df = pd.read_csv(
                os.path.join(note_dir, patient),
                sep="\t",
                header=0,
                engine="python",
            )
            col = "TEXT" if "TEXT" in p_df.columns else args.clean_column
            notes = p_df[col].tolist()

            for note in notes:
                fixed_note = getText(note)
                for word in fixed_note.split(" "):
                    if len(word) == 0:
                        continue
                    word_freq[word] += 1

    if len(word_freq) == 0:
        raise ValueError("word_freq is empty. Check input data.")

    highbar = word_freq.most_common(args.most_common)[-1][1]
    print(f"  > filter word freq in [{args.min_freq} ~ {highbar}]")

    print(f'2. Cleaning notes & saving to "{args.clean_column}"')
    all_sents = []

    for split in ["test", "train"]:
        partition = split + "_note"
        note_dir = os.path.join(args.raw_path, args.task, partition)
        if not os.path.isdir(note_dir):
            continue

        patients = [
            f for f in os.listdir(note_dir)
            if "episode" in f and f.endswith(".csv")
        ]

        for patient in tqdm(patients, desc=f"Cleaning {split}"):
            p_df = pd.read_csv(
                os.path.join(note_dir, patient),
                sep="\t",
                engine="python",
                header=0,
            )

            notes = p_df["TEXT"].tolist()
            notes_ = []

            for note in notes:
                note_ = []
                sentences = getText(note).split("\n")
                for sent in sentences:
                    sent_ = []
                    for word in sent.split(" "):
                        if len(word) == 0:
                            continue
                        if (
                            word_freq[word] >= args.min_freq
                            and word_freq[word] < highbar
                        ):
                            sent_.append(word)
                    if len(sent_) > 0:
                        all_sents.append(sent_)
                        note_.append(" ".join(sent_))
                notes_.append("\n".join(note_))

            p_df["fixed TEXT"] = notes_

            p_df.to_csv(
                os.path.join(note_dir, patient),
                sep="\t",
                index=False,
            )

    print("3. Building token-level embeddings (GatorTronS backend)")
    vocab_words = [
        w
        for w, c in word_freq.items()
        if (c >= args.min_freq and c < highbar and len(w) > 0)
    ]
    vocab_words = sorted(set(vocab_words))

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.hf_model_name)
    model.to(device)
    model.eval()

    emb_dict = {}
    with torch.no_grad():
        for w in tqdm(vocab_words, desc="   > HF token embeddings"):
            tokens = tokenizer(w, return_tensors="pt", add_special_tokens=False)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            if tokens["input_ids"].shape[1] == 0:
                continue

            outputs = model(**tokens)
            vec = (
                outputs.last_hidden_state.mean(dim=1)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype("float32")
            )
            emb_dict[w] = vec

    np.save(args.token_embedding_path + ".npy", emb_dict, allow_pickle=True)
    with open(args.vocab_output_path, "w") as f:
        f.write("\n".join(vocab_words))

    print("Complete!")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DEFAULT_RAW_PATH = os.path.join(DATA_DIR, "DATA_RAW")
    DEFAULT_PRE_PATH = os.path.join(DATA_DIR, "DATA_PRE")

    parser = argparse.ArgumentParser(description="Extract cleaned notes.")
    parser.add_argument("--raw_path", type=str, default=DEFAULT_RAW_PATH)
    parser.add_argument("--task", type=str, default="in-hospital-mortality")

    # 🔥 기본값을 GatorTronS로 변경
    parser.add_argument("--tokenizer", type=str, default="gatortron")
    parser.add_argument("--dimension", type=str, default="1024")
    parser.add_argument("--hf_model_name", type=str, default="UFNLP/gatortronS")

    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--pre_path", type=str, default=DEFAULT_PRE_PATH)

    parser.add_argument("--most_common", type=int, default=10)
    parser.add_argument("--min_freq", type=int, default=10)

    args, _ = parser.parse_known_args()

    os.makedirs(os.path.join(args.raw_path, "root"), exist_ok=True)
    os.makedirs(os.path.join(args.raw_path, args.task), exist_ok=True)
    os.makedirs(args.pre_path, exist_ok=True)

    args.vocab_output_path = os.path.join(args.raw_path, "root", "vocab.txt")
    args.token_embedding_path = os.path.join(
        args.raw_path, "root", f"{args.tokenizer}_{args.dimension}"
    )

    args.clean_column = "fixed TEXT"

    clean_docs(args)
