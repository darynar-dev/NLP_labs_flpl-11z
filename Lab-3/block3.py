import os
import re
from collections import Counter, defaultdict


def parse_file_with_metadata(raw):
    # Extract genre from YAML-like header and return (clean_text, genre)
    genre = "unknown"
    parts = raw.split("---")
    if len(parts) >= 3:
        header = parts[1]
        body = "---".join(parts[2:])
        match = re.search(r"genre:\s*(\S+)", header)
        if match:
            genre = match.group(1).strip().lower()
    else:
        body = raw

    clean_text = re.sub(r"\s+", " ", body)
    clean_text = re.sub(r"[’ʼ`]", "'", clean_text).strip()
    return clean_text, genre


class Corpus:
    def __init__(self):
        self.docs = []

    def add(self, text, genre="unknown", filename=""):
        self.docs.append({"text": text, "genre": genre, "filename": filename})

    def tokens(self):
        toks = []
        for d in self.docs:
            toks += re.findall(r"[а-яА-ЯіїєґІЇЄҐ']+", d["text"].lower())
        return toks

    def stats(self):
        toks = self.tokens()
        types = set(toks)
        ttr = len(types) / len(toks) if toks else 0
        return {
            "docs": len(self.docs),
            "tokens": len(toks),
            "types": len(types),
            "ttr": round(ttr, 3)
        }

    def frequency(self, n=15):
        return Counter(self.tokens()).most_common(n)

    def kwic(self, keyword, window=3):
        print(f"KWIC for '{keyword}':")
        found = 0
        for d in self.docs:
            words = d["text"].split()
            for i, w in enumerate(words):
                if keyword.lower() in w.lower():
                    left = " ".join(words[max(0, i - window):i])
                    right = " ".join(words[i + 1:i + 1 + window])
                    print(f"  [{d['genre']:7} | {d['filename']}] ...{left} [{w}] {right}...")
                    found += 1
        if found == 0:
            print("  No matches found.")
        print("------------------------")


def load_corpus_from_folder(folder="data"):
    corpus = Corpus()
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text, genre = parse_file_with_metadata(f.read())
            corpus.add(text, genre=genre, filename=filename)
    return corpus


def compare_genre_ttr(corpus):
    genre_tokens = defaultdict(list)
    for d in corpus.docs:
        toks = re.findall(r"[а-яА-ЯіїєґІЇЄҐ']+", d["text"].lower())
        genre_tokens[d["genre"]] += toks

    print("Genre TTR comparison:")
    for genre, toks in genre_tokens.items():
        types = set(toks)
        ttr = len(types) / len(toks) if toks else 0
        print(f"  {genre:10}: TTR = {ttr:.3f} (tokens={len(toks)}, types={len(types)})")
    print("------------------------")


corpus = load_corpus_from_folder("data")

print("General corpus stats:")
print(corpus.stats())
print("------------------------")

print("Top-15 frequent words (Zipf's law check):")
for rank, (word, freq) in enumerate(corpus.frequency(15), 1):
    print(f"  {rank:2}. {word:15} : {freq}")
print("------------------------")

corpus.kwic("мозк", window=3)
corpus.kwic("компані", window=3)

compare_genre_ttr(corpus)