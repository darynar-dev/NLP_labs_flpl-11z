import re
import nltk
import spacy
import stanza
from pathlib import Path


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter


def preprocess(raw):
    t = re.sub(r'\s+', ' ', raw)   # згорнути пробіли й переноси
    t = re.sub(r"[’ʼ`]", "'", t) # нормалізація апострофа
    return t.strip()


directory = Path(__file__).resolve().parent
with open(directory/"text.txt", "r", encoding="utf-8") as f:
    loaded_text = f.read()

clean = preprocess(loaded_text)



def run_regex_search():
    numbers = re.findall(r'\d+[.,]\d+|\d+', clean)
    apostrophe_words = re.findall(r"[а-яА-ЯіїєґІЇЄҐ]+'[а-яА-ЯіїєґІЇЄҐ]+", clean)
    hyphen_words = re.findall(r"[а-яА-ЯіїєґІЇЄҐa-zA-Z0-9]+-[а-яА-ЯіїєґІЇЄҐa-zA-Z0-9]+", clean)

    print("Числа:", numbers)
    # print("Кількість чисел:", len(numbers))
    print("\nСлова з апострофом:", apostrophe_words)
    print("\nСлова з дефісом:", hyphen_words)
    print("\n------------------------\n")


def my_tokenize(text):
    pattern = r"[а-яА-ЯіїєґІЇЄҐ'\-]+|\d+[.,]?\d*|[.,!?;:—»«()]"
    return re.findall(pattern, text)


def run_my_tokenizer():
    tokens = my_tokenize(clean)
    print(f"Власний regex-токенізатор — кількість токенів: {len(tokens)}")
    print("------------------------")



def compare_tokenizers():
    nltk_sents = sent_tokenize(clean)
    nltk_words = word_tokenize(clean)

    nlp_spacy = spacy.load('uk_core_news_sm')
    doc_spacy = nlp_spacy(clean)
    spacy_sents = list(doc_spacy.sents)
    spacy_words = [t.text for t in nlp_spacy(clean)]
    

    nlp_stanza = stanza.Pipeline('uk', processors='tokenize')
    doc_stanza = nlp_stanza(clean)
    stanza_words = []
    for sent in doc_stanza.sentences:
        for w in sent.words:
            stanza_words.append(w.text)

    print(f"NLTK:   речень = {len(nltk_sents)}, токенів = {len(nltk_words)}")
    print(f"spaCy:  речень = {len(spacy_sents)}, токенів = {len(spacy_words)}")
    print(f"Stanza: речень = {len(doc_stanza.sentences)}, токенів = {len(stanza_words)}")
    

    diff = (set(nltk_words) | set(spacy_words) | set(stanza_words)) - (set(nltk_words) & set(spacy_words) & set(stanza_words))

    print("Є тільки в NLTK:  ", set(nltk_words) - set(spacy_words) - set(stanza_words))
    print("Є тільки в spaCy: ", set(spacy_words) - set(nltk_words) - set(stanza_words))
    print("Є тільки в Stanza:", set(stanza_words) - set(nltk_words) - set(spacy_words))

    print(diff)
    print("------------------------")



def run_bpe_simulation():
    corpus = [
        'г р а в і т а ц і я </w>',
        'г р а в і т а ц і й н е </w>',
        'г р а в і т а ц і й н о г о </w>'
    ]

    for step in range(3):
        pairs = Counter()
        for word in corpus:
            s = word.split()
            for i in range(len(s)-1):
                pair = (s[i], s[i+1])
                if pair in pairs.keys():
                    pairs[pair] += 1
                else:
                    pairs[pair] = 1

        best_pair = pairs.most_common(1)[0][0]
        print(f"Крок {step}: найчастіша пара {best_pair} (частота {pairs[best_pair]}) -> об'єднати в '{best_pair[0] + best_pair[1]}'")

        new_corpus = []
        old_str = best_pair[0] + " " + best_pair[1]
        new_str = best_pair[0] + best_pair[1]
        for word in corpus:
            new_corpus.append(word.replace(old_str, new_str))
        corpus = new_corpus

    print("Результат після 3 кроків:", corpus)


run_regex_search()
run_my_tokenizer()
compare_tokenizers()
run_bpe_simulation()