import re
import stanza
import nltk
import spacy
nltk.download('punkt')
nltk.download('punkt_tab')
from pathlib import Path

from nltk.tokenize import word_tokenize, sent_tokenize


def preprocess(raw):
    t = re.sub(r'\s+', ' ', raw)   # згорнути пробіли й переноси
    return t.strip()

directory = Path(__file__).resolve().parent
with open(directory/"text.txt", "r", encoding="utf-8") as f:
    loaded_text = f.read()

clean = preprocess(loaded_text)


# Наївний метод

def split_sentences(t):
    parts = re.split(r'(?<=[.!?])\s+', t)   # ділити після . ! ?
    return [s for s in parts if s]

def run_naive_method():
    sents = split_sentences(clean)
    tokens = clean.split()
    print(f'Речень (наївний метод): {len(sents)}')
    print(f'Токенів (наївний метод): {len(tokens)}')
    for i, s in enumerate(sents, 1):
        print(f'  {i}. {s}')
    print('\nСписок токенів (наївний метод):')    
    print(tokens)    



# stanza

def run_stanza_with_freq():
    nlp = stanza.Pipeline('uk', processors='tokenize,pos,lemma,depparse')
    doc = nlp(clean)

    total_tokens = sum(len(s.words) for s in doc.sentences)
   
    print(f'Речень (Stanza): {len(doc.sentences)}\n')
    print(f'Токенів (Stanza): {total_tokens}\n')

    parts_of_speech_freq = {} # frequency

    for sent in doc.sentences:  
        for w in sent.words:
            if w.upos in parts_of_speech_freq.keys():
                parts_of_speech_freq[w.upos] =  parts_of_speech_freq[w.upos] + 1
            else:
                parts_of_speech_freq[w.upos] = 1

    print(parts_of_speech_freq)
    print("------------------------")


def compare_complex_words():
    complex_words = ["ударно-розвідувальний", "радіокерований", "FPV-дрони", \
                     "багаторазовий", "БПЛА"] 
    
    nltk_words = word_tokenize(clean)

    nlp_stanza = stanza.Pipeline('uk', processors='tokenize,pos,lemma,depparse')
    doc_stanza = nlp_stanza(clean)
    tokens_stanza = [word for sent in doc_stanza.sentences for word in sent.words]

    nlp_spacy = spacy.load('uk_core_news_sm')
    doc_spacy = nlp_spacy(clean)

    print(f"Речень (NLTK): {len(sent_tokenize(clean))}")
    print(f"Речень (spaCy): {len(list(doc_spacy.sents))}")
    print(f"Речень (Stanza): {len(doc_stanza.sentences)}\n")  
  
    for target in complex_words:

        parts = target.split('-')
        
        nltk_found = [w for w in nltk_words if w == target or w in parts]
        stanza_found = [f"{w.text} | лема: {w.lemma} | POS: {w.upos}" for w in tokens_stanza if w.text == target or w.text in parts]
        spacy_found = [f"{w.text} | лема: {w.lemma_} | POS: {w.pos_}" for w in doc_spacy if w.text == target or w.text in parts]

        #stanza_found = [w.text for w in tokens_stanza  if w.text == target or w.text in parts]
        #spacy_found = [w for w in doc_spacy if w.text == target or w.text in parts]

        print(f"Target word: {target}\nNLTK: {nltk_found}\nStanza: {stanza_found}\nSpacy: {spacy_found}\n\n")





run_naive_method()
run_stanza_with_freq()
compare_complex_words()
