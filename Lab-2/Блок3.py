import re
import stanza
import spacy

from collections import defaultdict


def preprocess(raw):
    t = re.sub(r'\s+', ' ', raw)
    t = re.sub(r"[’ʼ`]", "'", t)
    return t.strip()


with open("text.txt", "r", encoding="utf-8") as f:
    loaded_text = f.read()

clean = preprocess(loaded_text)



def compute_metrics(gold, pred, target_class):
    tp = sum(1 for g,p in zip(gold,pred)
             if g==target_class and p==target_class) # true positive
    
    fp = sum(1 for g,p in zip(gold,pred)
             if g!=target_class and p==target_class) # false positive
    
    fn = sum(1 for g,p in zip(gold,pred)
             if g==target_class and p!=target_class) # false negative
    
    precision = tp/(tp+fp) if (tp+fp) else 0 # точність
    recall = tp/(tp+fn) if (tp+fn) else 0 # повнота
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0 # гармонічне середнє

    return precision, recall, f1


def morph_amalysis():
    nlp_st = stanza.Pipeline('uk', processors='tokenize,pos')
    nlp_sp = spacy.load('uk_core_news_sm')

    doc_st = nlp_st(clean)
    doc_sp = nlp_sp(clean)

    words_st = []
    for sent in doc_st.sentences:
        for w in sent.words:
            words_st.append((w.start_char, w.text, w.upos))

    words_sp = []
    for w in doc_sp:
        words_sp.append((w.idx, w.text, w.pos_))

    return words_st, words_sp


def compare_st_vs_sp(words_st, words_sp):
    gold_tags = []   # Stanza як умовний еталон
    pred_tags = []   # spaCy
    disagreements = []

    sp_tags = {(w[0], w[1]): w[2] for w in words_sp} # (idx, text): pos

    for w in words_st:
        key = (w[0], w[1]) # inx, text

        if key not in sp_tags: # spaCy розбив слово інакше
            continue

        st_pos = w[2] # pos
        sp_pos = sp_tags[key]

        gold_tags.append(st_pos)
        pred_tags.append(sp_pos)

        if st_pos != sp_pos:
            disagreements.append((w[1], st_pos, sp_pos))
                
    print(f"Порівняно спільних токенів: {len(gold_tags)}")
    print(f"Знайдено розбіжностей: {len(disagreements)}\n")

    print("Усі розбіжності (Слово | Stanza | spaCy)")
    for idx, (word, st_pos, sp_pos) in enumerate(disagreements, 1):
        print(f"{idx:2}. {word:20} | Stanza: {st_pos:6} | spaCy: {sp_pos:6}")

    return gold_tags, pred_tags   


def calculate_sp_metrics_against_st(gold_tags, pred_tags):
    print("\nМетрики spaCy відносно Stanza")
    for tag in ['NOUN', 'VERB', 'ADJ']:
        p, r, f1 = compute_metrics(gold_tags, pred_tags, tag)
        print(f"{tag:5} -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")


def build_error_matrix(gold, pred):
    print("\nМатриця помилок")

    m = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, pred):
        m[g][p] += 1

    labels = sorted(set(gold) | set(pred))
    print(f"{'gold \\ pred':12}", end="")
    for p in labels:
        print(f"{p:>6}", end="")
    print()

    for g in labels:
        print(f"{g:12}", end="")
        for p in labels:
            print(f"{m[g][p]:>6}", end="")
        print()



words_st, words_sp = morph_amalysis()
gold_tags, pred_tags = compare_st_vs_sp(words_st, words_sp)
calculate_sp_metrics_against_st(gold_tags, pred_tags)
build_error_matrix(gold_tags, pred_tags)