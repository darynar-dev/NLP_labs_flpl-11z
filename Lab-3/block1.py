import re
import os
import stanza
import spacy
from spacy import displacy
from collections import Counter


def strip_metadata(raw):
    # Remove YAML-like metadata block between --- and ---
    parts = raw.split('---')
    if len(parts) >= 3:
        return '---'.join(parts[2:])
    return raw


def preprocess(raw):
    t = strip_metadata(raw)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r"[’ʼ`]", "'", t)
    return t.strip()


# --------

files = [
    "data/science_03_language_models_aphasia.txt",
    # "data/news_02_tesla_roadster.txt" - якщо прибираєш це, заміняєш відкриття файла
    # на то як робили в попередніх лабах
]

texts = []
for filename in files:
    with open(filename, "r", encoding="utf-8") as f:
        texts.append(preprocess(f.read()))

clean = " ".join(texts)

# ------переглянути чи потрібні метадані в тексті та чи вистачає 10 речень в одному тексті


class DepTree:
    def __init__(self, words):
        # words: list of tuples (id, text, head, deprel)
        self.words = words

    def root(self):
        return next(w for w in self.words if w[2] == 0)

    def children(self, word_id):
        return [w for w in self.words if w[2] == word_id]

    def depth(self, word_id=None):
        if word_id is None:
            word_id = self.root()[0]
        kids = self.children(word_id)
        if not kids:
            return 0
        return 1 + max(self.depth(k[0]) for k in kids)

    def find_crossings(self):
        # Find pairs of crossing arcs ((head1, dep1), (head2, dep2))
        arcs = [(w[2], w[0]) for w in self.words if w[2] != 0]
        crossings = []
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                i1, j1 = sorted(arcs[i])
                i2, j2 = sorted(arcs[j])
                if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                    crossings.append((arcs[i], arcs[j]))
        return crossings

    
         #переглянути метод find_crossings(self) можливо переписати



    def is_projective(self):
        crossings = self.find_crossings()
        return len(crossings) == 0, crossings


def parse_with_stanza():
    nlp = stanza.Pipeline('uk', processors='tokenize,pos,lemma,depparse')
    doc = nlp(clean)
    print(f"Total sentences parsed (Stanza): {len(doc.sentences)}")
    print("------------------------")
    return nlp, doc



def analyze_trees_and_deprels(doc):
    deprel_counts = Counter()
    non_projective_found = []

    print("Sentence tree analysis:")
    for idx, sent in enumerate(doc.sentences, 1):
        words = [(w.id, w.text, w.head, w.deprel) for w in sent.words]
        tree = DepTree(words)
        root_word = tree.root()
        direct_deps = tree.children(root_word[0])
        tree_depth = tree.depth()
        proj, crossings = tree.is_projective()

        print(f"Sent {idx:2}: root='{root_word[1]:14}' | direct_deps={len(direct_deps):2} | depth={tree_depth} | projective={proj}")

        if not proj:
            non_projective_found.append((idx, sent.text, crossings, words))

        for w in sent.words:
            deprel_counts[w.deprel] += 1

    print("\nDependency relations frequency:")
    for rel, cnt in deprel_counts.most_common():
        print(f"{rel:14} : {cnt}")
    print("------------------------")

    return non_projective_found


def visualize_trees(count=3):
    os.makedirs("outputs", exist_ok=True)
    nlp_sp = spacy.load('uk_core_news_sm')
    doc_sp = nlp_sp(clean)
    sents = list(doc_sp.sents)[:count]

    for i, sent in enumerate(sents, 1):
        svg = displacy.render(sent, style='dep', jupyter=False)
        filename = f"outputs/tree_{i}.svg"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"Saved visualization: {filename}")
    print("------------------------")



def print_crossings_readable(words, crossings):
    id_to_text = {w[0]: w[1] for w in words}
    for (h1, d1), (h2, d2) in crossings:
        print(f"  Arc ({id_to_text[h1]} -> {id_to_text[d1]}) crosses ({id_to_text[h2]} -> {id_to_text[d2]})")


def check_non_projective_and_ambiguity(nlp, non_projective_found):
    print("Non-projective tree check:")
    if non_projective_found:
        idx, text, crossings, words = non_projective_found[0]
        print(f"Found in text (Sentence {idx}): {text}")
        print_crossings_readable(words, crossings)
    else:
        print("No non-projective trees found in text. Testing constructed sentence:")
        test_sent = "Модель ми вчора перевірили мовну."
        doc_test = nlp(test_sent)
        words = [(w.id, w.text, w.head, w.deprel) for w in doc_test.sentences[0].words]
        tree = DepTree(words)
        print(f"Sentence: {test_sent}")
        for w in words:
            print(f"  {w[0]}: {w[1]:14} -> head={w[2]} ({w[3]})")
        print(f"Projective: {tree.is_projective()[0]}")
        print_crossings_readable(words, tree.find_crossings())

    print("\nSyntactic ambiguity analysis (Stanza):")
    ambig_sent = "Дослідники зі США використали мапи уражень мозку як незалежний орієнтир для аналізу цих збоїв у роботі моделі."
    doc_ambig = nlp(ambig_sent)
    sent_words = doc_ambig.sentences[0].words
    for w in sent_words:
        head_text = sent_words[w.head - 1].text if w.head > 0 else "ROOT"
        print(f"  {w.id:2}: {w.text:14} -> {head_text:14} ({w.deprel})")
    print("------------------------")


nlp_stanza, doc_stanza = parse_with_stanza()
non_proj = analyze_trees_and_deprels(doc_stanza)
visualize_trees(3)
check_non_projective_and_ambiguity(nlp_stanza, non_proj)