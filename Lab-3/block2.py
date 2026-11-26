import stanza
import spacy
import pandas as pd 
from spacy import displacy
from collections import defaultdict


# 5 sentences from 5 different user texts with manual gold standard: (word, head, deprel)
GOLD_DATA = [
    (
        "Дослідники зі США використали мапи уражень мозку як незалежний орієнтир для аналізу цих збоїв у роботі моделі.",
        [
            ("Дослідники", 4, "nsubj"), ("зі", 3, "case"), ("США", 1, "nmod"),
            ("використали", 0, "root"), ("мапи", 4, "obj"), ("уражень", 5, "nmod"),
            ("мозку", 6, "nmod"), ("як", 10, "mark"), ("незалежний", 10, "amod"),
            ("орієнтир", 4, "xcomp:pred"), ("для", 12, "case"), ("аналізу", 10, "nmod"),
            ("цих", 14, "det"), ("збоїв", 12, "nmod"), ("у", 16, "case"),
            ("роботі", 14, "nmod"), ("моделі", 16, "nmod"), (".", 4, "punct")
        ]
    ),
    (
        "Водночас на дугоподібній рампі авто тримали не двигуни, а вентилятори.",
        [
            ("Водночас", 6, "advmod"), ("на", 4, "case"), ("дугоподібній", 4, "amod"),
            ("рампі", 6, "obl"), ("авто", 6, "obj"), ("тримали", 0, "root"),
            ("не", 8, "advmod"), ("двигуни", 6, "nsubj"), (",", 11, "punct"),
            ("а", 11, "cc"), ("вентилятори", 8, "conj"), (".", 6, "punct")
        ]
    ),
    (
        "Водночас рада директорів компанії розпочала пошук постійного керівника серед кандидатів усередині компанії та поза нею.",
        [
            ("Водночас", 5, "advmod"), ("рада", 5, "nsubj"), ("директорів", 2, "nmod"),
            ("компанії", 3, "nmod"), ("розпочала", 0, "root"), ("пошук", 5, "obj"),
            ("постійного", 8, "amod"), ("керівника", 6, "nmod"), ("серед", 10, "case"),
            ("кандидатів", 6, "nmod"), ("усередині", 12, "case"), ("компанії", 10, "nmod"),
            ("та", 15, "cc"), ("поза", 15, "case"), ("нею", 12, "conj"), (".", 5, "punct")
        ]
    ),
    (
        "Натомість за депресії зміни у різних ділянках мозку частіше були подібними.",
        [
            ("Натомість", 11, "advmod"), ("за", 3, "case"), ("депресії", 11, "obl"),
            ("зміни", 11, "nsubj"), ("у", 7, "case"), ("різних", 7, "amod"),
            ("ділянках", 4, "nmod"), ("мозку", 7, "nmod"), ("частіше", 11, "advmod"),
            ("були", 11, "cop"), ("подібними", 0, "root"), (".", 11, "punct")
        ]
    ),
    (
        "На дорогах через опади й низьку температуру повітря очікується ожеледиця.",
        [
            ("На", 2, "case"), ("дорогах", 9, "obl"), ("через", 4, "case"),
            ("опади", 9, "obl"), ("й", 7, "cc"), ("низьку", 7, "amod"),
            ("температуру", 4, "conj"), ("повітря", 7, "nmod"), ("очікується", 0, "root"),
            ("ожеледиця", 9, "nsubj"), (".", 9, "punct")
        ]
    )
]


def parse_sentences():
    nlp_st = stanza.Pipeline('uk', processors='tokenize,pos,lemma,depparse')
    nlp_sp = spacy.load('uk_core_news_sm')

    stanza_results = []
    spacy_results = []

    for text, _ in GOLD_DATA:
        doc_st = nlp_st(text)
        st_sent = [(w.text, w.head, w.deprel) for s in doc_st.sentences for w in s.words]
        stanza_results.append(st_sent)

        doc_sp = nlp_sp(text)
        # In spaCy head.i starts from 0, so add +1 (and 0 for root)
        sp_sent = [(tok.text, tok.head.i + 1 if tok.head != tok else 0, tok.dep_.lower()) for tok in doc_sp]
        spacy_results.append(sp_sent)

    return stanza_results, spacy_results



def compute_uas_las(gold_all, sys_all):
    correct_head = 0
    correct_both = 0
    total = 0

    for gold_sent, sys_sent in zip(gold_all, sys_all):
        for g, s in zip(gold_sent, sys_sent):
            total += 1
            if g[1] == s[1]:
                correct_head += 1
                if g[2] == s[2]:
                    correct_both += 1

    uas = correct_head / total if total else 0
    las = correct_both / total if total else 0
    return uas, las


# переглянути метод compute_uas_las(gold_all, sys_all): можливо розбити на два методи як в методі



def evaluate_parser(name, gold_all, sys_all):
    uas, las = compute_uas_las(gold_all, sys_all)

    print(f"=== {name} ===")
    print(f"UAS = {uas:.3f} | LAS = {las:.3f}")

    head_errors = 0
    label_errors = 0

    # Store all label pairs when the predicted head is correct
    label_matrix = defaultdict(lambda: defaultdict(int))

    for sent_idx, (gold_sent, sys_sent) in enumerate(
        zip(gold_all, sys_all), 1
    ):
        for g, s in zip(gold_sent, sys_sent):

            if g[1] != s[1]:
                head_errors += 1

                print(
                    f"[Sent {sent_idx}] {g[0]}: "
                    f"HEAD error (gold={g[1]}, sys={s[1]})"
                )

            else:
                label_matrix[g[2]][s[2]] += 1

                if g[2] != s[2]:
                    label_errors += 1

                    print(
                        f"[Sent {sent_idx}] {g[0]}: "
                        f"LABEL error (gold={g[2]}, sys={s[2]})"
                    )

    print(f"Total HEAD errors: {head_errors}")
    print(f"Total LABEL errors: {label_errors}")

    # Build confusion matrix
    labels = sorted({
        label
        for gold_label, predictions in label_matrix.items()
        for label in [gold_label] + list(predictions.keys())
    })

    matrix = pd.DataFrame(
        0,
        index=labels,
        columns=labels
    )

    for gold_label, predictions in label_matrix.items():
        for predicted_label, count in predictions.items():
            matrix.loc[gold_label, predicted_label] = count

    print("\nLabel confusion matrix:")
    print(matrix)

    # Find the most frequently confused relation
    errors = matrix.copy()

    for label in labels:
        errors.loc[label, label] = 0

    if errors.values.sum() > 0:
        most_common = errors.stack().idxmax()

        print(
            f"\nMost confused relation: "
            f"{most_common[0]} -> {most_common[1]} "
            f"({errors.loc[most_common[0], most_common[1]]} errors)"
        )
    else:
        print("\nNo label errors.")

    print("------------------------")



gold_sentences = [item[1] for item in GOLD_DATA]
st_res, sp_res = parse_sentences()

evaluate_parser("Stanza", gold_sentences, st_res)
evaluate_parser("spaCy", gold_sentences, sp_res)

