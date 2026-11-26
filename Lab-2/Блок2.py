import re
import stanza


def preprocess(raw):
    t = re.sub(r'\s+', ' ', raw)
    t = re.sub(r"[’ʼ`]", "'", t)
    return t.strip()


with open("text.txt", "r", encoding="utf-8") as f:
    loaded_text = f.read()

clean = preprocess(loaded_text)


def rule_pos(word):
    w = word.lower()
    if re.search(r'(ти|тися)$', w):
        return 'VERB'
    if re.search(r'(ий|ій|их|ого)$', w):
        return 'ADJ'
    if re.search(r'(ання|ення|ість)$', w):
        return 'NOUN'
    if re.search(r'(но|ко|чно)$', w):
        return 'ADV'
    return 'X'  # не визначено


def full_morph_analysis():
    nlp = stanza.Pipeline('uk', processors='tokenize,pos,lemma')
    doc = nlp(clean)
    
    print("Повний морфоаналіз Stanza")
    for s in doc.sentences:
        for w in s.words:
            print(f"{w.text:16} {w.upos:6} лема={w.lemma:16} {w.feats}")
    print("------------------------")
    return doc


def get_frequencies(doc):
    parts_of_speech_freq = {}
    total_tokens = 0

    for sent in doc.sentences:
        for w in sent.words:
            total_tokens = total_tokens + 1
            if w.upos in parts_of_speech_freq.keys():
                parts_of_speech_freq[w.upos] = parts_of_speech_freq[w.upos] + 1
            else:
                parts_of_speech_freq[w.upos] = 1

    print(f"Частотний розподіл")
    for pos, cnt in parts_of_speech_freq.items():
        percent = round(100 * cnt / total_tokens, 1)
        print(f"{pos:6} : {cnt:3} ({percent}%)")
    print("------------------------")


def compare_baseline_vs_stanza(doc):
    test_words = [
        "покинути", "перетнути", "відомих", "сонячних", "світлових",
        "сингулярність", "радіовипромінювання", "зображення", "приблизно", "одночасно",
        "діра", "світло", "створили", "зникає", "об'єкти"
    ]

    # збираємо еталонні теги Stanza (з нормалізацією апострофа для надійного збігу)
    stanza_gold = {}
    for sent in doc.sentences:
        for w in sent.words:
            word_lower = w.text.lower()
            if word_lower in test_words and word_lower not in stanza_gold.keys():
                stanza_gold[word_lower] = w.upos

    print("Порівняння Baseline vs Stanza")
    correct = 0
    for w in test_words:
        gold_tag = stanza_gold.get(w, "НЕ ЗНАЙДЕНО")
        pred_tag = rule_pos(w)
        if pred_tag == gold_tag:
            correct = correct + 1
        else:
            print(f"{w:20} | Stanza: {gold_tag:5} | Baseline: {pred_tag:5}")

    accuracy = round(100 * correct / len(test_words), 2)
    print(f"\nТочність Baseline: {correct}/{len(test_words)} = {accuracy}%")




doc = full_morph_analysis()
get_frequencies(doc)
compare_baseline_vs_stanza(doc)