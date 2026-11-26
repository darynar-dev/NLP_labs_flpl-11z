import re
import spacy


def preprocess(raw):
    t = re.sub(r'\s+', ' ', raw)    # згорнути пробіли й переноси
    t = re.sub(r"[’ʼ`]", "'", t)  # нормалізація апострофа
    return t.strip()


# ------------------------------------------------------------
# Частина А: WSD за алгоритмом Леска (код з методички)
# ------------------------------------------------------------
class LeskWSD:
    def __init__(self):
        self.senses = {}  # слово -> {значення: множина ключових слів}

    def add_sense(self, word, sense_id, keywords):
        self.senses.setdefault(word, {})[sense_id] = set(keywords)

    def disambiguate(self, word, context):
        ctx = set(context.lower().split())  # слова контексту
        # обираємо значення, у якого найбільше спільних слів із контекстом
        return max(self.senses.get(word, {}),
                   key=lambda s: len(ctx & self.senses[word][s]))


# Завдання 1. WSD: 3 омоніми, 6 речень
def task1_wsd():
    print("=== Завдання 1. WSD (алгоритм Леска) ===")
    wsd = LeskWSD()
    # Алгоритм порівнює слова "як є", без лем, тому додаємо кілька форм слова
    wsd.add_sense('лист', 'письмо', ['пошта', 'поштою', 'конверт', 'конверті', 'написав',
                                     'надіслала', 'надіслав', 'прочитав', 'адресат'])
    wsd.add_sense('лист', 'листок', ['дерево', 'дерева', 'клена', 'дуба', 'восени',
                                     'осінній', 'жовтий', 'зелений', 'гілки', 'упав'])

    wsd.add_sense('коса', 'зачіска', ['волосся', 'дівчина', 'дівчини', 'заплела',
                                      'стрічка', 'стрічку', 'руса', 'русу'])
    wsd.add_sense('коса', 'знаряддя', ['косити', 'косив', 'косар', 'трава', 'траву',
                                       'сінокіс', 'гостра', 'наточив'])
    wsd.add_sense('коса', 'берег', ['море', 'моря', 'пісок', 'піщана',
                                    'берег', 'затока', 'мис'])

    wsd.add_sense('бал', 'оцінка', ['оцінка', 'іспит', 'іспиті', 'тест', 'тесті',
                                    'отримав', 'набрав', 'студент', 'учень'])
    wsd.add_sense('бал', 'танці', ['танці', 'танцювали', 'сукня', 'сукні', 'вальс',
                                   'палац', 'палаці', 'бальна', 'кавалер'])

    # (слово, речення, правильне значення); речення без розділових знаків,
    # бо split() залишає крапку приклеєною до слова
    tests = [
        ('лист', 'бабуся надіслала довгий лист поштою', 'письмо'),
        ('лист', 'восени жовтий лист упав з клена', 'листок'),
        ('коса', 'дівчина заплела русу косу і вплела стрічку', 'зачіска'),
        ('коса', 'піщана коса тягнеться далеко в море', 'берег'),
        ('бал', 'на іспиті студент набрав найвищий бал', 'оцінка'),
        ('бал', 'увечері в ратуші відбувся пишний бал', 'танці'),  # складний випадок
    ]

    correct = 0
    for word, sentence, gold in tests:
        pred = wsd.disambiguate(word, sentence)
        if pred == gold:
            correct = correct + 1
            mark = "правильно"
        else:
            mark = "ПОМИЛКА"
        print(f"{sentence:45} | {word:5} -> {pred:9} (еталон: {gold}) {mark}")

    print(f"\nТочність Леска: {correct}/{len(tests)} = {round(100 * correct / len(tests), 2)}%")
    # В останньому реченні немає жодного ключового слова (ратуші, пишний),
    # усі значення мають 0 збігів, і max() бере перше значення - 'оцінка'.
    # Це головне обмеження Леска: він бачить лише точні збіги слів.
    print("------------------------\n")


# ------------------------------------------------------------
# Частина Б: семантичні ролі (функція з методички)
# ------------------------------------------------------------
def label_roles(predicate, args):
    print(f'Предикат: {predicate}')
    for word, role in args:
        print(f'   {word} -> {role}')


# Завдання 2. Ролі для 4 речень (одне пасивне)
def task2_roles():
    print("=== Завдання 2. Семантичні ролі ===")

    print("1) Марія подарувала братові книгу")
    label_roles('подарувати', [('Марія', 'A0 (агент)'),
                               ('братові', 'A2 (отримувач)'),
                               ('книгу', 'A1 (тема)')])

    print("2) Учитель пояснив учням нову тему")
    label_roles('пояснити', [('Учитель', 'A0 (агент)'),
                             ('учням', 'A2 (отримувач)'),
                             ('нову тему', 'A1 (тема)')])

    # активна / пасивна пара
    print("3) Студенти написали статтю (АКТИВНА)")
    active = [('студенти', 'A0 (агент)'), ('статтю', 'A1 (пацієнт)')]
    label_roles('написати', active)

    print("4) Стаття була написана студентами (ПАСИВНА)")
    passive = [('студентами', 'A0 (агент)'), ('стаття', 'A1 (пацієнт)')]
    label_roles('написати', passive)

    # порівнюємо лише ролі (слова мають різні відмінкові форми: студенти/студентами)
    active_roles = sorted(role for word, role in active)
    passive_roles = sorted(role for word, role in passive)
    if active_roles == passive_roles:
        print("\nРолі в активному й пасивному реченні однакові:", active_roles)
    # Синтаксично 'студенти' - підмет, а 'студентами' - додаток в орудному відмінку,
    # 'статтю' - додаток, а 'стаття' - підмет. Але агентом (хто писав) в обох
    # реченнях залишаються студенти, а пацієнтом (що написано) - стаття.
    print("------------------------\n")


# ------------------------------------------------------------
# Частина В: BIO-розмітка (функція з методички)
# ------------------------------------------------------------
def extract_entities(bio_tokens):
    entities, current, ctype = [], [], None
    for token, tag in bio_tokens:
        if tag.startswith('B-'):          # початок нової сутності
            if current:
                entities.append((' '.join(current), ctype))
            current, ctype = [token], tag[2:]
        elif tag.startswith('I-') and current:  # продовження сутності
            current.append(token)
        else:                              # 'O' - не сутність
            if current:
                entities.append((' '.join(current), ctype))
            current, ctype = [], None
    if current:
        entities.append((' '.join(current), ctype))
    return entities


# Precision / Recall / F1 на рівні сутностей.
# Сутність вважається правильною, лише якщо збігаються І текст (межі), І тип.
def compute_entity_metrics(gold, pred):
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)  # true positive - є і в еталоні, і в моделі
    fp = len(pred_set - gold_set)  # false positive - модель знайшла зайве
    fn = len(gold_set - pred_set)  # false negative - модель пропустила

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1


# ------------------------------------------------------------
# Частина Г: реальний NER
# Використано spaCy uk_core_news_sm: LiBERTa з методички не навчена для NER.
# Ця модель знає лише PER, LOC, ORG, MISC, тому дати шукаємо регулярним виразом
# ------------------------------------------------------------
nlp = spacy.load('uk_core_news_sm')


def find_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # простий regex для дат: рік + 'року' / 'році' (день і місяць НЕ ловить)
    for date in re.findall(r'\d{4} року|\d{4} році', text):
        entities.append((date, 'DATE'))
    return entities


with open("text.txt", "r", encoding="utf-8") as f:
    loaded_text = f.read()

clean = preprocess(loaded_text)


# Завдання 3. NER на тексті з 13 речень: збираємо всі сутності
def task3_ner():
    print("=== Завдання 3. NER на тексті ===")
    entities = find_entities(clean)
    print(f"Усього знайдено сутностей: {len(entities)}")
    for text, label in entities:
        print(f"   {text:45} -> {label}")
    print("------------------------\n")


# Завдання 4. Оцінювання на 5 реченнях із ручним золотим стандартом
def task4_evaluation():
    print("=== Завдання 4. Оцінювання NER (P/R/F1) ===")
    # Золотий стандарт записано у BIO-розмітці вручну,
    # сутності з нього дістаємо функцією extract_entities
    gold_bio = [
        # 1. Іван Пулюй народився 2 лютого 1845 року в містечку Гримайлів на Тернопільщині.
        [('Іван', 'B-PER'), ('Пулюй', 'I-PER'), ('народився', 'O'),
         ('2', 'B-DATE'), ('лютого', 'I-DATE'), ('1845', 'I-DATE'), ('року', 'I-DATE'),
         ('в', 'O'), ('містечку', 'O'), ('Гримайлів', 'B-LOC'), ('на', 'O'),
         ('Тернопільщині', 'B-LOC'), ('.', 'O')],
        # 2. У 1865 році він закінчив класичну гімназію в Тернополі.
        [('У', 'O'), ('1865', 'B-DATE'), ('році', 'I-DATE'), ('він', 'O'),
         ('закінчив', 'O'), ('класичну', 'O'), ('гімназію', 'O'), ('в', 'O'),
         ('Тернополі', 'B-LOC'), ('.', 'O')],
        # 3. Пулюй вступив на богословський факультет Віденського університету, ...
        [('Пулюй', 'B-PER'), ('вступив', 'O'), ('на', 'O'), ('богословський', 'O'),
         ('факультет', 'O'), ('Віденського', 'B-ORG'), ('університету', 'I-ORG'),
         (',', 'O'), ('але', 'O'), ('згодом', 'O'), ('обрав', 'O'), ('фізику', 'O'),
         ('та', 'O'), ('математику', 'O'), ('.', 'O')],
        # 4. У Відні він познайомився з Пантелеймоном Кулішем і ...
        [('У', 'O'), ('Відні', 'B-LOC'), ('він', 'O'), ('познайомився', 'O'), ('з', 'O'),
         ('Пантелеймоном', 'B-PER'), ('Кулішем', 'I-PER'), ('і', 'O'), ('разом', 'O'),
         ('із', 'O'), ('ним', 'O'), ('узявся', 'O'), ('перекладати', 'O'),
         ('Біблію', 'O'), ('українською', 'O'), ('мовою', 'O'), ('.', 'O')],
        # 5. У 1876 році Пулюй здобув ступінь доктора філософії в Страсбурзькому університеті.
        [('У', 'O'), ('1876', 'B-DATE'), ('році', 'I-DATE'), ('Пулюй', 'B-PER'),
         ('здобув', 'O'), ('ступінь', 'O'), ('доктора', 'O'), ('філософії', 'O'), ('в', 'O'),
         ('Страсбурзькому', 'B-ORG'), ('університеті', 'I-ORG'), ('.', 'O')],
    ]

    all_gold = []
    all_pred = []
    for bio in gold_bio:
        # відновлюємо текст речення з токенів (пробіл перед розділовими знаками прибираємо)
        sentence = ' '.join(token for token, tag in bio)
        sentence = sentence.replace(' .', '.').replace(' ,', ',')

        gold = extract_entities(bio)
        pred = find_entities(sentence)
        all_gold.extend(gold)
        all_pred.extend(pred)

        print(sentence)
        print("   еталон:", gold)
        print("   модель:", pred)

    p, r, f1 = compute_entity_metrics(all_gold, all_pred)
    print(f"\nPrecision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")

    # Помилки меж: тип однаковий, а текст лише частково збігається
    # (одна сутність є частиною іншої, як '1856' vs '1856 року')
    print("\nПомилки меж:")
    for g_text, g_label in all_gold:
        for p_text, p_label in all_pred:
            if g_label == p_label and g_text != p_text and (g_text in p_text or p_text in g_text):
                print(f"   еталон: '{g_text}'  |  модель: '{p_text}'  ({g_label})")
    print("------------------------\n")


# Завдання 5. Вилучення фактів: таблиця особа | дата | місце | організація
def task5_facts():
    print("=== Завдання 5. Таблиця фактів ===")
    doc = nlp(clean)
    facts = 0
    print(f"{'Особа':25} | {'Дата':20} | {'Місце':20} | Організація")
    for sent in doc.sents:
        row = {'PER': [], 'DATE': [], 'LOC': [], 'ORG': []}
        for text, label in find_entities(sent.text):
            if label in row:  # MISC у таблицю не беремо
                row[label].append(text)

        # фактом вважаємо речення, де знайдено хоча б 2 різні типи сутностей
        # (наприклад, хто + коли, або хто + де)
        filled = sum(1 for values in row.values() if values)
        if filled >= 2:
            facts = facts + 1
            print(f"{', '.join(row['PER']):25} | {', '.join(row['DATE']):20} | "
                  f"{', '.join(row['LOC']):20} | {', '.join(row['ORG'])}")
    print(f"\nВитягнуто фактів: {facts}")
    print("------------------------\n")


task1_wsd()
task2_roles()
task3_ner()
task4_evaluation()
task5_facts()
