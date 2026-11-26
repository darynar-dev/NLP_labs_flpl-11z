import numpy as np
from gensim.models.fasttext import load_facebook_vectors



class WordVectors:
    def __init__(self):
        self.vecs = {}

    def add(self, word, vec):
        self.vecs[word] = np.array(vec, dtype=float)

    def cosine(self, w1, w2):
        # косинусна подібність = скалярний добуток / (довжина a * довжина b)
        a, b = self.vecs[w1], self.vecs[w2]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def most_similar(self, word, n=3):
        t = self.vecs[word]
        sims = []
        for w, v in self.vecs.items():
            if w == word:
                continue
            c = np.dot(t, v) / (np.linalg.norm(t) * np.linalg.norm(v))
            sims.append((w, float(c)))
        return sorted(sims, key=lambda x: -x[1])[:n]

    def analogy(self, a, b, c, n=1):
        # 'a до b, як c до ?' -> b - a + c
        result = self.vecs[b] - self.vecs[a] + self.vecs[c]
        sims = []
        for w, v in self.vecs.items():
            if w in (a, b, c):
                continue
            cos = np.dot(result, v) / (np.linalg.norm(result) * np.linalg.norm(v))
            sims.append((w, float(cos)))
        return sorted(sims, key=lambda x: -x[1])[:n]


def avg_similarity(wv, group):
    sims = []
    for i, w1 in enumerate(group):
        for w2 in group[i + 1:]:
            sims.append(wv.cosine(w1, w2))
    return sum(sims) / len(sims)


# Середня подібність МІЖ двома групами (кожне слово групи 1 з кожним словом групи 2)
def avg_between(wv, group1, group2):
    sims = []
    for w1 in group1:
        for w2 in group2:
            sims.append(wv.cosine(w1, w2))
    return sum(sims) / len(sims)


# ------------------------------------------------------------
# Частина Г: готова модель FastText для української
# Файл cc.uk.300.bin треба завантажити заздалегідь:
# https://fasttext.cc/docs/en/crawl-vectors.html
# Увага: модель велика (~7 ГБ), завантаження займає кілька хвилин
# ------------------------------------------------------------
print("Завантаження моделі FastText...")
ft = load_facebook_vectors('cc.uk.300.bin')
print("Модель завантажено\n")


# Завдання 1. 10 слів з різних семантичних груп -> 5 найсхожіших
def task1_most_similar():
    print("=== Завдання 1. Найсхожіші слова ===")
    words = [
        'кіт', 'собака',      # тварини
        'лікар', 'вчитель',   # професії
        'радість', 'страх',   # емоції
        'автобус',            # транспорт
        'хліб',               # їжа
        'річка',              # природа
        'Львів',              # місто
    ]
    for w in words:
        similar = ft.most_similar(w, topn=5)
        # округлюємо числа до 2 знаків, щоб вивід був коротшим
        similar_short = [(s, round(c, 2)) for s, c in similar]
        print(f"{w:10} -> {similar_short}")
    print("------------------------\n")


# Завдання 2. Косинусна подібність для 5 пар слів
def task2_pairs():
    print("=== Завдання 2. Косинусна подібність пар ===")
    pairs = [
        ('гарний', 'красивий', 'синоніми'),
        ('сміливий', 'хоробрий', 'синоніми'),
        ('гарячий', 'холодний', 'антоніми'),
        ('день', 'ніч', 'антоніми'),
        ('банан', 'парламент', 'незв\'язані'),
    ]
    print(f"{'Пара':25} | {'Тип':10} | Косинус")
    for w1, w2, kind in pairs:
        sim = ft.similarity(w1, w2)
        print(f"{w1 + ' - ' + w2:25} | {kind:10} | {sim:.3f}")
    print("------------------------\n")


# Завдання 3. 5 аналогій: 'a до b, як c до ?'  ->  b - a + c
def task3_analogies():
    print("=== Завдання 3. Аналогії ===")
    # (a, b, c, правильна відповідь)
    analogies = [
        ('Україна', 'Київ', 'Франція', 'Париж'),     # країна -> столиця
        ('Львів', 'Україна', 'Варшава', 'Польща'),   # місто -> країна
        ('чоловік', 'король', 'жінка', 'королева'),  # рід
        ('великий', 'більший', 'малий', 'менший'),   # ступені порівняння
        ('лікар', 'лікарня', 'вчитель', 'школа'),    # професія -> місце роботи
    ]
    correct = 0
    for a, b, c, expected in analogies:
        # positive = слова, які додаємо; negative = слово, яке віднімаємо
        result = ft.most_similar(positive=[b, c], negative=[a], topn=3)
        answer = result[0][0]  # найкращий варіант моделі
        if answer == expected:
            correct = correct + 1
            mark = "правильно"
        else:
            mark = "неправильно"
        print(f"{a}:{b} = {c}:? -> {answer:12} (очікували {expected}) {mark}")
        print(f"    топ-3: {[(s, round(x, 2)) for s, x in result]}")
    print(f"\nРозв'язано правильно: {correct}/{len(analogies)} = {100 * correct / len(analogies)}%")
    print("------------------------\n")


# Завдання 4. Кластеризація: подібність усередині груп і між групами
def task4_clusters():
    print("=== Завдання 4. Кластеризація ===")
    animals = ['кіт', 'собака', 'кінь', 'корова']
    transport = ['автобус', 'потяг', 'літак', 'трамвай']

    # кладемо справжні вектори FastText у наш клас WordVectors,
    # щоб використати функції avg_similarity та avg_between
    wv = WordVectors()
    for w in animals + transport:
        wv.add(w, ft[w])

    inside_animals = avg_similarity(wv, animals)
    inside_transport = avg_similarity(wv, transport)
    between = avg_between(wv, animals, transport)

    print(f"Усередині 'тварини':   {inside_animals:.3f}")
    print(f"Усередині 'транспорт': {inside_transport:.3f}")
    print(f"Між групами:           {between:.3f}")
    if inside_animals > between and inside_transport > between:
        print("Кластеризацію підтверджено: усередині груп подібність вища, ніж між ними")
    else:
        print("Кластеризацію НЕ підтверджено")
    print("------------------------\n")


# Завдання 5. Приклади, де модель помиляється або дає несподіваний результат
def task5_errors():
    print("=== Завдання 5. Помилки та несподівані результати ===")

    # 1) Упередження в даних: чоловік:програміст = жінка:?
    #    Якщо модель дасть щось на кшталт 'бухгалтерка' чи 'секретарка',
    #    це гендерний стереотип із текстів, на яких її навчали
    result = ft.most_similar(positive=['програміст', 'жінка'], negative=['чоловік'], topn=5)
    print("чоловік:програміст = жінка:? ->", [(s, round(x, 2)) for s, x in result])

    # 2) Рідкісне (діалектне) слово: 'пательня' = сковорода.
    #    FastText будує вектор з частин слова, тому серед "схожих"
    #    часто опиняються слова, схожі за написанням, а не за змістом
    result = ft.most_similar('пательня', topn=5)
    print("пательня ->", [(s, round(x, 2)) for s, x in result])
    print("------------------------\n")


# Завдання 6. Омоніми: яке значення "перемогло" в одному векторі
def task6_homonyms():
    print("=== Завдання 6. Омоніми ===")
    # омонім -> два слова-підказки для двох різних значень
    homonyms = {
        'коса': ('волосся', 'сінокіс'),     # зачіска / знаряддя
        'ключ': ('двері', 'джерело'),       # від замка / джерело води
        'замок': ('фортеця', 'двері'),      # споруда / пристрій
    }
    for word, (sense1, sense2) in homonyms.items():
        similar = ft.most_similar(word, topn=10)
        print(f"{word} -> {[s for s, x in similar]}")
        # порівнюємо, до якого значення вектор слова ближчий
        s1 = ft.similarity(word, sense1)
        s2 = ft.similarity(word, sense2)
        print(f"    cos({word}, {sense1}) = {s1:.3f}   cos({word}, {sense2}) = {s2:.3f}")
        if s1 > s2:
            print(f"    'переміг' сенс, пов'язаний зі словом '{sense1}'")
        else:
            print(f"    'переміг' сенс, пов'язаний зі словом '{sense2}'")
    print("Один вектор на слово змішує всі значення: перемагає те, що частіше в текстах.")
    print("------------------------\n")


task1_most_similar()
task2_pairs()
task3_analogies()
task4_clusters()
task5_errors()
task6_homonyms()
