import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize
from pathlib import Path


directory = Path(__file__).resolve().parent
with open(directory/"text.txt", "r", encoding="utf-8") as f:
    loaded_text = f.read()
 
sentences = sent_tokenize(loaded_text)
words = word_tokenize(loaded_text)

print(f"Кількість речень: {len(sentences)}")
print(f"Кількість токенів: {len(words)}\n\n")
print(f"Речення: {sentences}\n\n")
print(f"Токени: {words}")


# записуєм список в одну стрінгу, 
# кожен елемент з нового рядка
sentences_text = ""
for s in sentences:
    sentences_text = sentences_text + s + "\n"

words_text = ""
for w in words:
    words_text = words_text + w + "\n"

with open("sentences.txt", "w", encoding="utf-8") as f:
    f.write(str(sentences_text))

with open("words.txt", "w", encoding="utf-8") as f:
    f.write(str(words_text))

