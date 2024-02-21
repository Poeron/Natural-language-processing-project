from collections import Counter
import re
from gensim.models import Word2Vec

def preprocess_text(text):
    # Veriyi küçük harfe çevir ve kelimeleri ayır
    words = re.findall(r'\b\w+\b', text.lower())
    return words

def most_common_words(file_path, limit=20):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            words = preprocess_text(text)
            word_counts = Counter(words)
            most_common_words = word_counts.most_common(limit)

            print(f"En çok geçen {limit} kelime:")
            for word, count in most_common_words:
                print(f"{word}: {count} kez")

            # En çok geçen kelimeleri bir listeye ekleyelim
            most_common_words_list = [word for word, _ in most_common_words]

            return most_common_words_list
    except FileNotFoundError:
        print(f"{file_path} dosyası bulunamadı.")
        return []

# Kullanım örneği:
file_name = 'corpus.txt'
most_common_words_list = most_common_words(file_name)

# Word2Vec modelini kullan
with open(file_name, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

preprocessed_sentences = [preprocess_text(sentence) for sentence in sentences]

# Word2Vec modelini kullan
model = Word2Vec(sentences=preprocessed_sentences, vector_size=100, window=5, min_count=1, workers=4)

# En çok geçen 20 kelimenin her biri için benzer kelimeleri bulalım
for word in most_common_words_list:
    similar_words = model.wv.most_similar(word, topn=5)
    
    print(f"\n{word} kelimesine benzer kelimeler:")
    for similar_word, score in similar_words:
        print(f"{similar_word}: {score}")
