from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def calculate_similarity(tfidf_vectorizer, query_sentences, preprocessed_sentences):
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_sentences)
    tfidf_vectors_query = tfidf_vectorizer.transform(query_sentences)
    similarity_scores = cosine_similarity(tfidf_vectors_query, tfidf_matrix)
    return similarity_scores

def find_most_similar_sentences(similarity_scores, preprocessed_sentences):
    most_similar_indices = similarity_scores.argsort(axis=1)[:, ::-1][:, 1:4]
    most_similar_sentences = [(indices, [preprocessed_sentences[i] for i in indices]) for indices in most_similar_indices]
    return most_similar_sentences

def main():
    corpus_file = 'corpus.txt'
    query_sentences = [
        "reported positive result",
        "many souls already lost lives due way covid",
        "million americans died still",
        "find free covid vaccine",
        "looks pretty ridiculous people dont collapse dead spot covid"
    ]

    preprocessed_sentences = read_corpus(corpus_file)

    tfidf_vectorizer = TfidfVectorizer()
    similarity_scores = calculate_similarity(tfidf_vectorizer, query_sentences, preprocessed_sentences)

    most_similar_sentences = find_most_similar_sentences(similarity_scores, preprocessed_sentences)

    for i, (query_sentence, (indices, similar_sentences)) in enumerate(zip(query_sentences, most_similar_sentences), 1):
        print(f"\nCümle {i}: {query_sentence}")
        print("En Benzer 3 Cümle ve Satır Numaraları:")
        for j, (sentence, index) in enumerate(zip(similar_sentences, indices), start=1):
            print(f"{j}. Satır {index + 1}: {sentence}")

if __name__ == "__main__":
    main()
