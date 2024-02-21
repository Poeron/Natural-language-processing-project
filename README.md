# NLP Project

I pulled 10,000 sentences related to COVID-19 from Twitter for the project assignment of my 3rd-grade natural language processing course. Using the Counter function of the collections library in the project, I identified the 20 most frequently occurring words and found 5 words similar to each of the top 20 words using Word2Vec.

Additionally, by utilizing the following libraries, I identified 3 sentences similar to 5 randomly selected sentences:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

Furthermore, I conducted sentiment analysis on the selected 5 sentences and their similar 3 sentences using a pre-trained BERT model.

### Clone the Repository


   ```bash
   git clone https://github.com/Poeron/Natural-language-processing-project.git
   ```
   