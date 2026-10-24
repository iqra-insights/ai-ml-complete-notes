[Previous: 6. Convolutional Neural Network (CNN)](06-convolutional-neural-network-cnn.md) | [Home](../README.md) | [Next: 8. Sequence Models and Fake News Detection](08-sequence-models-and-fake-news-detection.md)

---

## 7. NLP: Sentiment Analysis on Amazon Reviews

NLP = machines ko human language samajhna sikhana.

### 7.1 NLP Pipeline

<p align="center"><img src="../images/09-nlp-pipeline.svg" alt="NLP pipeline · 5 stages" width="800"><br><sub><i>NLP pipeline · 5 stages</i></sub></p>

### 7.2 Text Preprocessing

| Step | What | Example |
| --- | --- | --- |
| Lowercasing | Sab small | 'Great' becomes 'great' |
| Remove punctuation/numbers/HTML/URLs | Noise hatao |  |
| Tokenization | Words mein todo | 'good phone' becomes ['good','phone'] |
| Stopword removal | is, the, a hatao | Note: 'not' ko sentiment mein mat hatao |
| Stemming | Root cut (fast, crude) | 'running' becomes 'run', 'studies' becomes 'studi' |
| Lemmatization | Proper dictionary root (better) | 'better' becomes 'good' |
| POS tagging / NER | Grammar tags / named entities |  |

### 7.3 Text Vectorization

| Method | Idea | Pros / Cons |
| --- | --- | --- |
| Bag of Words | Word counts | Simple; order aur meaning lost |
| N-grams | Word pairs/triples | Thoda context; bada vocabulary |
| TF-IDF | TF x IDF; common words ka weight kam | Better than BoW; still no semantics |
| Word2Vec / GloVe | Dense vectors, similar words paas | Semantics capture; ek word = ek vector |
| Transformer embeddings (BERT) | Context-aware vectors | Best; heavy |

**TF-IDF:** `TF = count of word in doc / total words`, `IDF = log(N / docs containing word)`

**Classification models:** Naive Bayes, Logistic Regression, SVM, Random Forest (TF-IDF ke saath fast baseline).

### 7.4 Embedding Layer

Embedding layer = integer word ID ko **dense learnable vector** mein badalta hai.

```
'good' -> ID 42 -> Embedding Matrix (vocab x dim) -> [0.12, -0.5, 0.8, ...]
```

Train ke dauran ye vectors seekhe jate hain, similar words paas aate hain. Pretrained GloVe/Word2Vec bhi load kar sakte ho.

### 7.5 Sentiment Analysis (Amazon Reviews)

Task: review ko Positive / Negative / Neutral classify karna.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['label'], test_size=0.2, stratify=df['label'])
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(X_train), y_train)
print(classification_report(y_test, clf.predict(vec.transform(X_test))))
```

**Practical tips:** class imbalance dhyan se (F1 dekho, sirf accuracy nahi); rating 4-5 = positive, 1-2 = negative, 3 = neutral/drop; negation handle karo.

**DL version:** Tokenizer -> pad\_sequences -> Embedding -> LSTM/GRU -> Dense(sigmoid).

### 7.6 Email Spam Detection

Binary classification: spam vs ham. Pipeline: clean -> TF-IDF -> Multinomial Naive Bayes (classic best baseline). **Metrics:** Precision important (ham ko spam mat bolo), Recall, F1, Confusion matrix.

### 7.7 Machine Translation

Source language ko target language mein convert karna.

<p align="center"><img src="../images/10-seq2seq-attention.svg" alt="seq2seq translation with attention" width="800"><br><sub><i>seq2seq translation with attention</i></sub></p>

**Metric:** BLEU score. **Evolution:** Rule-based -> Statistical -> Seq2Seq -> Attention -> Transformer.

### 7.8 Introduction to LLMs and the OpenAI API

LLM = bahut bade Transformer models jo huge text par train hue hain, next token predict karke text generate karte hain.

**Transformer core idea:** Self-Attention (har word dekhta hai ki baaki words kitne relevant hain), poori sequence parallel process hoti hai.

**Training stages:** Pretraining -> Instruction fine-tuning -> RLHF (human feedback).

**Key parameters:** `temperature` (randomness), `max_tokens`, `top_p`, `system / user / assistant` roles.

```python
from openai import OpenAI
client = OpenAI(api_key='YOUR_KEY')   # key hamesha environment variable mein rakho
resp = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Explain overfitting in 2 lines.'}
    ],
    temperature=0.7, max_tokens=150
)
print(resp.choices[0].message.content)
```

**Prompt engineering:** clear instruction, role, examples (few-shot), step-by-step reasoning, output format specify karo.

---

[Previous: 6. Convolutional Neural Network (CNN)](06-convolutional-neural-network-cnn.md) | [Home](../README.md) | [Next: 8. Sequence Models and Fake News Detection](08-sequence-models-and-fake-news-detection.md)
