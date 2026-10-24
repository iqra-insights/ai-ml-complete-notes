[Previous: 15. Deep Dive: Computer Vision, OpenCV and CNN](15-deep-dive-computer-vision-opencv-and-cnn.md) | [Home](../README.md) | [Next: 17. Deep Dive: RNN, LSTM, GRU, Fake News and Chatbot](17-deep-dive-rnn-lstm-gru-fake-news-and-chatbot.md)

---

## 16. Deep Dive: NLP, Sentiment, Spam, Translation and LLMs

### 16.1 Preprocessing code (NLTK)

```python
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords'); nltk.download('wordnet')

stop = set(stopwords.words('english')) - {'not', 'no', 'nor'}   # negation rakho
lemma = WordNetLemmatizer()

def clean(text):
    text = re.sub(r'<.*?>|http\S+|[^a-zA-Z\s]', ' ', text.lower())
    words = [lemma.lemmatize(w) for w in text.split() if w not in stop]
    return ' '.join(words)

df['clean'] = df['review'].apply(clean)
```

**Common mistakes:** negation words hata dena ('not good' ka matlab ulta ho jata hai), test data par vectorizer fit karna (leakage), emojis ignore karna jab sentiment ke liye useful hon.

### 16.2 TF-IDF worked example

Do documents: D1 = 'good phone', D2 = 'good camera good battery'.

```math
TF(t,d)=\frac{\text{count of }t\text{ in }d}{\text{words in }d},\quad IDF(t)=\log\frac{N}{df(t)},\quad TFIDF=TF\cdot IDF
```

- 'good' dono documents mein hai, df = 2, N = 2, IDF = log(1) = 0. Isliye 'good' ka TF-IDF 0 (common word, koi information nahi).
- 'camera' sirf D2 mein, IDF = log(2) = 0.69, isliye D2 ke liye important.

**Cosine similarity** (documents ki similarity):

```math
\cos\theta=\frac{A\cdot B}{\|A\|\,\|B\|}
```

### 16.3 Word embeddings in more detail

| Method | Idea |
| --- | --- |
| Word2Vec CBOW | Context words se beech ka word predict |
| Word2Vec Skip-gram | Ek word se aas-paas ke words predict (rare words ke liye achha) |
| GloVe | Co-occurrence matrix par based global vectors |
| FastText | Subword (character n-gram) vectors, spelling mistakes handle |
| BERT embeddings | Context ke hisaab se alag vector ('bank' river vs money) |

Famous example: `king - man + woman ≈ queen`. Typical embedding size 50-300 (Word2Vec/GloVe), 768 (BERT base).

### 16.4 Sentiment analysis: complete DL example

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tok = Tokenizer(num_words=10000, oov_token='<OOV>')
tok.fit_on_texts(X_train)
Xtr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=150, padding='post')
Xte = pad_sequences(tok.texts_to_sequences(X_test), maxlen=150, padding='post')

model = keras.Sequential([
    keras.layers.Embedding(10000, 64, input_length=150),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtr, y_train, validation_split=0.1, epochs=10, batch_size=64)
```

**Imbalance handling:** `class_weight`, oversampling (SMOTE for vectors), undersampling, F1/PR-AUC dekho. **Error analysis:** misclassified reviews padho: sarcasm, negation, mixed sentiment common reasons hain.

### 16.5 Naive Bayes for spam (the math)

```math
P(\text{spam}\mid w_1..w_n)\propto P(\text{spam})\prod_i P(w_i\mid \text{spam})
```

'Naive' isliye kyunki words ko independent maanta hai. Fir bhi text par bahut achha chalta hai. **Laplace smoothing** (alpha = 1) zero probability problem rokta hai.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
pipe = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB(alpha=1.0))
pipe.fit(X_train, y_train)
```

### 16.6 Machine translation in more detail

- **Seq2Seq:** encoder poori source sentence ko ek context vector mein compress karta hai. Bahut lambi sentences mein ye bottleneck ban jata hai.
- **Attention** har decoder step par encoder ke sab states dekhne deta hai (weighted sum), bottleneck khatam.
- **Teacher forcing:** training mein decoder ko pichla *sahi* word input dete hain (inference mein apna predicted word).
- **Beam search:** top-k candidate sentences saath rakhta hai, greedy se behtar translation.
- **BLEU:** machine translation aur reference ke n-gram overlap par score (0 se 1).

### 16.7 Transformers and self-attention

```math
\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
```

- **Q (query), K (key), V (value)** har word ke vectors hain. Q aur K ka dot product batata hai kaun sa word kis par dhyan de. Softmax weights banata hai, V ka weighted sum output hai.
- **Multi-head attention:** kai attention heads parallel, har head alag relation seekhta hai.
- **Positional encoding:** word order batata hai (kyunki attention khud order nahi jaanta).
- RNN ke muqable Transformer poori sequence **parallel** process karta hai, isliye training fast aur long-range dependency behtar.

|  | BERT | GPT |
| --- | --- | --- |
| Type | Encoder only | Decoder only |
| Reads | Dono directions (bidirectional) | Left to right |
| Training | Masked language modelling | Next token prediction |
| Best for | Classification, NER, QA | Text generation, chat |

### 16.8 LLM and OpenAI API in more detail

**Roles:** `system` (behaviour set karta hai), `user` (sawal), `assistant` (jawab). Conversation memory ke liye pichli messages har request mein bhejni padti hain (model khud yaad nahi rakhta).

| Parameter | Effect |
| --- | --- |
| temperature | 0 = focused, deterministic; 1+ = creative, random |
| top\_p | Sirf top probability mass ke tokens mein se choose |
| max\_tokens | Output length limit (cost control) |
| stream | Tokens aate hi dikhao (chat jaisa) |

**Prompt techniques:** zero-shot, few-shot (examples do), chain-of-thought ('step by step soch'), role prompting, output format specify (JSON), delimiters se input alag karo.

**RAG pipeline (apne documents par QA):**

```
Documents -> chunk -> embed -> vector store
Question  -> embed -> top-k similar chunks -> prompt = question + chunks -> LLM answer
```

**Hallucination kam karne ke tareeke:** RAG, 'only use the context' instruction, low temperature, sources dikhana.

**Safety note:** API key ko code mein hard-code mat karo, `os.environ['OPENAI_API_KEY']` use karo aur GitHub par kabhi push mat karo.

---

[Previous: 15. Deep Dive: Computer Vision, OpenCV and CNN](15-deep-dive-computer-vision-opencv-and-cnn.md) | [Home](../README.md) | [Next: 17. Deep Dive: RNN, LSTM, GRU, Fake News and Chatbot](17-deep-dive-rnn-lstm-gru-fake-news-and-chatbot.md)
