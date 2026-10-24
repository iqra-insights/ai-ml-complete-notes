# Reference snippets from notes/16-deep-dive-nlp-sentiment-spam-translation-and-llm.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
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

# ---- Snippet 2 ----
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

# ---- Snippet 3 ----
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
pipe = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB(alpha=1.0))
pipe.fit(X_train, y_train)

