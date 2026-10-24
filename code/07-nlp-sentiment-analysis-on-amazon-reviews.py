# Reference snippets from notes/07-nlp-sentiment-analysis-on-amazon-reviews.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(df['clean_review'], df['label'], test_size=0.2, stratify=df['label'])
vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(X_train), y_train)
print(classification_report(y_test, clf.predict(vec.transform(X_test))))

# ---- Snippet 2 ----
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

