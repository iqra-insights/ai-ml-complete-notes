[Previous: 16. Deep Dive: NLP, Sentiment, Spam, Translation and LLMs](16-deep-dive-nlp-sentiment-spam-translation-and-llm.md) | [Home](../README.md) | [Next: 18. Deep Dive: Time Series, ARIMA/SARIMA and LSTM Forecasting](18-deep-dive-time-series-arima-sarima-and-lstm-fore.md)

---

## 17. Deep Dive: RNN, LSTM, GRU, Fake News and Chatbot

### 17.1 LSTM equations

```math
f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad i_t=\sigma(W_i[h_{t-1},x_t]+b_i),\quad o_t=\sigma(W_o[h_{t-1},x_t]+b_o)
```

```math
\tilde C_t=\tanh(W_C[h_{t-1},x_t]+b_C),\quad C_t=f_t\odot C_{t-1}+i_t\odot\tilde C_t,\quad h_t=o_t\odot\tanh(C_t)
```

Cell state `C_t` mein addition hota hai (multiplication nahi), isliye gradient lambi doori tak bachta hai.

### 17.2 GRU equations

```math
z_t=\sigma(W_z[h_{t-1},x_t]),\quad r_t=\sigma(W_r[h_{t-1},x_t]),\quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tanh(W[r_t\odot h_{t-1},x_t])
```

**Parameter count (units = u, input = d):** LSTM = 4 x (u x (u + d) + u), GRU = 3 x (u x (u + d) + u), simple RNN = 1 x (...). Example: u = 64, d = 64 gives LSTM = 4 x (64 x 128 + 64) = **33,024**.

### 17.3 Practical sequence-model tips

| Topic | Tip |
| --- | --- |
| Padding | `pad_sequences`; `Masking` / `mask_zero=True` taaki padding ignore ho |
| Exploding gradients | `optimizer=Adam(clipnorm=1.0)` |
| Stacking | Agli LSTM layer ke liye pichli mein `return_sequences=True` |
| Bidirectional | Text classification mein aksar accuracy badhata hai |
| Overfitting | Dropout + recurrent\_dropout, early stopping, kam units |
| Long text | Truncate/chunk, ya Transformer use karo |
| Speed | GRU LSTM se tez, accuracy aksar similar |

### 17.4 Fake news detection: full approach

1. **Data:** title + text + label (real/fake). Duplicates hatao, class balance check.
2. **Leakage check:** source ka naam, date format ya 'Reuters' jaise tokens se model shortcut seekh sakta hai. Aise tokens hatao.
3. **Baseline:** TF-IDF + Logistic Regression ya PassiveAggressiveClassifier.
4. **DL model:** Tokenizer, pad, Embedding (ya GloVe), Bidirectional LSTM/GRU, Dense sigmoid.
5. **Evaluate:** Precision, Recall, F1, confusion matrix. Real news ko fake bolna aur fake ko real bolna dono ki cost alag hai, isse threshold choose karo.
6. **Explain:** top TF-IDF coefficients ya LIME se dikhao model kin words par depend karta hai.
7. **Limitation likho:** model language style pakadta hai, fact-checking nahi karta.

### 17.5 Chatbot in more detail

**intents.json structure:**

```json
{"intents": [
  {"tag": "greeting", "patterns": ["hi", "hello", "hey there"], "responses": ["Hello! How can I help?"]},
  {"tag": "hours", "patterns": ["what are your timings", "when do you open"], "responses": ["We are open 9 to 6, Mon to Sat."]}
]}
```

**Response logic:**

```python
import random, numpy as np
def reply(text, threshold=0.7):
    vec = vectorizer.transform([clean(text)])
    probs = model.predict(vec.toarray())[0]
    i = int(np.argmax(probs))
    if probs[i] < threshold:
        return 'Sorry, samajh nahi aaya. Kya aap dobara bata sakte hain?'
    return random.choice(responses[tags[i]])
```

**LLM chatbot with memory:**

```python
history = [{'role': 'system', 'content': 'You are a helpful support assistant. Answer briefly.'}]
def chat(user_msg):
    history.append({'role': 'user', 'content': user_msg})
    r = client.chat.completions.create(model='gpt-4o-mini', messages=history, temperature=0.3)
    ans = r.choices[0].message.content
    history.append({'role': 'assistant', 'content': ans})
    return ans
```

**Streamlit UI (minimal):**

```python
import streamlit as st
st.title('Support Bot')
if 'msgs' not in st.session_state: st.session_state.msgs = []
for m in st.session_state.msgs: st.chat_message(m['role']).write(m['content'])
if q := st.chat_input('Ask something'):
    st.session_state.msgs.append({'role': 'user', 'content': q})
    st.session_state.msgs.append({'role': 'assistant', 'content': chat(q)})
    st.rerun()
```

**Evaluate chatbot:** intent accuracy, fallback rate, response relevance (human rating), latency.

---

[Previous: 16. Deep Dive: NLP, Sentiment, Spam, Translation and LLMs](16-deep-dive-nlp-sentiment-spam-translation-and-llm.md) | [Home](../README.md) | [Next: 18. Deep Dive: Time Series, ARIMA/SARIMA and LSTM Forecasting](18-deep-dive-time-series-arima-sarima-and-lstm-fore.md)
