# Reference snippets from notes/17-deep-dive-rnn-lstm-gru-fake-news-and-chatbot.md
# These are study snippets, run them piece by piece with your own data (X, y, df, etc. are placeholders).

# ---- Snippet 1 ----
import random, numpy as np
def reply(text, threshold=0.7):
    vec = vectorizer.transform([clean(text)])
    probs = model.predict(vec.toarray())[0]
    i = int(np.argmax(probs))
    if probs[i] < threshold:
        return 'Sorry, samajh nahi aaya. Kya aap dobara bata sakte hain?'
    return random.choice(responses[tags[i]])

# ---- Snippet 2 ----
history = [{'role': 'system', 'content': 'You are a helpful support assistant. Answer briefly.'}]
def chat(user_msg):
    history.append({'role': 'user', 'content': user_msg})
    r = client.chat.completions.create(model='gpt-4o-mini', messages=history, temperature=0.3)
    ans = r.choices[0].message.content
    history.append({'role': 'assistant', 'content': ans})
    return ans

# ---- Snippet 3 ----
import streamlit as st
st.title('Support Bot')
if 'msgs' not in st.session_state: st.session_state.msgs = []
for m in st.session_state.msgs: st.chat_message(m['role']).write(m['content'])
if q := st.chat_input('Ask something'):
    st.session_state.msgs.append({'role': 'user', 'content': q})
    st.session_state.msgs.append({'role': 'assistant', 'content': chat(q)})
    st.rerun()

