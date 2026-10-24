[Previous: 8. Sequence Models and Fake News Detection](08-sequence-models-and-fake-news-detection.md) | [Home](../README.md) | [Next: 10. Time Series Analysis](10-time-series-analysis.md)

---

## 9. Chatbot

| Type | How | Example |
| --- | --- | --- |
| Rule-based | If-else / pattern matching | FAQ bot |
| Retrieval-based | Predefined responses mein se best chunta hai | Intent classifier |
| Generative | Naya response generate (Seq2Seq / LLM) | ChatGPT-style |

**Architecture**

<p align="center"><img src="../images/14-chatbot-architecture.svg" alt="chatbot architecture with fallback branch" width="800"><br><sub><i>chatbot architecture with fallback branch</i></sub></p>

**Intent-based chatbot steps**

1. `intents.json` banao (tag, patterns, responses)
2. Patterns ko tokenize + lemmatize
3. Bag-of-words / embeddings banao
4. Neural network train karo (intent classify)
5. Predict karke us tag ka response return karo
6. Confidence threshold set karo (low confidence par 'Samajh nahi aaya')

**LLM chatbot:** conversation history messages list mein bhejo (memory), system prompt se personality/rules set karo. **RAG** se apne documents par answer do. Deploy: Streamlit / Flask.

---

[Previous: 8. Sequence Models and Fake News Detection](08-sequence-models-and-fake-news-detection.md) | [Home](../README.md) | [Next: 10. Time Series Analysis](10-time-series-analysis.md)
