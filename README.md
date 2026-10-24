# AI, Machine Learning, Deep Learning, CV, NLP & Time Series: Complete Notes

Complete study notes from the **IT Vedant** course *Data Science & Analytics with Artificial Intelligence*. Covers definitions, diagrams, formulas, worked examples, code and interview questions in one place. Notes are written in Hinglish (Hindi + English) for fast revision.

> Author: **Iqra Siddiqui** | [LinkedIn](https://linkedin.com/in/siddiquiiqra) | [GitHub](https://github.com/iqra-insights)

## What is inside

- **19 module notes** in [`notes/`](notes/), each with previous/next navigation
- **15 diagrams** in [`images/`](images/) (PNG and SVG)
- **Runnable-style code snippets** for every module in [`code/`](code/)
- **One-file version**: [`AI_ML_Complete_Notes.md`](AI_ML_Complete_Notes.md)
- **PDF and Word downloads** in [`downloads/`](downloads/)
- Formulas are written in LaTeX and render directly on GitHub

<p align="center"><img src="images/01-ai-ml-dl-genai-nested.png" width="700" alt="AI, ML, DL and GenAI nested diagram"></p>

## Part 1: Concepts and diagrams

| # | Module |
|---|--------|
| 1 | [Understanding the World of AI](notes/01-understanding-the-world-of-ai.md) |
| 2 | [Introduction to Deep Learning](notes/02-introduction-to-deep-learning.md) |
| 3 | [Artificial Neural Network Deep Dive](notes/03-artificial-neural-network-deep-dive.md) |
| 4 | [Computer Vision](notes/04-computer-vision.md) |
| 5 | [Computer Vision with OpenCV: Images, Video, OCR, MediaPipe](notes/05-computer-vision-with-opencv-images-video-ocr-med.md) |
| 6 | [Convolutional Neural Network (CNN)](notes/06-convolutional-neural-network-cnn.md) |
| 7 | [NLP: Sentiment Analysis on Amazon Reviews](notes/07-nlp-sentiment-analysis-on-amazon-reviews.md) |
| 8 | [Sequence Models and Fake News Detection](notes/08-sequence-models-and-fake-news-detection.md) |
| 9 | [Chatbot](notes/09-chatbot.md) |
| 10 | [Time Series Analysis](notes/10-time-series-analysis.md) |
| 11 | [Weather Forecasting using ARIMA and SARIMA](notes/11-weather-forecasting-using-arima-and-sarima.md) |
| 12 | [Stock Price Prediction using Sequence Models (LSTM)](notes/12-stock-price-prediction-using-sequence-models-lst.md) |
| 13 | [Quick Revision Cheat Sheet](notes/13-quick-revision-cheat-sheet.md) |

## Part 2: Deep dives (math, worked examples, extra code)

| # | Module |
|---|--------|
| 14 | [Deep Dive: ML Basics, Perceptron and ANN](notes/14-deep-dive-ml-basics-perceptron-and-ann.md) |
| 15 | [Deep Dive: Computer Vision, OpenCV and CNN](notes/15-deep-dive-computer-vision-opencv-and-cnn.md) |
| 16 | [Deep Dive: NLP, Sentiment, Spam, Translation and LLMs](notes/16-deep-dive-nlp-sentiment-spam-translation-and-llm.md) |
| 17 | [Deep Dive: RNN, LSTM, GRU, Fake News and Chatbot](notes/17-deep-dive-rnn-lstm-gru-fake-news-and-chatbot.md) |
| 18 | [Deep Dive: Time Series, ARIMA/SARIMA and LSTM Forecasting](notes/18-deep-dive-time-series-arima-sarima-and-lstm-fore.md) |
| 19 | [Revision Plan and Mini-Project Checklist](notes/19-revision-plan-and-mini-project-checklist.md) |

## Course topics covered

Understanding AI, Introduction to DL, ANN deep dive (activation, loss, optimizers, regularization, callbacks), Computer Vision (detection, region proposals, segmentation), OpenCV (images, video, OCR, MediaPipe), CNN, NLP (preprocessing, vectorization, embeddings, sentiment, spam, translation, LLMs), RNN/LSTM/GRU, Fake News Detection, Chatbot, Time Series, ARIMA/SARIMA weather forecasting, LSTM stock forecasting.

## Suggested study order

Modules 1, 2, 3 (foundation), then 6, 7, 8, 10, 11. For Data Analyst and Data Scientist interviews, modules 3, 7, 10, 11 and the evaluation metrics in module 14 are asked most often.

## Repository structure

```
.
├── README.md
├── AI_ML_Complete_Notes.md     # everything in one file
├── notes/                      # 19 module notes
├── images/                     # 15 diagrams (PNG + SVG)
├── code/                       # Python snippets per module
├── downloads/                  # PDF and DOCX versions
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Using the code

The files in `code/` are reference snippets. Variables such as `X`, `y`, `df`, `train` and `test` are placeholders for your own data.

```bash
pip install -r requirements.txt
```

## Honesty note

Stock price models here are for learning only, not trading advice. Always compare forecasts with a naive baseline.

## License

MIT. See [LICENSE](LICENSE).
