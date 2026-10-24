[Home](../README.md) | [Next: 2. Introduction to Deep Learning](02-introduction-to-deep-learning.md)

---

## 1. Understanding the World of AI

### 1.1 What is AI?

AI (Artificial Intelligence) = machines ko aisa banana ki wo human jaisi intelligence dikhayein: seekhna, reason karna, decision lena, language samajhna.

**Types of AI (capability ke basis par)**

| Type | Meaning | Example |
| --- | --- | --- |
| Narrow AI (ANI) | Ek specific kaam mein expert | Siri, Netflix recommendations, spam filter |
| General AI (AGI) | Insaan jaisa har kaam kar sake | Abhi exist nahi karta |
| Super AI (ASI) | Insaan se zyada smart | Sirf theory / sci-fi |

**Types of AI (functionality ke basis par):** Reactive Machines (Deep Blue chess), Limited Memory (self-driving cars), Theory of Mind (research stage), Self-Aware (hypothetical).

### 1.2 AI vs ML vs DL

<p align="center"><img src="../images/01-ai-ml-dl-genai-nested.png" alt="AI, ML, DL and GenAI · nested diagram" width="800"><br><sub><i>AI, ML, DL and GenAI · nested diagram</i></sub></p>

| Point | AI | ML | DL |
| --- | --- | --- | --- |
| Scope | Sabse bada | AI ka subset | ML ka subset |
| Data need | Varies | Medium | Bahut zyada |
| Feature engineering | Manual rules | Manual | Automatic |
| Hardware | Normal | CPU ok | GPU/TPU chahiye |
| Example | Rule-based expert system | Random Forest, SVM | CNN, LSTM, Transformer |

**ML ke 3 types:** Supervised (labelled data: regression, classification), Unsupervised (unlabelled: clustering, PCA), Reinforcement (reward/penalty se seekhna).

### 1.3 Applications of AI

- **Healthcare:** disease detection from X-ray/MRI, drug discovery
- **Finance:** fraud detection, credit scoring, algorithmic trading
- **Retail/E-commerce:** recommendation engines, demand forecasting
- **Transport:** self-driving cars, route optimisation
- **Education:** personalised learning, auto-grading
- **Agriculture:** crop disease detection, yield prediction
- **Customer service:** chatbots, sentiment analysis
- **Security:** face recognition, anomaly detection

### 1.4 Introduction to GenAI

Generative AI = naya content create karta hai (text, image, audio, code, video) training data ke patterns seekhkar.

<p align="center"><img src="../images/02-genai-lifecycle.png" alt="GenAI lifecycle · build once, prompt many times" width="800"><br><sub><i>GenAI lifecycle · build once, prompt many times</i></sub></p>

| Model family | Kaam | Example |
| --- | --- | --- |
| LLM (Transformer) | Text/code generation | GPT, Claude, Gemini |
| Diffusion | Image generation (noise se image) | Stable Diffusion, DALL-E |
| GAN | Generator vs Discriminator ka game | Deepfakes, image synthesis |
| VAE | Latent space se generate | Image/anomaly tasks |

**Key terms:** Prompt, Token, Context window, Hallucination (galat but confident output), Fine-tuning, RAG (Retrieval Augmented Generation: external documents se facts lana).

**How an LLM works (step by step)**

1. Text ko **tokens** mein todte hain (words ya word-pieces).
2. Har token ek **embedding vector** ban jata hai.
3. **Transformer** layers (self-attention) context samajhti hain.
4. Model agle token ki probability predict karta hai, ek token chunta hai, aur ye loop chalta rehta hai jab tak jawab poora na ho.

**Training stages:** Pretraining (internet-scale text par next-token prediction), Instruction fine-tuning (sawal-jawab examples), RLHF ya preference tuning (insaani feedback se behtar aur safe jawab).

**Diffusion models (images):** Training mein image par dheere dheere noise add karte hain, aur model seekhta hai noise hatana. Generation mein pure noise se shuru karke step by step saaf image banate hain, text prompt se guide hokar.

**GAN:** Do networks ka game. Generator nakli data banata hai, Discriminator asli aur nakli mein fark pakadta hai.

```math
\min_G\max_D\; \mathbb{E}[\log D(x)]+\mathbb{E}[\log(1-D(G(z)))]
```

GAN training unstable hoti hai (mode collapse). Isliye image generation mein ab diffusion zyada common hai.

**VAE:** Data ko chhote latent space mein compress karke wapas banata hai. Anomaly detection aur smooth interpolation ke liye useful.

**GenAI use cases**

| Domain | Use |
| --- | --- |
| Text | Chatbots, summarisation, translation, email drafting |
| Code | Autocomplete, bug fixing, test generation |
| Images | Design, marketing creatives, product mockups |
| Audio/Video | Voice cloning, dubbing, video generation |
| Data work | SQL generation, report summaries, data cleaning scripts |

**Limitations:** hallucination, outdated knowledge, bias from training data, prompt injection, privacy leaks if sensitive data paste kiya, copyright questions. **Hamesha important facts verify karo.**

**Adapting a model:** Prompting (sabse sasta) -> RAG (apne documents) -> Fine-tuning (style/behaviour badalna, LoRA se sasta) -> Training from scratch (bahut mehnga).

**Extra glossary:** Multimodal (text + image + audio ek model mein), Agent (model jo tools use karke multi-step kaam kare), Embedding (meaning ka vector), Vector database (embeddings search), LoRA (kam parameters train karke fine-tune), Prompt injection (hidden instructions se model ko bahkana).

### 1.5 Future of AI

- AI agents jo multi-step tasks khud karein
- Multimodal models (text + image + audio ek saath)
- Edge AI (device par hi chalne wala AI)
- AI in healthcare/science (drug discovery, protein folding)
- Challenges: bias, privacy, job displacement, deepfakes, regulation, energy use
- **Responsible AI:** fairness, transparency, accountability, safety

**Major trends to know**

| Trend | Matlab |
| --- | --- |
| AI agents | Model plan banata hai, tools (search, code, APIs) use karta hai, multi-step kaam khud karta hai |
| Multimodal AI | Text, image, audio, video ek saath samajhna aur banana |
| Reasoning models | Jawab dene se pehle step-by-step sochne wale models |
| Small and on-device models | Phone ya laptop par chalne wale halke models, privacy aur speed |
| AI for science | Drug discovery, protein structure, weather and climate modelling |
| Robotics (embodied AI) | Vision + language models se robots ko instructions samjhana |
| AI in analytics | Natural-language se SQL, automated dashboards, insight summaries |

**Risks aur mitigation**

| Risk | Example | Mitigation |
| --- | --- | --- |
| Bias | Hiring model kisi group ko disadvantage de | Diverse data, fairness metrics, audits |
| Privacy | Personal data training ya prompts mein leak | Anonymisation, access control, data policies |
| Misinformation | Deepfakes, fake news at scale | Detection tools, watermarking, media literacy |
| Job impact | Routine tasks automate | Reskilling, AI as assistant, human in the loop |
| Security | Prompt injection, model misuse | Input filtering, red-teaming, least-privilege tools |
| Energy use | Bade models bahut power lete hain | Efficient models, better hardware |

**Regulation (high level):** Duniya bhar mein AI rules ban rahe hain, jaise EU ka AI Act (risk-based approach) aur India ka Digital Personal Data Protection Act, 2023 (personal data ke liye). Interview mein itna jaanna kaafi hai ki regulation risk aur data privacy par focus karti hai.

**Responsible AI principles:** Fairness, Transparency (explainability), Accountability, Privacy, Safety, Human oversight.

**Career angle: data aur AI roles**

| Role | Focus | Key skills |
| --- | --- | --- |
| Data Analyst | Insights, dashboards, reports | SQL, Excel, Power BI/Tableau, stats |
| Data Scientist | Modelling, experiments | Python, ML, statistics, communication |
| ML Engineer | Models ko production mein lana | Python, MLOps, APIs, cloud |
| AI / LLM Engineer | LLM apps, RAG, agents | Prompting, vector DBs, APIs, evaluation |
| MLOps Engineer | Deployment, monitoring, pipelines | Docker, CI/CD, cloud, monitoring |

**Future-proof skills:** strong fundamentals (stats, SQL, Python), problem framing, data storytelling, aur AI tools ko apne kaam mein effectively use karna.

---

[Home](../README.md) | [Next: 2. Introduction to Deep Learning](02-introduction-to-deep-learning.md)
