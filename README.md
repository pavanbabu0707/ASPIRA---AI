# 🎯 CareerPath AI — Intelligent Career Recommendation System  

> 🚀 *An AI-powered web app that transforms your skills, education, and interests into personalized career suggestions using NLP and real-world data.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit)
![NLP](https://img.shields.io/badge/NLP-Feature_Engineering-green?style=for-the-badge&logo=spacy)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

---

## 🧠 Overview  

**CareerPath AI** is an **AI-driven career guidance platform** that leverages **Natural Language Processing (NLP)** to analyze user inputs — such as skills, interests, and background — and provide **data-backed career suggestions**.  

The model is trained using **O\*NET**, **Kaggle**, and **CareerCon** datasets, combining occupational data, skill clusters, and career trajectories into one intelligent recommendation system.  

---

## 🖥️ Demo Preview  

*(Insert your Streamlit demo GIF or screenshot here)*  
![Demo Placeholder](https://github.com/yourusername/CareerPath-AI/assets/demo.gif)

<img width="1876" height="871" alt="Screenshot 2025-10-06 160718" src="https://github.com/user-attachments/assets/cfedc578-ef52-40ba-a894-0e676be5cb6f" />


---

## 🚀 Key Features  

✅ **NLP-Based Input Understanding** — Processes free-form text using tokenization, embeddings, and semantic similarity.  
✅ **AI-Driven Recommendations** — Suggests top-matching career paths using feature mapping and similarity scores.  
✅ **Real-World Data Sources** — Trained using O\*NET occupational data and Kaggle/CareerCon career datasets.  
✅ **Interactive Web App** — Clean, responsive Streamlit/Flask interface for quick and intuitive exploration.  
✅ **Explainable Insights** — Displays feature similarities and career fit scores.  

---

## 🧩 System Architecture  

User Input
↓
NLP Preprocessing (Tokenization, Lemmatization, Embeddings)
↓
Feature Mapping & Model Inference
↓
Career Recommendation Engine
↓
Web Interface (Streamlit/Flask)     


*(You can replace this diagram with a visual architecture image — e.g., `assets/architecture.png`)*  

---

## 📊 Datasets  

| Source | Description |
|--------|-------------|
| **O\*NET** | Occupational Information Network – detailed job, skills, and education data. |
| **Kaggle** | Career and job prediction datasets for training and benchmarking. |
| **CareerCon** | CareerCon 2019 dataset – professional attributes and data science career paths. |

---

## 🧮 Model Pipeline  

1. **Data Preprocessing**
   - Tokenization, stopword removal, lemmatization  
   - Embeddings: TF-IDF, Word2Vec, or Sentence-BERT  

2. **Feature Engineering**
   - Mapping user input features (skills, roles, degrees) to job clusters  

3. **Model Training**
   - Supervised (classification) or unsupervised (clustering + similarity) models  

4. **Recommendation Engine**
   - Ranks career matches using cosine similarity or confidence scores  

---

## ⚙️ Tech Stack  

| Category | Technologies |
|-----------|--------------|
| **Languages** | Python, JavaScript |
| **Frameworks** | Streamlit / Flask / FastAPI |
| **ML/NLP Libraries** | Scikit-learn, Transformers, SpaCy, NLTK, TensorFlow / PyTorch |
| **Data Tools** | Pandas, NumPy, SQL |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Deployment** | Streamlit Cloud / AWS / Docker |

---

## 🧰 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/CareerPath-AI.git
cd CareerPath-AI


