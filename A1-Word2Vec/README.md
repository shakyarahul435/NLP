# NLP Word Embedding Models & Web Application

This project implements various Natural Language Processing (NLP) models for generating word embeddings, integrated into a full-stack web application. The system consists of a research module with Jupyter Notebooks (`A1`), a Django backend for serving the models, and a React frontend for user interaction.

## 📂 Project Structure

```
├── A1/                              # Research & Model Implementation
│   ├── 01 - Word2Vec (Skipgram).ipynb
│   ├── 02 - Word2Vec (Neg Sampling).ipynb
│   ├── 03 - GloVe from Scratch.ipynb
│   └── 04 - GloVe (Gensim).ipynb
│
├── backend/                         # Django API Server
│   ├── manage.py
│   ├── db.sqlite3
│   └── core/
│       └── views.py                 # Model serving 
│
└── frontend/                        # React Client 
    ├── public/
    ├── src/
    │   └── App.js                   # Main interface
    └── package.json
```

## 🧠 Model Implementations (A1)

### 01 - Word2Vec (Skipgram)
- **Architecture**: Skip-gram with context prediction
- **Dataset**: Brown Corpus (News category)
- **Framework**: PyTorch
- **Key Feature**: Generates (center, context) pairs with dynamic window size

### 02 - Word2Vec (Negative Sampling)
- **Architecture**: Optimized Skip-gram variant
- **Dataset**: Brown Corpus (News category)
- **Framework**: PyTorch
- **Key Feature**: Efficient training via negative sampling instead of full softmax

### 03 - GloVe (Global Vectors) from Scratch
- **Architecture**: Co-occurrence matrix factorization
- **Dataset**: Brown Corpus
- **Framework**: PyTorch
- **Key Feature**: Combines global matrix statistics with local context windows

### 04 - GloVe (Gensim)
- **Model**: glove-wiki-gigaword-100
- **Dimensions**: 100-dimensional vectors
- **Pre-trained**: Wikipedia & Gigaword corpus
- **Features**: Semantic similarity, analogy resolution, word cluster visualization


## 📊 Model Performance Summary

The following table summarizes the comparative performance and characteristics of the implemented word embedding models.
*Values will be filled after experimentation and evaluation.*

| Model          | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy |
| -------------- | ----------: | ------------: | ------------: | -----------------: | ----------------: |
| Skipgram       |         2   |     7.2026    | 3239.11 sec  |        0            |     0             |
| Skipgram (NEG) |        2    |      3.61      |   3271.41 sec|          0        |      0            |
| GloVe          |        2    |      5.86     |      600 sec   |           0        |     0            |
| GloVe (Gensim) |         5|             - |             - |             0.55      |          0.89     |



## ⚙️ Backend (Django)

The backend exposes REST API endpoints to serve embeddings and compute similarities.

**Key Endpoint**: `POST /api/similarity/`
- Request: `{ word: string, model_type: string }`
- Response: `{ results: [{word: string, score: number}] }`

## 💻 Frontend (React)

Modern React SPA with an intuitive UI for exploring word similarities across all models.

**Features**:
- Model selection dropdown
- Real-time similarity search
- Visual score display (percentage format)
- Error handling & loading states

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+

### Installation

**1. Models & Backend**
```bash
# Install Python dependencies
pip install torch numpy matplotlib nltk gensim django djangorestframework

# Train models (optional - run notebooks in A1/)
cd A1
jupyter notebook

# Start Django server
cd backend
python manage.py migrate
python manage.py runserver
```

**2. Frontend**
```bash
cd frontend
npm install
npm start
```

The app will open at `http://localhost:3000` and connect to the backend at `http://localhost:8000`.

## 🛠️ Technologies

| Layer | Technology |
|-------|------------|
| **ML/NLP** | PyTorch, Gensim, NLTK |
| **Backend** | Django, Django REST Framework |
| **Frontend** | React, Axios |
| **Styling** | CSS-in-JS |

## 📝 Notes

- Ensure Django backend is running before starting the React frontend
- The backend expects models to be trained and saved in the designated paths
- CORS may need configuration if running on different ports

---

**Author**: Rahul Shakya - AIT NLP Course  
**Last Updated**: 2026