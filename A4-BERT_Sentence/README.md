# BERT Sentence Similarity and Classification App

This project implements BERT (Bidirectional Encoder Representations from Transformers) for sentence similarity and natural language inference tasks. It includes both from-scratch implementations and fine-tuned models using Hugging Face Transformers, along with a web application for interactive use.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
- [Web Application](#web-application)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [References](#references)
- [License](#license)

## Features

- **From-Scratch BERT Implementation**: Complete BERT model built from scratch with masked language modeling and next sentence prediction
- **Sentence-BERT Fine-tuning**: Fine-tuned BERT for sentence similarity and entailment classification
- **Web Application**: Interactive web app with React frontend and Django backend
- **Interactive Demos**: Visualizations and demonstrations of model capabilities
- **Educational Focus**: Comprehensive implementations for learning BERT internals

## Project Structure

```
A4-BERT_Sentence/
├── A4.ipynb                    # From-scratch BERT implementation
├── A4_2.ipynb                  # Sentence-BERT fine-tuning
├── bert-sentence-app/          # Web application
│   ├── backend/                # Django REST API
│   │   ├── requirements.txt
│   │   └── manage.py
│   └── frontend/               # React application
│       └── src/
├── model/                      # Saved model files
├── results/                    # PNG, MP4, GIF results
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 14+
- pip
- npm

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd bert-sentence-app/backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run database migrations:
   ```bash
   python manage.py migrate
   ```

4. Start the Django server:
   ```bash
   python manage.py runserver
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd bert-sentence-app/frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

## Usage

### Running the Web Application

1. Start the backend server (port 8000)
2. Start the frontend server (port 3000)
3. Open your browser to `http://localhost:3000`
4. Enter premise and hypothesis sentences to get entailment predictions

### Running Notebooks

1. Install Jupyter Notebook:
   ```bash
   pip install jupyter
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `A4.ipynb` or `A4_2.ipynb` and run the cells

## Notebooks

### A4.ipynb - BERT from Scratch

- Implements complete BERT architecture from scratch
- Includes tokenization, embedding layers, transformer encoder
- Trains on Wikipedia dataset with MLM and NSP objectives
- Small model configuration (2 layers, 4 heads, 256 hidden size)

### A4_2.ipynb - Sentence-BERT Fine-tuning

- Fine-tunes pre-trained BERT on SNLI and MNLI datasets
- Implements sentence similarity and entailment classification
- Includes evaluation metrics and comparison with pre-trained models
- Web application integration

## Web Application

The web application provides an interactive interface for:

- **Sentence Similarity**: Compare semantic similarity between sentence pairs
- **Entailment Classification**: Predict entailment, contradiction, or neutral relationships
- **Masked Language Modeling**: Fill in masked tokens in sentences

### API Endpoints

- `POST /api/sentence-similarity/`: Calculate similarity between two sentences
- `POST /api/masked-prediction/`: Predict masked tokens
- `POST /api/entailment/`: Classify entailment relationships

## Results

The `results/` directory contains visualizations and demonstrations:

- **PNG Images**: Training curves, attention visualizations, model architectures
- **MP4 Videos**: Interactive model demonstrations, training progress
- **GIF Animations**: Transformer attention mechanisms, token embeddings

Key performance metrics:
- Sentence similarity accuracy on SNLI/MNLI validation sets
- Entailment classification F1 scores
- Training convergence plots

## Technologies Used

- **Backend**: Django, Django REST Framework, PyTorch
- **Frontend**: React, JavaScript, CSS
- **Machine Learning**: Hugging Face Transformers, PyTorch
- **Data Processing**: Datasets library, NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Deployment**: Django development server, React development server

## References

1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
   - Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
   - Venue: NAACL-HLT 2019
   - Link: https://arxiv.org/abs/1810.04805

2. **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**
   - Authors: Nils Reimers, Iryna Gurevych
   - Link: https://arxiv.org/abs/1908.10084

3. **Attention Is All You Need**
   - Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
   - Venue: NeurIPS 2017
   - Link: https://arxiv.org/abs/1706.03762

## Acknowledgments

This implementation is based on the original BERT paper and various educational resources. Special thanks to the open-source community for providing the tools and datasets that made this project possible.

## License

This project and implementation are provided for educational purposes. Please refer to the original papers and licenses of the datasets used.

## Author

Rahul Shakya <br />
st125982<br />
Asian Institute of Technology - AIT