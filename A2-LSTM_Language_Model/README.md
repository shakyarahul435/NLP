# LSTM Language Model for Nepali Text

A full-stack web application featuring an LSTM-based language model for generating Nepali text, with an English-to-Nepali transliteration tool.

## 🚀 Features

- **Language Model Generation**: Generate coherent Nepali text using a trained LSTM model
- **English-to-Nepali Transliteration**: Convert English text to Nepali script using a custom mapping dictionary
- **Interactive Web Interface**: Modern React frontend with real-time transliteration
- **REST API**: Django REST Framework backend serving the language model
- **Configurable Parameters**: Adjust temperature and maximum length for text generation

## 🛠 Tech Stack

### Frontend
- **React** - User interface
- **Axios** - HTTP client for API calls
- **CSS** - Styling with modern gradients and responsive design

### Backend
- **Django** - Web framework
- **Django REST Framework** - API development
- **PyTorch** - Deep learning framework for LSTM model
- **NumPy** - Numerical computations

### Model
- **LSTM Architecture**: 2-layer LSTM with embedding and dropout
- **Vocabulary**: Custom tokenization for Nepali text
- **Training**: PyTorch-based training on Nepali text corpus

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+
- pip (Python package manager)
- npm (Node.js package manager)

## 🔧 Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd lstm-language_model_app/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env
   # On Windows:
   env\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. The model files should be in `backend/model/`:
   - `vocab.pkl` - Vocabulary dictionary
   - `itos.pkl` - Index to string mapping
   - `best-model.pt` - Trained PyTorch model

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd lstm-language_model_app/frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

## 🚀 Running the Application

### Start the Backend

1. Activate the virtual environment:
   ```bash
   # On Windows:
   env\Scripts\activate
   ```

2. Start the Django server:
   ```bash
   python manage.py runserver
   ```

   The API will be available at `http://127.0.0.1:8000`

### Start the Frontend

1. In a new terminal, navigate to the frontend directory:
   ```bash
   cd lstm-language_model_app/frontend
   ```

2. Start the React development server:
   ```bash
   npm start
   ```

   The application will open at `http://localhost:3000`

## 📖 Usage

### Text Generation
1. Enter a Nepali prompt in the "Prompt" field
2. Adjust the temperature (0.1-2.0) for creativity control
3. Set maximum length for generated text
4. Click "Generate" to create text

### Transliteration
1. Type English text in the transliteration input
2. See real-time conversion to Nepali script
3. Supports common Nepali words and phrases

## 🔌 API Endpoints

### Generate Text
- **URL**: `POST /api/generate/`
- **Body**:
  ```json
  {
    "prompt": "नेपाल",
    "temperature": 1.0,
    "max_len": 20,
    "seed": 42
  }
  ```
- **Response**:
  ```json
  {
    "generated": "नेपाल एक सुन्दर देश हो"
  }
  ```

### Transliterate Text
- **URL**: `POST /api/transliterate/`
- **Body**:
  ```json
  {
    "text": "namaste nepal"
  }
  ```
- **Response**:
  ```json
  {
    "transliterated": "नमस्ते नेपाल"
  }
  ```

## 🏗 Project Structure

```
A2/
├── lstm-language_model_app/
│   ├── backend/
│   │   ├── lm/
│   │   │   ├── models.py
│   │   │   ├── views.py
│   │   │   ├── urls.py
│   │   │   └── apps.py
│   │   ├── backend/
│   │   │   ├── settings.py
│   │   │   ├── urls.py
│   │   │   └── wsgi.py
│   │   ├── model/
│   │   │   ├── vocab.pkl
│   │   │   ├── itos.pkl
│   │   │   └── best-model.pt
│   │   ├── manage.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── public/
│       ├── src/
│       │   ├── App.js
│       │   ├── App.css
│       │   └── index.js
│       ├── package.json
│       └── node_modules/
├── A2.ipynb
├── LSTM LM.ipynb
├── best-model.pt
├── README.md
└── .gitignore
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Built for NLP course assignment
- Uses PyTorch for deep learning
- Inspired by modern language model architectures
- Nepali text corpus for training

## Author
Rahul Shakya <br />
st125982<br />
Asian Institute of Technology - AIT
