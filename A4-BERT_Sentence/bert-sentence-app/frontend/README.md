# BERT Sentence App Frontend

A React-based frontend for the BERT Sentence Similarity and Classification application. This interface allows users to interact with BERT models for natural language understanding tasks including sentence similarity comparison and entailment classification.

## Features

- **Sentence Similarity**: Compare semantic similarity between two input sentences
- **Entailment Classification**: Determine if one sentence entails, contradicts, or is neutral to another
- **Masked Language Modeling**: Predict masked tokens in sentences (future feature)
- **Clean UI**: Modern, responsive interface with tabbed navigation
- **Real-time Results**: Instant predictions from the backend API

## Technology Stack

- **React**: Frontend framework
- **JavaScript**: Programming language
- **CSS**: Styling and layout
- **Fetch API**: HTTP requests to Django backend
- **Create React App**: Build tool and development server

## Getting Started

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Running Django backend server

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd bert-sentence-app/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Backend Connection

Ensure the Django backend is running on `http://localhost:8000`. The frontend will automatically connect to the API endpoints.

## Usage

### Sentence Similarity Tab

1. Enter two sentences in the input fields
2. Click "Calculate Similarity"
3. View the cosine similarity score

### Entailment Classification Tab

1. Enter a premise sentence
2. Enter a hypothesis sentence
3. Click "Classify Relationship"
4. View the predicted relationship (Entailment/Contradiction/Neutral) with confidence score

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/
│   │   ├── App.js          # Main application component
│   │   ├── SimilarityTab.js # Sentence similarity interface
│   │   ├── EntailmentTab.js # Entailment classification interface
│   │   └── MaskTab.js      # Masked prediction interface (future)
│   ├── App.css             # Main styles
│   ├── index.js            # Application entry point
│   └── index.css           # Global styles
├── package.json
└── README.md
```

## API Integration

The frontend communicates with the Django backend through REST API calls:

- `POST /api/sentence-similarity/`: Calculates similarity between sentences
- `POST /api/entailment/`: Classifies entailment relationships
- `POST /api/masked-prediction/`: Predicts masked tokens (planned)

## Development

### Available Scripts

- `npm start`: Runs the app in development mode
- `npm test`: Launches the test runner
- `npm run build`: Builds the app for production
- `npm run eject`: Ejects from Create React App (not recommended)

### Adding New Features

1. Create new components in `src/components/`
2. Update `App.js` to include new tabs or routes
3. Add corresponding API calls in the component
4. Update styles in `App.css`

## Styling

The application uses custom CSS for a clean, modern look:

- Responsive design that works on desktop and mobile
- Tabbed interface for different functionalities
- Form inputs with validation
- Result display with confidence scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Related Projects

- **Backend**: Django REST API with PyTorch BERT models
- **Notebooks**: Jupyter notebooks with BERT implementations
  - `A4.ipynb`: BERT from scratch
  - `A4_2.ipynb`: Sentence-BERT fine-tuning

## License

This project is for educational purposes. See the main project README for full licensing information.
