import React, { useState } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('similarity');
  const [sentenceA, setSentenceA] = useState('');
  const [sentenceB, setSentenceB] = useState('');
  const [maskedSentence, setMaskedSentence] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSimilarity = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/similarity/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sentence_a: sentenceA,
          sentence_b: sentenceB,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setResult(`Similarity: ${data.similarity.toFixed(4)}`);
      } else {
        setResult(`Error: ${data.error}`);
      }
    } catch (error) {
      setResult('Error: Unable to connect to server');
    }
    setLoading(false);
  };

  const handleMaskPrediction = async () => {
    let sentence = maskedSentence.trim();
    if (!sentence.includes('[MASK]')) {
      sentence += ' [MASK]';
    }
    if (!sentence.endsWith('.')) {
      sentence += '.';
    }
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/mask/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sentence: sentence,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setResult(`Predicted: ${data.predictions[0].token_str}`);
      } else {
        setResult(`Error: ${data.error}`);
      }
    } catch (error) {
      setResult('Error: Unable to connect to server');
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>BERT Sentence App</h1>
        <div className="tabs">
          <button
            className={activeTab === 'similarity' ? 'active' : ''}
            onClick={() => setActiveTab('similarity')}
          >
            Sentence Similarity
          </button>
          <button
            className={activeTab === 'mask' ? 'active' : ''}
            onClick={() => setActiveTab('mask')}
          >
            Masked Prediction
          </button>
        </div>
        {activeTab === 'similarity' && (
          <div className="tab-content">
            <h2>Sentence Similarity (A4_2 Model)</h2>
            <textarea
              placeholder="Enter first sentence"
              value={sentenceA}
              onChange={(e) => setSentenceA(e.target.value)}
            />
            <textarea
              placeholder="Enter second sentence"
              value={sentenceB}
              onChange={(e) => setSentenceB(e.target.value)}
            />
            <button onClick={handleSimilarity} disabled={loading}>
              {loading ? 'Computing...' : 'Compute Similarity'}
            </button>
          </div>
        )}
        {activeTab === 'mask' && (
          <div className="tab-content">
            <h2>Masked Prediction (A4 Model)</h2>
            <textarea
              placeholder="Enter sentence with [MASK]"
              value={maskedSentence}
              onChange={(e) => setMaskedSentence(e.target.value)}
            />
            <p>Example: "The capital of France is [MASK]."</p>
            <button onClick={handleMaskPrediction} disabled={loading}>
              {loading ? 'Predicting...' : 'Predict Mask'}
            </button>
          </div>
        )}
        <div className="result">
          <p>{result}</p>
        </div>
      </header>
    </div>
  );
}

export default App;
