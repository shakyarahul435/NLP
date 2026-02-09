import { useState } from 'react';
import axios from 'axios';

function App() {
  const [word, setWord] = useState('');
  const [model, setModel] = useState('word2vec-negative');
  const [results, setResults] = useState([]);
  const [corpus, setCorpus] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const styles = {
    container: {
      minHeight: '100vh',
      backgroundColor: '#f0f2f5',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      padding: '20px',
      fontFamily: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif'
    },
    card: {
      backgroundColor: 'white',
      width: '100%',
      maxWidth: '450px',
      padding: '40px',
      borderRadius: '15px',
      boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
      textAlign: 'center'
    },
    title: { color: '#1a1a1a', marginBottom: '10px', fontSize: '24px' },
    subtitle: { color: '#666', fontSize: '14px', marginBottom: '30px' },
    label: { 
      display: 'block', 
      textAlign: 'left', 
      fontSize: '11px', 
      fontWeight: 'bold', 
      color: '#999', 
      textTransform: 'uppercase',
      marginBottom: '5px'
    },
    select: {
      width: '100%',
      padding: '12px',
      marginBottom: '20px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      fontSize: '16px',
      outline: 'none'
    },
    inputGroup: { display: 'flex', gap: '10px', marginBottom: '20px' },
    input: {
      flex: 1,
      padding: '12px',
      borderRadius: '8px',
      border: '1px solid #ddd',
      fontSize: '16px'
    },
    button: {
      backgroundColor: '#007bff',
      color: 'white',
      border: 'none',
      padding: '12px 20px',
      borderRadius: '8px',
      cursor: 'pointer',
      fontWeight: 'bold',
      transition: 'background 0.3s'
    },
    sectionTitle: { color: '#1a1a1a', margin: '20px 0 10px', fontSize: '16px' }
  };

  const getSimilarWords = async () => {
    if (!word) return;
    setLoading(true);
    setError('');

    try {
      const response = await axios.post(
        'http://localhost:8000/api/similarity/',
        { word, topn: 10 }
      );

      setResults(Array.isArray(response.data?.results) ? response.data.results : []);

      setCorpus(
        Array.isArray(response.data?.corpus)
          ? response.data.corpus
              .join(' ')
              .replace(/\s([.,!?;:])/g, '$1')
          : ''
      );

    } catch (err) {
      setError(err.response?.data?.error || 'Error connecting to server');
      setResults([]);
      setCorpus('');
    }

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>Word Similarities</h1>
        <p style={styles.subtitle}>NLP model explorer (Skip-gram Negative Sampling)</p>

        <label style={styles.label}>Select Embedding Model</label>
        <select
          style={styles.select}
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="word2vec-negative">Word2Vec Negative Sampling</option>
        </select>

        <div style={styles.inputGroup}>
          <input
            style={styles.input}
            type="text"
            placeholder="Type a word..."
            value={word}
            onChange={(e) => setWord(e.target.value)}
          />
          <button
            style={{ ...styles.button, backgroundColor: loading ? '#ccc' : '#007bff' }}
            onClick={getSimilarWords}
            disabled={loading}
          >
            {loading ? '...' : 'Find'}
          </button>
        </div>

        {error && <p style={{ color: 'red', fontSize: '13px' }}>{error}</p>}

        <h3 style={styles.sectionTitle}>Similar Words</h3>
        <div style={{ textAlign: 'left' }}>
          {results.length ? (
            results.map((item, index) => (
              <div key={index} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '12px',
                backgroundColor: '#f8f9ff',
                borderRadius: '10px',
                marginBottom: '10px',
                border: '1px solid #eef0ff'
              }}>
                <span style={{ fontWeight: 'bold', color: '#2c3e50' }}>
                  {item.word}
                </span>
                <span style={{
                  backgroundColor: '#e1e7ff',
                  color: '#4a6cf7',
                  padding: '4px 10px',
                  borderRadius: '20px',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  {(item.score * 100).toFixed(1)}%
                </span>
              </div>
            ))
          ) : (
            !error && <p style={{ color: '#777', fontSize: '13px' }}>No results yet.</p>
          )}
        </div>

        <h3 style={styles.sectionTitle}>Corpus</h3>
        <div
          style={{
            textAlign: 'left',
            fontSize: '14px',
            lineHeight: '1.6',
            color: '#333',
            backgroundColor: '#f8f9ff',
            padding: '12px',
            borderRadius: '10px',
            border: '1px solid #eef0ff',
            maxHeight: '200px',
            overflowY: 'auto'
          }}
        >
          {corpus || <p style={{ color: '#777', fontSize: '13px' }}>No corpus loaded.</p>}
        </div>
      </div>
    </div>
  );
}

export default App;