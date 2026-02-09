import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  return (
    <div>
      <div style={{backgroundColor: '#2b76ef', fontSize: '36px', textAlign: 'center', color: 'white', fontWeight: 'bold'}}>LSTM Language Model for Nepali Text</div>
      <div className="App">
        <Translator />
        <LanguageGenerator />
      </div>
    </div>
  );
}

function Translator() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');

  const transliterate = (text) => {
    const mapping = {
      'hamro': 'हाम्रो',
      'hamra': 'हाम्रा',
      'nepal': 'नेपाल',
      'ho': 'हो',
      'ek': 'एक',
      'sundar': 'सुन्दर',
      'desh': 'देश',
      'ma': 'म',
      'timro': 'तिम्रो',
      'mero': 'मेरो',
      'k': 'के',
      'ko': 'को',
      'lai': 'लाई',
      'bata': 'बाट',
      'sanga': 'सँग',
      'cha': 'छ',
      'thiyo': 'थियो',
      'garchu': 'गर्छु',
      'gara': 'गर',
      'deu': 'देऊ',
      'linu': 'लिनु',
      'aaja': 'आज',
      'kal': 'काल',
      'bihana': 'बिहान',
      'sanjh': 'साँझ',
      'namaste': 'नमस्ते',
      'dhanyabad': 'धन्यवाद',
      'kasto': 'कस्तो',
      'ramro': 'राम्रो',
      'naya': 'नया',
      'puro': 'पुरो',
      'chhoti': 'छोटी',
      'thulo': 'ठूलो',
      'kalo': 'कालो',
      'seto': 'सेतो',
      'hariyo': 'हरियो',
      'pani': 'पानी',
      'khana': 'खाना',
      'ghar': 'घर',
      'kitab': 'किताब',
      'school': 'स्कूल',
      'bazaar': 'बजार',
      'bhai': 'भाई',
      'bahan': 'बहिनी',
      'ama': 'आमा',
      'baba': 'बाबा',
      'didi': 'दिदी',
      'dai': 'दाइ',
      'keta': 'कता',
      'yaha': 'यहाँ',
      'tyaha': 'त्यहाँ',
      'kaha': 'कहाँ',
      'kahile': 'कहिले',
      'kati': 'कति',
      'kati ota': 'कति ओटा',
      'sathi': 'साथी',
      'priya': 'प्रिय',
      'sathi ho': 'साथी हो',
      'k cha': 'के छ',
      'k garne': 'के गर्ने',
      'khanu': 'खानु',
      'khau': 'खाऊ',
      'piunu': 'पिउनु',
      'piu': 'पिऊ',
      'nau': 'नाऊ',
      'ja': 'जा',
      'aa': 'आ',
      'bas': 'बस',
      'uth': 'उठ',
      'baith': 'बैठ',
      'son': 'सुन',
      'bol': 'बोल',
      'padh': 'पढ',
      'lekh': 'लेख',
      'khel': 'खेल'
    };
    const words = text.split(' ');
    return words.map(word => mapping[word.toLowerCase()] || word).join(' ');
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    setInput(value);
    setOutput(transliterate(value));
  };

  return (
    <div className="section">
      <h2>English to Nepali Translation</h2>
      <input
        type="text"
        value={input}
        onChange={handleInputChange}
        placeholder="Type English text, e.g., hamra"
      />
      <p>Output: {output}</p>
    </div>
  );
}

function LanguageGenerator() {
  const [prompt, setPrompt] = useState('');
  const [generated, setGenerated] = useState('');
  const [temperature, setTemperature] = useState(1.0);
  const [maxLen, setMaxLen] = useState(20);

  const handleGenerate = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/generate/', {
        prompt,
        temperature: parseFloat(temperature),
        max_len: parseInt(maxLen),
        seed: 42
      });
      console.log('Response:', response.data);
      setGenerated(response.data.generated);
    } catch (error) {
      console.error('Error generating:', error);
      setGenerated('Error: ' + (error.response ? error.response.data : error.message));
    }
  };

  return (
    <div className="section">
      <h2>Language Model Generation</h2>
      <h3>Generation Parameters</h3>
      <div>
        <label>Prompt:</label>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter prompt in Nepali"
        />
      </div>
      <div>
        <label>Temperature:</label>
        <input
          type="number"
          value={temperature}
          onChange={(e) => setTemperature(e.target.value)}
          step="0.1"
          min="0.1"
          max="2.0"
        />
      </div>
      <div>
        <label>Max Length:</label>
        <input
          type="number"
          value={maxLen}
          onChange={(e) => setMaxLen(e.target.value)}
        />
      </div>
      <button onClick={handleGenerate}>Generate</button>
      <p>Generated: {generated}</p>
    </div>
  );
}

export default App;
