import { useState } from 'react'

const API_BASE = 'http://127.0.0.1:8000/api'

export default function App() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  const [evaluationTable, setEvaluationTable] = useState(null)

  async function askQuestion() {
    const q = question.trim()
    if (!q) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API_BASE}/ask/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      })

      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || 'Request failed')
      }
      setEvaluationTable(data.evaluation_table)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <div className="card">
        <h1>A6 Contextual Retrieval Chatbot</h1>
        <p className="subtitle">Ask questions about Chapter 2 (Words and Tokens)</p>

        <div className="inputRow">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Type your question here..."
            rows={4}
          />
          <button onClick={askQuestion} disabled={loading}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </div>

        {error && <p className="error">{error}</p>}

        {evaluationTable && (
          <div className="resultBox">
            <h2>Evaluation Table</h2>
            <div className="tableWrap">
              <table>
                <thead>
                  <tr>
                    <th>Method</th>
                    <th>ROUGE-1</th>
                    <th>ROUGE-2</th>
                    <th>ROUGE-L</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluationTable.rows.map((row) => (
                    <tr key={row.Method}>
                      <td>{row.Method}</td>
                      <td>{row['ROUGE-1'].toFixed(4)}</td>
                      <td>{row['ROUGE-2'].toFixed(4)}</td>
                      <td>{row['ROUGE-L'].toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {result && (
          <div className="resultBox">
            <h2>Answer</h2>
            <p>{result.answer}</p>

            <h3><u>Source Citation</u></h3>
            <p><strong>Document:</strong> {result.source.document}</p>
            <p><strong>Chunk ID:</strong> {result.source.chunk_id}</p>

            <h3>Chunk Before / After</h3>
            <p><strong>BEFORE:</strong></p>
            <pre>{result.source.before}</pre>
            <p><strong>AFTER:</strong></p>
            <pre>{result.source.after}</pre>
          </div>
        )}
      </div>
    </div>
  )
}
