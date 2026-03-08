import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000/api";

function StatCard({ title, value, hint }) {
  return (
    <div className="stat-card">
      <p className="stat-title">{title}</p>
      <h3>{value}</h3>
      <small>{hint}</small>
    </div>
  );
}

function LossLineChart({ points, title, stroke }) {
  if (!points.length) {
    return (
      <div className="chart-card">
        <h3>{title}</h3>
        <p className="muted">No data available</p>
      </div>
    );
  }

  const width = 560;
  const height = 200;
  const pad = 24;
  const steps = points.map((p) => p.step);
  const values = points.map((p) => p.value);
  const minStep = Math.min(...steps);
  const maxStep = Math.max(...steps);
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);

  const xScale = (step) => {
    if (maxStep === minStep) return width / 2;
    return pad + ((step - minStep) / (maxStep - minStep)) * (width - pad * 2);
  };
  const yScale = (val) => {
    if (maxVal === minVal) return height / 2;
    return height - pad - ((val - minVal) / (maxVal - minVal)) * (height - pad * 2);
  };

  const linePath = points
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${xScale(p.step).toFixed(1)} ${yScale(p.value).toFixed(1)}`)
    .join(" ");

  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img" aria-label={`${title} chart`}>
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="axis" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="axis" />
        <path d={linePath} fill="none" stroke={stroke} strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      <p className="muted">
        Step {minStep} to {maxStep} | min {minVal.toFixed(4)} | max {maxVal.toFixed(4)}
      </p>
    </div>
  );
}

function AccuracyChart({ accuracy }) {
  const bounded = Math.max(0, Math.min(100, Number(accuracy) || 0));
  return (
    <div className="chart-card">
      <h3>Model B Accuracy (Win Rate)</h3>
      <div className="accuracy-track">
        <div className="accuracy-fill" style={{ width: `${bounded}%` }} />
      </div>
      <p className="muted">{bounded.toFixed(2)}% (formula uses tie = 0.5)</p>
    </div>
  );
}

function App() {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [prompt, setPrompt] = useState("Is earth flat? Answer in one sentence.");
  const [generation, setGeneration] = useState("");
  const [history, setHistory] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const run = async () => {
      try {
        const res = await fetch(`${API_BASE}/report/`);
        if (!res.ok) {
          throw new Error("Failed to load report data.");
        }
        const data = await res.json();
        setReport(data);
      } catch (e) {
        setError(e.message || "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    run();
  }, []);

  const metrics = report?.metrics || {};
  const winners = report?.judge_results || [];
  const lossLogs = report?.loss_logs || [];

  const { trainLossPoints, evalLossPoints } = useMemo(() => {
    const clean = (key) =>
      lossLogs
        .map((row) => ({ step: Number(row.step), value: Number(row[key]) }))
        .filter((row) => Number.isFinite(row.step) && Number.isFinite(row.value))
        .sort((a, b) => a.step - b.step);

    return {
      trainLossPoints: clean("train_loss"),
      evalLossPoints: clean("eval_loss")
    };
  }, [lossLogs]);

  const winnerBreakdown = useMemo(() => {
    const count = { "Model A": 0, "Model B": 0, Tie: 0 };
    winners.forEach((x) => {
      if (count[x.winner] !== undefined) {
        count[x.winner] += 1;
      }
    });
    return count;
  }, [winners]);

  const generate = async () => {
    setIsGenerating(true);
    setGeneration("");
    setError("");

    try {
      const res = await fetch(`${API_BASE}/generate/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: 180 })
      });

      if (!res.ok) {
        throw new Error("Generation failed. Ensure backend model files exist.");
      }

      const data = await res.json();
      const reply = data.response || "";
      setGeneration(reply);
      setHistory((prev) => [
        {
          id: Date.now(),
          prompt,
          reply,
        },
        ...prev,
      ]);
    } catch (e) {
      setError(e.message || "Unknown generation error");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <main className="page">
      <header className="hero">
        <p className="chip">NLP 2026 - Assignment A5</p>
        <h1>Human Preference Alignment Dashboard</h1>
        <p>Serving your saved DPO model and evaluation outputs from <code>a5_outputs</code> without retraining.</p>
      </header>

      {loading ? <p>Loading report...</p> : null}
      {error ? <p className="error">{error}</p> : null}

      {!loading && report ? (
        <>
          <section className="stats-grid">
            <StatCard title="Model B Win Rate" value={`${metrics.win_rate_percent ?? 0}%`} hint="tie = 0.5" />
            <StatCard title="Model B Wins" value={metrics.model_b_wins ?? 0} hint="Judge verdict counts" />
            <StatCard title="Ties" value={metrics.ties ?? 0} hint="Equal quality verdicts" />
            <StatCard title="Total Valid" value={metrics.total_valid ?? 0} hint="Total judged samples" />
          </section>

          <section className="panel">
            <h2>Judge Distribution</h2>
            <div className="bar-wrap">
              {["Model A", "Model B", "Tie"].map((key) => (
                <div className="bar-row" key={key}>
                  <span>{key}</span>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{ width: `${winners.length ? (winnerBreakdown[key] / winners.length) * 100 : 0}%` }}
                    />
                  </div>
                  <strong>{winnerBreakdown[key]}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel">
            <h2>Accuracy and Loss Charts</h2>
            <div className="charts-grid">
              <AccuracyChart accuracy={metrics.win_rate_percent} />
              <LossLineChart points={trainLossPoints} title="Training Loss" stroke="#0f766e" />
              <LossLineChart points={evalLossPoints} title="Evaluation Loss" stroke="#ca8a04" />
            </div>
          </section>

          <section className="panel">
            <h2>Sample Judge Results</h2>
            <div className="table-shell">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Instruction</th>
                    <th>Winner</th>
                  </tr>
                </thead>
                <tbody>
                  {winners.slice(0, 20).map((row) => (
                    <tr key={row.sample_id}>
                      <td>{row.sample_id}</td>
                      <td>{row.instruction_truncated}</td>
                      <td>{row.winner}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="panel">
            <h2>DPO Prompt Playground</h2>
            <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={6} />
            <button onClick={generate} disabled={isGenerating || !prompt.trim()}>
              {isGenerating ? "Generating..." : "Generate from DPO model"}
            </button>

            {generation ? (
              <div className="reply-card">
                <p className="reply-label">Latest Reply</p>
                <pre>{generation}</pre>
              </div>
            ) : null}

            {history.length > 1 ? (
              <div className="history-list">
                <h3>Previous Prompt Replies</h3>
                {history.slice(1).map((item) => (
                  <article className="history-item" key={item.id}>
                    <p className="history-title">Prompt</p>
                    <p className="history-text">{item.prompt}</p>
                    <p className="history-title">Reply</p>
                    <pre>{item.reply}</pre>
                  </article>
                ))}
              </div>
            ) : null}
          </section>
        </>
      ) : null}
    </main>
  );
}

export default App;
