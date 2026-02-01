import { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";

const SAMPLE_PROMPT =
  "Draft a 4-line status update in the voice of a city power grid operator.";

const formatNumber = (value, digits = 1) => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }
  return Number(value).toFixed(digits);
};

const badgeStyle = (ok) => {
  if (ok === null) {
    return { background: "rgba(10, 26, 47, 0.08)" };
  }
  return {
    background: ok ? "rgba(46, 229, 157, 0.25)" : "rgba(255, 122, 89, 0.25)",
  };
};

function App() {
  const [statusBadge, setStatusBadge] = useState({ text: "Linking...", ok: null });
  const [healthBadge, setHealthBadge] = useState({ text: "checking", ok: null });
  const [workersOk, setWorkersOk] = useState("--");
  const [workersMiss, setWorkersMiss] = useState("--");
  const [statusHint, setStatusHint] = useState("Listening for the city grid.");
  const [throughput, setThroughput] = useState("-- tok/s avg");
  const [liveRate, setLiveRate] = useState("-- tok/s live");
  const [feedState, setFeedState] = useState("idle");
  const [feedHint, setFeedHint] = useState(
    "Responses will appear here once the cluster answers."
  );
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(160);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [panelsCollapsed, setPanelsCollapsed] = useState(
    typeof window !== "undefined" ? window.innerWidth < 900 : false
  );

  const messagesRef = useRef(null);
  const conversationRef = useRef([]);
  const liveTokenCountRef = useRef(0);
  const tokenSeriesRef = useRef([]);
  const chartRef = useRef(null);
  const chartInstanceRef = useRef(null);

  useEffect(() => {
    document.body.classList.toggle("panels-collapsed", panelsCollapsed);
  }, [panelsCollapsed]);

  useEffect(() => {
    if (!chartRef.current) {
      return undefined;
    }
    const ctx = chartRef.current.getContext("2d");
    if (!ctx) {
      return undefined;
    }
    chartInstanceRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            data: [],
            borderColor: "rgba(0, 194, 255, 0.95)",
            borderWidth: 2,
            fill: false,
            tension: 0.35,
            pointRadius: (context) =>
              context.dataIndex === context.dataset.data.length - 1 ? 2.5 : 0,
            pointBackgroundColor: "rgba(46, 229, 157, 0.9)",
            pointHoverRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
        },
        scales: {
          x: { display: false, grid: { display: false, drawBorder: false } },
          y: { display: false, grid: { display: false, drawBorder: false } },
        },
      },
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const tps = liveTokenCountRef.current;
      const rate = tps / 1;
      const series = tokenSeriesRef.current;
      series.push(rate);
      if (series.length > 40) {
        series.shift();
      }
      tokenSeriesRef.current = series;
      setLiveRate(`${formatNumber(rate, 1)} tok/s live`);
      if (chartInstanceRef.current) {
        chartInstanceRef.current.data.labels = series.map((_, i) => i + 1);
        chartInstanceRef.current.data.datasets[0].data = series;
        chartInstanceRef.current.update("none");
      }
      liveTokenCountRef.current = 0;
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  const refreshStatus = async () => {
    try {
      const response = await fetch("/api/status");
      if (!response.ok) {
        throw new Error("Status unavailable");
      }
      const data = await response.json();
      const workers = data.workers ?? 0;
      const expected = data.expected_workers ?? 0;
      setWorkersOk(expected ? `${workers}` : "--");
      setWorkersMiss(expected ? `${Math.max(expected - workers, 0)}` : "--");
      setStatusBadge({
        text: data.cluster_ready ? "linked" : "offline",
        ok: data.cluster_ready,
      });
      setHealthBadge({
        text: data.cluster_healthy ? "healthy" : "degraded",
        ok: data.cluster_healthy,
      });
      setStatusHint(
        data.cluster_ready ? "Cluster ready for inference." : "Waiting for workers to report in."
      );
    } catch (error) {
      setWorkersOk("--");
      setWorkersMiss("--");
      setStatusBadge({ text: "offline", ok: false });
      setHealthBadge({ text: "unknown", ok: false });
      setStatusHint("Coordinator not reachable.");
    }
  };

  const refreshMetrics = async () => {
    try {
      const response = await fetch("/api/metrics");
      if (!response.ok) {
        throw new Error("Metrics unavailable");
      }
      const data = await response.json();
      const summary = data.summary || {};
      const avgThroughput = formatNumber(summary.avg_throughput_tokens_per_sec, 1);
      setThroughput(`${avgThroughput} tok/s avg`);
    } catch (error) {
      setThroughput("-- tok/s avg");
    }
  };

  useEffect(() => {
    refreshStatus();
    refreshMetrics();
    const statusInterval = setInterval(refreshStatus, 5000);
    const metricsInterval = setInterval(refreshMetrics, 8000);
    return () => {
      clearInterval(statusInterval);
      clearInterval(metricsInterval);
    };
  }, []);

  const setBusy = (state) => {
    setFeedState(state ? "sending" : "idle");
  };

  const addMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  const updateMessage = (id, updater) => {
    setMessages((prev) =>
      prev.map((message) => (message.id === id ? updater(message) : message))
    );
  };

  const sendPrompt = async () => {
    const trimmed = prompt.trim();
    if (!trimmed) {
      return;
    }

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      meta: "you",
      content: trimmed,
    };
    addMessage(userMessage);
    conversationRef.current = [
      ...conversationRef.current,
      { role: "user", content: trimmed },
    ];
    setPrompt("");
    setBusy(true);
    setFeedHint("Awaiting response from the grid...");

    const assistantId = `assistant-${Date.now()}`;
    addMessage({ id: assistantId, role: "assistant", meta: "grid", content: "" });
    let fullText = "";

    try {
      const payload = {
        prompt: trimmed,
        messages: conversationRef.current.slice(-12),
        max_new_tokens: Number(maxTokens || 160),
        temperature: Number(temperature || 0.7),
        top_p: Number(topP || 0.9),
      };

      const response = await fetch("/api/inference/text/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok || !response.body) {
        throw new Error("Inference failed");
      }

      setFeedHint("Streaming response from the grid...");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: token")) {
            continue;
          }
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            try {
              const parsed = JSON.parse(data);
              if (parsed.token) {
                fullText += parsed.token;
                liveTokenCountRef.current += 1;
                updateMessage(assistantId, (message) => ({
                  ...message,
                  content: message.content + parsed.token,
                }));
              } else if (parsed.error) {
                throw new Error(parsed.error);
              }
            } catch (err) {
              if (err.message !== "Unexpected end of JSON input") {
                throw err;
              }
            }
          }
        }
      }

      if (!fullText) {
        updateMessage(assistantId, (message) => ({
          ...message,
          content: "No text output returned.",
        }));
      } else {
        conversationRef.current = [
          ...conversationRef.current,
          { role: "assistant", content: fullText },
        ];
      }
    } catch (error) {
      updateMessage(assistantId, (message) => ({
        ...message,
        content: `${message.content}\n\nError: ${error.message}`,
      }));
    } finally {
      setFeedHint("Responses will appear here once the cluster answers.");
      setBusy(false);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (feedState === "sending") {
      return;
    }
    sendPrompt();
  };

  const handleSample = () => {
    setPrompt(SAMPLE_PROMPT);
  };

  const handleClear = () => {
    setMessages([]);
    conversationRef.current = [];
  };

  const handleRefresh = () => {
    refreshStatus();
    refreshMetrics();
  };

  return (
    <div className="scene">
      <div className="orb orb-a"></div>
      <div className="orb orb-b"></div>
      <div className="orb orb-c"></div>
      <header className="hero">
        <div className="hero-top">
          <span className="eyebrow">ShardCompute Coordinator</span>
          <span className="badge" style={badgeStyle(statusBadge.ok)}>
            {statusBadge.text}
          </span>
        </div>
        <h1>Metropolis Control Deck</h1>
        <p>
          A skyline-inspired console for orchestrating distributed inference. Monitor
          the grid, tune the flow, and launch prompts from a single neon-lit hub.
        </p>
        <div className="hero-actions">
          <button className="ghost" id="refresh-btn" type="button" onClick={handleRefresh}>
            Refresh
          </button>
          <button className="ghost" id="clear-btn" type="button" onClick={handleClear}>
            Clear Feed
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="card status" style={{ "--delay": "0.05s" }}>
          <div className="card-header">
            <h2>Cluster Pulse</h2>
            <span className="chip" style={badgeStyle(healthBadge.ok)}>
              {healthBadge.text}
            </span>
          </div>
          <div className="status-grid">
            <div className="metric">
              <p className="label">Workers</p>
              <div className="worker-dots">
                <span className="dot dot-ok"></span>
                <span className="dot-count">{workersOk}</span>
                <span className="dot dot-miss"></span>
                <span className="dot-count">{workersMiss}</span>
              </div>
            </div>
          </div>
          <div className="chart">
            <div className="chart-header">
              <span className="label">Token Flux</span>
              <div className="chart-meta">
                <span className="chart-value">{throughput}</span>
                <span className="chart-divider">•</span>
                <span className="chart-value">{liveRate}</span>
              </div>
            </div>
            <canvas ref={chartRef} height="120"></canvas>
          </div>
          <div className="hint">{statusHint}</div>
        </section>

        <section className="card control" style={{ "--delay": "0.12s" }}>
          <div className="card-header">
            <h2>Prompt Launch</h2>
            <span className="chip ghost">text mode</span>
          </div>
          <div className="settings">
            <div>
              <label className="label" htmlFor="max-tokens">
                Max tokens
              </label>
              <input
                id="max-tokens"
                type="number"
                min="16"
                max="512"
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </div>
            <div>
              <label className="label" htmlFor="temperature">
                Temperature <span id="temp-value">{temperature.toFixed(2)}</span>
              </label>
              <input
                id="temperature"
                type="range"
                min="0.1"
                max="1.5"
                step="0.05"
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
              />
            </div>
            <div>
              <label className="label" htmlFor="top-p">
                Top P <span id="top-p-value">{topP.toFixed(2)}</span>
              </label>
              <input
                id="top-p"
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={topP}
                onChange={(event) => setTopP(Number(event.target.value))}
              />
            </div>
          </div>
        </section>

        <section className="card stream" style={{ "--delay": "0.2s" }}>
          <div className="card-header">
            <h2>City Feed</h2>
            <div className="stream-actions">
              <span
                className="chip"
                style={{
                  background:
                    feedState === "sending"
                      ? "rgba(0, 194, 255, 0.25)"
                      : "rgba(10, 26, 47, 0.08)",
                }}
              >
                {feedState}
              </span>
              <button
                className="ghost toggle-panels"
                type="button"
                onClick={() => setPanelsCollapsed((prev) => !prev)}
              >
                {panelsCollapsed ? "Show Panels" : "Hide Panels"}
              </button>
            </div>
          </div>
          <div ref={messagesRef} className="messages" aria-live="polite">
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                {message.meta ? <div className="meta">{message.meta}</div> : null}
                <div>{message.content}</div>
              </div>
            ))}
          </div>
          <div className="hint">{feedHint}</div>
          <form className="prompt-dock" onSubmit={handleSubmit}>
            <label className="label" htmlFor="prompt-input">
              Command
            </label>
            <div className="prompt-row">
              <textarea
                id="prompt-input"
                rows="3"
                placeholder="Draft a prompt for the cluster..."
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
              ></textarea>
              <div className="prompt-actions">
                <button
                  type="submit"
                  className="icon-btn"
                  aria-label="Send"
                  disabled={feedState === "sending"}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M4 12l16-7-7 16-2.5-6L4 12z" fill="currentColor" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="icon-btn ghost"
                  aria-label="Load sample"
                  onClick={handleSample}
                >
                  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                    <path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" fill="currentColor" />
                  </svg>
                </button>
              </div>
            </div>
          </form>
        </section>
      </main>

      <footer className="footer">
        <span>ShardCompute UI</span>
        <span className="dot">•</span>
        <span>Metropolis build</span>
      </footer>
    </div>
  );
}

export default App;
