const statusBadge = document.getElementById("status-badge");
const healthChip = document.getElementById("health-chip");
const statusValue = document.getElementById("status-value");
const workersValue = document.getElementById("workers-value");
const expectedValue = document.getElementById("expected-value");
const latencyValue = document.getElementById("latency-value");
const throughputValue = document.getElementById("throughput-value");
const requestsValue = document.getElementById("requests-value");
const statusHint = document.getElementById("status-hint");
const messagesEl = document.getElementById("messages");
const feedChip = document.getElementById("feed-chip");
const feedHint = document.getElementById("feed-hint");
const tokenChart = document.getElementById("token-chart");
const tokensPerSecEl = document.getElementById("tokens-per-sec");

const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const sendBtn = document.getElementById("send-btn");
const sampleBtn = document.getElementById("sample-btn");
const clearBtn = document.getElementById("clear-btn");
const refreshBtn = document.getElementById("refresh-btn");

const maxTokensInput = document.getElementById("max-tokens");
const temperatureInput = document.getElementById("temperature");
const topPInput = document.getElementById("top-p");
const tempValue = document.getElementById("temp-value");
const topPValue = document.getElementById("top-p-value");

const tokenizerNote = document.getElementById("tokenizer-note");

const conversation = [];
let busy = false;
let liveTokenCount = 0;
const tokenSeries = [];
const maxSeriesPoints = 40;
const tokenSamplerMs = 1000;
let tokenChartInstance = null;

const SAMPLE_PROMPT =
  "Draft a 4-line status update in the voice of a city power grid operator.";

const formatNumber = (value, digits = 1) => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "--";
  }
  return Number(value).toFixed(digits);
};

const setBadge = (el, ok, text) => {
  el.textContent = text;
  el.style.background = ok
    ? "rgba(46, 229, 157, 0.25)"
    : "rgba(255, 122, 89, 0.25)";
};

const addMessage = (role, text, meta) => {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    wrapper.appendChild(metaEl);
  }

  const body = document.createElement("div");
  body.textContent = text;
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
};

const appendToMessage = (messageWrapper, text) => {
  const body = messageWrapper.querySelector("div:last-child");
  if (body) {
    body.textContent += text;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }
};

const refreshStatus = async () => {
  try {
    // Use root endpoint with JSON query param (matches COMMUNICATION_OUTLINE.md style)
    const response = await fetch("/?json=1");
    if (!response.ok) {
      throw new Error("Status unavailable");
    }
    const data = await response.json();

    statusValue.textContent = data.status || "unknown";
    workersValue.textContent = data.workers ?? "--";
    expectedValue.textContent = data.expected_workers ?? "--";
    setBadge(statusBadge, data.cluster_ready, data.cluster_ready ? "linked" : "offline");
    setBadge(healthChip, data.cluster_healthy, data.cluster_healthy ? "healthy" : "degraded");
    statusHint.textContent = data.cluster_ready
      ? "Cluster ready for inference."
      : "Waiting for workers to report in.";
  } catch (error) {
    statusValue.textContent = "offline";
    setBadge(statusBadge, false, "offline");
    setBadge(healthChip, false, "unknown");
    statusHint.textContent = "Coordinator not reachable.";
  }
};

const refreshMetrics = async () => {
  try {
    // Use /metrics endpoint (no /api prefix, matches COMMUNICATION_OUTLINE.md style)
    const response = await fetch("/metrics");
    if (!response.ok) {
      throw new Error("Metrics unavailable");
    }
    const data = await response.json();
    const summary = data.summary || {};

    latencyValue.textContent = `${formatNumber(summary.avg_latency_ms)} ms`;
    const avgThroughput = formatNumber(summary.avg_throughput_tokens_per_sec, 1);
    const lastThroughput = formatNumber(summary.last_throughput_tokens_per_sec, 1);
    throughputValue.textContent = `${avgThroughput} tok/s (prev: ${lastThroughput})`;
    requestsValue.textContent = summary.total_requests ?? "--";
  } catch (error) {
    latencyValue.textContent = "--";
    throughputValue.textContent = "--";
    requestsValue.textContent = "--";
  }
};

const updateRanges = () => {
  tempValue.textContent = Number(temperatureInput.value).toFixed(2);
  topPValue.textContent = Number(topPInput.value).toFixed(2);
};

const initTokenChart = () => {
  if (!tokenChart || typeof Chart === "undefined") {
    return;
  }
  const ctx = tokenChart.getContext("2d");
  if (!ctx) {
    return;
  }

  tokenChartInstance = new Chart(ctx, {
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
          pointRadius: (ctx) =>
            ctx.dataIndex === ctx.dataset.data.length - 1 ? 2.5 : 0,
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
        y: {
          display: false,
          grid: { display: false, drawBorder: false },
        },
      },
    },
  });
};

const updateTokenChart = () => {
  if (!tokenChartInstance) {
    return;
  }
  tokenChartInstance.data.labels = tokenSeries.map((_, index) => index + 1);
  tokenChartInstance.data.datasets[0].data = tokenSeries;
  tokenChartInstance.update("none");
};

const setBusy = (state) => {
  busy = state;
  sendBtn.disabled = state;
  feedChip.textContent = state ? "sending" : "idle";
  feedChip.style.background = state
    ? "rgba(0, 194, 255, 0.25)"
    : "rgba(10, 26, 47, 0.08)";
};

const sendPrompt = async (prompt) => {
  setBusy(true);
  feedHint.textContent = "Awaiting response from the grid...";

  const payload = {
    prompt,
    messages: conversation.slice(-12),
    max_new_tokens: Number(maxTokensInput.value || 160),
    temperature: Number(temperatureInput.value || 0.7),
    top_p: Number(topPInput.value || 0.9),
  };

  let messageWrapper = null;
  let fullText = "";

  try {
    // Use /inference/text/stream endpoint (no /api prefix, matches COMMUNICATION_OUTLINE.md style)
    const response = await fetch("/inference/text/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error("Inference failed");
    }

    feedHint.textContent = "Streaming response from the grid...";

    // Create message wrapper for streaming
    messageWrapper = addMessage("assistant", "", "grid");

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
        } else if (line.startsWith("data: ")) {
          const data = line.slice(6);
          try {
            const parsed = JSON.parse(data);
            if (parsed.token) {
              fullText += parsed.token;
              appendToMessage(messageWrapper, parsed.token);
              liveTokenCount += 1;
            } else if (parsed.error) {
              throw new Error(parsed.error);
            }
          } catch (e) {
            if (e.message !== "Unexpected end of JSON input") {
              console.error("Parse error:", e);
            }
          }
        }
      }
    }

    if (fullText) {
      conversation.push({ role: "assistant", content: fullText });
    } else {
      appendToMessage(messageWrapper, "No text output returned.");
    }
  } catch (error) {
    if (messageWrapper) {
      appendToMessage(messageWrapper, `\n\nError: ${error.message}`);
    } else {
      addMessage("assistant", error.message, "error");
    }
    if (error.message.includes("tokenizer")) {
      tokenizerNote.textContent =
        "Tokenizer missing. Set coordinator.tokenizer_path to enable text inference.";
    }
  } finally {
    feedHint.textContent = "Responses will appear here once the cluster answers.";
    setBusy(false);
  }
};

promptForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) {
    return;
  }
  const prompt = promptInput.value.trim();
  if (!prompt) {
    return;
  }
  addMessage("user", prompt, "you");
  conversation.push({ role: "user", content: prompt });
  promptInput.value = "";
  sendPrompt(prompt);
});

sampleBtn.addEventListener("click", () => {
  promptInput.value = SAMPLE_PROMPT;
  promptInput.focus();
});

clearBtn.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  conversation.length = 0;
});

refreshBtn.addEventListener("click", () => {
  refreshStatus();
  refreshMetrics();
});

temperatureInput.addEventListener("input", updateRanges);
topPInput.addEventListener("input", updateRanges);

updateRanges();
initTokenChart();
refreshStatus();
refreshMetrics();
setInterval(refreshStatus, 5000);
setInterval(refreshMetrics, 8000);
setInterval(() => {
  const tps = liveTokenCount / (tokenSamplerMs / 1000);
  tokenSeries.push(tps);
  if (tokenSeries.length > maxSeriesPoints) {
    tokenSeries.shift();
  }
  tokensPerSecEl.textContent = `${formatNumber(tps, 1)} tok/s`;
  updateTokenChart();
  liveTokenCount = 0;
}, tokenSamplerMs);
window.addEventListener("resize", updateTokenChart);
