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
};

const refreshStatus = async () => {
  try {
    const response = await fetch("/api/status");
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
    const response = await fetch("/api/metrics");
    if (!response.ok) {
      throw new Error("Metrics unavailable");
    }
    const data = await response.json();
    const summary = data.summary || {};

    latencyValue.textContent = `${formatNumber(summary.avg_latency_ms)} ms`;
    throughputValue.textContent = `${formatNumber(
      summary.avg_throughput_tokens_per_sec,
      1
    )} tok/s`;
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

  try {
    const response = await fetch("/api/inference/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Inference failed");
    }

    if (data.output_text) {
      addMessage("assistant", data.output_text, "grid");
      conversation.push({ role: "assistant", content: data.output_text });
    } else {
      addMessage("assistant", "No text output returned.", "grid");
    }
  } catch (error) {
    addMessage("assistant", error.message, "error");
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
refreshStatus();
refreshMetrics();
setInterval(refreshStatus, 5000);
setInterval(refreshMetrics, 8000);
