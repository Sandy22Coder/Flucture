const postureLabel = document.getElementById("postureLabel");
const reviewText = document.getElementById("reviewText");
const metricsGrid = document.getElementById("metricsGrid");
const trackerStatus = document.getElementById("trackerStatus");
const trackerError = document.getElementById("trackerError");
const reportStatus = document.getElementById("reportStatus");
const alertStatus = document.getElementById("alertStatus");
const reportActions = document.getElementById("reportActions");
const downloadPdfLink = document.getElementById("downloadPdfLink");
const notifyBtn = document.getElementById("notifyBtn");
const reportBtn = document.getElementById("reportBtn");
const postureBadge = document.getElementById("postureBadge");

const BASE_TITLE = "Flucture AI";
let lastPostureState = "unknown";
let lastNotificationAt = 0;

function svgFavicon(bg, text) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
      <rect width="64" height="64" rx="14" fill="${bg}"></rect>
      <text x="32" y="42" text-anchor="middle" font-size="34" font-family="Arial" fill="#ffffff">${text}</text>
    </svg>
  `;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function setFavicon(state) {
  let link = document.querySelector("link[rel='icon']");
  if (!link) {
    link = document.createElement("link");
    link.rel = "icon";
    document.head.appendChild(link);
  }

  if (state === "bad") {
    link.href = svgFavicon("#c95f3b", "!");
  } else if (state === "good") {
    link.href = svgFavicon("#2f8f5b", "✓");
  } else {
    link.href = svgFavicon("#1d2a2f", "F");
  }
}

function updateAttentionSignals(posture, review) {
  const lower = (posture || "").toLowerCase();
  const isBad = lower === "poor";
  const isGood = lower === "good";
  const nextState = isBad ? "bad" : isGood ? "good" : "neutral";

  document.title = isBad ? "Bad Posture Detected | Flucture AI" : isGood ? "Good Posture | Flucture AI" : BASE_TITLE;
  setFavicon(nextState);

  const now = Date.now();
  const shouldNotify =
    isBad &&
    document.hidden &&
    Notification.permission === "granted" &&
    lastPostureState !== "bad" &&
    now - lastNotificationAt > 45000;

  if (shouldNotify) {
    new Notification("Flucture AI posture alert", {
      body: review || "Poor posture detected. Sit upright and reset your shoulders.",
      silent: false,
    });
    lastNotificationAt = now;
  }

  lastPostureState = nextState;
}

async function postJson(path) {
  const response = await fetch(path, { method: "POST" });
  return response.json();
}

function updatePostureBadge(posture) {
  const lower = (posture || "").toLowerCase();
  if (lower === "good") {
    postureBadge.textContent = "Aligned";
    postureBadge.className = "posture-badge good";
    return;
  }
  if (lower === "poor") {
    postureBadge.textContent = "Needs Reset";
    postureBadge.className = "posture-badge poor";
    return;
  }
  if (lower === "tracker error") {
    postureBadge.textContent = "Issue";
    postureBadge.className = "posture-badge poor";
    return;
  }
  postureBadge.textContent = "Waiting";
  postureBadge.className = "posture-badge neutral";
}

function renderMetrics(metrics) {
  const keys = Object.keys(metrics || {});
  if (!keys.length) {
    metricsGrid.innerHTML = '<div class="list-block empty">No metrics yet.</div>';
    return;
  }

  metricsGrid.innerHTML = keys.map((key) => {
    const item = metrics[key];
    const delta = Number(item.actual) - Number(item.threshold);
    const state = delta > 0 ? "Above threshold" : "Within threshold";
    return `
      <div class="metric-card">
        <strong>${key}</strong>
        <p>Actual: ${item.actual}</p>
        <p>Threshold: ${item.threshold}</p>
        <p>${state}</p>
      </div>
    `;
  }).join("");
}

function renderCards(containerId, cards) {
  const root = document.getElementById(containerId);
  if (!cards || !cards.length) {
    root.className = "list-block empty";
    root.textContent = "No session evidence available.";
    return;
  }

  root.className = "list-block";
  root.innerHTML = cards.join("");
}

function renderReport(panel) {
  const progress = panel.progress_score || {};
  const summaryBlock = document.getElementById("summaryBlock");
  summaryBlock.className = "list-block";
  summaryBlock.innerHTML = `
    <div class="list-card">
      <strong>Overall Assessment</strong>
      <p>${panel.overall_assessment || "No summary available."}</p>
    </div>
    <div class="list-card">
      <strong>Risk Level</strong>
      <p>${panel.risk_level || "unknown"}</p>
    </div>
    <div class="list-card">
      <strong>Progress Score</strong>
      <p>Current: ${progress.current_score ?? 0} | Previous: ${progress.previous_score ?? 0} | Change: ${progress.change || "same"}</p>
    </div>
  `;

  renderCards("wrongList", panel.what_is_wrong.map((item) => `
    <div class="list-card">
      <strong>${item.issue} (${item.severity})</strong>
      <ul>${item.evidence.map((entry) => `<li>${entry}</li>`).join("")}</ul>
    </div>
  `));

  renderCards("riskList", panel.possible_consequences.map((item) => `
    <div class="list-card">
      <strong>${item.issue}</strong>
      <ul>${item.risks.map((risk) => `<li>${risk}</li>`).join("")}</ul>
    </div>
  `));

  renderCards("improvementList", panel.improvement_plan.map((item) => `
    <div class="list-card">
      <strong>Priority ${item.priority}: ${item.action}</strong>
      <p>${item.reason}</p>
    </div>
  `));

  const remedies = panel.remedies || {};
  renderCards("remedyList", [
    `<div class="list-card"><strong>Stretches</strong><ul>${(remedies.stretches || []).map((item) => `<li>${item}</li>`).join("")}</ul></div>`,
    `<div class="list-card"><strong>Strengthening</strong><ul>${(remedies.strengthening || []).map((item) => `<li>${item}</li>`).join("")}</ul></div>`,
    `<div class="list-card"><strong>Daily Habits</strong><ul>${(remedies.daily_habits || []).map((item) => `<li>${item}</li>`).join("")}</ul></div>`,
    `<div class="list-card"><strong>Ergonomics</strong><ul>${(remedies.ergonomic_corrections || []).map((item) => `<li>${item}</li>`).join("")}</ul></div>`
  ]);
}

async function refreshRealtime() {
  try {
    const [statusResponse, realtimeResponse] = await Promise.all([
      fetch("/status"),
      fetch("/realtime")
    ]);
    const statusPayload = await statusResponse.json();
    const realtimePayload = await realtimeResponse.json();

    trackerStatus.textContent = statusPayload.running ? "Tracking" : "Offline";
    postureLabel.textContent = realtimePayload.posture || "Calibrating";
    reviewText.textContent = realtimePayload.review || "Waiting for tracker...";
    updatePostureBadge(realtimePayload.posture);
    if (statusPayload.last_error) {
      trackerError.textContent = statusPayload.last_error;
      trackerError.classList.remove("hidden");
    } else {
      trackerError.textContent = "";
      trackerError.classList.add("hidden");
    }
    updateAttentionSignals(realtimePayload.posture, realtimePayload.review);
    renderMetrics(realtimePayload.metrics || {});
  } catch (error) {
    trackerStatus.textContent = "Unavailable";
    trackerError.textContent = "Could not reach the Flask app.";
    trackerError.classList.remove("hidden");
    updatePostureBadge("tracker error");
  }
}

document.getElementById("startBtn").addEventListener("click", async () => {
  await postJson("/start");
  refreshRealtime();
});

document.getElementById("stopBtn").addEventListener("click", async () => {
  await postJson("/stop");
  refreshRealtime();
});

reportBtn.addEventListener("click", async () => {
  reportStatus.textContent = "Generating report...";
  reportStatus.classList.remove("hidden");
  reportBtn.classList.add("loading");
  reportActions.classList.add("hidden");

  try {
    const response = await postJson("/generate_report");
    if (response.status === "ok") {
      renderReport(response.panel);
      const sourceLabel = response.generator === "openai" ? "OpenAI" : "fallback";
      reportStatus.textContent = `Report ready using ${sourceLabel}. Your session summary is now available and saved as ${response.report_file}.`;
      if (response.pdf_url) {
        downloadPdfLink.href = response.pdf_url;
        reportActions.classList.remove("hidden");
      }
      reportBtn.classList.remove("loading");
      return;
    }
    reportStatus.textContent = response.message || response.error || "Report generation failed.";
  } catch (error) {
    reportStatus.textContent = "Report generation failed. Check the Flask terminal for details.";
  }
  reportBtn.classList.remove("loading");
});

notifyBtn.addEventListener("click", async () => {
  if (!("Notification" in window)) {
    alertStatus.textContent = "Browser notifications are not supported here.";
    alertStatus.classList.remove("hidden");
    return;
  }

  if (Notification.permission === "granted") {
    alertStatus.textContent = "Background alerts are already enabled.";
    alertStatus.classList.remove("hidden");
    return;
  }

  const permission = await Notification.requestPermission();
  if (permission === "granted") {
    alertStatus.textContent = "Background alerts enabled. You will be notified when posture becomes poor in another tab.";
  } else {
    alertStatus.textContent = "Notification permission was not granted. Title and favicon alerts will still work.";
  }
  alertStatus.classList.remove("hidden");
});

setFavicon("neutral");
updatePostureBadge("neutral");
refreshRealtime();
setInterval(refreshRealtime, 2500);
