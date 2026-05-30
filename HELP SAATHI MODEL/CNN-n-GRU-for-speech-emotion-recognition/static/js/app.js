const chunkSeconds = Number(document.body.dataset.chunkSeconds || 4);
const visualizer = document.getElementById("visualizer");
const toggleBtn = document.getElementById("toggle-btn");
const monitorState = document.getElementById("monitor-state");
const transcriptLog = document.getElementById("transcript-log");
const errorBox = document.getElementById("error-box");
const decisionCard = document.getElementById("decision-card");
const decisionTitle = document.getElementById("decision-title");
const decisionBadge = document.getElementById("decision-badge");
const decisionCopy = document.getElementById("decision-copy");
const latestTranscript = document.getElementById("latest-transcript");
const keywordList = document.getElementById("keyword-list");
const emotionName = document.getElementById("emotion-name");
const emotionConfidence = document.getElementById("emotion-confidence");
const emotionBars = document.getElementById("emotion-bars");
const originalAudio = document.getElementById("original-audio");
const filteredAudio = document.getElementById("filtered-audio");
const filterChip = document.getElementById("filter-chip");
const fileInput = document.getElementById("file-input");
const modelStatus = document.getElementById("model-status");
const sttStatus = document.getElementById("stt-status");
const sttDot = document.getElementById("stt-dot");
const sosBanner = document.getElementById("sos-banner");

let isMonitoring = false;
let mediaRecorder = null;
let mediaStream = null;
let audioContext = null;
let analyser = null;
let animationId = null;
let recordTimeoutId = null;
let currentChunkParts = [];

for (let i = 0; i < 24; i += 1) {
  const bar = document.createElement("div");
  bar.className = "bar";
  visualizer.appendChild(bar);
}

const bars = [...document.querySelectorAll(".bar")];

function setError(message = "") {
  if (!message) {
    errorBox.className = "error-box";
    errorBox.textContent = "";
    return;
  }

  errorBox.className = "error-box visible";
  errorBox.textContent = message;
}

function updateDecisionIdle() {
  decisionCard.className = "decision-card neutral";
  decisionTitle.textContent = "Ready To Scan";
  decisionBadge.textContent = "Standby";
  decisionCopy.textContent = "Press the scan button to record one short clip and screen it for danger phrases.";
  latestTranscript.textContent = "No transcript yet.";
  keywordList.textContent = "None";
  emotionName.textContent = "Not evaluated";
  emotionConfidence.textContent = "0%";
  emotionBars.innerHTML = "";
  clearAudioPreviews();
  sosBanner.classList.remove("visible");
}

function clearAudioPreviews() {
  originalAudio.removeAttribute("src");
  filteredAudio.removeAttribute("src");
  originalAudio.load();
  filteredAudio.load();
  filterChip.textContent = "Filter inactive";
}

function updateAudioPreviews(result) {
  if (result.original_audio_preview) {
    originalAudio.src = result.original_audio_preview;
  } else {
    originalAudio.removeAttribute("src");
  }

  if (result.filtered_audio_preview) {
    filteredAudio.src = result.filtered_audio_preview;
  } else {
    filteredAudio.removeAttribute("src");
  }

  originalAudio.load();
  filteredAudio.load();
  filterChip.textContent = result.noise_filter_chain || "Filter active";
}

function updateDecisionFromResult(result) {
  const distressDetected = Boolean(result.danger_emotion_detected || result.emotion_result?.is_distress);
  latestTranscript.textContent = result.transcript || "No speech detected.";
  updateAudioPreviews(result);

  if (!result.keyword_hit) {
    decisionCard.className = "decision-card safe";
    decisionTitle.textContent = "No Danger Phrase Detected";
    decisionBadge.textContent = "Transcript only";
    decisionCopy.textContent = "This chunk was transcribed, but none of the tracked danger keywords were found, so emotion analysis was skipped.";
    keywordList.textContent = "None";
    emotionName.textContent = "Skipped";
    emotionConfidence.textContent = "0%";
    emotionBars.innerHTML = "";
    sosBanner.classList.remove("visible");
    return;
  }

  const emotion = result.emotion_result || {};
  const alertState = distressDetected ? "alert" : "waiting";
  decisionCard.className = `decision-card ${alertState}`;
  decisionTitle.textContent = distressDetected ? "SOS Triggered" : "Keyword Hit, Review Emotion";
  decisionBadge.textContent = distressDetected ? "Emergency" : "Needs review";
  decisionCopy.textContent = distressDetected
    ? "Danger keywords matched and the emotion model detected a distress emotion. SOS should be shown immediately."
    : "Danger keywords matched, but emotion confidence is not high enough for SOS yet.";
  keywordList.textContent = result.matched_keywords.join(", ");
  sosBanner.classList.toggle("visible", distressDetected);

  if (emotion.error) {
    emotionName.textContent = "Emotion error";
    emotionConfidence.textContent = "--";
    emotionBars.innerHTML = "";
    return;
  }

  emotionName.textContent = (emotion.emotion || "Unknown").toUpperCase();
  emotionConfidence.textContent = `${emotion.confidence || 0}%`;
  renderEmotionBars(emotion.all_probs || {}, emotion.emotion);
}

function renderEmotionBars(allProbs, topEmotion) {
  const sorted = Object.entries(allProbs).sort((a, b) => b[1] - a[1]);
  emotionBars.innerHTML = sorted.map(([label, value]) => `
    <div class="emotion-row">
      <div class="name">${label}</div>
      <div class="bar-track">
        <div class="bar-fill ${label === topEmotion ? "top" : ""}" style="width:${value}%"></div>
      </div>
      <div class="pct">${value}%</div>
    </div>
  `).join("");
}

function prependTranscriptCard(result) {
  const empty = transcriptLog.querySelector(".empty-state");
  if (empty) empty.remove();

  const card = document.createElement("article");
  card.className = "chunk-card";

  const flagClass = result.keyword_hit ? "alert" : "idle";
  const flagText = result.keyword_hit ? "Keyword hit" : "Clear";
  const time = new Date().toLocaleTimeString();

  card.innerHTML = `
    <div class="chunk-top">
      <div class="chunk-time">${time}</div>
      <div class="chunk-flag ${flagClass}">${flagText}</div>
    </div>
    <div class="chunk-text">${result.transcript || "No transcript returned."}</div>
    ${result.keyword_hit ? `<div class="chunk-keywords">Matched: ${result.matched_keywords.join(", ")}</div>` : ""}
  `;

  transcriptLog.prepend(card);
}

async function loadStatus() {
  try {
    const response = await fetch("/status");
    const data = await response.json();
    modelStatus.textContent = data.loaded ? `Model ready · ${data.device}` : "Emotion model not loaded";
    sttStatus.textContent = data.stt_reachable ? "STT server reachable" : `STT offline${data.stt_error ? " · " + data.stt_error : ""}`;
    sttDot.classList.toggle("down", !data.stt_reachable);
    filterChip.textContent = data.rnnoise_enabled ? "RNNoise active" : (data.noise_filter_chain || "Filter active");
  } catch (error) {
    modelStatus.textContent = "Status unavailable";
    sttStatus.textContent = "STT status unavailable";
    sttDot.classList.add("down");
    filterChip.textContent = "Filter status unavailable";
  }
}

function startVisualizer(stream) {
  audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 64;
  source.connect(analyser);
  const buffer = new Uint8Array(analyser.frequencyBinCount);

  const tick = () => {
    animationId = requestAnimationFrame(tick);
    analyser.getByteFrequencyData(buffer);
    bars.forEach((bar, index) => {
      const value = buffer[index % buffer.length] || 0;
      bar.style.height = `${Math.max(8, (value / 255) * 70)}px`;
      bar.classList.add("active");
    });
  };

  tick();
}

async function stopVisualizer() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  if (audioContext) {
    await audioContext.close();
  }

  bars.forEach((bar) => {
    bar.classList.remove("active");
    bar.style.height = "8px";
  });
}

async function sendChunk(blob, filename) {
  const formData = new FormData();
  formData.append("audio", blob, filename);

  const response = await fetch("/screen_audio", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "Failed to process audio chunk.");
  }

  prependTranscriptCard(result);
  updateDecisionFromResult(result);
  return result;
}

async function startMonitoring() {
  setError("");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    startVisualizer(mediaStream);

    isMonitoring = true;
    toggleBtn.textContent = "Recording...";
    toggleBtn.classList.add("stop");
    toggleBtn.disabled = true;
    monitorState.textContent = `Recording ${chunkSeconds}s clip...`;
    startSingleScanRecorder();
  } catch (error) {
    setError("Could not start microphone capture. Please allow mic access.");
  }
}

function startSingleScanRecorder() {
  if (!isMonitoring || !mediaStream) return;

  const options = MediaRecorder.isTypeSupported("audio/webm")
    ? { mimeType: "audio/webm" }
    : undefined;

  currentChunkParts = [];
  mediaRecorder = new MediaRecorder(mediaStream, options);

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      currentChunkParts.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    const blobType = currentChunkParts[0]?.type || mediaRecorder?.mimeType || "audio/webm";
    const blob = new Blob(currentChunkParts, { type: blobType });
    currentChunkParts = [];

    if (blob.size > 0) {
      try {
        monitorState.textContent = "Sending clip for transcription...";
        const extension = blobType.includes("mp4")
          ? "mp4"
          : blobType.includes("ogg")
            ? "ogg"
            : "webm";
        const result = await sendChunk(blob, `chunk-${Date.now()}.${extension}`);
        if (result.danger_emotion_detected) {
          await triggerSosStop();
        } else {
          await stopMonitoring();
          monitorState.textContent = "Scan complete. Press again to record another clip.";
        }
      } catch (error) {
        setError(error.message);
        await stopMonitoring();
        monitorState.textContent = "Scan failed. Press again to retry.";
      }
    }
  };

  mediaRecorder.start();
  recordTimeoutId = setTimeout(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, chunkSeconds * 1000);
}

async function stopMonitoring() {
  isMonitoring = false;

  if (recordTimeoutId) {
    clearTimeout(recordTimeoutId);
    recordTimeoutId = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  await stopVisualizer();

  mediaRecorder = null;
  mediaStream = null;
  toggleBtn.textContent = "Record 5s Scan";
  toggleBtn.classList.remove("stop");
  toggleBtn.disabled = false;
  monitorState.textContent = "Mic is idle. Press to record one clip.";
}

async function triggerSosStop() {
  if (isMonitoring) {
    await stopMonitoring();
  }
  monitorState.textContent = "SOS triggered. Press again for another scan if needed.";
  toggleBtn.textContent = "Record Again";
}

toggleBtn.addEventListener("click", async () => {
  if (!isMonitoring) {
    await startMonitoring();
  }
});

fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;

  setError("");
  try {
    monitorState.textContent = "Uploading file for screening...";
    const result = await sendChunk(file, file.name);
    monitorState.textContent = "File processed.";
    return result;
  } catch (error) {
    setError(error.message);
    monitorState.textContent = "Mic is idle.";
  } finally {
    event.target.value = "";
  }
});

loadStatus();
updateDecisionIdle();
