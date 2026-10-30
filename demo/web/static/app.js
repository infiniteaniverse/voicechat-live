// Browser mic capture → WebSocket → audio playback, with a real audio-reactive
// visualizer (Web Audio AnalyserNode — not a decorative fake animation).
// Implementation: Part 8 (stretch goal), redesigned in Part 9

const STATUS_TEXT = {
  idle: "Click to start",
  listening: "Listening...",
  processing: "Thinking...",
  speaking: "Speaking...",
  error: "Something went wrong",
};

// Red = your turn (listening), blue = the assistant's turn (speaking) — the
// visualizer and glow both key off this per state.
const STATE_COLOR = {
  idle: "148, 163, 184",
  listening: "224, 38, 63",
  processing: "139, 92, 246",
  speaking: "59, 130, 246",
  error: "255, 154, 61",
};

// If no new status arrives this long after "listening", assume the pipeline
// is now processing (STT/guardrails/LLM) — the server has no clean single
// point to announce this without extra hooks into tested Part 7 code, so the
// client infers it instead. See docs/tradeoffs.md.
const PROCESSING_INFER_DELAY_MS = 500;

const button = document.getElementById("session-btn");
const statusEl = document.getElementById("status");
const latencyEl = document.getElementById("latency");
const canvas = document.getElementById("visualizer");
const canvasCtx = canvas.getContext("2d");

let audioContext = null;
let micStream = null;
let workletNode = null;
let socket = null;
let nextPlaybackTime = 0;
let processingTimer = null;
let currentState = "idle";

let micAnalyser = null;
let playbackAnalyser = null;
let visualizerFrame = null;

function setState(state) {
  currentState = state;
  button.dataset.state = state;
  statusEl.textContent = STATUS_TEXT[state] ?? state;
  button.setAttribute(
    "aria-label",
    state === "idle" ? "Start conversation" : "End conversation"
  );
}

function armProcessingInference() {
  clearTimeout(processingTimer);
  processingTimer = setTimeout(() => setState("processing"), PROCESSING_INFER_DELAY_MS);
}

// ── Visualizer ────────────────────────────────────────────────────────────────
// Draws real frequency-bar data from whichever AnalyserNode is relevant to the
// current state: the mic input while listening, the TTS playback while
// speaking. Idle/processing get a slow ambient breathing ring instead, since
// there's no live audio signal to show.

function drawBars(analyser, rgb) {
  const bufferLength = analyser.frequencyBinCount;
  const data = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(data);

  const { width, height } = canvas;
  const cx = width / 2;
  const cy = height / 2;
  const baseRadius = 92;
  const barCount = 64;
  const step = Math.floor(bufferLength / barCount);

  canvasCtx.clearRect(0, 0, width, height);
  for (let i = 0; i < barCount; i++) {
    const value = data[i * step] / 255;
    const barLen = 6 + value * 70;
    const angle = (i / barCount) * Math.PI * 2 - Math.PI / 2;
    const x1 = cx + Math.cos(angle) * baseRadius;
    const y1 = cy + Math.sin(angle) * baseRadius;
    const x2 = cx + Math.cos(angle) * (baseRadius + barLen);
    const y2 = cy + Math.sin(angle) * (baseRadius + barLen);

    canvasCtx.strokeStyle = `rgba(${rgb}, ${0.35 + value * 0.65})`;
    canvasCtx.lineWidth = 3;
    canvasCtx.lineCap = "round";
    canvasCtx.beginPath();
    canvasCtx.moveTo(x1, y1);
    canvasCtx.lineTo(x2, y2);
    canvasCtx.stroke();
  }
}

function drawIdleRing(rgb, t) {
  const { width, height } = canvas;
  const cx = width / 2;
  const cy = height / 2;
  const pulse = 92 + Math.sin(t / 700) * 4;

  canvasCtx.clearRect(0, 0, width, height);
  canvasCtx.strokeStyle = `rgba(${rgb}, 0.25)`;
  canvasCtx.lineWidth = 2;
  canvasCtx.beginPath();
  canvasCtx.arc(cx, cy, pulse, 0, Math.PI * 2);
  canvasCtx.stroke();
}

function visualizerLoop(timestamp) {
  const rgb = STATE_COLOR[currentState] ?? STATE_COLOR.idle;

  if (currentState === "listening" && micAnalyser) {
    drawBars(micAnalyser, rgb);
  } else if (currentState === "speaking" && playbackAnalyser) {
    drawBars(playbackAnalyser, rgb);
  } else {
    drawIdleRing(rgb, timestamp);
  }

  visualizerFrame = requestAnimationFrame(visualizerLoop);
}

function startVisualizer() {
  if (visualizerFrame === null) {
    visualizerFrame = requestAnimationFrame(visualizerLoop);
  }
}

function stopVisualizer() {
  if (visualizerFrame !== null) {
    cancelAnimationFrame(visualizerFrame);
    visualizerFrame = null;
  }
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}

// ── Playback ─────────────────────────────────────────────────────────────────

function playChunk(float32Audio, sampleRate = 24000) {
  const buffer = audioContext.createBuffer(1, float32Audio.length, sampleRate);
  buffer.copyToChannel(float32Audio, 0);
  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackAnalyser);
  const startAt = Math.max(audioContext.currentTime, nextPlaybackTime);
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;
}

function handleServerMessage(event) {
  if (typeof event.data === "string") {
    const msg = JSON.parse(event.data);
    if (msg.type === "status") {
      clearTimeout(processingTimer);
      setState(msg.state);
      if (msg.state === "listening") {
        armProcessingInference();
      }
    } else if (msg.type === "latency") {
      const stageText = Object.entries(msg.stages)
        .map(([stage, ms]) => `${stage}: ${ms.toFixed(0)}ms`)
        .join(" | ");
      latencyEl.textContent = `${stageText} | total: ${msg.total_ms.toFixed(0)}ms`;
      latencyEl.hidden = false;
    } else if (msg.type === "error") {
      console.error("Server error:", msg.message);
      setState("error");
      setTimeout(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          setState("listening");
          armProcessingInference();
        }
      }, 1500);
    }
    return;
  }

  clearTimeout(processingTimer);
  playChunk(new Float32Array(event.data));
}

// ── Session lifecycle ───────────────────────────────────────────────────────

async function startSession() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    console.error("Microphone permission denied:", err);
    setState("error");
    return;
  }

  audioContext = new AudioContext();
  nextPlaybackTime = 0;
  await audioContext.audioWorklet.addModule("/static/mic-worklet.js");

  const micSource = audioContext.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(audioContext, "mic-capture-processor", {
    processorOptions: { frameMs: 32 },
  });
  micSource.connect(workletNode);

  // Analysers are visualization taps only — connecting an AnalyserNode
  // doesn't route audio anywhere by itself (it has to be .connect()ed
  // onward), so neither of these causes you to hear your own mic or
  // double-plays the response; they just observe the signal.
  micAnalyser = audioContext.createAnalyser();
  micAnalyser.fftSize = 256;
  micSource.connect(micAnalyser);

  playbackAnalyser = audioContext.createAnalyser();
  playbackAnalyser.fftSize = 256;
  playbackAnalyser.connect(audioContext.destination);

  startVisualizer();

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.binaryType = "arraybuffer";

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "session_start", input_sample_rate: audioContext.sampleRate }));
    workletNode.port.onmessage = (event) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(event.data.buffer);
      }
    };
    setState("listening");
    armProcessingInference();
  });

  socket.addEventListener("message", handleServerMessage);

  socket.addEventListener("close", () => {
    if (button.dataset.state !== "idle") {
      endSession();
    }
  });

  socket.addEventListener("error", (err) => {
    console.error("WebSocket error:", err);
  });
}

function endSession() {
  clearTimeout(processingTimer);
  stopVisualizer();
  if (socket) {
    socket.close();
    socket = null;
  }
  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  workletNode = null;
  micAnalyser = null;
  playbackAnalyser = null;
  latencyEl.hidden = true;
  setState("idle");
}

button.addEventListener("click", () => {
  if (button.dataset.state === "idle") {
    setState("listening"); // immediate feedback while mic permission/connection are pending
    startSession();
  } else {
    endSession();
  }
});

// Idle ambient ring animates even before a session starts, so the page
// doesn't look static/dead on first load.
startVisualizer();
