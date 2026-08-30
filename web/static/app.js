(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    source: "record",
    recording: false,
    recordingStartedAt: 0,
    notes: [],
    activeNotes: new Map(),
    voices: new Map(),
    duration: 10,
    temperature: 1,
    category: "auto",
    modelVersion: null,
    modelReady: null,
    generating: false,
    generationTimer: null,
    audioContext: null,
    timer: null,
    resultNotes: [],
    resultMotifEnd: 0,
    resultMidiUrl: null,
    playbackTimers: [],
  };

  const pitchNames = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  const keyboardMap = { a: 60, w: 61, s: 62, e: 63, d: 64, f: 65, t: 66, g: 67, y: 68, h: 69, u: 70, j: 71, k: 72 };
  const blackPitchClasses = new Set([1, 3, 6, 8, 10]);
  const heldKeys = new Set();

  function noteName(pitch) {
    return `${pitchNames[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
  }

  function ensureAudio() {
    if (!state.audioContext) state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (state.audioContext.state === "suspended") state.audioContext.resume();
    return state.audioContext;
  }

  function startVoice(pitch, velocity = 100) {
    if (state.voices.has(pitch)) return;
    const context = ensureAudio();
    const oscillator = context.createOscillator();
    const overtone = context.createOscillator();
    const gain = context.createGain();
    const tone = context.createBiquadFilter();
    const frequency = 440 * Math.pow(2, (pitch - 69) / 12);
    oscillator.type = "triangle";
    overtone.type = "sine";
    oscillator.frequency.value = frequency;
    overtone.frequency.value = frequency * 2;
    tone.type = "lowpass";
    tone.frequency.value = 2200;
    const volume = Math.min(0.16, (velocity / 127) * 0.14);
    gain.gain.setValueAtTime(0.001, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(volume, context.currentTime + 0.018);
    oscillator.connect(tone);
    overtone.connect(tone);
    tone.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    overtone.start();
    state.voices.set(pitch, { oscillator, overtone, gain });
  }

  function stopVoice(pitch) {
    const voice = state.voices.get(pitch);
    if (!voice) return;
    const context = ensureAudio();
    voice.gain.gain.cancelScheduledValues(context.currentTime);
    voice.gain.gain.setValueAtTime(Math.max(voice.gain.gain.value, 0.001), context.currentTime);
    voice.gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + 0.18);
    voice.oscillator.stop(context.currentTime + 0.2);
    voice.overtone.stop(context.currentTime + 0.2);
    state.voices.delete(pitch);
  }

  function setKeyActive(pitch, active) {
    document.querySelectorAll(`[data-pitch="${pitch}"]`).forEach((key) => key.classList.toggle("active", active));
  }

  function noteOn(pitch, velocity = 100) {
    startVoice(pitch, velocity);
    setKeyActive(pitch, true);
    if (state.recording && !state.activeNotes.has(pitch)) {
      state.activeNotes.set(pitch, { pitch, velocity, start: (performance.now() - state.recordingStartedAt) / 1000 });
    }
  }

  function noteOff(pitch) {
    stopVoice(pitch);
    setKeyActive(pitch, false);
    const active = state.activeNotes.get(pitch);
    if (active) {
      active.end = Math.max(active.start + 0.03, (performance.now() - state.recordingStartedAt) / 1000);
      state.notes.push(active);
      state.notes.sort((a, b) => a.start - b.start);
      state.activeNotes.delete(pitch);
      updateRecorder();
    }
  }

  function buildPiano() {
    const piano = $("#piano");
    const minPitch = 48;
    const maxPitch = 72;
    const whitePitches = [];
    for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {
      if (!blackPitchClasses.has(pitch % 12)) whitePitches.push(pitch);
    }
    whitePitches.forEach((pitch, whiteIndex) => {
      const key = document.createElement("button");
      key.type = "button";
      key.className = "piano-key white";
      key.dataset.pitch = pitch;
      key.setAttribute("aria-label", noteName(pitch));
      const mapped = Object.entries(keyboardMap).find(([, value]) => value === pitch);
      if (mapped) key.innerHTML = `<span class="key-label">${mapped[0].toUpperCase()}</span>`;
      wirePointerKey(key, pitch);
      piano.appendChild(key);

      const nextPitch = pitch + 1;
      if (nextPitch <= maxPitch && blackPitchClasses.has(nextPitch % 12)) {
        const black = document.createElement("button");
        black.type = "button";
        black.className = "piano-key black";
        black.style.left = `calc(12px + ${(whiteIndex + 1) * (100 / whitePitches.length)}% - ${(whiteIndex + 1) * (24 / whitePitches.length)}px)`;
        black.dataset.pitch = nextPitch;
        black.setAttribute("aria-label", noteName(nextPitch));
        wirePointerKey(black, nextPitch);
        piano.appendChild(black);
      }
    });
  }

  function wirePointerKey(key, pitch) {
    key.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      key.setPointerCapture(event.pointerId);
      noteOn(pitch);
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => key.addEventListener(name, () => noteOff(pitch)));
  }

  function closeActiveRecordedNotes() {
    [...state.activeNotes.keys()].forEach(noteOff);
  }

  function startRecording() {
    stopPlayback();
    state.notes = [];
    state.activeNotes.clear();
    state.recording = true;
    state.recordingStartedAt = performance.now();
    $("#record-button").classList.add("recording");
    $("#record-label").textContent = "Stop recording";
    state.timer = window.setInterval(updateRecorder, 50);
    updateRecorder();
  }

  function stopRecording() {
    closeActiveRecordedNotes();
    state.recording = false;
    window.clearInterval(state.timer);
    $("#record-button").classList.remove("recording");
    $("#record-label").textContent = "Record again";
    updateRecorder();
  }

  function updateRecorder() {
    const elapsed = state.recording ? (performance.now() - state.recordingStartedAt) / 1000 : Math.max(0, ...state.notes.map((n) => n.end));
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const seconds = (elapsed % 60).toFixed(1).padStart(4, "0");
    $("#record-time").textContent = `${minutes}:${seconds}`;
    $("#note-count").textContent = `${state.notes.length} note${state.notes.length === 1 ? "" : "s"}`;
    $("#preview-motif-button").disabled = state.notes.length === 0;
    if (elapsed >= 30 && state.recording) stopRecording();
  }

  function clearRecording() {
    if (state.recording) stopRecording();
    stopPlayback();
    state.notes = [];
    state.activeNotes.clear();
    $("#record-label").textContent = "Start recording";
    updateRecorder();
  }

  function stopPlayback() {
    state.playbackTimers.forEach(window.clearTimeout);
    state.playbackTimers = [];
    [...state.voices.keys()].forEach(stopVoice);
    $("#play-progress").style.transition = "none";
    $("#play-progress").style.width = "0";
  }

  function playNotes(notes) {
    stopPlayback();
    if (!notes.length) return;
    ensureAudio();
    const offset = Math.min(...notes.map((note) => note.start));
    const total = Math.max(...notes.map((note) => note.end)) - offset;
    notes.forEach((note) => {
      state.playbackTimers.push(window.setTimeout(() => noteOn(note.pitch, note.velocity), Math.max(0, (note.start - offset) * 1000)));
      state.playbackTimers.push(window.setTimeout(() => noteOff(note.pitch), Math.max(30, (note.end - offset) * 1000)));
    });
    requestAnimationFrame(() => {
      $("#play-progress").style.transition = `width ${total}s linear`;
      $("#play-progress").style.width = "100%";
    });
    state.playbackTimers.push(window.setTimeout(stopPlayback, (total + 0.25) * 1000));
  }

  function switchSource(source) {
    state.source = source;
    const record = source === "record";
    $("#record-tab").classList.toggle("active", record);
    $("#upload-tab").classList.toggle("active", !record);
    $("#record-tab").setAttribute("aria-selected", String(record));
    $("#upload-tab").setAttribute("aria-selected", String(!record));
    $("#record-panel").classList.toggle("hidden", !record);
    $("#upload-panel").classList.toggle("hidden", record);
  }

  function temperatureLabel(value) {
    if (value <= 0.8) return "Focused";
    if (value >= 1.2) return "Exploratory";
    return "Balanced";
  }

  function showNotice(message) {
    const notice = $("#notice");
    notice.textContent = message;
    notice.classList.remove("hidden");
    notice.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideNotice() { $("#notice").classList.add("hidden"); }

  const categoryLabels = {
    auto: "Motif-led · Auto",
    baroque_classical: "Baroque / Classical",
    romantic: "Romantic",
    impressionist_modern: "Impressionist / Modern",
  };

  const textureLabels = {
    monophonic: "Single-note motif",
    light_polyphonic: "Lightly polyphonic motif",
    full_polyphonic: "Full two-hand texture",
  };

  function syncGenerateAvailability() {
    const button = $("#generate-button");
    button.disabled = state.generating || state.modelReady === false;
    if (!state.generating) {
      button.querySelector("span:first-child").textContent = state.modelReady === false ? "Model unavailable" : "Generate continuation";
    }
  }

  function configureCategoryControl() {
    const select = $("#category");
    const help = $("#category-help");
    const supportsCategories = state.modelReady && state.modelVersion === "v2";
    select.disabled = !supportsCategories;
    if (supportsCategories) {
      help.textContent = "Auto follows the motif; a period choice gently guides the continuation.";
      return;
    }
    select.value = "auto";
    state.category = "auto";
    help.textContent = state.modelVersion === "v1"
      ? "The deployed v1 checkpoint follows the motif but does not support style categories yet."
      : "Style choices become available when the category-aware v2 checkpoint is loaded.";
  }

  function generationPhase(elapsed) {
    if (elapsed < 2) return "Reading your motif…";
    if (elapsed < 6) return "Listening for rhythm, timing and texture…";
    if (elapsed < 18) return "Composing the continuation note by note…";
    if (elapsed < 32) return "Shaping dynamics and closing active notes…";
    return "Finishing the best continuation within the CPU limit…";
  }

  function startGenerationProgress() {
    const startedAt = performance.now();
    const progress = $("#generation-progress");
    progress.classList.remove("hidden");
    const update = () => {
      const elapsed = Math.floor((performance.now() - startedAt) / 1000);
      $("#generation-phase").textContent = generationPhase(elapsed);
      $("#generation-detail").textContent = `${elapsed} second${elapsed === 1 ? "" : "s"} elapsed · CPU generation can take a little while`;
      $("#generate-button span:first-child").textContent = `Composing · ${elapsed}s`;
    };
    update();
    state.generationTimer = window.setInterval(update, 250);
  }

  function stopGenerationProgress() {
    window.clearInterval(state.generationTimer);
    state.generationTimer = null;
    $("#generation-progress").classList.add("hidden");
  }

  function updateResultContext(payload) {
    $("#result-model").textContent = `Model ${payload.model_version || "unknown"}`;
    $("#result-category").textContent = payload.category_applied
      ? (categoryLabels[payload.category] || payload.category)
      : "Motif-led · v1";
    const texture = $("#result-texture");
    if (payload.inferred_texture) {
      texture.textContent = textureLabels[payload.inferred_texture] || payload.inferred_texture.replaceAll("_", " ");
      texture.classList.remove("hidden");
    } else {
      texture.classList.add("hidden");
    }
    const resultMessage = $("#result-message");
    if (payload.timed_out || payload.reached_target_duration === false) {
      resultMessage.textContent = "The CPU time limit was reached, so this is a playable partial continuation. Try 5 seconds for a more reliable complete result on the free service.";
      resultMessage.classList.remove("hidden");
    } else {
      resultMessage.classList.add("hidden");
    }
  }

  async function generate() {
    hideNotice();
    const file = $("#midi-file").files[0];
    if (state.source === "record" && state.recording) stopRecording();
    if (state.source === "record" && state.notes.length < 2) return showNotice("Record at least two notes before generating.");
    if (state.source === "upload" && !file) return showNotice("Choose a MIDI file before generating.");

    const button = $("#generate-button");
    state.generating = true;
    syncGenerateAvailability();
    startGenerationProgress();
    const form = new FormData();
    form.append("duration_seconds", String(state.duration));
    form.append("temperature", String(state.temperature));
    form.append("category", state.category);
    if (state.source === "record") form.append("motif_json", JSON.stringify(state.notes));
    else form.append("midi_file", file);

    const controller = new AbortController();
    const clientTimeout = window.setTimeout(() => controller.abort(), 90000);
    try {
      const response = await fetch("/api/generate", { method: "POST", body: form, signal: controller.signal });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "The model could not generate this time.");
      state.resultNotes = payload.notes;
      if (state.resultMidiUrl) URL.revokeObjectURL(state.resultMidiUrl);
      const binary = atob(payload.midi_base64);
      const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0));
      state.resultMidiUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
      $("#download-button").href = state.resultMidiUrl;
      $("#result-duration").textContent = `${Number(payload.duration_seconds).toFixed(1)} seconds`;
      $("#result-card").classList.remove("hidden");
      state.resultMotifEnd = Number(payload.motif_end_seconds) || 0;
      updateResultContext(payload);
      drawPianoRoll(payload.notes, payload.motif_end_seconds);
      $("#result-card").scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (error) {
      if (error.name === "AbortError") {
        showNotice("Generation took longer than 90 seconds and the browser stopped waiting. The free service may be waking up; try a 5-second continuation once it is ready.");
      } else {
        showNotice(error.message.includes("fetch") ? "The free service may still be waking up. Wait a moment and try again." : error.message);
      }
    } finally {
      window.clearTimeout(clientTimeout);
      stopGenerationProgress();
      state.generating = false;
      syncGenerateAvailability();
    }
  }

  function drawPianoRoll(notes, motifEnd = 0) {
    const canvas = $("#piano-roll");
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * ratio;
    canvas.height = rect.height * ratio;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    const width = rect.width;
    const height = rect.height;
    context.fillStyle = "#11130f";
    context.fillRect(0, 0, width, height);
    if (!notes.length) return;
    const minPitch = Math.min(...notes.map((note) => note.pitch)) - 2;
    const maxPitch = Math.max(...notes.map((note) => note.pitch)) + 2;
    const maxTime = Math.max(...notes.map((note) => note.end));
    context.strokeStyle = "rgba(242,239,232,.06)";
    context.lineWidth = 1;
    for (let i = 1; i < 8; i += 1) {
      context.beginPath();
      context.moveTo((width / 8) * i, 0);
      context.lineTo((width / 8) * i, height);
      context.stroke();
    }
    if (motifEnd > 0) {
      const x = (motifEnd / maxTime) * width;
      context.fillStyle = "rgba(157,140,255,.08)";
      context.fillRect(0, 0, x, height);
      context.strokeStyle = "rgba(157,140,255,.65)";
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }
    notes.forEach((note) => {
      const x = (note.start / maxTime) * width;
      const y = height - ((note.pitch - minPitch + 1) / (maxPitch - minPitch + 2)) * height;
      const noteWidth = Math.max(3, ((note.end - note.start) / maxTime) * width);
      context.fillStyle = note.start < motifEnd ? "#9d8cff" : "#d8ff62";
      context.globalAlpha = 0.55 + (note.velocity / 127) * 0.4;
      context.fillRect(x, y, noteWidth, Math.max(3, height / (maxPitch - minPitch + 5)));
    });
    context.globalAlpha = 1;
  }

  async function checkHealth() {
    try {
      const response = await fetch("/health", { cache: "no-store" });
      if (!response.ok) throw new Error("Health check failed");
      const health = await response.json();
      state.modelReady = Boolean(health.model_ready);
      state.modelVersion = health.model_version || null;
      $("#model-dot").className = `status-dot ${health.model_ready ? "ready" : "error"}`;
      $("#model-status").textContent = health.model_ready ? `Model ${state.modelVersion || ""} ready`.replace("  ", " ") : "Model unavailable";
      if (health.repository_url) $("#source-link").href = health.repository_url;
      configureCategoryControl();
      syncGenerateAvailability();
    } catch {
      state.modelReady = null;
      state.modelVersion = null;
      $("#model-dot").className = "status-dot error";
      $("#model-status").textContent = "Service waking";
      configureCategoryControl();
      syncGenerateAvailability();
      window.setTimeout(checkHealth, 8000);
    }
  }

  buildPiano();
  $("#record-button").addEventListener("click", () => state.recording ? stopRecording() : startRecording());
  $("#clear-button").addEventListener("click", clearRecording);
  $("#preview-motif-button").addEventListener("click", () => playNotes(state.notes));
  $("#record-tab").addEventListener("click", () => switchSource("record"));
  $("#upload-tab").addEventListener("click", () => switchSource("upload"));
  $("#generate-button").addEventListener("click", generate);
  $("#play-result-button").addEventListener("click", () => playNotes(state.resultNotes));
  $("#temperature").addEventListener("input", (event) => {
    state.temperature = Number(event.target.value);
    $("#temperature-value").textContent = `${temperatureLabel(state.temperature)} · ${state.temperature.toFixed(1)}`;
  });
  $("#category").addEventListener("change", (event) => {
    state.category = event.target.value;
  });
  $("#duration-control").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-duration]");
    if (!button) return;
    state.duration = Number(button.dataset.duration);
    document.querySelectorAll("[data-duration]").forEach((item) => item.classList.toggle("active", item === button));
  });
  $("#midi-file").addEventListener("change", (event) => {
    const file = event.target.files[0];
    $("#file-name").textContent = file ? file.name : "";
  });
  const uploadZone = $("#upload-zone");
  ["dragenter", "dragover"].forEach((name) => uploadZone.addEventListener(name, (event) => { event.preventDefault(); uploadZone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => uploadZone.addEventListener(name, () => uploadZone.classList.remove("dragging")));
  uploadZone.addEventListener("drop", (event) => {
    event.preventDefault();
    if (event.dataTransfer.files.length) {
      $("#midi-file").files = event.dataTransfer.files;
      $("#file-name").textContent = event.dataTransfer.files[0].name;
    }
  });

  window.addEventListener("keydown", (event) => {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    const pitch = keyboardMap[event.key.toLowerCase()];
    if (pitch === undefined || heldKeys.has(event.key.toLowerCase())) return;
    event.preventDefault();
    heldKeys.add(event.key.toLowerCase());
    noteOn(pitch);
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    const pitch = keyboardMap[key];
    if (pitch === undefined) return;
    heldKeys.delete(key);
    noteOff(pitch);
  });
  window.addEventListener("blur", () => {
    heldKeys.clear();
    [...state.voices.keys()].forEach(noteOff);
  });
  window.addEventListener("resize", () => {
    if (state.resultNotes.length) drawPianoRoll(state.resultNotes, state.resultMotifEnd);
  });
  checkHealth();
})();
