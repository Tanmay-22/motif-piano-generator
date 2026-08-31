(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    source: "record", recording: false, countingIn: false, countInEnabled: false,
    countInToken: 0, metronomeEnabled: false, metronomeTimer: null,
    recordingStartedAt: 0, keyboardOctave: 4, notes: [], uploadNotes: [],
    activeNotes: new Map(), voices: new Map(), duration: 10, temperature: 1,
    category: "auto", modelVersion: null, modelReady: null, generating: false,
    generationTimer: null, audioContext: null, audioInput: null, timer: null,
    motifAnalysis: null, analysisController: null, resultNotes: [], resultMotifEnd: 0,
    resultMidiUrl: null, resultPayload: null, playbackTimers: [], playbackSession: 0,
    playbackNotes: [], playbackKind: null, playbackOffset: 0, playbackStartedAt: 0,
    playbackDuration: 0, playbackSpeed: 1, playing: false, playbackFrame: null, rollZoom: 1,
  };

  const pitchNames = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  const keyboardOffsets = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12 };
  const minimumKeyboardOctave = 2;
  const maximumKeyboardOctave = 7;
  const blackPitchClasses = new Set([1, 3, 6, 8, 10]);
  const heldKeys = new Map();
  const categoryLabels = {
    auto: "Motif-led · Auto", baroque_classical: "Baroque / Classical",
    romantic: "Romantic", impressionist_modern: "Impressionist / Modern",
  };
  const textureLabels = {
    monophonic: "Single-note phrase", light_polyphonic: "Light polyphony",
    full_polyphonic: "Full two-hand texture",
  };
  const samples = {
    melody: [
      { pitch: 60, start: 0, end: 0.34, velocity: 82 },
      { pitch: 64, start: 0.48, end: 0.84, velocity: 92 },
      { pitch: 67, start: 0.96, end: 1.32, velocity: 99 },
      { pitch: 69, start: 1.44, end: 1.78, velocity: 88 },
      { pitch: 67, start: 1.92, end: 2.45, velocity: 78 },
    ],
    two_hands: [
      { pitch: 48, start: 0, end: 0.82, velocity: 75 },
      { pitch: 55, start: 0, end: 0.82, velocity: 70 },
      { pitch: 64, start: 0.05, end: 0.42, velocity: 88 },
      { pitch: 67, start: 0.48, end: 0.86, velocity: 94 },
      { pitch: 50, start: 0.96, end: 1.82, velocity: 73 },
      { pitch: 57, start: 0.96, end: 1.82, velocity: 69 },
      { pitch: 65, start: 1.01, end: 1.38, velocity: 91 },
      { pitch: 69, start: 1.45, end: 1.88, velocity: 98 },
      { pitch: 52, start: 1.94, end: 2.8, velocity: 71 },
      { pitch: 59, start: 1.94, end: 2.8, velocity: 68 },
      { pitch: 67, start: 1.99, end: 2.38, velocity: 88 },
      { pitch: 72, start: 2.43, end: 2.86, velocity: 101 },
    ],
    chords: [
      { pitch: 48, start: 0, end: 0.86, velocity: 72 }, { pitch: 60, start: 0, end: 0.86, velocity: 82 }, { pitch: 64, start: 0, end: 0.86, velocity: 78 }, { pitch: 67, start: 0, end: 0.86, velocity: 75 },
      { pitch: 45, start: 1, end: 1.86, velocity: 70 }, { pitch: 57, start: 1, end: 1.86, velocity: 80 }, { pitch: 60, start: 1, end: 1.86, velocity: 76 }, { pitch: 64, start: 1, end: 1.86, velocity: 73 },
      { pitch: 41, start: 2, end: 2.86, velocity: 74 }, { pitch: 53, start: 2, end: 2.86, velocity: 84 }, { pitch: 57, start: 2, end: 2.86, velocity: 80 }, { pitch: 60, start: 2, end: 2.86, velocity: 76 },
      { pitch: 43, start: 3, end: 4.05, velocity: 76 }, { pitch: 55, start: 3, end: 4.05, velocity: 86 }, { pitch: 59, start: 3, end: 4.05, velocity: 82 }, { pitch: 62, start: 3, end: 4.05, velocity: 78 },
    ],
  };

  function noteName(pitch) {
    return `${pitchNames[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
  }

  function octaveStartPitch(octave) {
    return (octave + 1) * 12;
  }

  function keyboardPageStartOctave(octave = state.keyboardOctave) {
    return 4 + 2 * Math.floor((octave - 4) / 2);
  }

  function keyboardPitch(keyName) {
    const offset = keyboardOffsets[keyName];
    return offset === undefined ? undefined : octaveStartPitch(state.keyboardOctave) + offset;
  }

  function formatTime(seconds) {
    const safe = Math.max(0, Number(seconds) || 0);
    return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, "0")}`;
  }

  function ensureAudio() {
    if (!state.audioContext) {
      const context = new (window.AudioContext || window.webkitAudioContext)();
      const input = context.createGain();
      const compressor = context.createDynamicsCompressor();
      const reverb = context.createConvolver();
      const wet = context.createGain();
      const dry = context.createGain();
      const impulseLength = Math.floor(context.sampleRate * 0.65);
      const impulse = context.createBuffer(2, impulseLength, context.sampleRate);
      for (let channel = 0; channel < 2; channel += 1) {
        const data = impulse.getChannelData(channel);
        for (let index = 0; index < impulseLength; index += 1) {
          data[index] = (Math.random() * 2 - 1) * Math.pow(1 - index / impulseLength, 3.5);
        }
      }
      reverb.buffer = impulse;
      wet.gain.value = 0.13;
      dry.gain.value = 0.88;
      compressor.threshold.value = -18;
      compressor.knee.value = 16;
      compressor.ratio.value = 5;
      input.connect(dry).connect(compressor);
      input.connect(reverb).connect(wet).connect(compressor);
      compressor.connect(context.destination);
      state.audioContext = context;
      state.audioInput = input;
    }
    if (state.audioContext.state === "suspended") state.audioContext.resume();
    return state.audioContext;
  }

  function syncKeyActive(pitch) {
    const active = [...state.voices.values()].some((voice) => voice.pitch === pitch);
    document.querySelectorAll(`[data-pitch="${pitch}"]`).forEach((key) => key.classList.toggle("active", active));
  }

  function startVoice(pitch, velocity = 100, voiceId = `manual:${pitch}`) {
    if (state.voices.has(voiceId)) return;
    const context = ensureAudio();
    const frequency = 440 * Math.pow(2, (pitch - 69) / 12);
    const envelope = context.createGain();
    const tone = context.createBiquadFilter();
    const volume = Math.min(0.13, (velocity / 127) * 0.11);
    const partials = [
      { ratio: 1, level: 1, type: "triangle", detune: -1.5 },
      { ratio: 2, level: 0.32, type: "sine", detune: 1.8 },
      { ratio: 3, level: 0.12, type: "sine", detune: -2.4 },
      { ratio: 4.02, level: 0.055, type: "sine", detune: 2.1 },
    ];
    const oscillators = partials.map((partial) => {
      const oscillator = context.createOscillator();
      const partialGain = context.createGain();
      oscillator.type = partial.type;
      oscillator.frequency.value = frequency * partial.ratio;
      oscillator.detune.value = partial.detune;
      partialGain.gain.value = partial.level;
      oscillator.connect(partialGain).connect(tone);
      oscillator.start();
      return oscillator;
    });
    tone.type = "lowpass";
    tone.frequency.value = Math.min(5200, 1500 + frequency * 5.5);
    tone.Q.value = 0.7;
    envelope.gain.setValueAtTime(0.0001, context.currentTime);
    envelope.gain.exponentialRampToValueAtTime(volume, context.currentTime + 0.012);
    envelope.gain.exponentialRampToValueAtTime(Math.max(0.0001, volume * 0.42), context.currentTime + 0.58);
    tone.connect(envelope).connect(state.audioInput);
    state.voices.set(voiceId, { pitch, oscillators, envelope });
    syncKeyActive(pitch);
  }

  function stopVoice(voiceId) {
    const voice = state.voices.get(voiceId);
    if (!voice) return;
    const context = ensureAudio();
    voice.envelope.gain.cancelScheduledValues(context.currentTime);
    voice.envelope.gain.setValueAtTime(Math.max(voice.envelope.gain.value, 0.0001), context.currentTime);
    voice.envelope.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.24);
    voice.oscillators.forEach((oscillator) => oscillator.stop(context.currentTime + 0.27));
    state.voices.delete(voiceId);
    window.setTimeout(() => syncKeyActive(voice.pitch), 280);
  }

  function stopAllVoices() {
    [...state.voices.keys()].forEach(stopVoice);
  }

  function playClick(accent = false) {
    const context = ensureAudio();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = accent ? 1250 : 880;
    gain.gain.setValueAtTime(0.055, context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, context.currentTime + 0.055);
    oscillator.connect(gain).connect(state.audioInput);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.06);
  }

  function noteOn(pitch, velocity = 100, voiceId = `manual:${pitch}`) {
    startVoice(pitch, velocity, voiceId);
    if (state.recording && voiceId.startsWith("manual:") && !state.activeNotes.has(pitch)) {
      state.activeNotes.set(pitch, { pitch, velocity, start: (performance.now() - state.recordingStartedAt) / 1000 });
      updateRecordingStatus();
    }
  }

  function noteOff(pitch, voiceId = `manual:${pitch}`) {
    stopVoice(voiceId);
    const active = state.activeNotes.get(pitch);
    if (active && voiceId.startsWith("manual:")) {
      active.end = Math.max(active.start + 0.03, (performance.now() - state.recordingStartedAt) / 1000);
      state.notes.push(active);
      state.notes.sort((left, right) => left.start - right.start || left.pitch - right.pitch);
      state.activeNotes.delete(pitch);
      updateRecorder();
      updateRecordingStatus();
    }
  }

  function buildPiano() {
    const piano = $("#piano");
    const pageStartOctave = keyboardPageStartOctave();
    const minPitch = octaveStartPitch(pageStartOctave);
    const maxPitch = minPitch + 24;
    piano.replaceChildren();
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
      markComputerKey(key, pitch);
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
        markComputerKey(black, nextPitch);
        wirePointerKey(black, nextPitch);
        piano.appendChild(black);
      }
    });
    syncOctaveControls();
    new Set([...state.voices.values()].map((voice) => voice.pitch)).forEach(syncKeyActive);
  }

  function markComputerKey(key, pitch) {
    const offset = pitch - octaveStartPitch(state.keyboardOctave);
    const mapped = Object.entries(keyboardOffsets).find(([, value]) => value === offset);
    if (!mapped) return;
    key.classList.add("computer-mapped");
    key.dataset.computerKey = mapped[0];
    key.setAttribute("aria-label", `${noteName(pitch)}, computer key ${mapped[0].toUpperCase()}`);
    key.innerHTML = `<span class="key-label">${mapped[0].toUpperCase()}</span>`;
  }

  function syncOctaveControls() {
    const pageStartOctave = keyboardPageStartOctave();
    const position = state.keyboardOctave === pageStartOctave ? "Left 13 keys" : "Right 13 keys";
    $("#octave-display").textContent = `Octave ${state.keyboardOctave}`;
    $("#octave-position").textContent = position;
    $("#piano-range").textContent = `Page C${pageStartOctave}–C${pageStartOctave + 2} · active C${state.keyboardOctave}–C${state.keyboardOctave + 1}`;
    $("#octave-down-button").disabled = state.keyboardOctave <= minimumKeyboardOctave;
    $("#octave-up-button").disabled = state.keyboardOctave >= maximumKeyboardOctave;
  }

  function releaseManualVoices() {
    [...state.voices.entries()]
      .filter(([voiceId]) => voiceId.startsWith("manual:"))
      .forEach(([voiceId, voice]) => noteOff(voice.pitch, voiceId));
    heldKeys.clear();
  }

  function shiftKeyboardOctave(direction) {
    const nextOctave = Math.max(minimumKeyboardOctave, Math.min(maximumKeyboardOctave, state.keyboardOctave + direction));
    if (nextOctave === state.keyboardOctave) {
      updateRecordingStatus(`The computer-key range stops at octave ${state.keyboardOctave}.`);
      return;
    }
    const previousPage = keyboardPageStartOctave();
    releaseManualVoices();
    state.keyboardOctave = nextOctave;
    buildPiano();
    const pageChanged = previousPage !== keyboardPageStartOctave();
    updateRecordingStatus(`Computer keys now play C${nextOctave}–C${nextOctave + 1}${pageChanged ? ` on the C${keyboardPageStartOctave()}–C${keyboardPageStartOctave() + 2} page` : ""}.`);
  }

  function wirePointerKey(key, pitch) {
    key.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      key.setPointerCapture(event.pointerId);
      noteOn(pitch);
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => key.addEventListener(name, () => noteOff(pitch)));
  }

  function updateRecordingStatus(message = null) {
    const status = $("#recording-status");
    status.classList.toggle("recording-live", state.recording || state.countingIn);
    if (message) status.textContent = message;
    else if (state.countingIn) status.textContent = "Count-in… get ready.";
    else if (state.recording && state.activeNotes.size) status.textContent = `${state.activeNotes.size} key${state.activeNotes.size === 1 ? " is" : "s are"} held — release to finish the note.`;
    else if (state.recording) status.textContent = "Recording — play your phrase, then stop.";
    else if (state.notes.length) status.textContent = "Motif ready. Preview it, undo the last onset, or compose.";
    else status.textContent = "Ready when you are.";
  }

  function closeActiveRecordedNotes() {
    [...state.activeNotes.keys()].forEach((pitch) => noteOff(pitch));
  }

  function startMetronome() {
    stopMetronome();
    if (!state.metronomeEnabled) return;
    let beat = 0;
    playClick(true);
    state.metronomeTimer = window.setInterval(() => {
      beat = (beat + 1) % 4;
      playClick(beat === 0);
    }, 500);
  }

  function stopMetronome() {
    window.clearInterval(state.metronomeTimer);
    state.metronomeTimer = null;
  }

  function actualStartRecording() {
    stopPlayback(true);
    state.notes = [];
    state.activeNotes.clear();
    state.recording = true;
    state.recordingStartedAt = performance.now();
    $("#record-button").classList.add("recording");
    $("#record-label").textContent = "Stop recording";
    state.timer = window.setInterval(updateRecorder, 50);
    startMetronome();
    updateRecorder();
    updateRecordingStatus();
  }

  async function prepareRecording() {
    if (!state.countInEnabled) return actualStartRecording();
    stopPlayback(true);
    state.countingIn = true;
    const token = ++state.countInToken;
    $("#record-button").classList.add("recording");
    for (let beat = 4; beat >= 1; beat -= 1) {
      if (token !== state.countInToken) return;
      $("#record-label").textContent = `Cancel · ${beat}`;
      updateRecordingStatus(`Count-in ${beat}…`);
      playClick(beat === 4);
      await new Promise((resolve) => window.setTimeout(resolve, 500));
    }
    if (token !== state.countInToken) return;
    state.countingIn = false;
    actualStartRecording();
  }

  function cancelCountIn() {
    state.countInToken += 1;
    state.countingIn = false;
    $("#record-button").classList.remove("recording");
    $("#record-label").textContent = state.notes.length ? "Record again" : "Start recording";
    updateRecordingStatus("Count-in cancelled.");
  }

  function trimLeadingSilence() {
    if (!state.notes.length) return;
    const firstOnset = Math.min(...state.notes.map((note) => note.start));
    state.notes = state.notes.map((note) => ({ ...note, start: Math.max(0, note.start - firstOnset), end: note.end - firstOnset }));
  }

  function stopRecording() {
    closeActiveRecordedNotes();
    state.recording = false;
    window.clearInterval(state.timer);
    stopMetronome();
    trimLeadingSilence();
    $("#record-button").classList.remove("recording");
    $("#record-label").textContent = "Record again";
    updateRecorder();
    updateRecordingStatus();
    analyzeCurrentMotif();
  }

  function updateRecorder() {
    const elapsed = state.recording ? (performance.now() - state.recordingStartedAt) / 1000 : Math.max(0, ...state.notes.map((note) => note.end));
    const minutes = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const seconds = (elapsed % 60).toFixed(1).padStart(4, "0");
    $("#record-time").textContent = `${minutes}:${seconds}`;
    $("#note-count").textContent = `${state.notes.length} note${state.notes.length === 1 ? "" : "s"}`;
    $("#preview-motif-button").disabled = state.notes.length === 0;
    $("#undo-button").disabled = state.notes.length === 0 || state.recording;
    $("#motif-timeline").classList.toggle("hidden", state.notes.length === 0);
    if (state.notes.length && state.playbackKind === "motif") drawMotifRoll(currentPlaybackTime());
    if (elapsed >= 30 && state.recording) stopRecording();
  }

  function clearRecording() {
    if (state.recording) stopRecording();
    if (state.countingIn) cancelCountIn();
    stopPlayback(true);
    state.notes = [];
    state.activeNotes.clear();
    state.motifAnalysis = null;
    $("#record-label").textContent = "Start recording";
    $("#motif-analysis").classList.add("hidden");
    updateRecorder();
    updateRecordingStatus();
  }

  function undoLastOnset() {
    if (!state.notes.length || state.recording) return;
    stopPlayback(true);
    const lastOnset = Math.max(...state.notes.map((note) => note.start));
    state.notes = state.notes.filter((note) => Math.abs(note.start - lastOnset) > 0.035);
    updateRecorder();
    updateRecordingStatus(state.notes.length ? "Removed the last note or chord." : "Motif cleared.");
    if (state.notes.length >= 2) analyzeCurrentMotif();
    else $("#motif-analysis").classList.add("hidden");
  }

  function loadSample(name) {
    const notes = samples[name];
    if (!notes) return;
    if (state.recording) stopRecording();
    switchSource("record");
    stopPlayback(true);
    state.notes = notes.map((note) => ({ ...note }));
    $("#record-label").textContent = "Record again";
    updateRecorder();
    updateRecordingStatus("Example loaded. Listen, edit, or compose from it.");
    analyzeCurrentMotif();
    startPlayback(state.notes, 0, "motif");
  }

  function clearPlaybackTimers() {
    state.playbackTimers.forEach(window.clearTimeout);
    state.playbackTimers = [];
    if (state.playbackFrame) cancelAnimationFrame(state.playbackFrame);
    state.playbackFrame = null;
    state.playbackSession += 1;
    stopAllVoices();
  }

  function currentPlaybackTime() {
    if (!state.playing) return state.playbackOffset;
    return Math.min(state.playbackDuration, state.playbackOffset + ((performance.now() - state.playbackStartedAt) / 1000) * state.playbackSpeed);
  }

  function stopPlayback(reset = true) {
    clearPlaybackTimers();
    state.playing = false;
    if (reset) state.playbackOffset = 0;
    updatePlaybackVisuals();
  }

  function pausePlayback() {
    if (!state.playing) return;
    state.playbackOffset = currentPlaybackTime();
    clearPlaybackTimers();
    state.playing = false;
    updatePlaybackVisuals();
  }

  function startPlayback(notes, offset = 0, kind = "result") {
    if (!notes.length) return;
    clearPlaybackTimers();
    ensureAudio();
    state.playbackNotes = notes;
    state.playbackKind = kind;
    state.playbackDuration = Math.max(...notes.map((note) => Number(note.end)));
    state.playbackOffset = Math.max(0, Math.min(offset, Math.max(0, state.playbackDuration - 0.001)));
    state.playbackStartedAt = performance.now();
    state.playing = true;
    const session = state.playbackSession;
    notes.forEach((note, index) => {
      if (note.end <= state.playbackOffset) return;
      const voiceId = `play:${session}:${index}`;
      const startDelay = Math.max(0, ((note.start - state.playbackOffset) / state.playbackSpeed) * 1000);
      const endDelay = Math.max(30, ((note.end - state.playbackOffset) / state.playbackSpeed) * 1000);
      state.playbackTimers.push(window.setTimeout(() => startVoice(note.pitch, note.velocity, voiceId), startDelay));
      state.playbackTimers.push(window.setTimeout(() => stopVoice(voiceId), endDelay));
    });
    animatePlayback();
    updatePlaybackVisuals();
  }

  function animatePlayback() {
    if (!state.playing) return;
    const current = currentPlaybackTime();
    if (current >= state.playbackDuration - 0.002) {
      state.playbackOffset = state.playbackDuration;
      clearPlaybackTimers();
      state.playing = false;
      updatePlaybackVisuals();
      return;
    }
    updatePlaybackVisuals();
    if (state.playbackKind === "result" && state.rollZoom > 1) followPlayhead(current);
    state.playbackFrame = requestAnimationFrame(animatePlayback);
  }

  function toggleResultPlayback() {
    if (!state.resultNotes.length) return;
    if (state.playing && state.playbackKind === "result") return pausePlayback();
    const offset = state.playbackKind === "result" && state.playbackOffset < state.playbackDuration ? state.playbackOffset : 0;
    startPlayback(state.resultNotes, offset, "result");
  }

  function restartResultPlayback() {
    if (state.resultNotes.length) startPlayback(state.resultNotes, 0, "result");
  }

  function updatePlaybackVisuals() {
    const current = state.playbackKind === "result" ? currentPlaybackTime() : 0;
    const button = $("#play-result-button");
    const isResultPlaying = state.playing && state.playbackKind === "result";
    button.querySelector("span:first-child").textContent = isResultPlaying ? "Ⅱ" : "▶";
    button.querySelector("span:last-child").textContent = isResultPlaying ? "Pause" : (state.playbackKind === "result" && state.playbackOffset > 0 && state.playbackOffset < state.playbackDuration ? "Resume" : "Play result");
    const total = state.resultNotes.length ? Math.max(...state.resultNotes.map((note) => Number(note.end))) : 0;
    $("#play-time").textContent = `${formatTime(current)} / ${formatTime(total)}`;
    if (state.resultNotes.length) drawPianoRoll(isResultPlaying || state.playbackKind === "result" ? current : null);
    if (state.notes.length) drawMotifRoll(state.playbackKind === "motif" ? currentPlaybackTime() : null);
  }

  function followPlayhead(current) {
    const scroll = $("#piano-roll-scroll");
    const canvas = $("#piano-roll");
    const x = (current / Math.max(state.playbackDuration, 0.01)) * canvas.clientWidth;
    const rightEdge = scroll.scrollLeft + scroll.clientWidth * 0.82;
    const leftEdge = scroll.scrollLeft + scroll.clientWidth * 0.18;
    if (x > rightEdge || x < leftEdge) scroll.scrollTo({ left: Math.max(0, x - scroll.clientWidth * 0.28), behavior: "smooth" });
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
    if (record && state.notes.length >= 2) analyzeCurrentMotif();
    if (!record && $("#midi-file").files[0]) analyzeCurrentMotif();
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

  function syncGenerateAvailability() {
    const button = $("#generate-button");
    button.disabled = state.generating || state.modelReady === false;
    if (!state.generating) button.querySelector("span:first-child").textContent = state.modelReady === false ? "Model unavailable" : "Generate continuation";
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
    help.textContent = state.modelVersion === "v1" ? "The deployed v1 checkpoint follows the motif but does not support style categories yet." : "Style choices become available when the category-aware v2 checkpoint is loaded.";
  }

  function renderMotifAnalysis(features) {
    state.motifAnalysis = features;
    $("#motif-analysis").classList.remove("hidden");
    $("#analysis-texture").textContent = textureLabels[features.texture] || features.texture.replaceAll("_", " ");
    $("#analysis-range").textContent = `${noteName(features.pitch_min)}–${noteName(features.pitch_max)} · ${features.pitch_span} semitones`;
    $("#analysis-rhythm").textContent = features.median_onset_gap > 0 ? `${features.median_onset_gap.toFixed(2)}s median gap` : "Sustained chord";
    $("#analysis-dynamics").textContent = `Velocity ${Math.round(features.velocity_mean)} · spread ${features.velocity_range}`;
    $("#analysis-duration").textContent = `${features.note_count} notes · ${features.duration_seconds.toFixed(1)}s`;
    $("#analysis-density").textContent = `${features.note_density.toFixed(1)} notes/s · peak ${features.peak_polyphony}`;
    const texture = features.bass_and_treble ? "a wide bass-and-treble register" : textureLabels[features.texture].toLowerCase();
    $("#analysis-summary").textContent = `The phrase uses ${texture}; its exact delays, durations, pitches and velocities become the model's context.`;
  }

  async function analyzeCurrentMotif() {
    const file = $("#midi-file").files[0];
    if ((state.source === "record" && state.notes.length < 2) || (state.source === "upload" && !file)) {
      $("#motif-analysis").classList.add("hidden");
      return;
    }
    if (state.analysisController) state.analysisController.abort();
    state.analysisController = new AbortController();
    const form = new FormData();
    if (state.source === "record") form.append("motif_json", JSON.stringify(state.notes));
    else form.append("midi_file", file);
    try {
      const response = await fetch("/api/analyze", { method: "POST", body: form, signal: state.analysisController.signal });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || "This motif could not be analyzed.");
      if (state.source === "upload") {
        state.uploadNotes = payload.notes;
        $("#preview-upload-button").disabled = false;
      }
      renderMotifAnalysis(payload.features);
    } catch (error) {
      if (error.name !== "AbortError") showNotice(error.message);
    }
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
    $("#generation-progress").classList.remove("hidden");
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
    $("#result-category").textContent = payload.category_applied ? (categoryLabels[payload.category] || payload.category) : "Motif-led · v1";
    const texture = $("#result-texture");
    if (payload.inferred_texture) {
      texture.textContent = textureLabels[payload.inferred_texture] || payload.inferred_texture.replaceAll("_", " ");
      texture.classList.remove("hidden");
    } else texture.classList.add("hidden");
    const message = $("#result-message");
    if (payload.timed_out || payload.reached_target_duration === false) {
      message.textContent = "The CPU time limit was reached, so this is a playable partial continuation. Try 5 seconds for a more reliable complete result on the free service.";
      message.classList.remove("hidden");
    } else message.classList.add("hidden");
  }

  function updateGenerationInsight(payload) {
    const motifNotes = payload.notes.filter((note) => Number(note.start) < Number(payload.motif_end_seconds) - 0.001);
    const continuationNotes = payload.notes.length - motifNotes.length;
    $("#insight-input").textContent = `${motifNotes.length} notes · ${Number(payload.motif_end_seconds).toFixed(1)}s`;
    $("#insight-output").textContent = `${continuationNotes} notes`;
    $("#insight-direction").textContent = payload.category_applied ? (categoryLabels[payload.category] || payload.category) : "Motif-led";
    $("#insight-creativity").textContent = `${temperatureLabel(state.temperature)} · ${state.temperature.toFixed(1)}`;
    const feature = state.motifAnalysis;
    if (feature) {
      const register = feature.bass_and_treble ? "wide two-register shape" : `${noteName(feature.pitch_min)}–${noteName(feature.pitch_max)} range`;
      $("#insight-summary").textContent = `The model received every onset delay, duration, pitch and velocity, then continued the ${textureLabels[feature.texture].toLowerCase()} and ${register}.`;
    } else $("#insight-summary").textContent = "The model continued from the motif's note timing, pitch, duration, velocity and inferred texture.";
  }

  async function generate() {
    hideNotice();
    const file = $("#midi-file").files[0];
    if (state.source === "record" && state.recording) stopRecording();
    if (state.source === "record" && state.notes.length < 2) return showNotice("Record at least two notes before generating.");
    if (state.source === "upload" && !file) return showNotice("Choose a MIDI file before generating.");
    stopPlayback(true);
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
      state.resultPayload = payload;
      state.resultNotes = payload.notes;
      state.resultMotifEnd = Number(payload.motif_end_seconds) || 0;
      state.playbackKind = "result";
      state.playbackOffset = 0;
      state.playbackDuration = Math.max(0, ...state.resultNotes.map((note) => Number(note.end)));
      if (state.resultMidiUrl) URL.revokeObjectURL(state.resultMidiUrl);
      const binary = atob(payload.midi_base64);
      const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
      state.resultMidiUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/midi" }));
      $("#download-button").href = state.resultMidiUrl;
      $("#result-duration").textContent = `${Number(payload.duration_seconds).toFixed(1)} seconds`;
      $("#result-card").classList.remove("hidden");
      updateResultContext(payload);
      updateGenerationInsight(payload);
      updatePlaybackVisuals();
      $("#result-card").scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (error) {
      if (error.name === "AbortError") showNotice("Generation took longer than 90 seconds and the browser stopped waiting. The free service may be waking up; try a 5-second continuation once it is ready.");
      else showNotice(error.message.includes("fetch") ? "The free service may still be waking up. Wait a moment and try again." : error.message);
    } finally {
      window.clearTimeout(clientTimeout);
      stopGenerationProgress();
      state.generating = false;
      syncGenerateAvailability();
    }
  }

  function paintPianoRoll(context, width, height, notes, motifEnd = 0, playhead = null) {
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#11130f";
    context.fillRect(0, 0, width, height);
    if (!notes.length) return;
    const minPitch = Math.max(21, Math.min(...notes.map((note) => note.pitch)) - 2);
    const maxPitch = Math.min(108, Math.max(...notes.map((note) => note.pitch)) + 2);
    const pitchSpan = Math.max(5, maxPitch - minPitch + 1);
    const maxTime = Math.max(0.1, ...notes.map((note) => Number(note.end)));
    context.lineWidth = 1;
    for (let pitch = minPitch; pitch <= maxPitch; pitch += 1) {
      const y = height - ((pitch - minPitch + 0.5) / pitchSpan) * height;
      context.strokeStyle = pitch % 12 === 0 ? "rgba(242,239,232,.1)" : "rgba(242,239,232,.035)";
      context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
    }
    const seconds = Math.ceil(maxTime);
    const gridStep = seconds <= 12 ? 1 : seconds <= 30 ? 2 : 5;
    for (let second = 0; second <= seconds; second += gridStep) {
      const x = (second / maxTime) * width;
      context.strokeStyle = "rgba(242,239,232,.07)";
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
      context.fillStyle = "rgba(242,239,232,.35)";
      context.font = "9px DM Mono, monospace";
      context.fillText(`${second}s`, x + 4, 12);
    }
    if (motifEnd > 0) {
      const x = (motifEnd / maxTime) * width;
      context.fillStyle = "rgba(157,140,255,.075)";
      context.fillRect(0, 0, x, height);
      context.strokeStyle = "rgba(157,140,255,.7)";
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    }
    const noteHeight = Math.max(3.5, Math.min(12, (height / pitchSpan) * 0.68));
    notes.forEach((note) => {
      const x = (note.start / maxTime) * width;
      const y = height - ((note.pitch - minPitch + 0.7) / pitchSpan) * height;
      const noteWidth = Math.max(3, ((note.end - note.start) / maxTime) * width);
      const active = playhead !== null && note.start <= playhead && note.end > playhead;
      context.save();
      context.globalAlpha = active ? 1 : 0.52 + (note.velocity / 127) * 0.42;
      context.fillStyle = note.start < motifEnd ? "#9d8cff" : "#d8ff62";
      if (active) { context.shadowColor = context.fillStyle; context.shadowBlur = 12; }
      context.fillRect(x, y, noteWidth, noteHeight);
      if (active) { context.strokeStyle = "#ffffff"; context.lineWidth = 1; context.strokeRect(x - 1, y - 1, noteWidth + 2, noteHeight + 2); }
      context.restore();
    });
    if (playhead !== null) {
      const x = (Math.max(0, Math.min(playhead, maxTime)) / maxTime) * width;
      context.strokeStyle = "rgba(255,255,255,.92)";
      context.lineWidth = 1.5;
      context.shadowColor = "rgba(216,255,98,.8)";
      context.shadowBlur = 8;
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
      context.shadowBlur = 0;
      context.fillStyle = "#ffffff";
      context.beginPath(); context.arc(x, 8, 3, 0, Math.PI * 2); context.fill();
    }
  }

  function drawCanvas(canvas, notes, motifEnd, playhead) {
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const pixelWidth = Math.round(rect.width * ratio);
    const pixelHeight = Math.round(rect.height * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    paintPianoRoll(context, rect.width, rect.height, notes, motifEnd, playhead);
  }

  function drawPianoRoll(playhead = null) { drawCanvas($("#piano-roll"), state.resultNotes, state.resultMotifEnd, playhead); }

  function drawMotifRoll(playhead = null) {
    const end = Math.max(0, ...state.notes.map((note) => note.end));
    drawCanvas($("#motif-roll"), state.notes, end + 0.01, playhead);
  }

  function seekResult(event) {
    if (!state.resultNotes.length) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const total = Math.max(...state.resultNotes.map((note) => Number(note.end)));
    const target = Math.max(0, Math.min(total, ((event.clientX - rect.left) / rect.width) * total));
    const wasPlaying = state.playing && state.playbackKind === "result";
    if (wasPlaying) startPlayback(state.resultNotes, target, "result");
    else {
      clearPlaybackTimers(); state.playbackKind = "result"; state.playbackDuration = total;
      state.playbackOffset = target; state.playing = false; updatePlaybackVisuals();
    }
  }

  function toggleZoom() {
    state.rollZoom = state.rollZoom === 1 ? 2 : state.rollZoom === 2 ? 4 : 1;
    const scroll = $("#piano-roll-scroll");
    scroll.classList.toggle("zoom-2", state.rollZoom === 2);
    scroll.classList.toggle("zoom-4", state.rollZoom === 4);
    $("#zoom-button").textContent = `Zoom ${state.rollZoom}×`;
    requestAnimationFrame(updatePlaybackVisuals);
  }

  async function toggleFullscreen() {
    const stage = $("#piano-roll-stage");
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await stage.requestFullscreen();
    } catch { showNotice("Fullscreen is not available in this browser, but zoom still works."); }
    syncFullscreenButton();
    window.setTimeout(syncFullscreenButton, 80);
    window.setTimeout(syncFullscreenButton, 500);
    window.setTimeout(syncFullscreenButton, 1500);
  }

  function syncFullscreenButton() {
    $("#fullscreen-button").textContent = document.fullscreenElement ? "Exit fullscreen" : "Fullscreen";
  }

  function exportResultImage() {
    if (!state.resultNotes.length) return;
    const canvas = document.createElement("canvas");
    canvas.width = 1600; canvas.height = 900;
    const context = canvas.getContext("2d");
    const gradient = context.createLinearGradient(0, 0, 1600, 900);
    gradient.addColorStop(0, "#171a16"); gradient.addColorStop(1, "#0c0e0c");
    context.fillStyle = gradient; context.fillRect(0, 0, 1600, 900);
    context.fillStyle = "#d8ff62"; context.font = "500 22px monospace";
    context.fillText("MOTIF · GENERATED PIANO CONTINUATION", 80, 82);
    context.fillStyle = "#f2efe8"; context.font = "600 58px system-ui, sans-serif";
    context.fillText("A new phrase, shaped by yours", 80, 154);
    context.fillStyle = "#98958f"; context.font = "20px system-ui, sans-serif";
    context.fillText(`${categoryLabels[state.resultPayload?.category] || "Motif-led"}  ·  ${state.resultNotes.length} notes  ·  ${state.playbackDuration.toFixed(1)} seconds`, 80, 196);
    context.save(); context.translate(80, 245);
    paintPianoRoll(context, 1440, 520, state.resultNotes, state.resultMotifEnd, null);
    context.restore();
    context.fillStyle = "#9d8cff"; context.fillRect(80, 806, 24, 6);
    context.fillStyle = "#98958f"; context.font = "16px monospace"; context.fillText("YOUR MOTIF", 116, 814);
    context.fillStyle = "#d8ff62"; context.fillRect(280, 806, 24, 6);
    context.fillStyle = "#98958f"; context.fillText("GENERATED CONTINUATION", 316, 814);
    context.fillText("motif-piano-generator", 1280, 814);
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
    link.download = "motif-piano-roll.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
    const button = $("#download-image-button");
    button.dataset.exported = "true";
    button.innerHTML = "Image saved <span aria-hidden=\"true\">✓</span>";
    window.setTimeout(() => {
      delete button.dataset.exported;
      button.innerHTML = "Save image <span aria-hidden=\"true\">↓</span>";
    }, 1800);
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
      configureCategoryControl(); syncGenerateAvailability();
      if (!health.model_ready) window.setTimeout(checkHealth, 12000);
    } catch {
      state.modelReady = null; state.modelVersion = null;
      $("#model-dot").className = "status-dot";
      $("#model-status").textContent = "Service waking…";
      configureCategoryControl(); syncGenerateAvailability();
      window.setTimeout(checkHealth, 8000);
    }
  }

  function isTypingTarget(target) {
    return target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement || target.isContentEditable;
  }

  buildPiano();
  $("#record-button").addEventListener("click", () => {
    if (state.countingIn) cancelCountIn(); else if (state.recording) stopRecording(); else prepareRecording();
  });
  $("#count-in-toggle").addEventListener("click", (event) => {
    state.countInEnabled = !state.countInEnabled;
    event.currentTarget.setAttribute("aria-pressed", String(state.countInEnabled));
    event.currentTarget.textContent = `Count-in ${state.countInEnabled ? "on" : "off"}`;
  });
  $("#metronome-toggle").addEventListener("click", (event) => {
    state.metronomeEnabled = !state.metronomeEnabled;
    event.currentTarget.setAttribute("aria-pressed", String(state.metronomeEnabled));
    event.currentTarget.textContent = `Metronome ${state.metronomeEnabled ? "on" : "off"}`;
    if (state.recording) startMetronome();
  });
  $("#octave-down-button").addEventListener("click", () => shiftKeyboardOctave(-1));
  $("#octave-up-button").addEventListener("click", () => shiftKeyboardOctave(1));
  $("#undo-button").addEventListener("click", undoLastOnset);
  $("#clear-button").addEventListener("click", clearRecording);
  $("#preview-motif-button").addEventListener("click", () => startPlayback(state.notes, 0, "motif"));
  $("#preview-upload-button").addEventListener("click", () => startPlayback(state.uploadNotes, 0, "motif"));
  $("#record-tab").addEventListener("click", () => switchSource("record"));
  $("#upload-tab").addEventListener("click", () => switchSource("upload"));
  $("#generate-button").addEventListener("click", generate);
  $("#play-result-button").addEventListener("click", toggleResultPlayback);
  $("#restart-result-button").addEventListener("click", restartResultPlayback);
  $("#piano-roll").addEventListener("click", seekResult);
  $("#zoom-button").addEventListener("click", toggleZoom);
  $("#fullscreen-button").addEventListener("click", toggleFullscreen);
  $("#download-image-button").addEventListener("click", exportResultImage);
  $(".sample-strip").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-sample]");
    if (button) loadSample(button.dataset.sample);
  });
  $(".speed-control").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-speed]");
    if (!button) return;
    const current = currentPlaybackTime();
    const wasPlaying = state.playing;
    state.playbackSpeed = Number(button.dataset.speed);
    document.querySelectorAll("[data-speed]").forEach((item) => item.classList.toggle("active", item === button));
    if (wasPlaying) startPlayback(state.playbackNotes, current, state.playbackKind);
  });
  $("#temperature").addEventListener("input", (event) => {
    state.temperature = Number(event.target.value);
    $("#temperature-value").textContent = `${temperatureLabel(state.temperature)} · ${state.temperature.toFixed(1)}`;
  });
  $("#category").addEventListener("change", (event) => { state.category = event.target.value; });
  $("#duration-control").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-duration]");
    if (!button) return;
    state.duration = Number(button.dataset.duration);
    document.querySelectorAll("[data-duration]").forEach((item) => item.classList.toggle("active", item === button));
  });

  function handleMidiFile(file) {
    $("#file-name").textContent = file ? file.name : "";
    state.uploadNotes = [];
    $("#preview-upload-button").disabled = true;
    if (file) analyzeCurrentMotif(); else $("#motif-analysis").classList.add("hidden");
  }

  $("#midi-file").addEventListener("change", (event) => handleMidiFile(event.target.files[0]));
  const uploadZone = $("#upload-zone");
  ["dragenter", "dragover"].forEach((name) => uploadZone.addEventListener(name, (event) => {
    event.preventDefault(); uploadZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => uploadZone.addEventListener(name, () => uploadZone.classList.remove("dragging")));
  uploadZone.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer(); transfer.items.add(file);
    $("#midi-file").files = transfer.files; handleMidiFile(file);
  });

  window.addEventListener("keydown", (event) => {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    const keyName = event.key.toLowerCase();
    if (!isTypingTarget(event.target) && (keyName === "z" || keyName === "x")) {
      event.preventDefault();
      shiftKeyboardOctave(keyName === "x" ? 1 : -1);
      return;
    }
    if (!isTypingTarget(event.target) && event.code === "Space" && !$("#result-card").classList.contains("hidden")) {
      event.preventDefault(); toggleResultPlayback(); return;
    }
    if (event.key === "Escape") { stopPlayback(true); return; }
    if (isTypingTarget(event.target)) return;
    const pitch = keyboardPitch(keyName);
    if (pitch === undefined || heldKeys.has(keyName)) return;
    if (pitch < 21 || pitch > 108) return;
    event.preventDefault(); heldKeys.set(keyName, pitch); noteOn(pitch);
  });
  window.addEventListener("keyup", (event) => {
    const keyName = event.key.toLowerCase();
    if (keyboardOffsets[keyName] === undefined) return;
    const pitch = heldKeys.get(keyName) ?? keyboardPitch(keyName);
    heldKeys.delete(keyName); noteOff(pitch);
  });
  window.addEventListener("blur", () => { heldKeys.clear(); closeActiveRecordedNotes(); stopAllVoices(); });
  window.addEventListener("resize", () => {
    if (state.resultNotes.length) updatePlaybackVisuals();
    if (state.notes.length) drawMotifRoll(state.playbackKind === "motif" ? currentPlaybackTime() : null);
  });
  document.addEventListener("fullscreenchange", () => {
    syncFullscreenButton();
    requestAnimationFrame(updatePlaybackVisuals);
  });

  updateRecorder();
  updateRecordingStatus();
  checkHealth();
})();
