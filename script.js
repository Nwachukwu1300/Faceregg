/*
  FaceEgg — browser demo logic (no backend, no database)
  - Loads face-api.js lightweight models from CDN
  - Extracts embeddings for two uploaded images
  - Computes Euclidean distance and compares against a threshold
  - Threshold is mapped from a "strictness" slider (0..100)
*/

const ui = {
  year: document.getElementById('year'),
  fileA: document.getElementById('fileA'),
  fileB: document.getElementById('fileB'),
  previewA: document.getElementById('previewA'),
  previewB: document.getElementById('previewB'),
  statusA: document.getElementById('statusA'),
  statusB: document.getElementById('statusB'),
  strictness: document.getElementById('strictness'),
  strictnessLabel: document.getElementById('strictnessLabel'),
  thresholdLabel: document.getElementById('thresholdLabel'),
  compareBtn: document.getElementById('compareBtn'),
  resetBtn: document.getElementById('resetBtn'),
  matchStatus: document.getElementById('matchStatus'),
  distanceVal: document.getElementById('distanceVal'),
  confidenceVal: document.getElementById('confidenceVal'),
  resultNote: document.getElementById('resultNote'),
};

const state = {
  modelLoaded: false,
  descriptorA: null,
  descriptorB: null,
};

function mapStrictnessToThreshold(value) {
  // Strictness 0..100 maps to threshold 0.8..0.25 (lower threshold = stricter)
  const minT = 0.25; // strict
  const maxT = 0.8;  // loose
  const alpha = 1 - (value / 100);
  return +(minT + (maxT - minT) * (1 - alpha)).toFixed(2);
}

function euclideanDistance(a, b) {
  if (!a || !b) return NaN;
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function distanceToConfidence(distance, threshold) {
  if (Number.isNaN(distance)) return 0;
  // Simple curve: confidence is 1 at 0, ~0 at 2x threshold
  const x = Math.max(0, Math.min(1, 1 - distance / (threshold * 2)));
  return +(x).toFixed(2);
}

async function loadModels() {
  // Use a public, reliable host for demo model weights
  const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  state.modelLoaded = true;
}

async function getDescriptorFromImage(imgEl) {
  const detection = await faceapi
    .detectSingleFace(imgEl, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();
  return detection?.descriptor || null;
}

function setPreview(file, imgEl, statusEl) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  imgEl.src = url;
  imgEl.style.display = 'block';
  statusEl.textContent = 'Processing…';
}

async function processSide(which) {
  const isA = which === 'A';
  const fileInput = isA ? ui.fileA : ui.fileB;
  const preview = isA ? ui.previewA : ui.previewB;
  const status = isA ? ui.statusA : ui.statusB;
  const file = fileInput.files?.[0];
  if (!file) return;
  setPreview(file, preview, status);
  await faceapi.tf.nextFrame();
  const descriptor = await getDescriptorFromImage(preview);
  if (isA) state.descriptorA = descriptor; else state.descriptorB = descriptor;
  if (!descriptor) {
    status.textContent = 'No face detected. Try another photo.';
  } else {
    status.textContent = 'Ready.';
  }
  // Enable compare as soon as two files are selected, even if embeddings are still computing
  const hasBothFiles = Boolean(ui.fileA.files?.length && ui.fileB.files?.length);
  ui.compareBtn.disabled = !hasBothFiles;
}

function updateThresholdLabels() {
  const sVal = Number(ui.strictness.value);
  ui.strictnessLabel.textContent = String(sVal);
  const t = mapStrictnessToThreshold(sVal);
  ui.thresholdLabel.textContent = String(t);
}

async function compare() {
  // If descriptors are not ready yet, compute them now from current previews
  if (!state.descriptorA && ui.previewA?.src) {
    ui.statusA.textContent = 'Processing…';
    await faceapi.tf.nextFrame();
    state.descriptorA = await getDescriptorFromImage(ui.previewA);
    ui.statusA.textContent = state.descriptorA ? 'Ready.' : 'No face detected. Try another photo.';
  }
  if (!state.descriptorB && ui.previewB?.src) {
    ui.statusB.textContent = 'Processing…';
    await faceapi.tf.nextFrame();
    state.descriptorB = await getDescriptorFromImage(ui.previewB);
    ui.statusB.textContent = state.descriptorB ? 'Ready.' : 'No face detected. Try another photo.';
  }

  if (!state.descriptorA || !state.descriptorB) {
    ui.resultNote.textContent = 'Couldn’t detect a single face in one or both images.';
    ui.matchStatus.textContent = '—';
    ui.distanceVal.textContent = '—';
    ui.confidenceVal.textContent = '—';
    return;
  }
  const sVal = Number(ui.strictness.value);
  const threshold = mapStrictnessToThreshold(sVal);
  const distance = euclideanDistance(state.descriptorA, state.descriptorB);
  const confidence = distanceToConfidence(distance, threshold);
  const matched = distance <= threshold;
  ui.matchStatus.textContent = matched ? 'Match' : 'Not a match';
  ui.matchStatus.style.color = matched ? 'var(--ok)' : 'var(--danger)';
  ui.distanceVal.textContent = distance.toFixed(3);
  ui.confidenceVal.textContent = String(confidence);
  ui.resultNote.textContent = matched
    ? 'These photos likely show the same person at this strictness.'
    : 'These photos are unlikely to be the same person at this strictness.';
}

function resetAll() {
  ui.fileA.value = '';
  ui.fileB.value = '';
  ui.previewA.src = '';
  ui.previewB.src = '';
  ui.previewA.style.display = 'none';
  ui.previewB.style.display = 'none';
  ui.statusA.textContent = 'Waiting for image…';
  ui.statusB.textContent = 'Waiting for image…';
  ui.matchStatus.textContent = '—';
  ui.matchStatus.style.color = '';
  ui.distanceVal.textContent = '—';
  ui.confidenceVal.textContent = '—';
  ui.resultNote.textContent = 'Load two photos to begin.';
  state.descriptorA = null;
  state.descriptorB = null;
  ui.compareBtn.disabled = true;
}

function attachEvents() {
  ui.fileA.addEventListener('change', () => processSide('A'));
  ui.fileB.addEventListener('change', () => processSide('B'));
  ui.strictness.addEventListener('input', updateThresholdLabels);
  ui.compareBtn.addEventListener('click', compare);
  ui.resetBtn.addEventListener('click', resetAll);
}

async function init() {
  if (ui.year) ui.year.textContent = new Date().getFullYear();
  attachEvents();
  updateThresholdLabels();
  try {
    await loadModels();
  } catch (e) {
    console.error('Model load failed', e);
    const note = 'Failed to load on‑device models. Check your connection or try again.';
    ui.statusA.textContent = note;
    ui.statusB.textContent = note;
  }
}

document.addEventListener('DOMContentLoaded', init);


