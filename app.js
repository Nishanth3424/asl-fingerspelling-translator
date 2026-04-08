// === WORD BUILDER DICTIONARY ===
// Top ~1200 common English words, lowercase, used for autocomplete suggestions
const WORD_DICT = ('a able about above across act add age ago agree air all allow alone along '
+ 'already also always am among and animal another answer any apart appear apple '
+ 'area arm around ask at back ball base be bear because become bed before begin '
+ 'being bell best better between big bird black blue body book both box boy bring '
+ 'brother build burn but buy by call came can car care carry case cause change '
+ 'check child children city class clear close color come common cool cost could '
+ 'country cover cross cut dark day dead deal deep did die different dinner do dog '
+ 'done door down draw drive drop dry during each early earth eat eight end enough '
+ 'even ever every example eye face fact fall family far fast feel feet fell felt '
+ 'few field fight fill find fire first fish five floor fly follow food for force '
+ 'form found four free friend from front full fun game get girl give glad go god '
+ 'gold good got great green grew ground grow had hand happen hard have he head '
+ 'hear heart heavy help her here high him his hold home hope horse hot hour house '
+ 'how human hundred idea if in its keep kind king know land large last late learn '
+ 'leave left less letter life light like line little live long look low made make '
+ 'man many may mean meet men mind miss more morning most mother move much must my '
+ 'name near need never new next night nine no north not now number of off old on '
+ 'once open or other our out over own page paper park part pass past peace people '
+ 'perhaps place plan plant play point poor put question quick rain read real red '
+ 'rest right rise river road rock room round run said same saw say school sea see '
+ 'seem seen self send set seven she show side since six sleep small so some son '
+ 'song soon south space speak stand start state stay step still stop story strong '
+ 'such summer sun sure take talk tell ten than thank that the their them then '
+ 'there these thing think this those though three through time today together told '
+ 'too took top town tree true try turn two under until up use very walk want war '
+ 'warm was watch water way we week well went were west what when where while white '
+ 'who why wide will wind with without woman women wood word work world would write '
+ 'year yet you young '
+ 'hello bye yes sorry please thank welcome great nice love miss call home feel free hi hey good morning evening afternoon night').split(' ').filter(Boolean);

// Build prefix index for fast lookup
const WORD_PREFIX_MAP = {};
WORD_DICT.forEach(word => {
  for (let i = 1; i <= word.length; i++) {
    const prefix = word.slice(0, i).toUpperCase();
    if (!WORD_PREFIX_MAP[prefix]) WORD_PREFIX_MAP[prefix] = [];
    WORD_PREFIX_MAP[prefix].push(word);
  }
});

const LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const NUM_LANDMARKS = 21;
const BASE_DIM = NUM_LANDMARKS * 3;

const FINGERTIP_IDS = [4, 8, 12, 16, 20];
const ANGLE_TRIPLETS = [
  [2, 3, 4], [5, 6, 7], [0, 5, 6],
  [9, 10, 11], [0, 9, 10], [13, 14, 15],
  [0, 13, 14], [17, 18, 19], [0, 17, 18]
];
const CURL_PAIRS = [[4, 2], [8, 5], [12, 9], [16, 13], [20, 17]];
const ENHANCED_DIM = BASE_DIM + 10 + 9 + 5 + 3 + 4 + 4 + 2; // 100
// Cooldown between letter commits.
// 30 frames at ~30 fps = ~1.0 second between commits (balanced for sentence building).
const COMMIT_COOLDOWN = 30;
const MIN_SAMPLES_PER_CLASS = 5;

let model = null;
let predictionBuffer = [];
let lastCommittedLetter = '';
let cooldownCounter = 0;
let translatedText = '';
let currentMode = 'recognize';

// Word Builder state
let currentWordLetters = '';   // letters being built for the current word
let completedSentence = '';    // full sentence of confirmed words
let collectedData = {};
let currentLandmarks = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let fpsValue = 0;
let lastEntropy = 0.5;
let useEnhanced = true;
let scalerMean = null;
let scalerStd = null;

// Motion tracking for J and Z detection
const MOTION_BUFFER_SIZE = 18;
const MOTION_THRESHOLD = 0.012;
let motionBuffer = [];

// Auto-space when hand is removed (faster = more responsive sentence building)
const AUTO_SPACE_FRAMES = 25; // ~0.8s at 30fps
let handAbsentFrames = 0;

// Confusion detection for smart calibration prompt
let recentPredictions = [];
let calibrationModel = null;
const CONFUSION_WINDOW = 60;

// Session stats
let sessionLetterCount = 0;
let sessionWordCount = 0;
let sessionStartTime = Date.now();

function updateSessionStats() {
  const lettersEl = document.getElementById('stat-letters');
  const wordsEl = document.getElementById('stat-words');
  const timeEl = document.getElementById('stat-time');
  if (lettersEl) lettersEl.textContent = sessionLetterCount;
  if (wordsEl) wordsEl.textContent = sessionWordCount;
  if (timeEl) {
    const mins = Math.floor((Date.now() - sessionStartTime) / 60000);
    timeEl.textContent = mins + 'm';
  }
}
setInterval(updateSessionStats, 10000); // update time every 10s

// Practice Mode state
let practiceTarget = 'A';
let practiceCorrectFrames = 0;
let practiceScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
let practiceLetterStats = {};
let practiceDrillMode = 'sequential';
let practiceDrillIndex = 0;
let practiceShowingResult = false;
const PRACTICE_HOLD_REQUIRED = 15;

const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('overlay');
const ctx = canvasEl.getContext('2d');
const letterEl = document.getElementById('predicted-letter');
const confidenceEl = document.getElementById('confidence');
const textEl = document.getElementById('text-content');
const handStatusEl = document.getElementById('hand-status');
const modelStatusEl = document.getElementById('model-status');
const fpsEl = document.getElementById('fps-display');
const latencyEl = document.getElementById('latency-display');
const letterSelect = document.getElementById('letter-select');
const trainBtn = document.getElementById('btn-train');
const confidenceBarEl = document.getElementById('confidence-bar');
const confidenceLevelEl = document.getElementById('confidence-level');

LABELS.forEach(l => {
  const opt = document.createElement('option');
  opt.value = l; opt.textContent = l;
  letterSelect.appendChild(opt);
});

function updateCollectProgress() {
  const container = document.getElementById('collect-progress');
  container.innerHTML = '';
  let totalReady = 0;
  LABELS.forEach(l => {
    const count = (collectedData[l] || []).length;
    const div = document.createElement('div');
    div.className = 'letter-count' +
      (count >= MIN_SAMPLES_PER_CLASS ? ' enough' : count > 0 ? ' has-data' : '');
    div.textContent = `${l}:${count}`;
    container.appendChild(div);
    if (count >= MIN_SAMPLES_PER_CLASS) totalReady++;
  });
  trainBtn.disabled = totalReady < 3;
}
updateCollectProgress();

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const raw = [];
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    raw.push(landmarks[i].x - wrist.x);
    raw.push(landmarks[i].y - wrist.y);
    raw.push(landmarks[i].z - wrist.z);
  }
  const mx = raw[9 * 3], my = raw[9 * 3 + 1], mz = raw[9 * 3 + 2];
  const scale = Math.sqrt(mx * mx + my * my + mz * mz) || 1;
  return raw.map(v => v / scale);
}

function computeEnhancedFeatures(base) {
  const pts = [];
  for (let i = 0; i < 21; i++)
    pts.push([base[i * 3], base[i * 3 + 1], base[i * 3 + 2]]);

  const d3 = (a, b) => {
    const dx = pts[a][0] - pts[b][0], dy = pts[a][1] - pts[b][1], dz = pts[a][2] - pts[b][2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  const dists = [];
  for (let i = 0; i < FINGERTIP_IDS.length; i++)
    for (let j = i + 1; j < FINGERTIP_IDS.length; j++)
      dists.push(d3(FINGERTIP_IDS[i], FINGERTIP_IDS[j]));

  const angles = ANGLE_TRIPLETS.map(([a, b, c]) => {
    const ba = [pts[a][0] - pts[b][0], pts[a][1] - pts[b][1], pts[a][2] - pts[b][2]];
    const bc = [pts[c][0] - pts[b][0], pts[c][1] - pts[b][1], pts[c][2] - pts[b][2]];
    const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    const mA = Math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2) || 1e-8;
    const mC = Math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2) || 1e-8;
    return Math.acos(Math.max(-1, Math.min(1, dot / (mA * mC)))) / Math.PI;
  });

  const curls = CURL_PAIRS.map(([tip, mcp]) => {
    const td = Math.sqrt(pts[tip][0] ** 2 + pts[tip][1] ** 2 + pts[tip][2] ** 2);
    const md = Math.sqrt(pts[mcp][0] ** 2 + pts[mcp][1] ** 2 + pts[mcp][2] ** 2) || 1e-8;
    return td / md;
  });

  const v1 = pts[5], v2 = pts[17];
  const nx = v1[1] * v2[2] - v1[2] * v2[1];
  const ny = v1[2] * v2[0] - v1[0] * v2[2];
  const nz = v1[0] * v2[1] - v1[1] * v2[0];
  const nmag = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1e-8;
  const palmNormal = [nx / nmag, ny / nmag, nz / nmag];

  const thumbDists = [8, 12, 16, 20].map(tip => d3(4, tip));

  const spreadPairs = [[4, 2, 8, 5], [8, 5, 12, 9], [12, 9, 16, 13], [16, 13, 20, 17]];
  const spreads = spreadPairs.map(([t1, b1, t2, b2]) => {
    const da = [pts[t1][0] - pts[b1][0], pts[t1][1] - pts[b1][1], pts[t1][2] - pts[b1][2]];
    const db = [pts[t2][0] - pts[b2][0], pts[t2][1] - pts[b2][1], pts[t2][2] - pts[b2][2]];
    const dot = da[0] * db[0] + da[1] * db[1] + da[2] * db[2];
    const m1 = Math.sqrt(da[0] ** 2 + da[1] ** 2 + da[2] ** 2) || 1e-8;
    const m2 = Math.sqrt(db[0] ** 2 + db[1] ** 2 + db[2] ** 2) || 1e-8;
    return Math.acos(Math.max(-1, Math.min(1, dot / (m1 * m2)))) / Math.PI;
  });

  const hx = pts[12][0], hy = pts[12][1];
  const hmag = Math.sqrt(hx * hx + hy * hy) || 1e-8;
  const handDir = [hx / hmag, hy / hmag];

  return [...base, ...dists, ...angles, ...curls,
          ...palmNormal, ...thumbDists, ...spreads, ...handDir];
}

function computeEntropy(probs) {
  let e = 0;
  for (let i = 0; i < probs.length; i++)
    if (probs[i] > 1e-10) e -= probs[i] * Math.log(probs[i]);
  return e / Math.log(probs.length);
}

function applyScaler(features) {
  if (!scalerMean || !scalerStd) return features;
  return features.map((v, i) => (v - scalerMean[i]) / (scalerStd[i] || 1));
}

function trackConfusion(label) {
  recentPredictions.push(label);
  if (recentPredictions.length > CONFUSION_WINDOW)
    recentPredictions.shift();

  if (recentPredictions.length < 30) return;

  const counts = {};
  recentPredictions.forEach(l => counts[l] = (counts[l] || 0) + 1);
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  if (sorted.length >= 2) {
    const [l1, c1] = sorted[0];
    const [l2, c2] = sorted[1];
    if (c1 > 8 && c2 > 8 && Math.abs(c1 - c2) < 10) {
      const prompt = document.getElementById('calibration-prompt');
      if (prompt) {
        prompt.textContent = `Tip: ${l1} and ${l2} look similar. Use Calibrate tab to record samples for both.`;
        prompt.style.display = 'block';
        setTimeout(() => { prompt.style.display = 'none'; }, 8000);
        recentPredictions = [];
      }
    }
  }
}

function blendPrediction(baseProbs, features) {
  if (!calibrationModel) return baseProbs;
  const calInput = tf.tensor2d([features]);
  const calPred = calibrationModel.predict(calInput);
  const calProbs = calPred.dataSync();
  calInput.dispose(); calPred.dispose();

  return baseProbs.map((p, i) => 0.7 * p + 0.3 * (calProbs[i] || 0));
}

function resolveConfusablePair(probs, base) {
  const pts = [];
  for (let i = 0; i < 21; i++)
    pts.push([base[i * 3], base[i * 3 + 1], base[i * 3 + 2]]);

  const sorted = Array.from(probs).map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const top1 = sorted[0], top2 = sorted[1];
  if (top1[0] - top2[0] > 0.15) return LABELS[top1[1]];

  const l1 = LABELS[top1[1]], l2 = LABELS[top2[1]];
  const pair = [l1, l2].sort().join('');

  const d3 = (a, b) => {
    const dx = pts[a][0] - pts[b][0], dy = pts[a][1] - pts[b][1], dz = pts[a][2] - pts[b][2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };

  if (pair === 'GP') {
    const v1 = pts[5], v2 = pts[17];
    const ny = v1[2] * v2[0] - v1[0] * v2[2];
    return ny < 0 ? 'P' : 'G';
  }
  if (pair === 'MN') {
    const ringCurl = d3(16, 0) / (d3(13, 0) || 1e-8);
    return ringCurl < 1.2 ? 'M' : 'N';
  }
  if (pair === 'DF') {
    const indexExt = pts[8][1];
    return indexExt < pts[5][1] ? 'D' : 'F';
  }
  if (pair === 'UV') {
    const spread = d3(8, 12);
    return spread < 0.15 ? 'U' : 'V';
  }
  if (pair === 'AS' || pair === 'AT' || pair === 'ST') {
    const thumbX = pts[4][0];
    const thumbY = pts[4][1];
    const indexMcpY = pts[5][1];
    if (Math.abs(thumbX) > 0.3) return 'A';
    if (thumbY < indexMcpY) return 'S';
    return 'T';
  }

  return LABELS[top1[1]];
}

function getMotionAmount(landmarks) {
  const wrist = landmarks[0];
  motionBuffer.push({ x: wrist.x, y: wrist.y });
  if (motionBuffer.length > MOTION_BUFFER_SIZE)
    motionBuffer.shift();
  if (motionBuffer.length < 8) return 0;

  let totalDist = 0;
  for (let i = 1; i < motionBuffer.length; i++) {
    const dx = motionBuffer[i].x - motionBuffer[i - 1].x;
    const dy = motionBuffer[i].y - motionBuffer[i - 1].y;
    totalDist += Math.sqrt(dx * dx + dy * dy);
  }
  return totalDist / (motionBuffer.length - 1);
}

function getAdaptiveParams(entropy) {
  // Balanced hold times: responsive enough for sentences, stable enough to avoid jitter.
  // Low entropy (confident) = faster commit. High entropy = slower, more careful.
  if (entropy < 0.3) return { window: 12, hold: 8, confThresh: 0.55 };
  if (entropy < 0.6) return { window: 16, hold: 12, confThresh: 0.65 };
  return { window: 22, hold: 16, confThresh: 0.75 };
}

function smoothPrediction(label, conf, entropy) {
  predictionBuffer.push({ label, conf });
  lastEntropy = entropy;
  const params = getAdaptiveParams(entropy);

  while (predictionBuffer.length > params.window)
    predictionBuffer.shift();

  const votes = {};
  const totalConf = {};
  predictionBuffer.forEach(p => {
    votes[p.label] = (votes[p.label] || 0) + 1;
    totalConf[p.label] = (totalConf[p.label] || 0) + p.conf;
  });

  let bestLabel = '', bestCount = 0;
  for (const [l, c] of Object.entries(votes))
    if (c > bestCount) { bestCount = c; bestLabel = l; }

  const avgConf = totalConf[bestLabel] / bestCount;
  return {
    label: bestLabel, count: bestCount, confidence: avgConf,
    holdNeeded: params.hold, confThresh: params.confThresh
  };
}

function updateConfidenceDisplay(conf, entropy) {
  let level, cls;
  if (conf >= 0.80 && entropy < 0.4) {
    level = 'High'; cls = 'conf-high';
  } else if (conf >= 0.55 && entropy < 0.7) {
    level = 'Medium'; cls = 'conf-med';
  } else {
    level = 'Low'; cls = 'conf-low';
  }

  confidenceEl.textContent = `Confidence: ${(conf * 100).toFixed(0)}%`;
  if (confidenceBarEl) {
    confidenceBarEl.style.width = `${conf * 100}%`;
    confidenceBarEl.className = `confidence-bar ${cls}`;
  }
  if (confidenceLevelEl) {
    confidenceLevelEl.textContent = level;
    confidenceLevelEl.className = `confidence-level ${cls}`;
  }
  letterEl.className = `predicted-letter ${cls}`;
}

function onResults(results) {
  const t0 = performance.now();

  canvasEl.width = canvasEl.clientWidth;
  canvasEl.height = canvasEl.clientHeight;
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsValue = frameCount; frameCount = 0; lastFpsTime = now;
    fpsEl.textContent = `FPS: ${fpsValue}`;
  }

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    handStatusEl.textContent = 'Hand detected';
    handStatusEl.classList.add('detected');
    canvasEl.closest('.video-container').classList.add('hand-active');
    currentLandmarks = landmarks;

    handAbsentFrames = 0;

    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#6366f180', lineWidth: 2 });
    drawLandmarks(ctx, landmarks, { color: '#a78bfa', lineWidth: 1, radius: 3 });

    if (cooldownCounter > 0) cooldownCounter--;

    if (model && (currentMode === 'recognize' || currentMode === 'practice')) {
      const base = normalizeLandmarks(landmarks);
      const rawFeatures = useEnhanced ? computeEnhancedFeatures(base) : base;
      const features = applyScaler(rawFeatures);
      const input = tf.tensor2d([features]);
      const pred = model.predict(input);
      const probs = pred.dataSync();
      input.dispose(); pred.dispose();

      const blended = blendPrediction(Array.from(probs), features);
      let rawLabel = resolveConfusablePair(blended, base);

      let maxIdx = LABELS.indexOf(rawLabel);
      if (maxIdx < 0) { maxIdx = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i; rawLabel = LABELS[maxIdx]; }
      const rawConf = probs[maxIdx];

      const motion = getMotionAmount(landmarks);
      if (motion > MOTION_THRESHOLD) {
        if (rawLabel === 'I') rawLabel = 'J';
        else if (rawLabel === 'D') rawLabel = 'Z';
      }
      const entropy = computeEntropy(Array.from(probs));
      const smoothed = smoothPrediction(rawLabel, rawConf, entropy);

      updateConfidenceDisplay(smoothed.confidence, entropy);
      trackConfusion(smoothed.label);

      if (smoothed.confidence >= smoothed.confThresh) {
        letterEl.textContent = smoothed.label;

        if (currentMode === 'practice') {
          handlePracticeDetection(smoothed.label, smoothed.confidence);
        } else if (smoothed.count >= smoothed.holdNeeded && cooldownCounter === 0) {
          // Add letter to word builder buffer
          currentWordLetters += smoothed.label;
          sessionLetterCount++;
          updateSessionStats();
          updateWordBuilderUI();
          lastCommittedLetter = smoothed.label;
          cooldownCounter = COMMIT_COOLDOWN;
          letterEl.classList.add('flash');
          setTimeout(() => letterEl.classList.remove('flash'), 200);
        }
      }

      latencyEl.textContent = `Latency: ${(performance.now() - t0).toFixed(1)}ms`;
    }
  } else {
    handAbsentFrames++;
    currentLandmarks = null;
    predictionBuffer = [];
    motionBuffer = [];
    canvasEl.closest('.video-container').classList.remove('hand-active');

    if (currentMode !== 'practice' && handAbsentFrames === AUTO_SPACE_FRAMES && currentWordLetters.length > 0) {
      // Commit current word to sentence when hand is removed
      commitCurrentWord();
      lastCommittedLetter = '';
      handStatusEl.textContent = 'Word confirmed! Show hand to keep signing';
      handStatusEl.classList.add('detected');
    } else if (handAbsentFrames > AUTO_SPACE_FRAMES) {
      handStatusEl.textContent = 'Drop hand = space | Show hand = sign';
      handStatusEl.classList.remove('detected');
    } else if (handAbsentFrames > 10 && currentWordLetters.length > 0) {
      const remaining = AUTO_SPACE_FRAMES - handAbsentFrames;
      const dots = '.'.repeat(Math.ceil(remaining / 10));
      handStatusEl.textContent = `Confirming word${dots}`;
      handStatusEl.classList.remove('detected');
    } else {
      handStatusEl.textContent = 'No hand detected';
      handStatusEl.classList.remove('detected');
    }

    letterEl.textContent = '-';
    updateConfidenceDisplay(0, 1);
  }
}

function recordSample() {
  if (!currentLandmarks) {
    alert('Hold your hand in view of the camera first!');
    return;
  }
  const letter = letterSelect.value;
  if (!collectedData[letter]) collectedData[letter] = [];
  const base = normalizeLandmarks(currentLandmarks);
  collectedData[letter].push(computeEnhancedFeatures(base));
  updateCollectProgress();

  const btn = document.getElementById('btn-record');
  btn.textContent = `Recorded! (${collectedData[letter].length})`;
  btn.style.background = '#22c55e';
  setTimeout(() => { btn.textContent = 'Record'; btn.style.background = ''; }, 500);
}

async function trainModel() {
  const btn = document.getElementById('btn-train');
  btn.textContent = 'Training...'; btn.disabled = true;
  modelStatusEl.textContent = 'Training model...';
  modelStatusEl.className = 'status-msg loading';
  await tf.nextFrame();

  const activeLabels = LABELS.filter(l =>
    (collectedData[l] || []).length >= MIN_SAMPLES_PER_CLASS);
  if (activeLabels.length < 3) {
    alert('Need at least 3 letters with 5+ samples each.');
    btn.textContent = 'Train Model'; btn.disabled = false;
    return;
  }

  const xs = [], ys = [];
  activeLabels.forEach(l => {
    collectedData[l].forEach(features => {
      xs.push(features);
      const oneHot = new Array(26).fill(0);
      oneHot[LABELS.indexOf(l)] = 1;
      ys.push(oneHot);
    });
  });

  const xTensor = tf.tensor2d(xs);
  const yTensor = tf.tensor2d(ys);

  if (model) model.dispose();
  model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [ENHANCED_DIM], units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 26, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  useEnhanced = true;

  await model.fit(xTensor, yTensor, {
    epochs: 80, batchSize: 16, shuffle: true, validationSplit: 0.15,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 10 === 0)
          btn.textContent = `Training... ${epoch}/80 (${(logs.acc * 100).toFixed(0)}%)`;
      }
    }
  });

  xTensor.dispose(); yTensor.dispose();

  if (model && scalerMean) {
    calibrationModel = tf.sequential();
    calibrationModel.add(tf.layers.dense({ inputShape: [ENHANCED_DIM], units: 128, activation: 'relu' }));
    calibrationModel.add(tf.layers.dense({ units: 26, activation: 'softmax' }));
    const cxs = tf.tensor2d(xs.map(f => applyScaler(f)));
    const cys = tf.tensor2d(ys);
    calibrationModel.compile({ optimizer: tf.train.adam(0.002), loss: 'categoricalCrossentropy' });
    await calibrationModel.fit(cxs, cys, { epochs: 40, batchSize: 8, shuffle: true });
    cxs.dispose(); cys.dispose();
  }

  btn.textContent = 'Train Model'; btn.disabled = false;
  modelStatusEl.textContent =
    `Model ready! Trained on ${activeLabels.length} letters, ${xs.length} samples (blended)`;
  modelStatusEl.className = 'status-msg ready';
  switchMode('recognize');
}

async function loadPretrainedModel() {
  try {
    // Check if opened as file:// — model fetch won't work
    if (window.location.protocol === 'file:') {
      modelStatusEl.innerHTML = 'ERROR: Open via HTTP server, not file://.<br>' +
        '<small>Run: python -m http.server 8000<br>Then open http://localhost:8000</small>';
      modelStatusEl.className = 'status-msg error';
      console.error('Cannot load model from file:// protocol. Use HTTP server.');
      return false;
    }

    modelStatusEl.textContent = 'Loading pre-trained model...';
    modelStatusEl.className = 'status-msg loading';
    console.log('[ASL] Loading model from ./model/model.json ...');
    model = await tf.loadLayersModel('./model/model.json');
    console.log('[ASL] Model loaded. Input shape:', model.inputs[0].shape);

    try {
      const resp = await fetch('./model/scaler.json');
      const data = await resp.json();
      scalerMean = data.mean;
      scalerStd = data.std;
      console.log('[ASL] Scaler loaded. Features:', scalerMean.length);
    } catch (e) {
      scalerMean = null;
      scalerStd = null;
      console.warn('[ASL] No scaler found, using raw features');
    }

    const inputDim = model.inputs[0].shape[1];
    if (inputDim === ENHANCED_DIM) {
      useEnhanced = true;
      const scalerStatus = scalerMean ? ' + scaler' : '';
      modelStatusEl.textContent = `Enhanced model loaded (100-D${scalerStatus}). Start signing!`;
    } else if (inputDim === 87) {
      useEnhanced = true;
      modelStatusEl.textContent = 'Legacy model loaded (87-D). Start signing!';
    } else {
      useEnhanced = false;
      modelStatusEl.textContent = 'Base model loaded (63-D). Start signing!';
    }
    modelStatusEl.className = 'status-msg ready';
    return true;
  } catch (e) {
    console.error('[ASL] Model load failed:', e);
    modelStatusEl.innerHTML = 'Model load failed: ' + e.message +
      '<br><small>Make sure you are running from an HTTP server.</small>';
    modelStatusEl.className = 'status-msg error';
    return false;
  }
}

function switchMode(mode) {
  currentMode = mode;
  document.getElementById('tab-recognize').classList.toggle('active', mode === 'recognize');
  document.getElementById('tab-collect').classList.toggle('active', mode === 'collect');
  document.getElementById('tab-practice').classList.toggle('active', mode === 'practice');
  document.getElementById('tab-learn').classList.toggle('active', mode === 'learn');
  document.getElementById('recognize-panel').classList.toggle('hidden', mode !== 'recognize');
  document.getElementById('collect-panel').classList.toggle('hidden', mode !== 'collect');
  document.getElementById('practice-panel').classList.toggle('hidden', mode !== 'practice');
  document.getElementById('learn-panel').classList.toggle('hidden', mode !== 'learn');
  if (mode === 'practice') {
    initPracticeGrid();
    nextPracticeTarget();
  }
  if (mode === 'learn') {
    initLearnGrid();
    selectLearnLetter(learnCurrentLetter);
  }
}

// === WORD BUILDER FUNCTIONS ===

function getSuggestions(prefix, max = 6) {
  if (!prefix || prefix.length === 0) return [];
  const matches = WORD_PREFIX_MAP[prefix.toUpperCase()] || [];
  // Sort: exact matches first, then by length (shorter = more common)
  return matches
    .sort((a, b) => {
      if (a.toUpperCase() === prefix.toUpperCase()) return -1;
      if (b.toUpperCase() === prefix.toUpperCase()) return 1;
      return a.length - b.length;
    })
    .slice(0, max);
}

function updateWordBuilderUI() {
  const wordEl = document.getElementById('current-word-letters');
  const sugPanel = document.getElementById('suggestions-panel');

  if (currentWordLetters.length === 0) {
    wordEl.textContent = '-';
    sugPanel.innerHTML = '';
    return;
  }

  wordEl.textContent = currentWordLetters;

  const suggestions = getSuggestions(currentWordLetters);
  sugPanel.innerHTML = '';
  suggestions.forEach((word, idx) => {
    const btn = document.createElement('button');
    btn.className = 'suggestion-btn' + (idx === 0 ? ' top-pick' : '');
    btn.textContent = word;
    btn.onclick = () => selectSuggestion(word);
    sugPanel.appendChild(btn);
  });

  // Also show the raw letters as an option if it's not in suggestions
  const rawLower = currentWordLetters.toLowerCase();
  const alreadyShown = suggestions.some(w => w.toLowerCase() === rawLower);
  if (!alreadyShown && suggestions.length > 0) {
    const rawBtn = document.createElement('button');
    rawBtn.className = 'suggestion-btn';
    rawBtn.textContent = currentWordLetters + ' (raw)';
    rawBtn.onclick = () => selectSuggestion(currentWordLetters);
    sugPanel.appendChild(rawBtn);
  }
}

function selectSuggestion(word) {
  // User picked a word — commit it to the sentence
  const toAdd = (completedSentence.length > 0 && !completedSentence.endsWith(' '))
    ? ' ' + word
    : word;
  completedSentence += toAdd;
  textEl.textContent = completedSentence;
  translatedText = completedSentence;
  currentWordLetters = '';
  updateWordBuilderUI();
}

// FIX 2: Compute Levenshtein edit distance for dictionary correction.
function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => i === 0 ? j : j === 0 ? i : 0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i-1] === b[j-1]
        ? dp[i-1][j-1]
        : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
  return dp[m][n];
}

// FIX 2: Find closest real dictionary word using edit distance.
// Only considers words starting with the same letter to prevent cross-letter corruption.
function findClosestWord(letters, maxDist = 1) {
  const lower = letters.toLowerCase();
  if (WORD_DICT.includes(lower)) return lower;
  const prefixMatches = WORD_PREFIX_MAP[letters.toUpperCase()] || [];
  if (prefixMatches.length > 0 && prefixMatches[0].toLowerCase() === lower) return prefixMatches[0];
  // Edit distance — only consider words starting with same letter
  const firstLetter = lower[0];
  let best = null, bestDist = maxDist + 1;
  for (const word of WORD_DICT) {
    if (word[0] !== firstLetter) continue;
    if (Math.abs(word.length - lower.length) > maxDist) continue;
    const d = levenshtein(lower, word);
    if (d < bestDist) { bestDist = d; best = word; }
  }
  return bestDist <= maxDist ? best : null;
}

function commitCurrentWord() {
  if (currentWordLetters.length === 0) return;
  const wordToAdd = currentWordLetters.toLowerCase();
  const toAdd = (completedSentence.length > 0 && !completedSentence.endsWith(' '))
    ? ' ' + wordToAdd : wordToAdd;
  completedSentence += toAdd;
  sessionWordCount++;
  updateSessionStats();
  // Flash the sentence display on word commit
  const sentenceEl = document.getElementById('translated-text');
  if (sentenceEl) {
    sentenceEl.classList.add('word-commit-flash');
    setTimeout(() => sentenceEl.classList.remove('word-commit-flash'), 350);
  }
  textEl.textContent = completedSentence;
  translatedText = completedSentence;
  currentWordLetters = '';
  updateWordBuilderUI();
}

function clearCurrentWord() {
  currentWordLetters = '';
  updateWordBuilderUI();
}

function clearText() {
  translatedText = '';
  completedSentence = '';
  currentWordLetters = '';
  textEl.textContent = '';
  lastCommittedLetter = '';
  updateWordBuilderUI();
}

function addSpace() {
  // Commit current word if any, then add space
  if (currentWordLetters.length > 0) commitCurrentWord();
  if (completedSentence.length > 0 && !completedSentence.endsWith(' ')) {
    completedSentence += ' ';
    textEl.textContent = completedSentence;
    translatedText = completedSentence;
  }
  lastCommittedLetter = '';
}

function deleteLast() {
  if (currentWordLetters.length > 0) {
    // Delete last letter from current word
    currentWordLetters = currentWordLetters.slice(0, -1);
    updateWordBuilderUI();
  } else {
    // Delete last character from sentence
    completedSentence = completedSentence.trimEnd();
    // Remove last word
    const words = completedSentence.split(' ');
    words.pop();
    completedSentence = words.join(' ');
    textEl.textContent = completedSentence;
    translatedText = completedSentence;
  }
  lastCommittedLetter = '';
}

// === PRACTICE MODE FUNCTIONS ===

LABELS.forEach(l => { practiceLetterStats[l] = { correct: 0, total: 0 }; });

function initPracticeGrid() {
  const grid = document.getElementById('practice-letter-grid');
  if (!grid) return;
  grid.innerHTML = '';
  LABELS.forEach(l => {
    const cell = document.createElement('div');
    cell.className = 'practice-grid-cell';
    const stats = practiceLetterStats[l];
    if (stats.total >= 3) cell.classList.add('good');
    else if (stats.total >= 1) cell.classList.add('ok');
    cell.textContent = l;
    if (stats.total > 0) cell.title = l + ': ' + stats.correct + '/' + stats.total;
    grid.appendChild(cell);
  });
}

function startDrill(mode) {
  practiceDrillMode = mode;
  practiceDrillIndex = 0;
  practiceScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
  updatePracticeScoreDisplay();
  nextPracticeTarget();
}

function nextPracticeTarget() {
  practiceShowingResult = false;
  practiceCorrectFrames = 0;

  if (practiceDrillMode === 'sequential') {
    practiceTarget = LABELS[practiceDrillIndex % LABELS.length];
    practiceDrillIndex++;
  } else if (practiceDrillMode === 'random') {
    practiceTarget = LABELS[Math.floor(Math.random() * LABELS.length)];
  } else if (practiceDrillMode === 'weak') {
    const weak = LABELS.filter(l => (practiceLetterStats[l].total || 0) < 3);
    practiceTarget = weak.length > 0
      ? weak[Math.floor(Math.random() * weak.length)]
      : LABELS[Math.floor(Math.random() * LABELS.length)];
  }

  const targetEl = document.getElementById('practice-target');
  if (targetEl) targetEl.textContent = practiceTarget;
  const resultEl = document.getElementById('practice-result');
  if (resultEl) { resultEl.textContent = ''; resultEl.className = 'practice-result'; }
  const qualityEl = document.getElementById('practice-quality');
  if (qualityEl) qualityEl.textContent = '';
}

function handlePracticeDetection(detectedLetter, confidence) {
  if (practiceShowingResult) return;

  const qualityEl = document.getElementById('practice-quality');
  const resultEl = document.getElementById('practice-result');

  // Quality feedback
  if (qualityEl) {
    if (confidence >= 0.90) {
      qualityEl.textContent = 'Excellent form!';
      qualityEl.className = 'practice-quality quality-high';
    } else if (confidence >= 0.70) {
      qualityEl.textContent = 'Good — hold steady';
      qualityEl.className = 'practice-quality quality-med';
    } else {
      qualityEl.textContent = 'Adjust your hand position';
      qualityEl.className = 'practice-quality quality-low';
    }
  }

  if (detectedLetter === practiceTarget) {
    practiceCorrectFrames++;
    if (practiceCorrectFrames >= PRACTICE_HOLD_REQUIRED) {
      // Correct!
      practiceShowingResult = true;
      practiceScore.correct++;
      practiceScore.total++;
      practiceScore.streak++;
      if (practiceScore.streak > practiceScore.bestStreak)
        practiceScore.bestStreak = practiceScore.streak;
      practiceLetterStats[practiceTarget].correct++;
      practiceLetterStats[practiceTarget].total++;

      if (resultEl) {
        resultEl.textContent = 'Correct!';
        resultEl.className = 'practice-result result-correct';
      }

      updatePracticeScoreDisplay();
      initPracticeGrid();
      setTimeout(function() { nextPracticeTarget(); }, 1200);
    }
  } else {
    practiceCorrectFrames = 0;
    if (resultEl) {
      const tip = getConfusionTip(practiceTarget, detectedLetter);
      resultEl.innerHTML = 'Detected: ' + detectedLetter + (tip ? '<br><small>' + tip + '</small>' : '');
      resultEl.className = 'practice-result result-wrong';
    }
  }
}

function getConfusionTip(target, detected) {
  var tips = {
    'GP': 'G faces side, P faces down',
    'MN': 'M = 3 fingers over thumb, N = 2',
    'DF': 'D = index up, F = circle with index+thumb',
    'UV': 'U = fingers together, V = fingers apart',
    'AS': 'A = thumb beside fist, S = thumb over fingers',
    'IJ': 'J = I with a hook motion',
  };
  var pair = [target, detected].sort().join('');
  return tips[pair] || '';
}

function updatePracticeScoreDisplay() {
  const scoreEl = document.getElementById('practice-score-display');
  const streakEl = document.getElementById('practice-streak-display');
  if (scoreEl) scoreEl.textContent = practiceScore.correct + ' / ' + practiceScore.total + ' correct';
  if (streakEl) streakEl.textContent = 'Streak: ' + practiceScore.streak;
}

function resetPracticeStats() {
  LABELS.forEach(l => { practiceLetterStats[l] = { correct: 0, total: 0 }; });
  practiceScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
  updatePracticeScoreDisplay();
  initPracticeGrid();
}

// === LEARN MODE ===
let learnCurrentLetter = 'A';
let learnLessonActive = false;
let learnLessonTimer = null;
let learnLessonIndex = 0;
let learnAnimFrame = null;
let learnCurrentPose = null;
let learnTargetPose = null;
let learnAnimProgress = 0;

const LEARN_HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

const LEARN_FINGERTIPS_SET = new Set([4, 8, 12, 16, 20]);

const ASL_POSES = {
  'A': [[110,250],[80,210],[65,185],[58,165],[52,150],[90,180],[88,150],[86,130],[84,115],[112,175],[110,148],[108,128],[106,115],[132,178],[130,152],[128,132],[126,118],[148,183],[147,158],[146,138],[145,125]],
  'B': [[110,250],[75,215],[60,195],[52,178],[48,162],[88,185],[86,145],[84,110],[82,80],[110,182],[110,142],[110,108],[110,78],[130,185],[130,145],[130,110],[130,80],[148,188],[148,150],[148,115],[148,85]],
  'C': [[110,250],[72,210],[58,188],[52,168],[50,152],[92,178],[85,148],[80,122],[76,100],[114,174],[108,144],[104,118],[100,96],[133,178],[128,148],[124,122],[120,100],[150,185],[146,155],[142,130],[138,110]],
  'D': [[110,250],[82,210],[72,195],[65,182],[60,172],[90,180],[86,142],[84,108],[82,80],[112,180],[114,158],[116,148],[112,142],[132,182],[134,162],[134,152],[130,145],[150,186],[150,165],[150,155],[148,148]],
  'E': [[110,250],[82,208],[70,195],[65,188],[68,182],[90,185],[88,162],[86,148],[88,140],[112,182],[110,158],[108,144],[110,136],[130,184],[128,160],[126,147],[128,138],[148,186],[146,163],[145,150],[147,142]],
  'F': [[110,250],[78,210],[68,195],[65,180],[70,168],[90,185],[84,165],[80,155],[76,165],[112,182],[110,145],[110,112],[110,82],[130,184],[130,147],[130,114],[130,84],[148,187],[148,150],[148,117],[148,87]],
  'G': [[110,250],[80,215],[65,200],[50,188],[38,178],[88,190],[70,178],[55,168],[42,160],[112,195],[110,185],[108,188],[107,195],[130,197],[128,188],[127,192],[126,198],[148,200],[146,192],[145,196],[144,202]],
  'H': [[110,250],[80,215],[65,200],[52,188],[42,178],[88,190],[70,178],[54,168],[40,158],[112,192],[94,180],[78,170],[64,162],[132,198],[130,192],[128,196],[127,202],[148,200],[146,194],[145,198],[144,204]],
  'I': [[110,250],[82,210],[72,196],[68,185],[70,175],[92,185],[90,162],[90,152],[92,145],[112,182],[110,158],[110,148],[112,140],[130,184],[128,162],[128,152],[130,144],[148,186],[147,158],[146,130],[145,105]],
  'J': [[110,250],[82,210],[72,196],[68,185],[70,175],[92,185],[90,162],[90,152],[92,145],[112,182],[110,158],[110,148],[112,140],[130,184],[128,162],[128,152],[130,144],[148,186],[147,155],[148,125],[155,100]],
  'K': [[110,250],[82,208],[72,190],[68,172],[72,155],[90,182],[86,145],[84,112],[82,82],[110,182],[108,145],[106,112],[104,82],[130,185],[130,165],[130,155],[130,148],[148,188],[147,168],[146,158],[146,152]],
  'L': [[110,250],[80,215],[62,202],[46,190],[34,180],[90,182],[86,145],[84,108],[82,78],[112,200],[112,185],[112,178],[112,172],[130,202],[130,188],[130,182],[130,176],[148,204],[148,190],[148,184],[148,178]],
  'M': [[110,250],[82,208],[78,198],[80,192],[84,198],[92,185],[90,162],[90,148],[92,140],[112,182],[110,158],[110,144],[112,136],[130,184],[128,162],[128,148],[130,140],[148,188],[147,168],[146,158],[148,152]],
  'N': [[110,250],[82,208],[78,198],[80,192],[84,198],[92,185],[90,162],[90,148],[92,140],[112,182],[110,158],[110,144],[112,136],[130,186],[128,165],[128,156],[130,150],[148,188],[147,168],[146,158],[148,152]],
  'O': [[110,250],[78,208],[66,190],[60,172],[62,155],[90,182],[84,158],[80,138],[76,120],[112,178],[108,154],[106,132],[104,118],[130,182],[126,158],[124,136],[122,122],[148,186],[145,162],[143,142],[142,128]],
  'P': [[110,250],[82,208],[72,198],[62,188],[54,178],[92,195],[88,210],[86,222],[84,235],[112,195],[110,210],[108,222],[106,232],[130,185],[128,165],[128,156],[130,150],[148,187],[147,168],[146,158],[148,152]],
  'Q': [[110,250],[82,205],[76,218],[72,228],[70,238],[92,195],[88,212],[86,225],[84,236],[112,182],[110,162],[110,152],[112,145],[130,184],[128,162],[128,152],[130,145],[148,186],[147,165],[146,155],[148,148]],
  'R': [[110,250],[80,215],[68,202],[62,188],[60,175],[90,185],[88,148],[92,112],[94,82],[112,185],[108,148],[104,112],[100,82],[130,186],[130,166],[130,156],[130,150],[148,188],[147,168],[146,158],[148,152]],
  'S': [[110,250],[82,210],[75,195],[72,185],[78,178],[92,185],[90,162],[90,150],[92,142],[112,182],[110,158],[110,148],[112,140],[130,184],[128,162],[128,150],[130,142],[148,186],[147,165],[146,155],[148,148]],
  'T': [[110,250],[82,208],[78,192],[82,178],[90,172],[92,185],[90,165],[90,155],[92,148],[112,183],[110,162],[110,152],[112,145],[130,185],[128,165],[128,155],[130,148],[148,187],[147,167],[146,157],[148,150]],
  'U': [[110,250],[78,215],[65,200],[58,188],[55,175],[92,185],[88,148],[86,112],[84,82],[112,185],[110,148],[108,112],[106,82],[130,188],[130,168],[130,158],[130,152],[148,190],[148,170],[148,160],[148,154]],
  'V': [[110,250],[78,215],[65,200],[58,188],[55,175],[92,185],[85,148],[80,112],[76,82],[112,185],[115,148],[116,112],[118,82],[130,188],[130,168],[130,158],[130,152],[148,190],[148,170],[148,160],[148,154]],
  'W': [[110,250],[78,215],[65,200],[58,188],[55,175],[88,185],[82,148],[78,112],[74,82],[110,183],[108,146],[106,110],[104,80],[130,185],[132,148],[134,112],[136,82],[150,192],[150,172],[150,162],[150,156]],
  'X': [[110,250],[78,215],[65,200],[58,188],[55,175],[90,185],[88,158],[88,140],[92,128],[112,188],[112,168],[112,158],[114,152],[130,190],[130,170],[130,160],[130,154],[148,192],[148,172],[148,162],[148,156]],
  'Y': [[110,250],[78,215],[60,205],[46,195],[36,185],[92,188],[90,168],[90,158],[92,152],[112,185],[110,165],[110,155],[112,148],[130,188],[128,168],[128,158],[130,152],[148,186],[147,160],[146,132],[145,108]],
  'Z': [[110,250],[78,215],[65,200],[58,188],[55,175],[90,185],[86,148],[84,115],[82,85],[112,192],[112,175],[112,168],[114,162],[130,194],[130,175],[130,168],[130,162],[148,196],[148,177],[148,170],[148,164]],
};

const ASL_DESCRIPTIONS = {
  'A': 'Make a fist. Thumb rests beside the index finger, pointing up.',
  'B': 'All four fingers extended straight up, close together. Thumb tucked across palm.',
  'C': 'Curve all fingers and thumb into a C-shape, like holding a cup.',
  'D': 'Index finger points up. Other fingers and thumb form a circle.',
  'E': 'All fingers bent/curled down. Thumb tucked under fingers.',
  'F': 'Index finger and thumb form a circle. Other three fingers point up.',
  'G': 'Index finger and thumb point sideways (like a gun pointing left).',
  'H': 'Index and middle fingers extended sideways, parallel, pointing left.',
  'I': 'Only the pinky finger extended up. Others form a fist.',
  'J': 'Like I, but trace a J-curve motion with your pinky.',
  'K': 'Index and middle fingers up in a V, thumb pointing up between them.',
  'L': 'Index finger points up, thumb points out \u2014 forming an L shape.',
  'M': 'Three fingers (index, middle, ring) fold over the tucked thumb.',
  'N': 'Index and middle fingers fold over the tucked thumb.',
  'O': 'All fingers and thumb curve to meet, forming an O shape.',
  'P': 'Like K but rotated \u2014 index points down, middle points forward.',
  'Q': 'Like G but pointing down \u2014 index and thumb point toward ground.',
  'R': 'Cross your index and middle fingers (like crossing fingers for luck).',
  'S': 'Make a fist. Thumb wraps over all fingers.',
  'T': 'Make a fist. Thumb pokes up between index and middle fingers.',
  'U': 'Index and middle fingers extended up and held together.',
  'V': 'Index and middle fingers extended up and spread apart (peace sign).',
  'W': 'Index, middle, and ring fingers extended up and spread apart.',
  'X': 'Index finger extended and bent/hooked like a hook.',
  'Y': 'Thumb and pinky extended out. Other fingers folded into palm.',
  'Z': 'Index finger extended. Trace the letter Z in the air.',
};

const LEARN_MNEMONICS = {
  'A': 'Think: a fist bumping "A" for Ace.',
  'B': 'B = Blocked fingers, like pressing a buzzer.',
  'C': 'C = Cup shape. Curl your hand like holding a cup.',
  'D': 'D = one Digit up, the rest form a circle.',
  'E': 'E = fingers curled like Eagle talons.',
  'F': 'F = circle with index + thumb, like an OK.',
  'G': 'G = Gun. Index + thumb point sideways.',
  'H': 'H = two fingers Horizontal, pointing left.',
  'I': 'I = only the Index... wait, Pinky! Just the pinky up.',
  'J': 'J = like I, but trace a J-curve motion.',
  'K': 'K = Knuckles: index + middle up in a V, thumb between.',
  'L': 'L = classic L shape with index + thumb.',
  'M': 'M = three fingers over thumb (M has 3 humps).',
  'N': 'N = two fingers over thumb (N has 2 humps).',
  'O': 'O = Oval. Tips touch to make an O.',
  'P': 'P = like K but rotated down, pointing to the floor.',
  'Q': 'Q = like G but pointing down.',
  'R': 'R = cross your fingers for luck.',
  'S': 'S = fist with thumb over all fingers (Stop sign fist).',
  'T': 'T = Thumb pokes between index + middle.',
  'U': 'U = two fingers Up, held together.',
  'V': 'V = Victory/peace sign, fingers spread.',
  'W': 'W = three fingers Wide, like a W shape.',
  'X': 'X = one finger hooked like an X-hook.',
  'Y': 'Y = thumb and pinky out (hang loose sign).',
  'Z': 'Z = index up, draw a Z shape in the air.',
};

const learnViewedLetters = new Set();

function practiceThisLetter() {
  const letter = learnCurrentLetter;
  switchMode('practice');
  // Set practice target to the letter being learned
  practiceDrillMode = 'sequential';
  practiceTarget = letter;
  practiceCorrectFrames = 0;
  practiceShowingResult = false;
  const targetEl = document.getElementById('practice-target');
  if (targetEl) targetEl.textContent = letter;
  const resultEl = document.getElementById('practice-result');
  if (resultEl) { resultEl.textContent = ''; resultEl.className = 'practice-result'; }
}

function initLearnGrid() {
  const grid = document.getElementById('learn-letter-grid');
  if (!grid) return;
  grid.innerHTML = '';
  LABELS.forEach(l => {
    const cell = document.createElement('div');
    cell.className = 'learn-grid-cell' + (l === learnCurrentLetter ? ' active' : '');
    cell.textContent = l;
    cell.onclick = () => selectLearnLetter(l);
    grid.appendChild(cell);
  });
}

function selectLearnLetter(letter) {
  const prev = learnCurrentLetter;
  learnCurrentLetter = letter;
  learnTargetPose = ASL_POSES[letter];
  learnCurrentPose = learnCurrentPose || ASL_POSES[prev];
  learnAnimProgress = 0;
  document.querySelectorAll('.learn-grid-cell').forEach((cell, i) => {
    cell.classList.toggle('active', LABELS[i] === letter);
  });
  learnViewedLetters.add(letter);
  // Update grid cell styles for viewed letters
  document.querySelectorAll('.learn-grid-cell').forEach((cell, i) => {
    cell.classList.toggle('viewed', learnViewedLetters.has(LABELS[i]));
  });
  const titleEl = document.getElementById('learn-letter-title');
  const descEl = document.getElementById('learn-description');
  const mnemonicEl = document.getElementById('learn-mnemonic');
  const progEl = document.getElementById('learn-progress');
  const photoEl = document.getElementById('learn-ref-photo');
  if (titleEl) titleEl.textContent = letter;
  if (descEl) descEl.textContent = ASL_DESCRIPTIONS[letter] || '';
  if (mnemonicEl) mnemonicEl.textContent = LEARN_MNEMONICS[letter] || '';
  if (progEl) progEl.textContent = 'Letter ' + (LABELS.indexOf(letter) + 1) + ' of 26  \u00b7  Seen: ' + learnViewedLetters.size + '/26';
  if (photoEl) {
    photoEl.classList.add('photo-fade');
    setTimeout(() => {
      photoEl.src = 'dataset/asl_alphabet_test/' + letter + '_test.jpg';
      photoEl.alt = 'ASL ' + letter;
      photoEl.classList.remove('photo-fade');
    }, 150);
  }
  animateLearnHand();
}

function animateLearnHand() {
  if (learnAnimFrame) cancelAnimationFrame(learnAnimFrame);
  const canvas = document.getElementById('learn-hand-canvas');
  if (!canvas) return;
  const ctx2 = canvas.getContext('2d');
  function lerp(a, b, t) { return a + (b - a) * t; }
  function easeInOut(t) { return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }
  function frame() {
    learnAnimProgress = Math.min(1, learnAnimProgress + 0.05);
    const t = easeInOut(learnAnimProgress);
    const from = learnCurrentPose || learnTargetPose;
    const to = learnTargetPose;
    const pts = to.map((p, i) => [
      lerp(from[i][0], p[0], t),
      lerp(from[i][1], p[1], t)
    ]);
    drawLearnHand(ctx2, canvas, pts);
    if (learnAnimProgress < 1) {
      learnAnimFrame = requestAnimationFrame(frame);
    } else {
      learnCurrentPose = learnTargetPose.map(p => [...p]);
    }
  }
  learnAnimFrame = requestAnimationFrame(frame);
}

// Finger colors: thumb, index, middle, ring, pinky
const FINGER_COLORS = ['#f59e0b', '#6ee7b7', '#60a5fa', '#f472b6', '#a78bfa'];
// Joint ranges per finger
const FINGER_JOINTS = [
  [1,2,3,4],    // thumb
  [5,6,7,8],    // index
  [9,10,11,12], // middle
  [13,14,15,16],// ring
  [17,18,19,20],// pinky
];
const FINGER_NAMES = ['Thumb','Index','Middle','Ring','Pinky'];
const FINGERTIP_IDS_LEARN = [4, 8, 12, 16, 20];

function drawLearnHand(ctx2, canvas, pts) {
  ctx2.clearRect(0, 0, canvas.width, canvas.height);
  // Subtle gradient background
  const grad = ctx2.createLinearGradient(0, 0, 0, canvas.height);
  grad.addColorStop(0, '#12122a');
  grad.addColorStop(1, '#0f0f1a');
  ctx2.fillStyle = grad;
  ctx2.fillRect(0, 0, canvas.width, canvas.height);

  // Palm area
  const palmPts = [pts[0], pts[5], pts[9], pts[13], pts[17]];
  ctx2.beginPath();
  ctx2.moveTo(palmPts[0][0], palmPts[0][1]);
  palmPts.forEach(p => ctx2.lineTo(p[0], p[1]));
  ctx2.closePath();
  ctx2.fillStyle = 'rgba(99,102,241,0.10)';
  ctx2.fill();
  ctx2.strokeStyle = 'rgba(99,102,241,0.35)';
  ctx2.lineWidth = 1;
  ctx2.stroke();

  // Palm cross-connections (knuckle bar)
  [[5,9],[9,13],[13,17]].forEach(([a, b]) => {
    ctx2.beginPath();
    ctx2.moveTo(pts[a][0], pts[a][1]);
    ctx2.lineTo(pts[b][0], pts[b][1]);
    ctx2.strokeStyle = 'rgba(99,102,241,0.3)';
    ctx2.lineWidth = 1.5;
    ctx2.stroke();
  });

  // Draw each finger's bones in its own color
  FINGER_JOINTS.forEach((joints, fi) => {
    const color = FINGER_COLORS[fi];
    ctx2.strokeStyle = color;
    ctx2.lineWidth = 2.8;
    ctx2.lineCap = 'round';
    // Wrist to MCP
    const mcp = joints[0];
    ctx2.beginPath();
    ctx2.moveTo(pts[0][0], pts[0][1]);
    ctx2.lineTo(pts[mcp][0], pts[mcp][1]);
    ctx2.globalAlpha = 0.45;
    ctx2.stroke();
    ctx2.globalAlpha = 1;
    // MCP to tip
    for (let k = 0; k < joints.length - 1; k++) {
      ctx2.beginPath();
      ctx2.moveTo(pts[joints[k]][0], pts[joints[k]][1]);
      ctx2.lineTo(pts[joints[k+1]][0], pts[joints[k+1]][1]);
      ctx2.stroke();
    }
  });

  // Joints
  pts.forEach((p, i) => {
    ctx2.beginPath();
    let color = '#a78bfa';
    let radius = 4;
    if (i === 0) { color = '#ffffff'; radius = 7; }
    else {
      const fi = FINGER_JOINTS.findIndex(j => j.includes(i));
      if (fi >= 0) {
        color = FINGER_COLORS[fi];
        radius = FINGERTIP_IDS_LEARN.includes(i) ? 6.5 : 4;
      }
    }
    ctx2.arc(p[0], p[1], radius, 0, Math.PI * 2);
    ctx2.fillStyle = color;
    ctx2.fill();
    ctx2.strokeStyle = '#0f0f1a';
    ctx2.lineWidth = 1.5;
    ctx2.stroke();
  });

  // Fingertip labels (small, near each fingertip)
  ctx2.font = 'bold 8px Segoe UI';
  ctx2.textAlign = 'center';
  FINGERTIP_IDS_LEARN.forEach((tipIdx, fi) => {
    const p = pts[tipIdx];
    const color = FINGER_COLORS[fi];
    ctx2.fillStyle = color;
    const name = FINGER_NAMES[fi];
    // Position label slightly above the fingertip
    const labelY = p[1] < 80 ? p[1] + 14 : p[1] - 10;
    ctx2.fillText(name, p[0], labelY);
  });
  ctx2.textAlign = 'left';

  // Letter badge top-left
  ctx2.fillStyle = 'rgba(99,102,241,0.75)';
  ctx2.fillRect(6, 6, 46, 20);
  ctx2.fillStyle = '#ffffff';
  ctx2.font = 'bold 12px Segoe UI';
  ctx2.fillText('ASL: ' + learnCurrentLetter, 10, 20);
}

function learnNextLetter() {
  const idx = LABELS.indexOf(learnCurrentLetter);
  selectLearnLetter(LABELS[(idx + 1) % 26]);
}

function learnPrevLetter() {
  const idx = LABELS.indexOf(learnCurrentLetter);
  selectLearnLetter(LABELS[(idx + 25) % 26]);
}

function toggleLessonMode() {
  learnLessonActive = !learnLessonActive;
  const btn = document.getElementById('btn-lesson-mode');
  if (learnLessonActive) {
    btn.textContent = 'Stop Lesson';
    btn.classList.add('active');
    learnLessonIndex = LABELS.indexOf(learnCurrentLetter);
    runLessonStep();
  } else {
    btn.textContent = 'Start Lesson';
    btn.classList.remove('active');
    if (learnLessonTimer) clearTimeout(learnLessonTimer);
  }
}

function runLessonStep() {
  if (!learnLessonActive) return;
  selectLearnLetter(LABELS[learnLessonIndex % 26]);
  learnLessonIndex++;
  learnLessonTimer = setTimeout(runLessonStep, 3000);
}

async function init() {
  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`
  });
  hands.setOptions({
    maxNumHands: 1, modelComplexity: 1,
    minDetectionConfidence: 0.7, minTrackingConfidence: 0.5
  });
  hands.onResults(onResults);

  const camera = new Camera(videoEl, {
    onFrame: async () => { await hands.send({ image: videoEl }); },
    width: 640, height: 480
  });
  await camera.start();
  await loadPretrainedModel();
}

init();
