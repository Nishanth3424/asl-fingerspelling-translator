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
let detectionSpeed = 'careful';

function setDetectionSpeed(speed) {
  detectionSpeed = speed;
  ['careful', 'normal'].forEach(s => {
    const btn = document.getElementById('speed-' + s);
    if (btn) btn.classList.toggle('active', s === speed);
  });
  const indicator = document.getElementById('speed-indicator');
  if (indicator) indicator.textContent = speed === 'careful' ? 'Careful' : 'Normal';
  predictionBuffer = [];
}

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
  if (detectionSpeed === 'careful') {
    return { window: 30, hold: 24, confThresh: 0.80 };
  }
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
    numHandAbsentFrames = 0;

    drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: '#6366f180', lineWidth: 2 });
    drawLandmarks(ctx, landmarks, { color: '#a78bfa', lineWidth: 1, radius: 3 });

    if (cooldownCounter > 0) cooldownCounter--;

    if (currentMode === 'numbers' && numModel) {
      handleNumbersDetection(landmarks);
    }

    if (model && (currentMode === 'recognize' || currentMode === 'practice' || currentMode === 'learn')) {
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

      // Commit progress bar for word builder
      if (currentMode === 'recognize') {
        const cpBar = document.getElementById('commit-progress-bar');
        const cpLabel = document.getElementById('commit-progress-label');
        if (cpBar) {
          if (smoothed.confidence >= smoothed.confThresh && cooldownCounter === 0) {
            const progress = Math.min(1, smoothed.count / smoothed.holdNeeded);
            cpBar.style.width = (progress * 100) + '%';
            cpBar.style.background = progress >= 1
              ? 'var(--success)'
              : 'linear-gradient(90deg, var(--teal), var(--teal-light))';
            if (cpLabel) cpLabel.textContent = progress >= 0.3 ? smoothed.label : '';
          } else {
            cpBar.style.width = cooldownCounter > 0 ? '100%' : '0%';
            cpBar.style.background = cooldownCounter > 0
              ? 'rgba(245,158,11,0.4)' : '';
            if (cpLabel) cpLabel.textContent = cooldownCounter > 0 ? 'Cooldown' : '';
          }
        }
      }

      if (smoothed.confidence >= smoothed.confThresh) {
        letterEl.textContent = smoothed.label;

        if (currentMode === 'practice') {
          handlePracticeDetection(smoothed.label, smoothed.confidence);
        } else if (currentMode === 'learn') {
          handleLearnDetection(smoothed.label, smoothed.confidence);
        } else if (smoothed.count >= smoothed.holdNeeded && cooldownCounter === 0) {
          currentWordLetters += smoothed.label;
          sessionLetterCount++;
          updateSessionStats();
          updateWordBuilderUI();
          lastCommittedLetter = smoothed.label;
          cooldownCounter = detectionSpeed === 'careful' ? 55 : COMMIT_COOLDOWN;
          letterEl.classList.add('flash');
          setTimeout(() => letterEl.classList.remove('flash'), 200);
        }
      } else if (currentMode === 'learn') {
        // Still update learn display even at low confidence
        handleLearnDetection(smoothed.label, smoothed.confidence);
      }

      latencyEl.textContent = `Latency: ${(performance.now() - t0).toFixed(1)}ms`;
    }
  } else {
    handAbsentFrames++;
    currentLandmarks = null;
    predictionBuffer = [];
    motionBuffer = [];
    canvasEl.closest('.video-container').classList.remove('hand-active');

    if (currentMode === 'numbers') {
      handleNumbersHandAbsent();
    }

    if (currentMode !== 'practice' && currentMode !== 'numbers' && handAbsentFrames === AUTO_SPACE_FRAMES && currentWordLetters.length > 0) {
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
    // Reset commit progress when hand absent
    const cpBar = document.getElementById('commit-progress-bar');
    const cpLabel = document.getElementById('commit-progress-label');
    if (cpBar) { cpBar.style.width = '0%'; }
    if (cpLabel) { cpLabel.textContent = ''; }
    if (currentMode === 'learn') {
      learnCorrectFrames = 0;
      // Cancel any pending lesson advance — hand must be present to advance
      if (learnLessonTimer) { clearTimeout(learnLessonTimer); learnLessonTimer = null; }
      clearLearnMatchDisplay();
      const detectedEl = document.getElementById('learn-cam-letter');
      if (detectedEl) detectedEl.textContent = '—';
    }
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

  // Nav buttons
  ['recognize','practice','learn','collect','numbers'].forEach(function(m) {
    var btn = document.getElementById('tab-' + m);
    if (btn) btn.classList.toggle('active', m === mode);
  });

  // Page panels (right column)
  ['recognize','practice','learn','collect','numbers'].forEach(function(m) {
    var panel = document.getElementById(m + '-panel');
    if (panel) panel.classList.toggle('hidden', m !== mode);
  });

  // Aux panels below video (left column)
  ['recognize','practice','learn','numbers'].forEach(function(m) {
    var aux = document.getElementById('aux-' + m);
    if (aux) aux.classList.toggle('hidden', m !== mode);
  });

  if (mode === 'practice') {
    initPracticeGrid();
    nextPracticeTarget();
  }
  if (mode === 'learn') {
    initLearnGrid();
    selectLearnLetter(learnCurrentLetter);
    learnCorrectFrames = 0;
    clearLearnMatchDisplay();
    switchLearnMode(currentLearnSubMode);
    updateLearnStatsUI();
    updateLearnRing();
  }
  if (mode === 'numbers') {
    numInitLearnGrid();
    numUpdateStatsUI();
    numUpdateRing();
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
let currentLearnSubMode = 'browse';
let learnStartTime = Date.now();

// Quiz state
let quizActive = false;
let quizType = 'sign';
let quizCurrentLetter = '';
let quizScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
let quizTimer = null;
let quizTimeLeft = 30;
let quizSpeedLetters = [];
let quizSpeedIdx = 0;

// Flashcard state
let fcDeck = [];
let fcIndex = 0;
let fcFlipped = false;
let fcCurrentFilter = 'all';

// Progress persisted in localStorage
let learnProgress = {};
function loadLearnProgress() {
  try {
    const saved = localStorage.getItem('asl_learn_progress');
    if (saved) learnProgress = JSON.parse(saved);
  } catch(e) {}
  LABELS.forEach(l => {
    if (!learnProgress[l]) learnProgress[l] = { level: 'new', attempts: 0, correct: 0 };
  });
}
function saveLearnProgress() {
  try { localStorage.setItem('asl_learn_progress', JSON.stringify(learnProgress)); } catch(e) {}
}
loadLearnProgress();

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

const ASL_REFERENCE_URLS = {};
LABELS.forEach(l => {
  ASL_REFERENCE_URLS[l] = 'https://commons.wikimedia.org/wiki/Special:FilePath/Sign_language_' + l + '.svg';
});

function setLearnRefImage(imgId, letter) {
  const img = document.getElementById(imgId);
  if (!img) return;
  const url = ASL_REFERENCE_URLS[letter];
  if (url) {
    img.src = url;
    img.alt = 'ASL sign for letter ' + letter;
    img.style.display = '';
    img.onerror = function() { this.style.opacity = '0.3'; };
    img.onload = function() { this.style.opacity = '1'; };
  }
}

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

const ASL_STEPS = {
  'A': ['Make a fist with all fingers curled in', 'Rest your thumb alongside the index finger', 'Thumb should point upward', 'Keep wrist straight, palm facing forward'],
  'B': ['Extend all four fingers straight up', 'Keep fingers pressed together', 'Fold thumb across the palm', 'Face palm outward'],
  'C': ['Curve all fingers together', 'Curve the thumb to match', 'Form a C-shape as if holding a cup', 'Fingers and thumb should not touch'],
  'D': ['Point index finger straight up', 'Touch thumb tip to middle finger tip', 'Ring and pinky also touch thumb', 'Keep the index clearly extended'],
  'E': ['Curl all four fingers down toward palm', 'Tuck thumb underneath the fingers', 'Fingertips rest against the palm', 'Hand forms a claw-like shape'],
  'F': ['Touch index finger tip to thumb tip', 'Form a small circle with index and thumb', 'Extend middle, ring, and pinky straight up', 'Spread the three raised fingers slightly'],
  'G': ['Point index finger sideways to the left', 'Extend thumb parallel to index finger', 'Other fingers curl into the palm', 'Hand is oriented sideways, not up'],
  'H': ['Extend index and middle fingers sideways', 'Point them to the left, held together', 'Thumb rests below the fingers', 'Other fingers curl into the palm'],
  'I': ['Make a fist with fingers curled', 'Extend only the pinky finger up', 'Thumb wraps over other fingers', 'Pinky should be straight and vertical'],
  'J': ['Start with an I hand shape (pinky up)', 'Trace a J-curve in the air with pinky', 'Scoop downward then curve back up', 'This is a motion-based letter'],
  'K': ['Extend index and middle fingers up in a V', 'Place thumb pointing up between them', 'Thumb tip touches middle finger', 'Ring and pinky curl into palm'],
  'L': ['Extend index finger straight up', 'Extend thumb straight out to the side', 'Form an L-shape with finger and thumb', 'Other three fingers curl into palm'],
  'M': ['Fold index, middle, and ring fingers over thumb', 'Thumb is tucked underneath all three', 'Pinky sits beside ring finger', 'Three fingers drape over the thumb'],
  'N': ['Fold index and middle fingers over thumb', 'Thumb is tucked under these two fingers', 'Ring and pinky curl beside them', 'Two fingers drape over the thumb'],
  'O': ['Curve all fingers forward', 'Touch fingertips to thumb tip', 'Form a round O shape', 'All five digits meet at their tips'],
  'P': ['Start like K (index+middle up, thumb between)', 'Rotate hand so fingers point downward', 'Index points toward the floor', 'Wrist bends to angle the hand down'],
  'Q': ['Start like G (index+thumb pointing sideways)', 'Rotate hand so fingers point downward', 'Index and thumb point at the ground', 'Other fingers remain curled in'],
  'R': ['Cross your index over middle finger', 'Like crossing fingers for luck', 'Other fingers curl into palm', 'Thumb wraps over ring finger'],
  'S': ['Make a tight fist', 'Wrap thumb over all four fingers', 'Thumb rests across the front of fingers', 'Different from A where thumb is beside'],
  'T': ['Make a fist', 'Poke thumb up between index and middle', 'Thumb tip peeks out between the two', 'Other fingers stay curled tight'],
  'U': ['Extend index and middle fingers up', 'Hold them pressed together', 'Other fingers curl into palm', 'Thumb wraps over ring finger'],
  'V': ['Extend index and middle fingers up', 'Spread them apart into a V shape', 'Like a peace sign', 'Other fingers curl in, thumb over them'],
  'W': ['Extend index, middle, and ring fingers up', 'Spread all three apart', 'Pinky curls in, thumb touches pinky', 'Three fingers form a W shape'],
  'X': ['Extend index finger and hook it', 'Bend it at the middle joint', 'Like a hook or beckoning gesture', 'Other fingers stay in a fist'],
  'Y': ['Extend thumb straight out', 'Extend pinky straight out', 'Curl middle three fingers into palm', 'Like the hang loose sign'],
  'Z': ['Extend index finger up', 'Draw a Z shape in the air', 'Move left, diagonal down, then right', 'This is a motion-based letter'],
};

const DIFFICULTY_MAP = {
  'A':'beginner','B':'beginner','C':'beginner','D':'intermediate','E':'intermediate',
  'F':'intermediate','G':'intermediate','H':'intermediate','I':'beginner','J':'advanced',
  'K':'intermediate','L':'beginner','M':'advanced','N':'advanced','O':'beginner',
  'P':'advanced','Q':'advanced','R':'intermediate','S':'beginner','T':'advanced',
  'U':'intermediate','V':'beginner','W':'beginner','X':'intermediate','Y':'beginner','Z':'advanced'
};

const CONFUSABLE_MAP = {
  'A':['S','T'],'S':['A','T'],'T':['A','S'],
  'G':['P','H'],'P':['G','Q'],'Q':['P','G'],
  'H':['G','U'],'M':['N'],'N':['M'],
  'D':['F','X'],'F':['D'],'U':['V','H'],'V':['U','K'],
  'I':['J','Y'],'J':['I'],'K':['V'],'X':['D'],
  'Y':['I'],
};

function updateLetterMastery(letter, wasCorrect) {
  const p = learnProgress[letter];
  p.attempts++;
  if (wasCorrect) p.correct++;
  const rate = p.attempts > 0 ? p.correct / p.attempts : 0;
  if (p.attempts >= 5 && rate >= 0.8) p.level = 'mastered';
  else if (p.attempts >= 2 && rate >= 0.5) p.level = 'practiced';
  else if (p.attempts >= 1) p.level = 'seen';
  saveLearnProgress();
  updateLearnStatsUI();
  updateLearnRing();
}

function updateLearnStatsUI() {
  let mastered = 0, totalAttempts = 0, totalCorrect = 0;
  LABELS.forEach(l => {
    const p = learnProgress[l];
    if (p.level === 'mastered') mastered++;
    totalAttempts += p.attempts;
    totalCorrect += p.correct;
  });
  const accEl = document.getElementById('ls-accuracy');
  const mastEl = document.getElementById('ls-mastered');
  const streakEl = document.getElementById('ls-streak');
  const timeEl = document.getElementById('ls-time');
  if (mastEl) mastEl.innerHTML = mastered + '<span class="learn-stat-unit">/26</span>';
  if (accEl) accEl.textContent = totalAttempts > 0 ? Math.round(totalCorrect / totalAttempts * 100) + '%' : '—';
  if (streakEl) streakEl.textContent = quizScore.bestStreak;
  if (timeEl) timeEl.textContent = Math.floor((Date.now() - learnStartTime) / 60000) + 'm';
}

function updateLearnRing() {
  let mastered = 0;
  LABELS.forEach(l => { if (learnProgress[l].level === 'mastered') mastered++; });
  const pct = Math.round(mastered / 26 * 100);
  const circumference = 213.6;
  const offset = circumference - (circumference * pct / 100);
  const ringEl = document.getElementById('learn-ring-fill');
  const pctEl = document.getElementById('learn-ring-pct');
  if (ringEl) ringEl.style.strokeDashoffset = offset;
  if (pctEl) pctEl.textContent = pct + '%';
}

function resetLearnProgress() {
  if (!confirm('Reset all learning progress?')) return;
  LABELS.forEach(l => { learnProgress[l] = { level: 'new', attempts: 0, correct: 0 }; });
  quizScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
  saveLearnProgress();
  initLearnGrid();
  updateLearnStatsUI();
  updateLearnRing();
}

function switchLearnMode(mode) {
  currentLearnSubMode = mode;
  ['browse','lesson','quiz','flashcard'].forEach(m => {
    const tab = document.getElementById('ltab-' + m);
    const view = document.getElementById('learn-view-' + m);
    if (tab) tab.classList.toggle('active', m === mode);
    if (view) view.style.display = (m === mode) ? '' : 'none';
  });
  if (mode === 'lesson') initLessonView();
  if (mode === 'quiz') initQuizView();
  if (mode === 'flashcard') initFlashcardView();
}

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
    const p = learnProgress[l];
    let cls = 'learn-grid-cell';
    if (l === learnCurrentLetter) cls += ' active';
    if (p && p.level !== 'new') cls += ' mastery-' + p.level;
    cell.className = cls;
    cell.textContent = l;
    cell.onclick = () => selectLearnLetter(l);
    grid.appendChild(cell);
  });
  updateLearnRing();
  updateLearnStatsUI();
}

function selectLearnLetter(letter) {
  const prev = learnCurrentLetter;
  learnCurrentLetter = letter;
  learnTargetPose = ASL_POSES[letter];
  learnCurrentPose = learnCurrentPose || ASL_POSES[prev];
  learnAnimProgress = 0;
  learnViewedLetters.add(letter);
  if (learnProgress[letter].level === 'new') {
    learnProgress[letter].level = 'seen';
    saveLearnProgress();
  }

  // Update grid
  document.querySelectorAll('.learn-grid-cell').forEach((cell, i) => {
    const l = LABELS[i];
    const p = learnProgress[l];
    let cls = 'learn-grid-cell';
    if (l === letter) cls += ' active';
    if (p && p.level !== 'new') cls += ' mastery-' + p.level;
    cell.className = cls;
  });

  // Update browse view
  const titleEl = document.getElementById('learn-letter-title');
  const mnemonicEl = document.getElementById('learn-mnemonic');
  const progEl = document.getElementById('learn-progress');
  const diffEl = document.getElementById('learn-diff-badge');
  const mastEl = document.getElementById('learn-mastery-badge');
  const stepsEl = document.getElementById('learn-step-list');
  const similarEl = document.getElementById('learn-similar');
  const tryitEl = document.getElementById('learn-tryit-msg');

  if (titleEl) titleEl.textContent = letter;
  if (mnemonicEl) mnemonicEl.textContent = LEARN_MNEMONICS[letter] || '';

  // Difficulty badge
  if (diffEl) {
    const diff = DIFFICULTY_MAP[letter] || 'beginner';
    diffEl.textContent = diff.charAt(0).toUpperCase() + diff.slice(1);
    diffEl.className = 'learn-badge diff' + (diff !== 'beginner' ? ' ' + diff : '');
  }

  // Mastery badge
  if (mastEl) {
    const p = learnProgress[letter];
    const lvl = p ? p.level : 'new';
    mastEl.textContent = lvl.charAt(0).toUpperCase() + lvl.slice(1);
    mastEl.className = 'learn-badge mast' + (lvl !== 'new' ? ' ' + lvl : '');
  }

  // Step-by-step instructions
  if (stepsEl) {
    stepsEl.innerHTML = '';
    const steps = ASL_STEPS[letter] || [ASL_DESCRIPTIONS[letter] || ''];
    steps.forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      stepsEl.appendChild(li);
    });
  }

  // Similar / confusable letters
  if (similarEl) {
    const confusables = CONFUSABLE_MAP[letter];
    if (confusables && confusables.length > 0) {
      similarEl.innerHTML = '<span class="learn-section-label">Watch Out</span> Often confused with: ' +
        confusables.map(c => '<span class="sim-tag">' + c + '</span>').join(' ');
    } else {
      similarEl.innerHTML = '';
    }
  }

  // Progress text
  if (progEl) {
    let seen = 0;
    LABELS.forEach(l => { if (learnProgress[l].level !== 'new') seen++; });
    progEl.textContent = 'Letter ' + (LABELS.indexOf(letter) + 1) + ' of 26  \u00b7  Seen: ' + seen + '/26';
  }

  // Reset try-it section
  if (tryitEl) {
    tryitEl.textContent = 'Show your hand to the camera and sign this letter';
    tryitEl.className = 'learn-tryit-msg';
  }
  const holdBar = document.getElementById('learn-hold-bar-browse');
  if (holdBar) holdBar.style.width = '0%';
  const holdPct = document.getElementById('learn-hold-pct');
  if (holdPct) holdPct.textContent = '0%';

  learnCorrectFrames = 0;
  setLearnRefImage('learn-ref-img', letter);
  updateLearnRing();
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

// === LESSON MODE ===
function initLessonView() {
  const letter = learnCurrentLetter;
  const targetEl = document.getElementById('lesson-target-letter');
  const descEl = document.getElementById('lesson-description');
  const stepsEl = document.getElementById('lesson-steps');
  if (targetEl) targetEl.textContent = letter;
  if (descEl) descEl.textContent = ASL_DESCRIPTIONS[letter] || '';
  if (stepsEl) {
    const steps = ASL_STEPS[letter] || [];
    stepsEl.innerHTML = steps.length > 0 ? '<ol>' + steps.map(s => '<li>' + s + '</li>').join('') + '</ol>' : '';
  }
  drawLearnHandOnCanvas('lesson-hand-canvas', letter);
  updateLessonProgress();
}

function updateLessonProgress() {
  const idx = learnLessonIndex;
  const labelEl = document.getElementById('lesson-step-label');
  const barEl = document.getElementById('lesson-bar-fill');
  if (labelEl) labelEl.textContent = 'Letter ' + (idx + 1) + ' of 26';
  if (barEl) barEl.style.width = ((idx / 26) * 100) + '%';
}

function toggleLessonMode() {
  learnLessonActive = !learnLessonActive;
  const btn = document.getElementById('btn-lesson-toggle');
  if (learnLessonActive) {
    btn.textContent = '\u25A0 Stop Lesson';
    btn.classList.add('stop');
    learnLessonIndex = LABELS.indexOf(learnCurrentLetter);
    learnCorrectFrames = 0;
    runLessonStep();
  } else {
    btn.textContent = '\u25B6 Start Lesson';
    btn.classList.remove('stop');
    if (learnLessonTimer) clearTimeout(learnLessonTimer);
    learnCorrectFrames = 0;
    clearLearnMatchDisplay();
    const fb = document.getElementById('lesson-feedback');
    if (fb) { fb.textContent = 'Lesson paused'; fb.className = 'lesson-feedback'; }
  }
}

function runLessonStep() {
  if (!learnLessonActive) return;
  if (learnLessonIndex >= 26) {
    learnLessonActive = false;
    const btn = document.getElementById('btn-lesson-toggle');
    if (btn) { btn.textContent = '\u25B6 Start Lesson'; btn.classList.remove('stop'); }
    const fb = document.getElementById('lesson-feedback');
    if (fb) { fb.textContent = '\u2728 Congratulations! You completed the alphabet!'; fb.className = 'lesson-feedback correct'; }
    return;
  }
  const letter = LABELS[learnLessonIndex % 26];
  selectLearnLetter(letter);
  const targetEl = document.getElementById('lesson-target-letter');
  const descEl = document.getElementById('lesson-description');
  const stepsEl = document.getElementById('lesson-steps');
  if (targetEl) targetEl.textContent = letter;
  if (descEl) descEl.textContent = ASL_DESCRIPTIONS[letter] || '';
  if (stepsEl) {
    const steps = ASL_STEPS[letter] || [];
    stepsEl.innerHTML = steps.length > 0 ? '<ol>' + steps.map(s => '<li>' + s + '</li>').join('') + '</ol>' : '';
  }
  drawLearnHandOnCanvas('lesson-hand-canvas', letter);
  updateLessonProgress();
  learnCorrectFrames = 0;
  const fb = document.getElementById('lesson-feedback');
  if (fb) { fb.textContent = 'Sign the letter ' + letter + ' to continue'; fb.className = 'lesson-feedback'; }
  const hb = document.getElementById('lesson-hold-bar');
  if (hb) hb.style.width = '0%';
}

function lessonSkip() {
  if (!learnLessonActive) {
    learnLessonActive = true;
    const btn = document.getElementById('btn-lesson-toggle');
    if (btn) { btn.textContent = '\u25A0 Stop Lesson'; btn.classList.add('stop'); }
    learnLessonIndex = LABELS.indexOf(learnCurrentLetter);
  }
  learnLessonIndex++;
  runLessonStep();
}

function lessonRestart() {
  learnLessonIndex = 0;
  learnLessonActive = true;
  const btn = document.getElementById('btn-lesson-toggle');
  if (btn) { btn.textContent = '\u25A0 Stop Lesson'; btn.classList.add('stop'); }
  runLessonStep();
}

function drawLearnHandOnCanvas(canvasId, letter) {
  const idMap = {
    'lesson-hand-canvas': 'lesson-ref-img',
    'quiz-hand-canvas': 'quiz-ref-img',
    'fc-hand-canvas': 'fc-ref-img',
    'learn-hand-canvas': 'learn-ref-img'
  };
  setLearnRefImage(idMap[canvasId] || canvasId, letter);
}

// === QUIZ MODE ===
function initQuizView() {
  const promptEl = document.getElementById('quiz-prompt');
  if (promptEl) promptEl.textContent = 'Choose a quiz type above to begin!';
  const optEl = document.getElementById('quiz-options');
  if (optEl) optEl.innerHTML = '';
  const fbEl = document.getElementById('quiz-feedback');
  if (fbEl) { fbEl.textContent = ''; fbEl.className = 'quiz-feedback'; }
  const resEl = document.getElementById('quiz-result');
  if (resEl) resEl.style.display = 'none';
}

function startQuiz(type) {
  quizType = type;
  quizActive = true;
  quizScore = { correct: 0, total: 0, streak: 0, bestStreak: quizScore.bestStreak || 0 };
  if (quizTimer) { clearInterval(quizTimer); quizTimer = null; }

  ['sign','identify','speed'].forEach(t => {
    const el = document.getElementById('qtype-' + t);
    if (el) el.classList.toggle('active', t === type);
  });

  const resEl = document.getElementById('quiz-result');
  if (resEl) resEl.style.display = 'none';
  const timerWrap = document.getElementById('quiz-timer-wrap');

  if (type === 'speed') {
    quizTimeLeft = 30;
    quizSpeedLetters = LABELS.slice().sort(() => Math.random() - 0.5);
    quizSpeedIdx = 0;
    if (timerWrap) timerWrap.style.display = '';
    quizTimer = setInterval(quizTimerTick, 1000);
    updateQuizTimerUI();
  } else {
    if (timerWrap) timerWrap.style.display = 'none';
  }

  updateQuizScoreUI();
  quizNext();
}

function quizNext() {
  if (!quizActive) return;
  const promptEl = document.getElementById('quiz-prompt');
  const optEl = document.getElementById('quiz-options');
  const fbEl = document.getElementById('quiz-feedback');
  const quizImgEl = document.getElementById('quiz-ref-img');
  const letterBig = document.getElementById('quiz-letter-big');

  if (fbEl) { fbEl.textContent = ''; fbEl.className = 'quiz-feedback'; }
  if (optEl) optEl.innerHTML = '';

  if (quizType === 'sign' || quizType === 'speed') {
    if (quizType === 'speed') {
      if (quizSpeedIdx >= quizSpeedLetters.length) quizSpeedLetters = LABELS.slice().sort(() => Math.random() - 0.5);
      quizCurrentLetter = quizSpeedLetters[quizSpeedIdx % quizSpeedLetters.length];
      quizSpeedIdx++;
    } else {
      quizCurrentLetter = LABELS[Math.floor(Math.random() * 26)];
    }
    if (promptEl) promptEl.textContent = 'Sign this letter:';
    if (letterBig) { letterBig.textContent = quizCurrentLetter; letterBig.style.display = ''; }
    if (quizImgEl) quizImgEl.style.display = 'none';
    learnCorrectFrames = 0;
  } else if (quizType === 'identify') {
    quizCurrentLetter = LABELS[Math.floor(Math.random() * 26)];
    if (promptEl) promptEl.textContent = 'What letter is this?';
    if (letterBig) letterBig.style.display = 'none';
    if (quizImgEl) {
      quizImgEl.style.display = '';
      setLearnRefImage('quiz-ref-img', quizCurrentLetter);
    }
    const options = generateQuizOptions(quizCurrentLetter, 4);
    if (optEl) {
      options.forEach(letter => {
        const btn = document.createElement('button');
        btn.className = 'quiz-opt-btn';
        btn.textContent = letter;
        btn.onclick = () => quizCheckIdentify(letter, btn);
        optEl.appendChild(btn);
      });
    }
  }
}

function generateQuizOptions(correct, count) {
  const opts = new Set([correct]);
  const confusables = CONFUSABLE_MAP[correct] || [];
  confusables.forEach(c => { if (opts.size < count) opts.add(c); });
  while (opts.size < count) opts.add(LABELS[Math.floor(Math.random() * 26)]);
  return Array.from(opts).sort(() => Math.random() - 0.5);
}

function quizCheckIdentify(letter, btnEl) {
  const optBtns = document.querySelectorAll('.quiz-opt-btn');
  optBtns.forEach(b => { b.disabled = true; });

  const correct = letter === quizCurrentLetter;
  quizScore.total++;
  if (correct) {
    quizScore.correct++;
    quizScore.streak++;
    if (quizScore.streak > quizScore.bestStreak) quizScore.bestStreak = quizScore.streak;
    btnEl.classList.add('correct');
    updateLetterMastery(quizCurrentLetter, true);
    showQuizFeedback(true, quizCurrentLetter);
  } else {
    quizScore.streak = 0;
    btnEl.classList.add('wrong');
    optBtns.forEach(b => { if (b.textContent === quizCurrentLetter) b.classList.add('correct'); });
    updateLetterMastery(quizCurrentLetter, false);
    showQuizFeedback(false, quizCurrentLetter);
  }
  updateQuizScoreUI();
  setTimeout(() => { if (quizActive) quizNext(); }, 1500);
}

function quizCheckSign() {
  quizScore.total++;
  quizScore.correct++;
  quizScore.streak++;
  if (quizScore.streak > quizScore.bestStreak) quizScore.bestStreak = quizScore.streak;
  updateLetterMastery(quizCurrentLetter, true);
  showQuizFeedback(true, quizCurrentLetter);
  updateQuizScoreUI();
  setTimeout(() => { if (quizActive) quizNext(); }, 1000);
}

function showQuizFeedback(correct, letter) {
  const fbEl = document.getElementById('quiz-feedback');
  if (!fbEl) return;
  if (correct) {
    fbEl.textContent = '\u2713 Correct! That\'s ' + letter;
    fbEl.className = 'quiz-feedback correct';
  } else {
    fbEl.textContent = '\u2717 The answer was ' + letter;
    fbEl.className = 'quiz-feedback wrong';
  }
}

function updateQuizScoreUI() {
  const scoreEl = document.getElementById('quiz-score');
  const streakEl = document.getElementById('quiz-streak');
  if (scoreEl) scoreEl.textContent = quizScore.correct + ' / ' + quizScore.total;
  if (streakEl) streakEl.textContent = '\uD83D\uDD25 ' + quizScore.streak;
}

function quizTimerTick() {
  quizTimeLeft--;
  updateQuizTimerUI();
  if (quizTimeLeft <= 0) {
    clearInterval(quizTimer);
    quizTimer = null;
    quizActive = false;
    showQuizResult();
  }
}

function updateQuizTimerUI() {
  const fillEl = document.getElementById('quiz-timer-fill');
  const textEl = document.getElementById('quiz-timer-text');
  if (fillEl) fillEl.style.width = ((quizTimeLeft / 30) * 100) + '%';
  if (textEl) textEl.textContent = quizTimeLeft + 's';
}

function showQuizResult() {
  const resEl = document.getElementById('quiz-result');
  if (!resEl) return;
  const rate = quizScore.total > 0 ? Math.round(quizScore.correct / quizScore.total * 100) : 0;
  resEl.innerHTML = '<h3>Quiz Complete!</h3>' +
    '<p class="qr-stat">Score: <strong>' + quizScore.correct + ' / ' + quizScore.total + '</strong> (' + rate + '%)</p>' +
    '<p class="qr-stat">Best Streak: <strong>' + quizScore.bestStreak + '</strong></p>' +
    '<button onclick="startQuiz(\'' + quizType + '\')">Play Again</button>';
  resEl.style.display = '';
}

// === FLASHCARD MODE ===
function initFlashcardView() {
  fcBuildDeck();
  fcIndex = 0;
  fcFlipped = false;
  fcRender();
}

function fcBuildDeck() {
  if (fcCurrentFilter === 'all') {
    fcDeck = LABELS.slice();
  } else if (fcCurrentFilter === 'new') {
    fcDeck = LABELS.filter(l => learnProgress[l].level === 'new' || learnProgress[l].level === 'seen');
  } else {
    fcDeck = LABELS.filter(l => learnProgress[l].level !== 'mastered');
  }
  if (fcDeck.length === 0) fcDeck = LABELS.slice();
  fcDeck.sort(() => Math.random() - 0.5);
}

function fcSetFilter(filter) {
  fcCurrentFilter = filter;
  ['all','new','weak'].forEach(f => {
    const el = document.getElementById('fc-f-' + f);
    if (el) el.classList.toggle('active', f === filter);
  });
  fcBuildDeck();
  fcIndex = 0;
  fcFlipped = false;
  fcRender();
}

function fcFlip() {
  fcFlipped = !fcFlipped;
  const card = document.getElementById('fc-card');
  if (card) card.classList.toggle('flipped', fcFlipped);
  if (fcFlipped) {
    const letter = fcDeck[fcIndex % fcDeck.length];
    setLearnRefImage('fc-ref-img', letter);
    const descEl = document.getElementById('fc-back-desc');
    if (descEl) descEl.textContent = ASL_DESCRIPTIONS[letter] || '';
  }
}

function fcNext() {
  fcIndex = (fcIndex + 1) % fcDeck.length;
  fcFlipped = false;
  const card = document.getElementById('fc-card');
  if (card) card.classList.remove('flipped');
  fcRender();
}

function fcPrev() {
  fcIndex = (fcIndex - 1 + fcDeck.length) % fcDeck.length;
  fcFlipped = false;
  const card = document.getElementById('fc-card');
  if (card) card.classList.remove('flipped');
  fcRender();
}

function fcKnow() {
  const letter = fcDeck[fcIndex % fcDeck.length];
  updateLetterMastery(letter, true);
  initLearnGrid();
  fcNext();
}

function fcDunno() {
  const letter = fcDeck[fcIndex % fcDeck.length];
  updateLetterMastery(letter, false);
  initLearnGrid();
  fcNext();
}

function fcRender() {
  if (fcDeck.length === 0) return;
  const letter = fcDeck[fcIndex % fcDeck.length];
  const frontEl = document.getElementById('fc-front-letter');
  const counterEl = document.getElementById('fc-counter');
  if (frontEl) frontEl.textContent = letter;
  if (counterEl) counterEl.textContent = (fcIndex + 1) + ' / ' + fcDeck.length;
}

// === LEARN MODE DETECTION ===
let learnCorrectFrames = 0;
const LEARN_HOLD_REQUIRED = 22;

function clearLearnMatchDisplay() {
  const matchEl = document.getElementById('learn-match-display');
  if (matchEl) { matchEl.className = 'learn-match-display'; matchEl.textContent = ''; }
  ['learn-hold-bar', 'learn-hold-bar-browse', 'lesson-hold-bar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.width = '0%';
  });
  const pctEl = document.getElementById('learn-hold-pct');
  if (pctEl) pctEl.textContent = '0%';
}

function handleLearnDetection(detectedLetter, confidence) {
  const matchEl = document.getElementById('learn-match-display');
  const detectedEl = document.getElementById('learn-cam-letter');

  if (detectedEl) detectedEl.textContent = detectedLetter;

  // Determine target based on sub-mode
  let targetLetter = learnCurrentLetter;
  if (currentLearnSubMode === 'quiz' && quizActive && (quizType === 'sign' || quizType === 'speed')) {
    targetLetter = quizCurrentLetter;
  }

  const isMatch = detectedLetter === targetLetter && confidence >= 0.60;

  if (isMatch) {
    learnCorrectFrames = Math.min(learnCorrectFrames + 1, LEARN_HOLD_REQUIRED);
  } else {
    learnCorrectFrames = Math.max(learnCorrectFrames - 1, 0);
  }

  const progress = learnCorrectFrames / LEARN_HOLD_REQUIRED;
  const pct = Math.round(progress * 100);

  // Update all hold bars
  ['learn-hold-bar', 'learn-hold-bar-browse', 'lesson-hold-bar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.width = pct + '%';
  });
  const pctEl = document.getElementById('learn-hold-pct');
  if (pctEl) pctEl.textContent = pct + '%';

  // Update match display
  if (matchEl) {
    if (learnCorrectFrames >= LEARN_HOLD_REQUIRED) {
      matchEl.className = 'learn-match-display match-correct';
      matchEl.textContent = '\u2713 Correct!';
    } else if (isMatch) {
      matchEl.className = 'learn-match-display match-holding';
      matchEl.textContent = 'Hold\u2026';
    } else if (detectedLetter && detectedLetter !== '-') {
      matchEl.className = 'learn-match-display match-wrong';
      matchEl.textContent = 'Keep trying';
    } else {
      matchEl.className = 'learn-match-display';
      matchEl.textContent = '';
    }
  }

  // Update browse try-it message
  if (currentLearnSubMode === 'browse') {
    const tryitEl = document.getElementById('learn-tryit-msg');
    if (tryitEl) {
      if (learnCorrectFrames >= LEARN_HOLD_REQUIRED) {
        tryitEl.textContent = '\u2713 Great job! You signed ' + targetLetter + ' correctly!';
        tryitEl.className = 'learn-tryit-msg success';
      } else if (isMatch) {
        tryitEl.textContent = 'Hold steady\u2026 ' + pct + '%';
        tryitEl.className = 'learn-tryit-msg holding';
      } else if (detectedLetter && detectedLetter !== '-') {
        tryitEl.textContent = 'Detected: ' + detectedLetter + ' \u2014 Try signing ' + targetLetter;
        tryitEl.className = 'learn-tryit-msg';
      }
    }
  }

  // Update lesson feedback
  if (currentLearnSubMode === 'lesson' && learnLessonActive) {
    const fb = document.getElementById('lesson-feedback');
    if (fb) {
      if (learnCorrectFrames >= LEARN_HOLD_REQUIRED) {
        fb.textContent = '\u2713 Correct!';
        fb.className = 'lesson-feedback correct';
      } else if (isMatch) {
        fb.textContent = 'Hold steady\u2026';
        fb.className = 'lesson-feedback holding';
      } else {
        fb.textContent = 'Sign the letter ' + targetLetter;
        fb.className = 'lesson-feedback';
      }
    }
  }

  // On successful hold
  if (learnCorrectFrames >= LEARN_HOLD_REQUIRED) {
    learnCorrectFrames = 0;
    updateLetterMastery(targetLetter, true);
    initLearnGrid();

    if (currentLearnSubMode === 'lesson' && learnLessonActive) {
      learnLessonIndex++;
      setTimeout(runLessonStep, 800);
    } else if (currentLearnSubMode === 'quiz' && quizActive && (quizType === 'sign' || quizType === 'speed')) {
      quizCheckSign();
    } else if (currentLearnSubMode === 'browse') {
      setTimeout(() => {
        const tryitEl = document.getElementById('learn-tryit-msg');
        if (tryitEl) {
          tryitEl.textContent = 'Show your hand to the camera and sign this letter';
          tryitEl.className = 'learn-tryit-msg';
        }
      }, 2000);
    }
  }
}

// Keyboard shortcuts for learn mode
document.addEventListener('keydown', function(e) {
  if (currentMode !== 'learn') return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowRight') { e.preventDefault(); learnNextLetter(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); learnPrevLetter(); }
  else if (e.key === ' ' && currentLearnSubMode === 'flashcard') { e.preventDefault(); fcFlip(); }
});



// ═══════════════════════════════════════════════════════════════
// NUMBERS MODE (0-9) — Separate model, parallel to A-Z
// ═══════════════════════════════════════════════════════════════

const NUM_LABELS = ['0','1','2','3','4','5','6','7','8','9','10'];
let numModel = null;
let numScalerMean = null;
let numScalerStd = null;
let numPredictionBuffer = [];
let numCurrentDigits = '';
let numCompletedText = '';
let numCooldownCounter = 0;
let numCorrectFrames = 0;
let numSessionDigitCount = 0;
let numSessionNumberCount = 0;
let numCurrentSubMode = 'recognize';
let numCurrentLearnSubMode = 'browse';
let numLearnCurrent = '0';
let numLessonActive = false;
let numLessonIndex = 0;
let numHandAbsentFrames = 0;

// Numbers practice state
let numPracticeTarget = '1';
let numPracticeCorrectFrames = 0;
let numPracticeScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
let numPracticeStats = {};
let numPracticeDrillMode = 'sequential';
let numPracticeDrillIndex = 0;
let numPracticeShowingResult = false;

// Numbers quiz state
let numQuizActive = false;
let numQuizType = 'sign';
let numQuizCurrent = '';
let numQuizScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
let numQuizTimer = null;
let numQuizTimeLeft = 30;

// Numbers flashcard state
let numFcDeck = [];
let numFcIndex = 0;
let numFcFlipped = false;
let numFcFilter = 'all';

// Numbers learn progress
let numLearnProgress = {};
function numLoadProgress() {
  try {
    const saved = localStorage.getItem('asl_num_learn_progress');
    if (saved) numLearnProgress = JSON.parse(saved);
  } catch(e) {}
  NUM_LABELS.forEach(d => {
    if (!numLearnProgress[d]) numLearnProgress[d] = { level: 'new', attempts: 0, correct: 0 };
  });
}
function numSaveProgress() {
  try { localStorage.setItem('asl_num_learn_progress', JSON.stringify(numLearnProgress)); } catch(e) {}
}
numLoadProgress();
NUM_LABELS.forEach(d => { numPracticeStats[d] = { correct: 0, total: 0 }; });

// Numbers descriptions and instructions
const NUM_DESCRIPTIONS = {
  '0': 'Form an O-shape with all fingers curving to meet the thumb.',
  '1': 'Index finger points up. Other fingers and thumb form a fist.',
  '2': 'Index and middle fingers point up, spread apart (like V/peace sign).',
  '3': 'Thumb, index, and middle fingers extended. Ring and pinky curled.',
  '4': 'All four fingers extended up and spread apart. Thumb tucked across palm.',
  '5': 'All five fingers extended up and spread apart.',
  '6': 'Thumb and pinky touch. Index, middle, and ring extended.',
  '7': 'Thumb and ring finger touch. Index, middle, and pinky extended.',
  '8': 'Thumb and middle finger touch. Index, ring, and pinky extended.',
  '9': 'Thumb and index finger touch. Middle, ring, and pinky extended.',
  '10': 'Make a fist with thumb pointing straight up (thumbs up) and shake.',
};

const NUM_STEPS = {
  '0': ['Curve all fingers toward the thumb', 'Touch fingertips to thumb tip', 'Form a round O/zero shape', 'Like making the letter O'],
  '1': ['Make a fist with all fingers', 'Extend only the index finger up', 'Keep thumb curled around other fingers', 'Index points straight up'],
  '2': ['Make a fist with all fingers', 'Extend index and middle fingers', 'Spread them apart in a V shape', 'Same as the letter V / peace sign'],
  '3': ['Extend thumb outward', 'Extend index and middle fingers up', 'Curl ring and pinky into palm', 'Thumb, index, middle all spread'],
  '4': ['Extend all four fingers straight up', 'Spread fingers apart', 'Tuck thumb across the palm', 'Same as the letter B but fingers spread'],
  '5': ['Extend all five fingers', 'Spread all fingers wide apart', 'Thumb extends to the side', 'Open hand, palm facing forward'],
  '6': ['Extend index, middle, and ring fingers up', 'Touch thumb tip to pinky tip', 'Three fingers stay up and spread', 'Looks like the letter W + thumb-pinky touch'],
  '7': ['Extend index, middle, and pinky fingers up', 'Touch thumb tip to ring finger tip', 'Three fingers remain extended', 'Ring finger folds to meet thumb'],
  '8': ['Extend index, ring, and pinky fingers up', 'Touch thumb tip to middle finger tip', 'Three fingers remain extended', 'Middle finger folds to meet thumb'],
  '9': ['Extend middle, ring, and pinky fingers up', 'Touch thumb tip to index finger tip', 'Three fingers remain extended', 'Index folds to meet thumb, like OK sign'],
  '10': ['Make a fist with all fingers curled in', 'Extend thumb straight upward', 'Shake or twist hand slightly', 'Like a thumbs-up gesture with a shake'],
};

const NUM_MNEMONICS = {
  '0': 'Zero = O shape, like a zero/circle.',
  '1': 'One = one finger up (index).',
  '2': 'Two = peace sign / V shape.',
  '3': 'Three = thumb + 2 fingers (like a 3-pronged fork).',
  '4': 'Four = four fingers up, thumb hidden.',
  '5': 'Five = full open hand, all fingers spread.',
  '6': 'Six = W-hand + thumb meets pinky (6th finger equivalent).',
  '7': 'Seven = thumb meets ring finger (4th finger = 7-4=3 up).',
  '8': 'Eight = thumb meets middle finger.',
  '9': 'Nine = thumb meets index (like OK sign).',
  '10': 'Ten = thumbs up + shake (like saying "great!" = 10/10).',
};

const NUM_DIFFICULTY = {
  '0':'beginner','1':'beginner','2':'beginner','3':'beginner','4':'beginner',
  '5':'beginner','6':'intermediate','7':'intermediate','8':'intermediate','9':'intermediate',
  '10':'intermediate'
};

const NUM_CONFUSABLE = {
  '0':['O'],'1':['D'],'2':['V'],'3':['6'],'4':['B'],
  '5':[],'6':['W','3'],'7':['8'],'8':['7'],'9':['F'],
  '10':['A','S']
};

// Reference images for numbers — inline SVG data URIs (never break, no network needed)
// Each SVG depicts the ASL hand shape for that number
function _numSvg(paths, vb) {
  return 'data:image/svg+xml,' + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="' + (vb||'0 0 200 200') + '" width="200" height="200">' +
    '<rect width="200" height="200" rx="16" fill="%231a2636"/>' +
    '<g transform="translate(100,105)" fill="none" stroke="%2300A896" stroke-width="4" stroke-linecap="round" stroke-linejoin="round">' +
    paths + '</g></svg>'
  );
}
// Palm = ellipse centered at 0,0. Fingers extend upward (neg y). Thumb extends left.
// ASL 0: O-shape (thumb+index touching in circle, others curled)
// ASL 1: Index up, others curled
// ASL 2: Index+middle up (V shape)
// ASL 3: Thumb+index+middle up
// ASL 4: Four fingers up, thumb across palm
// ASL 5: All five fingers spread
// ASL 6: Thumb+pinky touching, middle three up
// ASL 7: Thumb+ring touching, index+middle+pinky up
// ASL 8: Thumb+middle touching, index+ring+pinky up
// ASL 9: Thumb+index touching, middle+ring+pinky up
// ASL 10: Thumbs up (fist + thumb pointing up)
const NUM_REFERENCE_URLS = {
  '0': _numSvg(
    '<ellipse cx="0" cy="5" rx="30" ry="40" stroke-width="5"/>' +
    '<ellipse cx="0" cy="-30" rx="14" ry="14" stroke="%234DD0E1" stroke-width="3"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">0</text>'
  ),
  '1': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="0" y1="-17" x2="0" y2="-70"/>' +
    '<line x1="-10" y1="-10" x2="-25" y2="5"/><line x1="10" y1="-10" x2="25" y2="5"/><line x1="18" y1="-5" x2="30" y2="12"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">1</text>'
  ),
  '2': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-8" y1="-17" x2="-16" y2="-70"/><line x1="8" y1="-17" x2="16" y2="-70"/>' +
    '<line x1="-18" y1="-5" x2="-30" y2="5"/><line x1="18" y1="-5" x2="30" y2="12"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">2</text>'
  ),
  '3': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-8" y1="-17" x2="-12" y2="-70"/><line x1="8" y1="-17" x2="12" y2="-70"/>' +
    '<line x1="-30" y1="0" x2="-50" y2="-25"/>' +
    '<line x1="18" y1="-5" x2="30" y2="12"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">3</text>'
  ),
  '4': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-14" y1="-17" x2="-20" y2="-70"/><line x1="-4" y1="-17" x2="-6" y2="-72"/>' +
    '<line x1="6" y1="-17" x2="8" y2="-72"/><line x1="16" y1="-17" x2="22" y2="-70"/>' +
    '<path d="M-28,5 Q-35,-5 -25,10" stroke-width="3"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">4</text>'
  ),
  '5': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-16" y1="-17" x2="-26" y2="-68"/><line x1="-6" y1="-17" x2="-8" y2="-72"/>' +
    '<line x1="6" y1="-17" x2="8" y2="-72"/><line x1="16" y1="-17" x2="26" y2="-68"/>' +
    '<line x1="-30" y1="0" x2="-52" y2="-20"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">5</text>'
  ),
  '6': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-8" y1="-17" x2="-12" y2="-70"/><line x1="0" y1="-17" x2="0" y2="-72"/>' +
    '<line x1="8" y1="-17" x2="12" y2="-70"/>' +
    '<path d="M-28,2 Q-42,-15 -30,20" stroke="%234DD0E1" stroke-width="3"/>' +
    '<line x1="20" y1="-8" x2="28" y2="15"/><path d="M28,15 Q20,25 -28,18" stroke="%234DD0E1" stroke-width="3"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">6</text>'
  ),
  '7': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-12" y1="-17" x2="-18" y2="-70"/><line x1="-2" y1="-17" x2="-2" y2="-72"/>' +
    '<line x1="20" y1="-10" x2="28" y2="-65"/>' +
    '<path d="M-28,2 Q-42,-12 -28,18" stroke="%234DD0E1" stroke-width="3"/>' +
    '<line x1="10" y1="-12" x2="14" y2="8"/><path d="M14,8 Q8,18 -26,14" stroke="%234DD0E1" stroke-width="3"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">7</text>'
  ),
  '8': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-14" y1="-17" x2="-20" y2="-70"/>' +
    '<line x1="8" y1="-17" x2="12" y2="-70"/><line x1="20" y1="-10" x2="28" y2="-65"/>' +
    '<path d="M-28,2 Q-42,-12 -28,18" stroke="%234DD0E1" stroke-width="3"/>' +
    '<line x1="-2" y1="-14" x2="0" y2="5"/><path d="M0,5 Q-8,18 -26,14" stroke="%234DD0E1" stroke-width="3"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">8</text>'
  ),
  '9': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="32" stroke-width="5"/>' +
    '<line x1="-4" y1="-17" x2="-6" y2="-72"/><line x1="8" y1="-17" x2="12" y2="-70"/>' +
    '<line x1="20" y1="-10" x2="28" y2="-65"/>' +
    '<path d="M-28,2 Q-42,-15 -28,18" stroke="%234DD0E1" stroke-width="3"/>' +
    '<line x1="-14" y1="-14" x2="-12" y2="5"/><path d="M-12,5 Q-18,18 -28,14" stroke="%234DD0E1" stroke-width="3"/>' +
    '<text x="0" y="60" text-anchor="middle" fill="%238899AA" font-size="18" font-family="Calibri" stroke="none">9</text>'
  ),
  '10': _numSvg(
    '<ellipse cx="0" cy="15" rx="28" ry="35" stroke-width="5"/>' +
    '<line x1="-30" y1="0" x2="-30" y2="-55" stroke-width="5"/>' +
    '<circle cx="-30" cy="-58" r="5" fill="%2300A896" stroke="%2300A896"/>' +
    '<line x1="-10" y1="-10" x2="-22" y2="5"/><line x1="0" y1="-12" x2="-8" y2="5"/>' +
    '<line x1="10" y1="-10" x2="10" y2="8"/><line x1="20" y1="-5" x2="25" y2="12"/>' +
    '<path d="M-5,40 L5,40 M-5,36 L5,36" stroke="%234DD0E1" stroke-width="2"/>' +
    '<text x="0" y="65" text-anchor="middle" fill="%238899AA" font-size="16" font-family="Calibri" stroke="none">10 (shake)</text>'
  ),
};

function numSetRefImage(imgId, digit) {
  const img = document.getElementById(imgId);
  if (!img) return;
  img.src = NUM_REFERENCE_URLS[digit] || '';
  img.alt = 'ASL sign for number ' + digit;
  img.style.display = '';
  img.onerror = function() { this.style.opacity = '0.3'; };
  img.onload = function() { this.style.opacity = '1'; };
}

// Load numbers model
async function loadNumbersModel() {
  try {
    if (window.location.protocol === 'file:') return false;
    console.log('[NUM] Loading numbers model...');
    numModel = await tf.loadLayersModel('./model_numbers/model.json');
    console.log('[NUM] Numbers model loaded. Input:', numModel.inputs[0].shape);

    try {
      const resp = await fetch('./model_numbers/scaler.json');
      const data = await resp.json();
      numScalerMean = data.mean;
      numScalerStd = data.std;
      console.log('[NUM] Numbers scaler loaded.');
    } catch(e) {
      numScalerMean = null; numScalerStd = null;
    }
    return true;
  } catch(e) {
    console.warn('[NUM] Numbers model load failed:', e);
    return false;
  }
}

function numApplyScaler(features) {
  if (!numScalerMean || !numScalerStd) return features;
  return features.map((v, i) => (v - numScalerMean[i]) / (numScalerStd[i] || 1));
}

// Numbers detection pipeline
function handleNumbersDetection(landmarks) {
  const base = normalizeLandmarks(landmarks);
  const rawFeatures = computeEnhancedFeatures(base);
  const features = numApplyScaler(rawFeatures);
  const input = tf.tensor2d([features]);
  const pred = numModel.predict(input);
  const probs = pred.dataSync();
  input.dispose(); pred.dispose();

  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
  const rawLabel = NUM_LABELS[maxIdx];
  const rawConf = probs[maxIdx];

  const entropy = computeEntropy(Array.from(probs));

  // Smoothing
  numPredictionBuffer.push({ label: rawLabel, conf: rawConf });
  const params = getAdaptiveParams(entropy);
  while (numPredictionBuffer.length > params.window) numPredictionBuffer.shift();

  const votes = {};
  const totalConf = {};
  numPredictionBuffer.forEach(p => {
    votes[p.label] = (votes[p.label] || 0) + 1;
    totalConf[p.label] = (totalConf[p.label] || 0) + p.conf;
  });
  let bestLabel = '', bestCount = 0;
  for (const [l, c] of Object.entries(votes))
    if (c > bestCount) { bestCount = c; bestLabel = l; }
  const avgConf = totalConf[bestLabel] / bestCount;

  // Update aux display
  const liveEl = document.getElementById('numbers-live-digit');
  const confEl = document.getElementById('numbers-live-conf');
  if (liveEl) liveEl.textContent = bestLabel;
  if (confEl) confEl.textContent = (avgConf * 100).toFixed(0) + '%';

  if (numCurrentSubMode === 'recognize') {
    handleNumRecognize(bestLabel, avgConf, bestCount, params, entropy);
  } else if (numCurrentSubMode === 'practice') {
    handleNumPracticeDetection(bestLabel, avgConf);
  } else if (numCurrentSubMode === 'learn') {
    handleNumLearnDetection(bestLabel, avgConf);
  }
}

function handleNumRecognize(label, conf, count, params, entropy) {
  const predEl = document.getElementById('num-predicted');
  const confBarEl = document.getElementById('num-confidence-bar');
  const confLevelEl = document.getElementById('num-confidence-level');
  const confLabelEl = document.getElementById('num-confidence');

  if (numCooldownCounter > 0) numCooldownCounter--;

  // Update confidence display
  let level, cls;
  if (conf >= 0.80 && entropy < 0.4) { level = 'High'; cls = 'conf-high'; }
  else if (conf >= 0.55 && entropy < 0.7) { level = 'Medium'; cls = 'conf-med'; }
  else { level = 'Low'; cls = 'conf-low'; }

  if (confLabelEl) confLabelEl.textContent = `Confidence: ${(conf * 100).toFixed(0)}%`;
  if (confBarEl) { confBarEl.style.width = `${conf * 100}%`; confBarEl.className = `confidence-bar ${cls}`; }
  if (confLevelEl) { confLevelEl.textContent = level; confLevelEl.className = `confidence-level ${cls}`; }

  if (conf >= params.confThresh) {
    if (predEl) { predEl.textContent = label; predEl.className = `predicted-letter ${cls}`; }

    if (count >= params.hold && numCooldownCounter === 0) {
      numCurrentDigits += label;
      numSessionDigitCount++;
      numUpdateRecognizeUI();
      numCooldownCounter = detectionSpeed === 'careful' ? 55 : COMMIT_COOLDOWN;
      if (predEl) { predEl.classList.add('flash'); setTimeout(() => predEl.classList.remove('flash'), 200); }
    }
  }
}

function numUpdateRecognizeUI() {
  const digitsEl = document.getElementById('num-current-digits');
  const statDigitsEl = document.getElementById('num-stat-digits');
  const statNumsEl = document.getElementById('num-stat-numbers');
  if (digitsEl) digitsEl.textContent = numCurrentDigits || '—';
  if (statDigitsEl) statDigitsEl.textContent = numSessionDigitCount;
  if (statNumsEl) statNumsEl.textContent = numSessionNumberCount;
}

function numCommitNumber() {
  if (numCurrentDigits.length === 0) return;
  const sep = numCompletedText.length > 0 ? ' ' : '';
  numCompletedText += sep + numCurrentDigits;
  numSessionNumberCount++;
  const textEl = document.getElementById('num-text-content');
  if (textEl) textEl.textContent = numCompletedText;
  numCurrentDigits = '';
  numUpdateRecognizeUI();
}

function numClearText() {
  numCompletedText = '';
  numCurrentDigits = '';
  const textEl = document.getElementById('num-text-content');
  if (textEl) textEl.textContent = '';
  numUpdateRecognizeUI();
}

function numAddSpace() {
  if (numCurrentDigits.length > 0) numCommitNumber();
}

function numDeleteLast() {
  if (numCurrentDigits.length > 0) {
    numCurrentDigits = numCurrentDigits.slice(0, -1);
    numUpdateRecognizeUI();
  } else {
    const parts = numCompletedText.trim().split(' ');
    parts.pop();
    numCompletedText = parts.join(' ');
    const textEl = document.getElementById('num-text-content');
    if (textEl) textEl.textContent = numCompletedText;
  }
}

// Numbers sub-mode switching
function switchNumbersMode(mode) {
  numCurrentSubMode = mode;
  ['recognize','practice','learn'].forEach(m => {
    const btn = document.getElementById('numtab-' + m);
    const view = document.getElementById('numbers-view-' + m);
    if (btn) btn.classList.toggle('active', m === mode);
    if (view) view.style.display = (m === mode) ? '' : 'none';
  });
  if (mode === 'practice') {
    numInitPracticeGrid();
    numNextPracticeTarget();
  }
  if (mode === 'learn') {
    numInitLearnGrid();
    numSelectLearnDigit(numLearnCurrent);
    switchNumLearnMode(numCurrentLearnSubMode);
    numUpdateStatsUI();
    numUpdateRing();
  }
}

// === NUMBERS PRACTICE ===
function numInitPracticeGrid() {
  const grid = document.getElementById('num-practice-grid');
  if (!grid) return;
  grid.innerHTML = '';
  NUM_LABELS.forEach(d => {
    const cell = document.createElement('div');
    cell.className = 'practice-grid-cell';
    const stats = numPracticeStats[d];
    if (stats.total >= 3) cell.classList.add('good');
    else if (stats.total >= 1) cell.classList.add('ok');
    cell.textContent = d;
    if (stats.total > 0) cell.title = d + ': ' + stats.correct + '/' + stats.total;
    grid.appendChild(cell);
  });
}

function numStartDrill(mode) {
  numPracticeDrillMode = mode;
  numPracticeDrillIndex = 0;
  numPracticeScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
  numUpdatePracticeScore();
  numNextPracticeTarget();
}

function numNextPracticeTarget() {
  numPracticeShowingResult = false;
  numPracticeCorrectFrames = 0;

  if (numPracticeDrillMode === 'sequential') {
    numPracticeTarget = NUM_LABELS[numPracticeDrillIndex % NUM_LABELS.length];
    numPracticeDrillIndex++;
  } else if (numPracticeDrillMode === 'random') {
    numPracticeTarget = NUM_LABELS[Math.floor(Math.random() * NUM_LABELS.length)];
  } else {
    const weak = NUM_LABELS.filter(d => (numPracticeStats[d].total || 0) < 3);
    numPracticeTarget = weak.length > 0
      ? weak[Math.floor(Math.random() * weak.length)]
      : NUM_LABELS[Math.floor(Math.random() * NUM_LABELS.length)];
  }

  const targetEl = document.getElementById('num-practice-target');
  if (targetEl) targetEl.textContent = numPracticeTarget;
  const resultEl = document.getElementById('num-practice-result');
  if (resultEl) { resultEl.textContent = ''; resultEl.className = 'practice-result'; }
  const qualityEl = document.getElementById('num-practice-quality');
  if (qualityEl) qualityEl.textContent = '';
}

function handleNumPracticeDetection(detected, confidence) {
  if (numPracticeShowingResult) return;

  const qualityEl = document.getElementById('num-practice-quality');
  const resultEl = document.getElementById('num-practice-result');

  if (qualityEl) {
    if (confidence >= 0.90) { qualityEl.textContent = 'Excellent form!'; qualityEl.className = 'practice-quality quality-high'; }
    else if (confidence >= 0.70) { qualityEl.textContent = 'Good — hold steady'; qualityEl.className = 'practice-quality quality-med'; }
    else { qualityEl.textContent = 'Adjust your hand position'; qualityEl.className = 'practice-quality quality-low'; }
  }

  if (detected === numPracticeTarget) {
    numPracticeCorrectFrames++;
    if (numPracticeCorrectFrames >= PRACTICE_HOLD_REQUIRED) {
      numPracticeShowingResult = true;
      numPracticeScore.correct++;
      numPracticeScore.total++;
      numPracticeScore.streak++;
      if (numPracticeScore.streak > numPracticeScore.bestStreak)
        numPracticeScore.bestStreak = numPracticeScore.streak;
      numPracticeStats[numPracticeTarget].correct++;
      numPracticeStats[numPracticeTarget].total++;

      if (resultEl) { resultEl.textContent = 'Correct!'; resultEl.className = 'practice-result result-correct'; }
      numUpdatePracticeScore();
      numInitPracticeGrid();
      setTimeout(numNextPracticeTarget, 1200);
    }
  } else {
    numPracticeCorrectFrames = 0;
    if (resultEl) {
      resultEl.textContent = 'Detected: ' + detected;
      resultEl.className = 'practice-result result-wrong';
    }
  }
}

function numUpdatePracticeScore() {
  const scoreEl = document.getElementById('num-practice-score-display');
  const streakEl = document.getElementById('num-practice-streak-display');
  if (scoreEl) scoreEl.textContent = numPracticeScore.correct + ' / ' + numPracticeScore.total + ' correct';
  if (streakEl) streakEl.textContent = 'Streak: ' + numPracticeScore.streak;
}

function numResetPracticeStats() {
  NUM_LABELS.forEach(d => { numPracticeStats[d] = { correct: 0, total: 0 }; });
  numPracticeScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
  numUpdatePracticeScore();
  numInitPracticeGrid();
}

// === NUMBERS LEARN MODE ===
function numInitLearnGrid() {
  const grid = document.getElementById('num-learn-grid');
  if (!grid) return;
  grid.innerHTML = '';
  NUM_LABELS.forEach(d => {
    const cell = document.createElement('div');
    const p = numLearnProgress[d];
    let cls = 'learn-grid-cell';
    if (d === numLearnCurrent) cls += ' active';
    if (p && p.level !== 'new') cls += ' mastery-' + p.level;
    cell.className = cls;
    cell.textContent = d;
    cell.onclick = () => numSelectLearnDigit(d);
    grid.appendChild(cell);
  });
}

function numSelectLearnDigit(digit) {
  numLearnCurrent = digit;
  if (numLearnProgress[digit].level === 'new') {
    numLearnProgress[digit].level = 'seen';
    numSaveProgress();
  }

  // Update grid active state
  document.querySelectorAll('#num-learn-grid .learn-grid-cell').forEach((cell, i) => {
    const d = NUM_LABELS[i];
    const p = numLearnProgress[d];
    let cls = 'learn-grid-cell';
    if (d === digit) cls += ' active';
    if (p && p.level !== 'new') cls += ' mastery-' + p.level;
    cell.className = cls;
  });

  // Update browse view
  const titleEl = document.getElementById('num-learn-title');
  const mnemonicEl = document.getElementById('num-learn-mnemonic');
  const diffEl = document.getElementById('num-learn-diff-badge');
  const mastEl = document.getElementById('num-learn-mastery-badge');
  const stepsEl = document.getElementById('num-learn-step-list');
  const similarEl = document.getElementById('num-learn-similar');
  const tryitEl = document.getElementById('num-learn-tryit-msg');
  const progEl = document.getElementById('num-learn-progress');

  if (titleEl) titleEl.textContent = digit;
  if (mnemonicEl) mnemonicEl.textContent = NUM_MNEMONICS[digit] || '';

  if (diffEl) {
    const diff = NUM_DIFFICULTY[digit] || 'beginner';
    diffEl.textContent = diff.charAt(0).toUpperCase() + diff.slice(1);
    diffEl.className = 'learn-badge diff' + (diff !== 'beginner' ? ' ' + diff : '');
  }

  if (mastEl) {
    const p = numLearnProgress[digit];
    const lvl = p ? p.level : 'new';
    mastEl.textContent = lvl.charAt(0).toUpperCase() + lvl.slice(1);
    mastEl.className = 'learn-badge mast' + (lvl !== 'new' ? ' ' + lvl : '');
  }

  if (stepsEl) {
    stepsEl.innerHTML = '';
    const steps = NUM_STEPS[digit] || [NUM_DESCRIPTIONS[digit] || ''];
    steps.forEach(s => { const li = document.createElement('li'); li.textContent = s; stepsEl.appendChild(li); });
  }

  if (similarEl) {
    const conf = NUM_CONFUSABLE[digit];
    if (conf && conf.length > 0) {
      similarEl.innerHTML = '<span class="learn-section-label">Watch Out</span> Similar to: ' +
        conf.map(c => '<span class="sim-tag">' + c + '</span>').join(' ');
    } else {
      similarEl.innerHTML = '';
    }
  }

  if (progEl) {
    let seen = 0;
    NUM_LABELS.forEach(d => { if (numLearnProgress[d].level !== 'new') seen++; });
    progEl.textContent = 'Number ' + (NUM_LABELS.indexOf(digit) + 1) + ' of ' + NUM_LABELS.length + '  \u00b7  Seen: ' + seen + '/' + NUM_LABELS.length;
  }

  if (tryitEl) { tryitEl.textContent = 'Show your hand to the camera and sign this number'; tryitEl.className = 'learn-tryit-msg'; }
  const holdBar = document.getElementById('num-learn-hold-bar-browse');
  if (holdBar) holdBar.style.width = '0%';
  const holdPct = document.getElementById('num-learn-hold-pct');
  if (holdPct) holdPct.textContent = '0%';

  numCorrectFrames = 0;
  numSetRefImage('num-learn-ref-img', digit);
  numUpdateRing();
}

function numLearnNext() {
  const idx = NUM_LABELS.indexOf(numLearnCurrent);
  numSelectLearnDigit(NUM_LABELS[(idx + 1) % NUM_LABELS.length]);
}

function numLearnPrev() {
  const idx = NUM_LABELS.indexOf(numLearnCurrent);
  numSelectLearnDigit(NUM_LABELS[(idx + NUM_LABELS.length - 1) % NUM_LABELS.length]);
}

function numPracticeThis() {
  switchNumbersMode('practice');
  numPracticeDrillMode = 'sequential';
  numPracticeTarget = numLearnCurrent;
  numPracticeCorrectFrames = 0;
  numPracticeShowingResult = false;
  const targetEl = document.getElementById('num-practice-target');
  if (targetEl) targetEl.textContent = numPracticeTarget;
}

function switchNumLearnMode(mode) {
  numCurrentLearnSubMode = mode;
  ['browse','lesson','quiz','flashcard'].forEach(m => {
    const tab = document.getElementById('nltab-' + m);
    const view = document.getElementById('num-learn-view-' + m);
    if (tab) tab.classList.toggle('active', m === mode);
    if (view) view.style.display = (m === mode) ? '' : 'none';
  });
  if (mode === 'lesson') numInitLessonView();
  if (mode === 'quiz') numInitQuizView();
  if (mode === 'flashcard') numInitFlashcardView();
}

function numUpdateMastery(digit, wasCorrect) {
  const p = numLearnProgress[digit];
  p.attempts++;
  if (wasCorrect) p.correct++;
  const rate = p.attempts > 0 ? p.correct / p.attempts : 0;
  if (p.attempts >= 5 && rate >= 0.8) p.level = 'mastered';
  else if (p.attempts >= 2 && rate >= 0.5) p.level = 'practiced';
  else if (p.attempts >= 1) p.level = 'seen';
  numSaveProgress();
  numUpdateStatsUI();
  numUpdateRing();
}

function numUpdateStatsUI() {
  let mastered = 0, totalAttempts = 0, totalCorrect = 0;
  NUM_LABELS.forEach(d => {
    const p = numLearnProgress[d];
    if (p.level === 'mastered') mastered++;
    totalAttempts += p.attempts;
    totalCorrect += p.correct;
  });
  const mastEl = document.getElementById('num-ls-mastered');
  const accEl = document.getElementById('num-ls-accuracy');
  const streakEl = document.getElementById('num-ls-streak');
  if (mastEl) mastEl.innerHTML = mastered + '<span class="learn-stat-unit">/' + NUM_LABELS.length + '</span>';
  if (accEl) accEl.textContent = totalAttempts > 0 ? Math.round(totalCorrect / totalAttempts * 100) + '%' : '—';
  if (streakEl) streakEl.textContent = numQuizScore.bestStreak;
}

function numUpdateRing() {
  let mastered = 0;
  NUM_LABELS.forEach(d => { if (numLearnProgress[d].level === 'mastered') mastered++; });
  const pct = Math.round(mastered / NUM_LABELS.length * 100);
  const circumference = 213.6;
  const offset = circumference - (circumference * pct / 100);
  const ringEl = document.getElementById('num-learn-ring-fill');
  const pctEl = document.getElementById('num-learn-ring-pct');
  if (ringEl) ringEl.style.strokeDashoffset = offset;
  if (pctEl) pctEl.textContent = pct + '%';
}

function numResetLearnProgress() {
  if (!confirm('Reset all number learning progress?')) return;
  NUM_LABELS.forEach(d => { numLearnProgress[d] = { level: 'new', attempts: 0, correct: 0 }; });
  numQuizScore = { correct: 0, total: 0, streak: 0, bestStreak: 0 };
  numSaveProgress();
  numInitLearnGrid();
  numUpdateStatsUI();
  numUpdateRing();
}

// === NUMBERS LEARN DETECTION ===
function handleNumLearnDetection(detected, confidence) {
  const matchEl = document.getElementById('numbers-match-display');

  let targetDigit = numLearnCurrent;
  if (numCurrentLearnSubMode === 'quiz' && numQuizActive && (numQuizType === 'sign' || numQuizType === 'speed')) {
    targetDigit = numQuizCurrent;
  }

  const isMatch = detected === targetDigit && confidence >= 0.60;

  if (isMatch) {
    numCorrectFrames = Math.min(numCorrectFrames + 1, LEARN_HOLD_REQUIRED);
  } else {
    numCorrectFrames = Math.max(numCorrectFrames - 1, 0);
  }

  const progress = numCorrectFrames / LEARN_HOLD_REQUIRED;
  const pct = Math.round(progress * 100);

  // Update hold bars
  ['numbers-hold-bar', 'num-learn-hold-bar-browse', 'num-lesson-hold-bar'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.width = pct + '%';
  });
  const holdPctEl = document.getElementById('num-learn-hold-pct');
  if (holdPctEl) holdPctEl.textContent = pct + '%';

  if (matchEl) {
    if (numCorrectFrames >= LEARN_HOLD_REQUIRED) {
      matchEl.className = 'learn-match-display match-correct';
      matchEl.textContent = '\u2713 Correct!';
    } else if (isMatch) {
      matchEl.className = 'learn-match-display match-holding';
      matchEl.textContent = 'Hold\u2026';
    } else if (detected && detected !== '—') {
      matchEl.className = 'learn-match-display match-wrong';
      matchEl.textContent = 'Keep trying';
    } else {
      matchEl.className = 'learn-match-display';
      matchEl.textContent = '';
    }
  }

  // Browse try-it
  if (numCurrentLearnSubMode === 'browse') {
    const tryitEl = document.getElementById('num-learn-tryit-msg');
    if (tryitEl) {
      if (numCorrectFrames >= LEARN_HOLD_REQUIRED) {
        tryitEl.textContent = '\u2713 Great job! You signed ' + targetDigit + ' correctly!';
        tryitEl.className = 'learn-tryit-msg success';
      } else if (isMatch) {
        tryitEl.textContent = 'Hold steady\u2026 ' + pct + '%';
        tryitEl.className = 'learn-tryit-msg holding';
      } else if (detected && detected !== '—') {
        tryitEl.textContent = 'Detected: ' + detected + ' \u2014 Try signing ' + targetDigit;
        tryitEl.className = 'learn-tryit-msg';
      }
    }
  }

  // Lesson feedback
  if (numCurrentLearnSubMode === 'lesson' && numLessonActive) {
    const fb = document.getElementById('num-lesson-feedback');
    if (fb) {
      if (numCorrectFrames >= LEARN_HOLD_REQUIRED) { fb.textContent = '\u2713 Correct!'; fb.className = 'lesson-feedback correct'; }
      else if (isMatch) { fb.textContent = 'Hold steady\u2026'; fb.className = 'lesson-feedback holding'; }
      else { fb.textContent = 'Sign the number ' + targetDigit; fb.className = 'lesson-feedback'; }
    }
  }

  // On success
  if (numCorrectFrames >= LEARN_HOLD_REQUIRED) {
    numCorrectFrames = 0;
    numUpdateMastery(targetDigit, true);
    numInitLearnGrid();

    if (numCurrentLearnSubMode === 'lesson' && numLessonActive) {
      numLessonIndex++;
      setTimeout(numRunLessonStep, 800);
    } else if (numCurrentLearnSubMode === 'quiz' && numQuizActive && (numQuizType === 'sign' || numQuizType === 'speed')) {
      numQuizCheckSign();
    } else if (numCurrentLearnSubMode === 'browse') {
      setTimeout(() => {
        const tryitEl = document.getElementById('num-learn-tryit-msg');
        if (tryitEl) { tryitEl.textContent = 'Show your hand to the camera and sign this number'; tryitEl.className = 'learn-tryit-msg'; }
      }, 2000);
    }
  }
}

// === NUMBERS LESSON MODE ===
function numInitLessonView() {
  const digit = numLearnCurrent;
  const targetEl = document.getElementById('num-lesson-target');
  const descEl = document.getElementById('num-lesson-description');
  const stepsEl = document.getElementById('num-lesson-steps');
  if (targetEl) targetEl.textContent = digit;
  if (descEl) descEl.textContent = NUM_DESCRIPTIONS[digit] || '';
  if (stepsEl) {
    const steps = NUM_STEPS[digit] || [];
    stepsEl.innerHTML = steps.length > 0 ? '<ol>' + steps.map(s => '<li>' + s + '</li>').join('') + '</ol>' : '';
  }
  numSetRefImage('num-lesson-ref-img', digit);
  numUpdateLessonProgress();
}

function numUpdateLessonProgress() {
  const labelEl = document.getElementById('num-lesson-step-label');
  const barEl = document.getElementById('num-lesson-bar-fill');
  if (labelEl) labelEl.textContent = 'Number ' + (numLessonIndex + 1) + ' of ' + NUM_LABELS.length;
  if (barEl) barEl.style.width = ((numLessonIndex / NUM_LABELS.length) * 100) + '%';
}

function numToggleLesson() {
  numLessonActive = !numLessonActive;
  const btn = document.getElementById('num-btn-lesson-toggle');
  if (numLessonActive) {
    btn.textContent = '\u25A0 Stop Lesson';
    btn.classList.add('stop');
    numLessonIndex = NUM_LABELS.indexOf(numLearnCurrent);
    numCorrectFrames = 0;
    numRunLessonStep();
  } else {
    btn.textContent = '\u25B6 Start Lesson';
    btn.classList.remove('stop');
    numCorrectFrames = 0;
    const fb = document.getElementById('num-lesson-feedback');
    if (fb) { fb.textContent = 'Lesson paused'; fb.className = 'lesson-feedback'; }
  }
}

function numRunLessonStep() {
  if (!numLessonActive) return;
  if (numLessonIndex >= NUM_LABELS.length) {
    numLessonActive = false;
    const btn = document.getElementById('num-btn-lesson-toggle');
    if (btn) { btn.textContent = '\u25B6 Start Lesson'; btn.classList.remove('stop'); }
    const fb = document.getElementById('num-lesson-feedback');
    if (fb) { fb.textContent = '\u2728 Congratulations! You completed all numbers!'; fb.className = 'lesson-feedback correct'; }
    return;
  }
  const digit = NUM_LABELS[numLessonIndex % NUM_LABELS.length];
  numSelectLearnDigit(digit);
  const targetEl = document.getElementById('num-lesson-target');
  const descEl = document.getElementById('num-lesson-description');
  const stepsEl = document.getElementById('num-lesson-steps');
  if (targetEl) targetEl.textContent = digit;
  if (descEl) descEl.textContent = NUM_DESCRIPTIONS[digit] || '';
  if (stepsEl) {
    const steps = NUM_STEPS[digit] || [];
    stepsEl.innerHTML = steps.length > 0 ? '<ol>' + steps.map(s => '<li>' + s + '</li>').join('') + '</ol>' : '';
  }
  numSetRefImage('num-lesson-ref-img', digit);
  numUpdateLessonProgress();
  numCorrectFrames = 0;
  const fb = document.getElementById('num-lesson-feedback');
  if (fb) { fb.textContent = 'Sign the number ' + digit + ' to continue'; fb.className = 'lesson-feedback'; }
  const hb = document.getElementById('num-lesson-hold-bar');
  if (hb) hb.style.width = '0%';
}

function numLessonSkip() {
  if (!numLessonActive) {
    numLessonActive = true;
    const btn = document.getElementById('num-btn-lesson-toggle');
    if (btn) { btn.textContent = '\u25A0 Stop Lesson'; btn.classList.add('stop'); }
    numLessonIndex = NUM_LABELS.indexOf(numLearnCurrent);
  }
  numLessonIndex++;
  numRunLessonStep();
}

function numLessonRestart() {
  numLessonIndex = 0;
  numLessonActive = true;
  const btn = document.getElementById('num-btn-lesson-toggle');
  if (btn) { btn.textContent = '\u25A0 Stop Lesson'; btn.classList.add('stop'); }
  numRunLessonStep();
}

// === NUMBERS QUIZ MODE ===
function numInitQuizView() {
  const promptEl = document.getElementById('num-quiz-prompt');
  if (promptEl) promptEl.textContent = 'Choose a quiz type above to begin!';
  const optEl = document.getElementById('num-quiz-options');
  if (optEl) optEl.innerHTML = '';
  const fbEl = document.getElementById('num-quiz-feedback');
  if (fbEl) { fbEl.textContent = ''; fbEl.className = 'quiz-feedback'; }
}

function numStartQuiz(type) {
  numQuizType = type;
  numQuizActive = true;
  numQuizScore = { correct: 0, total: 0, streak: 0, bestStreak: numQuizScore.bestStreak || 0 };
  if (numQuizTimer) { clearInterval(numQuizTimer); numQuizTimer = null; }

  ['sign','identify','speed'].forEach(t => {
    const el = document.getElementById('nqtype-' + t);
    if (el) el.classList.toggle('active', t === type);
  });

  const resEl = document.getElementById('num-quiz-result');
  if (resEl) resEl.style.display = 'none';
  const timerWrap = document.getElementById('num-quiz-timer-wrap');

  if (type === 'speed') {
    numQuizTimeLeft = 30;
    if (timerWrap) timerWrap.style.display = '';
    numQuizTimer = setInterval(numQuizTimerTick, 1000);
    numUpdateQuizTimerUI();
  } else {
    if (timerWrap) timerWrap.style.display = 'none';
  }

  numUpdateQuizScoreUI();
  numQuizNext();
}

function numQuizNext() {
  if (!numQuizActive) return;
  const promptEl = document.getElementById('num-quiz-prompt');
  const optEl = document.getElementById('num-quiz-options');
  const fbEl = document.getElementById('num-quiz-feedback');
  const quizImgEl = document.getElementById('num-quiz-ref-img');
  const letterBig = document.getElementById('num-quiz-letter-big');

  if (fbEl) { fbEl.textContent = ''; fbEl.className = 'quiz-feedback'; }
  if (optEl) optEl.innerHTML = '';

  if (numQuizType === 'sign' || numQuizType === 'speed') {
    numQuizCurrent = NUM_LABELS[Math.floor(Math.random() * NUM_LABELS.length)];
    if (promptEl) promptEl.textContent = 'Sign this number:';
    if (letterBig) { letterBig.textContent = numQuizCurrent; letterBig.style.display = ''; }
    if (quizImgEl) quizImgEl.style.display = 'none';
    numCorrectFrames = 0;
  } else if (numQuizType === 'identify') {
    numQuizCurrent = NUM_LABELS[Math.floor(Math.random() * NUM_LABELS.length)];
    if (promptEl) promptEl.textContent = 'What number is this?';
    if (letterBig) letterBig.style.display = 'none';
    if (quizImgEl) { quizImgEl.style.display = ''; numSetRefImage('num-quiz-ref-img', numQuizCurrent); }
    const options = numGenerateQuizOptions(numQuizCurrent, 4);
    if (optEl) {
      options.forEach(d => {
        const btn = document.createElement('button');
        btn.className = 'quiz-opt-btn';
        btn.textContent = d;
        btn.onclick = () => numQuizCheckIdentify(d, btn);
        optEl.appendChild(btn);
      });
    }
  }
}

function numGenerateQuizOptions(correct, count) {
  const opts = new Set([correct]);
  while (opts.size < count) opts.add(NUM_LABELS[Math.floor(Math.random() * NUM_LABELS.length)]);
  return Array.from(opts).sort(() => Math.random() - 0.5);
}

function numQuizCheckIdentify(digit, btnEl) {
  const optBtns = document.querySelectorAll('#num-quiz-options .quiz-opt-btn');
  optBtns.forEach(b => b.disabled = true);

  const correct = digit === numQuizCurrent;
  numQuizScore.total++;
  if (correct) {
    numQuizScore.correct++;
    numQuizScore.streak++;
    if (numQuizScore.streak > numQuizScore.bestStreak) numQuizScore.bestStreak = numQuizScore.streak;
    btnEl.classList.add('correct');
    numUpdateMastery(numQuizCurrent, true);
    numShowQuizFeedback(true, numQuizCurrent);
  } else {
    numQuizScore.streak = 0;
    btnEl.classList.add('wrong');
    optBtns.forEach(b => { if (b.textContent === numQuizCurrent) b.classList.add('correct'); });
    numUpdateMastery(numQuizCurrent, false);
    numShowQuizFeedback(false, numQuizCurrent);
  }
  numUpdateQuizScoreUI();
  setTimeout(() => { if (numQuizActive) numQuizNext(); }, 1500);
}

function numQuizCheckSign() {
  numQuizScore.total++;
  numQuizScore.correct++;
  numQuizScore.streak++;
  if (numQuizScore.streak > numQuizScore.bestStreak) numQuizScore.bestStreak = numQuizScore.streak;
  numUpdateMastery(numQuizCurrent, true);
  numShowQuizFeedback(true, numQuizCurrent);
  numUpdateQuizScoreUI();
  setTimeout(() => { if (numQuizActive) numQuizNext(); }, 1000);
}

function numShowQuizFeedback(correct, digit) {
  const fbEl = document.getElementById('num-quiz-feedback');
  if (!fbEl) return;
  if (correct) { fbEl.textContent = '\u2713 Correct! That\'s ' + digit; fbEl.className = 'quiz-feedback correct'; }
  else { fbEl.textContent = '\u2717 The answer was ' + digit; fbEl.className = 'quiz-feedback wrong'; }
}

function numUpdateQuizScoreUI() {
  const scoreEl = document.getElementById('num-quiz-score');
  const streakEl = document.getElementById('num-quiz-streak');
  if (scoreEl) scoreEl.textContent = numQuizScore.correct + ' / ' + numQuizScore.total;
  if (streakEl) streakEl.textContent = '\uD83D\uDD25 ' + numQuizScore.streak;
}

function numQuizTimerTick() {
  numQuizTimeLeft--;
  numUpdateQuizTimerUI();
  if (numQuizTimeLeft <= 0) {
    clearInterval(numQuizTimer);
    numQuizTimer = null;
    numQuizActive = false;
    numShowQuizResult();
  }
}

function numUpdateQuizTimerUI() {
  const fillEl = document.getElementById('num-quiz-timer-fill');
  const textEl = document.getElementById('num-quiz-timer-text');
  if (fillEl) fillEl.style.width = ((numQuizTimeLeft / 30) * 100) + '%';
  if (textEl) textEl.textContent = numQuizTimeLeft + 's';
}

function numShowQuizResult() {
  const resEl = document.getElementById('num-quiz-result');
  if (!resEl) return;
  const rate = numQuizScore.total > 0 ? Math.round(numQuizScore.correct / numQuizScore.total * 100) : 0;
  resEl.innerHTML = '<h3>Quiz Complete!</h3>' +
    '<p class="qr-stat">Score: <strong>' + numQuizScore.correct + ' / ' + numQuizScore.total + '</strong> (' + rate + '%)</p>' +
    '<p class="qr-stat">Best Streak: <strong>' + numQuizScore.bestStreak + '</strong></p>' +
    '<button onclick="numStartQuiz(\'' + numQuizType + '\')">Play Again</button>';
  resEl.style.display = '';
}

// === NUMBERS FLASHCARD MODE ===
function numInitFlashcardView() {
  numFcBuildDeck();
  numFcIndex = 0;
  numFcFlipped = false;
  numFcRender();
}

function numFcBuildDeck() {
  if (numFcFilter === 'all') { numFcDeck = NUM_LABELS.slice(); }
  else if (numFcFilter === 'new') { numFcDeck = NUM_LABELS.filter(d => numLearnProgress[d].level === 'new' || numLearnProgress[d].level === 'seen'); }
  else { numFcDeck = NUM_LABELS.filter(d => numLearnProgress[d].level !== 'mastered'); }
  if (numFcDeck.length === 0) numFcDeck = NUM_LABELS.slice();
  numFcDeck.sort(() => Math.random() - 0.5);
}

function numFcSetFilter(filter) {
  numFcFilter = filter;
  ['all','new','weak'].forEach(f => {
    const el = document.getElementById('num-fc-f-' + f);
    if (el) el.classList.toggle('active', f === filter);
  });
  numFcBuildDeck();
  numFcIndex = 0;
  numFcFlipped = false;
  numFcRender();
}

function numFcFlip() {
  numFcFlipped = !numFcFlipped;
  const card = document.getElementById('num-fc-card');
  if (card) card.classList.toggle('flipped', numFcFlipped);
  if (numFcFlipped) {
    const digit = numFcDeck[numFcIndex % numFcDeck.length];
    numSetRefImage('num-fc-ref-img', digit);
    const descEl = document.getElementById('num-fc-back-desc');
    if (descEl) descEl.textContent = NUM_DESCRIPTIONS[digit] || '';
  }
}

function numFcNext() {
  numFcIndex = (numFcIndex + 1) % numFcDeck.length;
  numFcFlipped = false;
  const card = document.getElementById('num-fc-card');
  if (card) card.classList.remove('flipped');
  numFcRender();
}

function numFcPrev() {
  numFcIndex = (numFcIndex - 1 + numFcDeck.length) % numFcDeck.length;
  numFcFlipped = false;
  const card = document.getElementById('num-fc-card');
  if (card) card.classList.remove('flipped');
  numFcRender();
}

function numFcKnow() {
  const digit = numFcDeck[numFcIndex % numFcDeck.length];
  numUpdateMastery(digit, true);
  numInitLearnGrid();
  numFcNext();
}

function numFcDunno() {
  const digit = numFcDeck[numFcIndex % numFcDeck.length];
  numUpdateMastery(digit, false);
  numInitLearnGrid();
  numFcNext();
}

function numFcRender() {
  if (numFcDeck.length === 0) return;
  const digit = numFcDeck[numFcIndex % numFcDeck.length];
  const frontEl = document.getElementById('num-fc-front-letter');
  const counterEl = document.getElementById('num-fc-counter');
  if (frontEl) frontEl.textContent = digit;
  if (counterEl) counterEl.textContent = (numFcIndex + 1) + ' / ' + numFcDeck.length;
}

// Handle hand absent for numbers recognize mode
function handleNumbersHandAbsent() {
  numHandAbsentFrames++;
  numPredictionBuffer = [];
  if (numCurrentSubMode === 'recognize' && numHandAbsentFrames === AUTO_SPACE_FRAMES && numCurrentDigits.length > 0) {
    numCommitNumber();
  }
  const liveEl = document.getElementById('numbers-live-digit');
  if (liveEl) liveEl.textContent = '—';
  const confEl = document.getElementById('numbers-live-conf');
  if (confEl) confEl.textContent = '';
  if (numCurrentSubMode === 'learn') {
    numCorrectFrames = 0;
    const matchEl = document.getElementById('numbers-match-display');
    if (matchEl) { matchEl.className = 'learn-match-display'; matchEl.textContent = ''; }
  }
}

// ═══════════════════════════════════════════════════════════════
// END NUMBERS MODE
// ═══════════════════════════════════════════════════════════════


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
  await loadNumbersModel();
}

init();
