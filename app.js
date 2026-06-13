const $ = (selector) => document.querySelector(selector);

let DrawingUtils;
let FilesetResolver;
let PoseLandmarker;

const els = {
  video: $("#video"),
  mainCanvas: $("#mainCanvas"),
  pipCanvas: $("#pipCanvas"),
  cameraBtn: $("#cameraBtn"),
  switchBtn: $("#switchBtn"),
  delayBtn: $("#delayBtn"),
  pipBtn: $("#pipBtn"),
  poseBtn: $("#poseBtn"),
  delaySlider: $("#delaySlider"),
  delayValue: $("#delayValue"),
  drawHandSelect: $("#drawHandSelect"),
  cameraStatus: $("#cameraStatus"),
  modelStatus: $("#modelStatus"),
  canvasLabel: $("#canvasLabel"),
  metricBoard: $("#metricBoard"),
  phaseCard: $("#phaseCard"),
  phaseValue: $("#phaseValue"),
  phaseConfidence: $("#phaseConfidence"),
  eventLog: $("#eventLog"),
  rotateBtn: $("#rotateBtn")
};

const ctx = els.mainCanvas.getContext("2d");
const pipCtx = els.pipCanvas.getContext("2d");
const capture = document.createElement("canvas");
const captureCtx = capture.getContext("2d", { willReadFrequently: false });
const fallbackCapture = document.createElement("canvas");
const fallbackCaptureCtx = fallbackCapture.getContext("2d", { willReadFrequently: false });
const delayVideo = document.createElement("video");
delayVideo.muted = true;
delayVideo.playsInline = true;
delayVideo.preload = "auto";
delayVideo.setAttribute("muted", "");
delayVideo.setAttribute("playsinline", "");

const DELAY_SEGMENT_MS = 2000;
const DELAY_BUFFER_EXTRA_MS = 2500;
const DELAY_REPLAY_FORWARD_DRIFT_SEC = 0.65;
const FALLBACK_DELAY_FPS = 30;
const PORTRAIT_FRAME_SIZE = { width: 720, height: 1024 };
const LANDSCAPE_FRAME_SIZE = { width: 1024, height: 720 };
const FALLBACK_JPEG_QUALITY = 0.72;
const FACE_LANDMARK_INDEX_MAX = 10;
const FACE_CENTER_SMOOTHING_ALPHA = 0.42;
const MIN_DRAW_BEFORE_ANCHOR_MS = 220;
const MIN_ANCHOR_BEFORE_RELEASE_MS = 260;
const DRAW_TO_ANCHOR_CONFIRM_MS = 120;
const ANCHOR_TO_RELEASE_CONFIRM_MS = 160;
const RECORDER_MIME_TYPES = [
  "video/mp4;codecs=avc1.42E01E",
  "video/mp4",
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm"
];

const FULL_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task";

const TASKS_VISION_URLS = [
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs",
  "https://unpkg.com/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs",
  "https://unpkg.com/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs"
];

const LANDMARKS = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftEar: 7,
  rightEar: 8,
  mouthLeft: 9,
  mouthRight: 10,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24
};

const state = {
  stream: null,
  facingMode: "user",
  running: false,
  poseEnabled: true,
  delayEnabled: true,
  pipEnabled: false,
  delayMs: 3000,
  targetFps: 30,
  poseLandmarker: null,
  videoSegments: [],
  poseSamples: [],
  delayRecorder: null,
  delayRecorderTimer: null,
  delayRecorderMimeType: "",
  delayRecorderSupported: !!window.MediaRecorder && !isIOSDevice(),
  delayFallbackSupported: !!window.HTMLCanvasElement?.prototype?.toBlob,
  delayFallbackActive: false,
  delayFallbackWarningShown: false,
  delayFallbackDecodeWarningShown: false,
  fallbackFrames: [],
  fallbackEncoding: false,
  lastFallbackFrameAt: 0,
  fallbackDisplayFrame: null,
  fallbackDisplayImage: null,
  delaySupportWarningShown: false,
  delayRecorderGeneration: 0,
  replaySegment: null,
  replayPendingSeek: 0,
  hasReplayFrame: false,
  phase: "INIT",
  confidence: 0,
  lastPhaseAt: 0,
  lastAnalysisAt: 0,
  cameraAspectRatio: "",
  captureDrawRect: null,
  lastLandmarks: null,
  lastDrawWrist: null,
  smoothedFaceCenter: null,
  lastAnchorAt: 0,
  lastReleaseAt: 0,
  releaseResetSeen: true,
  anchorSnapshot: null,
  candidate: { phase: null, since: 0, count: 0 },
  lastMetrics: {},
  metricHistory: [],
  drag: null
};

const metricDefs = [
  { id: "phase", label: "階段", unit: "" },
  { id: "shoulderHipAngle", label: "肩髖平行", unit: "deg", good: (v) => v <= 12 },
  { id: "bowArmAngle", label: "持弓臂角度", unit: "deg", good: (v) => v >= 145 },
  { id: "drawElbowAngle", label: "拉弦肘角度", unit: "deg", good: (v) => v <= 95 },
  { id: "drawLength", label: "拉距", unit: "肩寬", good: (v) => v >= 1.15 },
  { id: "drawWristFaceDistance", label: "腕臉距", unit: "肩寬", good: (v) => v <= 0.55 },
  { id: "drawWristShoulderHeight", label: "腕肩高差", unit: "肩寬", good: (v) => Math.abs(v) <= 0.28 },
  { id: "drawWristSpeed", label: "拉弦腕速度", unit: "/s" }
];

function logEvent(text) {
  const p = document.createElement("p");
  p.textContent = `${new Date().toLocaleTimeString()} ${text}`;
  els.eventLog.prepend(p);
  while (els.eventLog.children.length > 8) els.eventLog.lastElementChild.remove();
}

function setStatus() {
  els.cameraStatus.textContent = state.running ? "相機運作中" : "相機未啟動";
  els.modelStatus.textContent = state.poseLandmarker ? "高精度模型已載入 · 30 FPS" : "高精度模型未載入";
  els.delayValue.textContent = `${(state.delayMs / 1000).toFixed(1)}s`;
  const canShowCompressedDelay =
    state.delayEnabled && state.delayRecorderSupported && state.delayMs >= DELAY_SEGMENT_MS;
  const canShowFallbackDelay =
    state.delayEnabled && state.delayFallbackActive && state.delayMs >= DELAY_SEGMENT_MS;
  els.canvasLabel.textContent =
    canShowCompressedDelay
      ? `延遲 ${(state.delayMs / 1000).toFixed(1)}s · 壓縮30FPS`
      : canShowFallbackDelay
        ? `延遲 ${(state.delayMs / 1000).toFixed(1)}s · iPhone相容30FPS`
      : "即時畫面";
  els.delayBtn.classList.toggle("active", canShowCompressedDelay || canShowFallbackDelay);
  els.pipBtn.classList.toggle("active", state.pipEnabled);
  els.poseBtn.classList.toggle("active", state.poseEnabled);
  els.pipCanvas.classList.toggle("visible", state.pipEnabled && state.running);
}

function cameraErrorMessage(error) {
  const name = error?.name || "Error";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "瀏覽器拒絕相機權限。請允許 localhost 使用相機後重新按啟動。";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "找不到可用攝影機。請確認攝影機已連接，且沒有被其他軟體占用。";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "攝影機目前無法讀取，可能被其他 App 使用中。";
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return "攝影機不支援目前解析度或鏡頭方向，已嘗試降級設定。";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "此瀏覽器不支援相機 API，或目前頁面不是安全來源。請使用 localhost 或 HTTPS。";
  }
  return error?.message || "相機啟動失敗。";
}

function isPermissionError(error) {
  return error?.name === "NotAllowedError" || error?.name === "SecurityError";
}

function frontCameraZoomConstraint() {
  return state.facingMode === "user" ? { zoom: { ideal: 1 } } : {};
}

async function requestCameraStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("此瀏覽器不支援 navigator.mediaDevices.getUserMedia");
  }

  const preferredSize = displayFrameSize();
  const preferred = {
    video: {
      facingMode: { ideal: state.facingMode },
      width: { ideal: preferredSize.width },
      height: { ideal: preferredSize.height },
      frameRate: { ideal: 30 },
      ...frontCameraZoomConstraint()
    },
    audio: false
  };
  const relaxed = {
    video: {
      facingMode: { ideal: state.facingMode },
      frameRate: { ideal: 30 },
      ...frontCameraZoomConstraint()
    },
    audio: false
  };
  const basic = { video: true, audio: false };

  try {
    return await navigator.mediaDevices.getUserMedia(preferred);
  } catch (firstError) {
    if (isPermissionError(firstError)) throw firstError;
    logEvent(`相機高解析設定不可用：${cameraErrorMessage(firstError)}`);
    try {
      return await navigator.mediaDevices.getUserMedia(relaxed);
    } catch (secondError) {
      if (isPermissionError(secondError)) throw secondError;
      logEvent(`指定鏡頭不可用，改用預設相機：${cameraErrorMessage(secondError)}`);
      return navigator.mediaDevices.getUserMedia(basic);
    }
  }
}

async function applyFrontCameraWideAnglePreference() {
  if (state.facingMode !== "user") return;
  const track = state.stream?.getVideoTracks?.()[0];
  if (!track?.getCapabilities || !track.applyConstraints) return;
  const capabilities = track.getCapabilities();
  const settings = track.getSettings?.() || {};
  const minZoom = capabilities.zoom?.min;
  if (!Number.isFinite(minZoom)) return;
  const currentZoom = Number.isFinite(settings.zoom) ? settings.zoom : Infinity;
  if (currentZoom <= minZoom + 0.001) return;
  try {
    await track.applyConstraints({ advanced: [{ zoom: minZoom }] });
    logEvent(`前鏡頭已套用最廣視角 zoom ${minZoom}`);
  } catch (error) {
    logEvent(`前鏡頭廣角設定不可用：${error.message || "zoom unsupported"}`);
  }
}

async function loadPoseModel() {
  if (state.poseLandmarker) return;
  els.modelStatus.textContent = "模型載入中...";
  if (state.poseLandmarker) {
    state.poseLandmarker.close();
    state.poseLandmarker = null;
  }
  if (!PoseLandmarker) {
    let lastError;
    for (const url of TASKS_VISION_URLS) {
      try {
        const mod = await import(url);
        DrawingUtils = mod.DrawingUtils;
        FilesetResolver = mod.FilesetResolver;
        PoseLandmarker = mod.PoseLandmarker;
        break;
      } catch (error) {
        lastError = error;
      }
    }
    if (!PoseLandmarker) throw lastError || new Error("MediaPipe 模組載入失敗");
  }
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
  );
  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: FULL_MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45
  });
  setStatus();
  logEvent("高精度姿態模型已載入 · 30 FPS");
}

function supportedRecorderMimeType() {
  if (!window.MediaRecorder?.isTypeSupported) return "";
  return RECORDER_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function delayRecorderBitrate() {
  const settings = state.stream?.getVideoTracks?.()[0]?.getSettings?.() || {};
  const width = settings.width || els.video.videoWidth || 1280;
  const height = settings.height || els.video.videoHeight || 720;
  return Math.round(Math.min(12_000_000, Math.max(3_000_000, width * height * 4)));
}

function revokeSegment(segment) {
  if (segment?.url) {
    URL.revokeObjectURL(segment.url);
    segment.url = "";
  }
}

function releaseFallbackDisplayImage() {
  state.fallbackDisplayImage?.close?.();
  state.fallbackDisplayImage = null;
  state.fallbackDisplayFrame = null;
}

function clearFallbackFrames() {
  state.fallbackFrames = [];
  state.fallbackEncoding = false;
  state.lastFallbackFrameAt = 0;
  state.delayFallbackDecodeWarningShown = false;
  releaseFallbackDisplayImage();
}

function clearDelayBuffer() {
  state.videoSegments.forEach(revokeSegment);
  state.videoSegments = [];
  clearFallbackFrames();
  state.poseSamples = [];
  state.replaySegment = null;
  state.replayPendingSeek = 0;
  state.hasReplayFrame = false;
  delayVideo.pause();
  delayVideo.playbackRate = 1;
  delayVideo.removeAttribute("src");
  delayVideo.load();
}

function pruneDelayBuffer(now) {
  const maxAge = state.delayMs + DELAY_BUFFER_EXTRA_MS + DELAY_SEGMENT_MS;
  const maxFallbackFrames = Math.ceil((maxAge / 1000) * FALLBACK_DELAY_FPS) + 4;
  const minTime = now - maxAge;
  while (state.videoSegments.length && state.videoSegments[0].end < minTime) {
    const removed = state.videoSegments.shift();
    if (removed === state.replaySegment) state.replaySegment = null;
    revokeSegment(removed);
  }
  while (state.fallbackFrames.length && state.fallbackFrames[0].time < minTime) {
    state.fallbackFrames.shift();
  }
  while (state.fallbackFrames.length > maxFallbackFrames) {
    state.fallbackFrames.shift();
  }
  while (state.poseSamples.length && state.poseSamples[0].time < minTime) {
    state.poseSamples.shift();
  }
}

function isIOSDevice() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
}

function isPortraitScreen() {
  return window.matchMedia?.("(orientation: portrait)")?.matches ?? window.innerHeight >= window.innerWidth;
}

function displayFrameSize() {
  return isPortraitScreen() ? PORTRAIT_FRAME_SIZE : LANDSCAPE_FRAME_SIZE;
}

function coverDrawRect(source, targetWidth, targetHeight) {
  const sourceWidth = source.videoWidth || source.width;
  const sourceHeight = source.videoHeight || source.height;
  if (!sourceWidth || !sourceHeight || !targetWidth || !targetHeight) return null;
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  return {
    x: (targetWidth - drawWidth) / 2,
    y: (targetHeight - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight
  };
}

function drawImageCover(targetCtx, source, targetWidth, targetHeight) {
  const rect = coverDrawRect(source, targetWidth, targetHeight);
  if (!rect) return null;
  targetCtx.fillStyle = "#020617";
  targetCtx.fillRect(0, 0, targetWidth, targetHeight);
  targetCtx.drawImage(source, rect.x, rect.y, rect.width, rect.height);
  return rect;
}

function projectLandmarksToRect(landmarks, rect, width, height) {
  if (!landmarks || !rect || !width || !height) return landmarks;
  return landmarks.map((lm) => ({
    ...lm,
    x: (rect.x + lm.x * rect.width) / width,
    y: (rect.y + lm.y * rect.height) / height
  }));
}

function projectRectToTarget(sourceRect, sourceWidth, sourceHeight, targetRect) {
  if (!sourceRect || !sourceWidth || !sourceHeight || !targetRect) return targetRect;
  return {
    x: targetRect.x + (sourceRect.x / sourceWidth) * targetRect.width,
    y: targetRect.y + (sourceRect.y / sourceHeight) * targetRect.height,
    width: (sourceRect.width / sourceWidth) * targetRect.width,
    height: (sourceRect.height / sourceHeight) * targetRect.height
  };
}

function enableFallbackDelay(reason) {
  stopDelayRecorder(false);
  state.delayRecorderSupported = false;
  state.delayFallbackActive = state.delayFallbackSupported;
  if (!state.delayFallbackActive) {
    if (!state.delaySupportWarningShown) {
      logEvent("此瀏覽器不支援延遲回看，改顯示即時畫面");
      state.delaySupportWarningShown = true;
    }
    setStatus();
    return;
  }
  if (!state.delayFallbackWarningShown) {
    logEvent(reason || "使用 iPhone 相容延遲回看");
    state.delayFallbackWarningShown = true;
  }
  setStatus();
}

function stopDelayRecorder(storeCurrentSegment = false) {
  clearTimeout(state.delayRecorderTimer);
  state.delayRecorderTimer = null;
  const recorder = state.delayRecorder;
  state.delayRecorder = null;
  if (!storeCurrentSegment) state.delayRecorderGeneration += 1;
  if (recorder && recorder.state !== "inactive") {
    try {
      recorder.stop();
    } catch {
      // The recorder may already be stopped by the browser when tracks end.
    }
  }
}

function startDelayRecorderSegment() {
  if (
    !state.running ||
    !state.delayEnabled ||
    state.delayMs < DELAY_SEGMENT_MS ||
    !state.delayRecorderSupported ||
    state.delayRecorder ||
    !state.stream
  ) {
    return;
  }
  if (!window.MediaRecorder) {
    enableFallbackDelay("此瀏覽器不支援壓縮錄影，已切換 iPhone 相容延遲");
    return;
  }

  const mimeType = state.delayRecorderMimeType || supportedRecorderMimeType();
  state.delayRecorderMimeType = mimeType;
  const chunks = [];
  const startedAt = performance.now();
  const generation = state.delayRecorderGeneration;

  try {
    const options = { videoBitsPerSecond: delayRecorderBitrate() };
    if (mimeType) options.mimeType = mimeType;
    const recorder = new MediaRecorder(state.stream, options);
    state.delayRecorder = recorder;

    recorder.addEventListener("dataavailable", (event) => {
      if (event.data?.size) chunks.push(event.data);
    });

    recorder.addEventListener("error", (event) => {
      logEvent(`壓縮回看錄製錯誤：${event.error?.message || "MediaRecorder error"}`);
      enableFallbackDelay("壓縮回看錄製錯誤，已切換 iPhone 相容延遲");
    });

    recorder.addEventListener("stop", () => {
      clearTimeout(state.delayRecorderTimer);
      state.delayRecorderTimer = null;
      if (state.delayRecorder === recorder) state.delayRecorder = null;

      const isCurrent = generation === state.delayRecorderGeneration;
      if (isCurrent && chunks.length && state.running && state.delayEnabled) {
        const endedAt = performance.now();
        const type = chunks[0].type || recorder.mimeType || mimeType || "video/webm";
        state.videoSegments.push({
          start: startedAt,
          end: endedAt,
          blob: new Blob(chunks, { type }),
          url: ""
        });
        pruneDelayBuffer(endedAt);
      }

      if (isCurrent && state.running && state.delayEnabled) {
        queueMicrotask(startDelayRecorderSegment);
      }
    });

    recorder.start();
    state.delayRecorderTimer = setTimeout(() => {
      if (state.delayRecorder === recorder && recorder.state === "recording") {
        try {
          recorder.requestData();
        } catch {
          // Some Safari builds do not allow explicit requestData before stop.
        }
        recorder.stop();
      }
    }, DELAY_SEGMENT_MS);
  } catch (error) {
    state.delayRecorder = null;
    enableFallbackDelay(`壓縮延遲回看不可用，已切換 iPhone 相容延遲：${error.message}`);
  }
}

function compactLandmarks(landmarks) {
  if (!landmarks) return null;
  return landmarks.map((lm) => ({
    x: lm.x,
    y: lm.y,
    z: lm.z,
    visibility: lm.visibility
  }));
}

function compactMetrics(metrics = {}) {
  return {
    phase: state.phase,
    shoulderHipAngle: metrics.shoulderHipAngle,
    bowArmAngle: metrics.bowArmAngle,
    drawElbowAngle: metrics.drawElbowAngle,
    drawLength: metrics.drawLength,
    drawWristFaceDistance: metrics.drawWristFaceDistance,
    drawWristShoulderHeight: metrics.drawWristShoulderHeight,
    drawWristSpeed: metrics.drawWristSpeed
  };
}

function pushPoseSample(time, landmarks, metrics) {
  if (!state.delayEnabled) return;
  state.poseSamples.push({
    time,
    landmarks: compactLandmarks(landmarks),
    metrics: compactMetrics(metrics)
  });
  pruneDelayBuffer(time);
}

function nearestPoseSample(targetTime) {
  if (!state.poseSamples.length) return null;
  let best = state.poseSamples[0];
  let bestDelta = Math.abs(best.time - targetTime);
  for (let i = state.poseSamples.length - 1; i >= 0; i -= 1) {
    const delta = Math.abs(state.poseSamples[i].time - targetTime);
    if (delta < bestDelta) {
      best = state.poseSamples[i];
      bestDelta = delta;
    }
    if (state.poseSamples[i].time < targetTime && delta > bestDelta) break;
  }
  return best;
}

function nearestFallbackFrame(targetTime) {
  if (!state.fallbackFrames.length) return null;
  let best = state.fallbackFrames[0];
  let bestDelta = Math.abs(best.time - targetTime);
  for (let i = state.fallbackFrames.length - 1; i >= 0; i -= 1) {
    const delta = Math.abs(state.fallbackFrames[i].time - targetTime);
    if (delta < bestDelta) {
      best = state.fallbackFrames[i];
      bestDelta = delta;
    }
    if (state.fallbackFrames[i].time < targetTime && delta > bestDelta) break;
  }
  return best;
}

function captureFallbackFrame(now, landmarks, metrics) {
  if (
    !state.delayEnabled ||
    !state.delayFallbackActive ||
    state.delayMs < DELAY_SEGMENT_MS ||
    state.fallbackEncoding ||
    now - state.lastFallbackFrameAt < 1000 / FALLBACK_DELAY_FPS ||
    !capture.width ||
    !capture.height
  ) {
    return;
  }

  const size = displayFrameSize();
  if (fallbackCapture.width !== size.width || fallbackCapture.height !== size.height) {
    fallbackCapture.width = size.width;
    fallbackCapture.height = size.height;
  }
  drawImageCover(fallbackCaptureCtx, capture, size.width, size.height);
  state.fallbackEncoding = true;
  state.lastFallbackFrameAt = now;
  fallbackCapture.toBlob((blob) => {
    state.fallbackEncoding = false;
    if (!blob || !state.running || !state.delayEnabled || !state.delayFallbackActive) return;
    state.fallbackFrames.push({
      time: now,
      blob,
      drawRect: state.captureDrawRect,
      frameSize: size,
      landmarks: compactLandmarks(landmarks),
      metrics: compactMetrics(metrics)
    });
    pruneDelayBuffer(performance.now());
  }, "image/jpeg", FALLBACK_JPEG_QUALITY);
}

async function decodeFallbackFrame(frame) {
  if (!frame) return null;
  if (state.fallbackDisplayFrame === frame && state.fallbackDisplayImage) {
    return state.fallbackDisplayImage;
  }
  releaseFallbackDisplayImage();
  if (window.createImageBitmap) {
    const bitmap = await createImageBitmap(frame.blob);
    state.fallbackDisplayFrame = frame;
    state.fallbackDisplayImage = {
      image: bitmap,
      close: () => bitmap.close?.()
    };
    return state.fallbackDisplayImage;
  }
  const url = URL.createObjectURL(frame.blob);
  const image = new Image();
  image.decoding = "async";
  image.src = url;
  if (image.decode) {
    await image.decode();
  } else {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
    });
  }
  state.fallbackDisplayFrame = frame;
  state.fallbackDisplayImage = {
    image,
    close: () => URL.revokeObjectURL(url)
  };
  return state.fallbackDisplayImage;
}

async function currentFallbackReplay(now) {
  if (!state.delayEnabled || !state.delayFallbackActive || state.delayMs < DELAY_SEGMENT_MS) return null;
  const targetTime = now - state.delayMs;
  const frame = nearestFallbackFrame(targetTime);
  if (!frame) return null;
  let decoded = null;
  try {
    decoded = await decodeFallbackFrame(frame);
  } catch (error) {
    if (!state.delayFallbackDecodeWarningShown) {
      logEvent(`iPhone相容回看解碼失敗，暫時顯示即時畫面：${error.message || "decode error"}`);
      state.delayFallbackDecodeWarningShown = true;
    }
    return null;
  }
  return decoded ? { frame, decoded, sample: frame } : null;
}

function segmentForTime(targetTime) {
  if (!state.videoSegments.length) return null;
  for (let i = state.videoSegments.length - 1; i >= 0; i -= 1) {
    const segment = state.videoSegments[i];
    if (segment.start <= targetTime && targetTime <= segment.end + 120) return segment;
  }
  const first = state.videoSegments[0];
  const last = state.videoSegments.at(-1);
  if (targetTime < first.start) return first;
  if (targetTime - last.end <= DELAY_SEGMENT_MS * 1.5) return last;
  return null;
}

function syncReplayVideo(segment, targetTime) {
  const rawOffset = (targetTime - segment.start) / 1000;
  const segmentDuration = Math.max(0.05, (segment.end - segment.start) / 1000);
  const offset = Math.max(0, Math.min(rawOffset, segmentDuration - 0.04));

  if (state.replaySegment !== segment) {
    if (!segment.url) segment.url = URL.createObjectURL(segment.blob);
    state.replaySegment = segment;
    state.replayPendingSeek = offset;
    delayVideo.src = segment.url;
    delayVideo.load();
    return false;
  }

  if (delayVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return false;
  const duration = Number.isFinite(delayVideo.duration) ? delayVideo.duration : segmentDuration;
  const safeOffset = Math.max(0, Math.min(offset, Math.max(0, duration - 0.04)));
  if (safeOffset - delayVideo.currentTime > DELAY_REPLAY_FORWARD_DRIFT_SEC) {
    try {
      delayVideo.currentTime = safeOffset;
    } catch {
      return false;
    }
  }
  if (delayVideo.currentTime - safeOffset > 0.6) {
    delayVideo.playbackRate = 0.92;
  } else if (safeOffset - delayVideo.currentTime > 0.25) {
    delayVideo.playbackRate = 1.08;
  } else {
    delayVideo.playbackRate = 1;
  }
  if (delayVideo.paused || delayVideo.ended) delayVideo.play().catch(() => {});
  return true;
}

function currentDelayReplay(now) {
  if (!state.delayEnabled || !state.delayRecorderSupported || state.delayMs < DELAY_SEGMENT_MS) return null;
  const targetTime = now - state.delayMs;
  const segment = segmentForTime(targetTime);
  if (!segment || !syncReplayVideo(segment, targetTime)) return null;
  return {
    video: delayVideo,
    sample: nearestPoseSample(targetTime)
  };
}

function shouldHoldReplayFrame() {
  return state.delayEnabled && (state.delayRecorderSupported || state.delayFallbackActive) && state.hasReplayFrame;
}

delayVideo.addEventListener("loadedmetadata", () => {
  try {
    const duration = Number.isFinite(delayVideo.duration) ? delayVideo.duration : state.replayPendingSeek;
    delayVideo.currentTime = Math.max(0, Math.min(state.replayPendingSeek, Math.max(0, duration - 0.04)));
  } catch {
    // Seeking before enough metadata is available can fail on some mobile browsers.
  }
});

delayVideo.addEventListener("canplay", () => {
  if (state.running && state.delayEnabled) delayVideo.play().catch(() => {});
});

async function startCamera() {
  if (state.running) {
    stopCamera();
    return;
  }
  els.cameraBtn.disabled = true;
  els.cameraBtn.textContent = "啟動中...";
  els.cameraStatus.textContent = "正在請求相機權限";
  state.stream = await requestCameraStream();
  els.video.srcObject = state.stream;
  await els.video.play();
  await applyFrontCameraWideAnglePreference();
  clearDelayBuffer();
  state.smoothedFaceCenter = null;
  state.delayRecorderSupported = !!window.MediaRecorder && !isIOSDevice();
  state.delayFallbackActive = state.delayEnabled && (!state.delayRecorderSupported || isIOSDevice()) && state.delayFallbackSupported;
  state.delaySupportWarningShown = false;
  state.delayFallbackWarningShown = false;
  state.delayRecorderGeneration += 1;
  state.running = true;
  els.cameraBtn.disabled = false;
  els.cameraBtn.textContent = "停止相機";
  setStatus();
  logEvent("相機已啟動");
  if (state.delayEnabled && state.delayFallbackActive && !state.delayFallbackWarningShown) {
    logEvent(isIOSDevice() ? "iPhone 使用相容延遲回看" : "壓縮錄影不可用，使用相容延遲回看");
    state.delayFallbackWarningShown = true;
    setStatus();
  } else if (state.delayEnabled && !state.delayRecorderSupported && !state.delayFallbackActive && !state.delaySupportWarningShown) {
    logEvent("此瀏覽器不支援延遲回看，改顯示即時畫面");
    state.delaySupportWarningShown = true;
    setStatus();
  } else if (state.delayEnabled) {
    startDelayRecorderSegment();
  }
  requestAnimationFrame(loop);

  loadPoseModel().catch((error) => {
    els.modelStatus.textContent = "模型載入失敗";
    logEvent(`模型載入失敗，僅顯示相機畫面：${error.message}`);
  });
}

function stopCamera() {
  state.running = false;
  stopDelayRecorder(false);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  clearDelayBuffer();
  state.lastLandmarks = null;
  state.captureDrawRect = null;
  state.smoothedFaceCenter = null;
  state.cameraAspectRatio = "";
  state.delayFallbackActive = false;
  els.cameraBtn.textContent = "啟動相機";
  clearCanvas();
  setStatus();
  logEvent("相機已停止");
}

async function switchCamera() {
  state.facingMode = state.facingMode === "user" ? "environment" : "user";
  if (!state.running) return;
  stopCamera();
  await startCamera();
}

function syncDisplayFrameAspect() {
  const { width, height } = displayFrameSize();
  const aspectRatio = `${width} / ${height}`;
  state.cameraAspectRatio = aspectRatio;
  els.mainCanvas.style.aspectRatio = aspectRatio;
  els.pipCanvas.style.aspectRatio = aspectRatio;
  els.mainCanvas.parentElement?.style.setProperty("--camera-aspect-ratio", aspectRatio);
  els.mainCanvas.parentElement?.style.setProperty("--camera-aspect-value", String(width / height));
  return { width, height };
}

function resizeCanvases() {
  const { width, height } = syncDisplayFrameAspect();
  for (const canvas of [els.mainCanvas, capture]) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }
  const pipWidth = Math.max(320, Math.round(width * 0.32));
  const pipHeight = Math.round(pipWidth * (height / width));
  if (els.pipCanvas.width !== pipWidth || els.pipCanvas.height !== pipHeight) {
    els.pipCanvas.width = pipWidth;
    els.pipCanvas.height = pipHeight;
  }
}

function clearCanvas() {
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, els.mainCanvas.width || 1280, els.mainCanvas.height || 720);
  pipCtx.clearRect(0, 0, els.pipCanvas.width, els.pipCanvas.height);
}

function point(landmarks, index) {
  const lm = landmarks?.[index];
  if (!lm || lm.visibility < 0.35) return null;
  return lm;
}

function distance(a, b) {
  if (!a || !b) return NaN;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a, b) {
  if (!a || !b) return null;
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z || 0) + (b.z || 0)) / 2,
    visibility: Math.min(a.visibility ?? 1, b.visibility ?? 1)
  };
}

function weightedPoint(parts) {
  let total = 0;
  let x = 0;
  let y = 0;
  let z = 0;
  let visibility = 0;
  for (const { point: part, weight } of parts) {
    if (!part || !Number.isFinite(part.x) || !Number.isFinite(part.y)) continue;
    total += weight;
    x += part.x * weight;
    y += part.y * weight;
    z += (part.z || 0) * weight;
    visibility = Math.max(visibility, part.visibility ?? 0);
  }
  if (!total) return null;
  return { x: x / total, y: y / total, z: z / total, visibility };
}

function estimateFaceCenter(landmarks, leftShoulder = null, rightShoulder = null, shoulderWidth = 1) {
  const nose = point(landmarks, LANDMARKS.nose);
  const eyeMid = midpoint(point(landmarks, LANDMARKS.leftEye), point(landmarks, LANDMARKS.rightEye));
  const mouthMid = midpoint(point(landmarks, LANDMARKS.mouthLeft), point(landmarks, LANDMARKS.mouthRight));
  const earMid = midpoint(point(landmarks, LANDMARKS.leftEar), point(landmarks, LANDMARKS.rightEar));
  const center = weightedPoint([
    { point: eyeMid, weight: 0.36 },
    { point: nose, weight: 0.28 },
    { point: mouthMid, weight: 0.26 },
    { point: earMid, weight: 0.1 }
  ]);
  if (center) return center;
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  if (!shoulderMid) return null;
  return {
    x: shoulderMid.x,
    y: shoulderMid.y - shoulderWidth * 0.65,
    z: shoulderMid.z,
    visibility: shoulderMid.visibility
  };
}

function smoothFaceCenter(faceCenter, now) {
  if (!faceCenter) {
    state.smoothedFaceCenter = null;
    return null;
  }
  const previous = state.smoothedFaceCenter;
  if (!previous || now - previous.now > 450) {
    state.smoothedFaceCenter = { ...faceCenter, now };
    return state.smoothedFaceCenter;
  }
  const alpha = FACE_CENTER_SMOOTHING_ALPHA;
  state.smoothedFaceCenter = {
    x: previous.x + (faceCenter.x - previous.x) * alpha,
    y: previous.y + (faceCenter.y - previous.y) * alpha,
    z: previous.z + ((faceCenter.z || 0) - (previous.z || 0)) * alpha,
    visibility: faceCenter.visibility,
    now
  };
  return state.smoothedFaceCenter;
}

function angle(a, b, c) {
  if (!a || !b || !c) return NaN;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!mag) return NaN;
  return Math.acos(Math.min(1, Math.max(-1, dot / mag))) * (180 / Math.PI);
}

function vectorAngle(a, b, c, d) {
  if (!a || !b || !c || !d) return NaN;
  const v1 = { x: b.x - a.x, y: b.y - a.y };
  const v2 = { x: d.x - c.x, y: d.y - c.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);
  if (!mag) return NaN;
  return Math.acos(Math.min(1, Math.max(-1, dot / mag))) * (180 / Math.PI);
}

function computeMetrics(landmarks, now) {
  const drawRight = els.drawHandSelect.value === "right";
  const draw = {
    shoulder: point(landmarks, drawRight ? LANDMARKS.rightShoulder : LANDMARKS.leftShoulder),
    elbow: point(landmarks, drawRight ? LANDMARKS.rightElbow : LANDMARKS.leftElbow),
    wrist: point(landmarks, drawRight ? LANDMARKS.rightWrist : LANDMARKS.leftWrist)
  };
  const bow = {
    shoulder: point(landmarks, drawRight ? LANDMARKS.leftShoulder : LANDMARKS.rightShoulder),
    elbow: point(landmarks, drawRight ? LANDMARKS.leftElbow : LANDMARKS.rightElbow),
    wrist: point(landmarks, drawRight ? LANDMARKS.leftWrist : LANDMARKS.rightWrist)
  };
  const leftShoulder = point(landmarks, LANDMARKS.leftShoulder);
  const rightShoulder = point(landmarks, LANDMARKS.rightShoulder);
  const leftHip = point(landmarks, LANDMARKS.leftHip);
  const rightHip = point(landmarks, LANDMARKS.rightHip);
  const shoulderWidth = distance(leftShoulder, rightShoulder) || 1;
  const faceCenter = smoothFaceCenter(estimateFaceCenter(landmarks, leftShoulder, rightShoulder, shoulderWidth), now);
  const last = state.lastDrawWrist;
  const dt = last ? Math.max(0.001, (now - last.now) / 1000) : 0;
  const speed = last && draw.wrist ? distance(draw.wrist, last) / dt / shoulderWidth : 0;
  const drawLength = distance(draw.wrist, bow.wrist) / shoulderWidth;
  if (draw.wrist) state.lastDrawWrist = { ...draw.wrist, now };

  return {
    shoulderHipAngle: Math.abs(vectorAngle(leftShoulder, rightShoulder, leftHip, rightHip)),
    bowArmAngle: angle(bow.shoulder, bow.elbow, bow.wrist),
    drawElbowAngle: angle(draw.shoulder, draw.elbow, draw.wrist),
    drawLength,
    drawWristFaceDistance: distance(draw.wrist, faceCenter) / shoulderWidth,
    drawWristShoulderHeight: draw.wrist && draw.shoulder ? (draw.wrist.y - draw.shoulder.y) / shoulderWidth : NaN,
    drawElbowY: draw.elbow && draw.shoulder ? (draw.elbow.y - draw.shoulder.y) / shoulderWidth : NaN,
    drawWristY: draw.wrist && draw.shoulder ? (draw.wrist.y - draw.shoulder.y) / shoulderWidth : NaN,
    bowElbowY: bow.elbow && bow.shoulder ? (bow.elbow.y - bow.shoulder.y) / shoulderWidth : NaN,
    bowWristY: bow.wrist && bow.shoulder ? (bow.wrist.y - bow.shoulder.y) / shoulderWidth : NaN,
    drawWristSpeed: speed,
    drawWrist: draw.wrist,
    bowWrist: bow.wrist,
    faceCenter,
    shoulderWidth
  };
}

function finite(value) {
  return Number.isFinite(value);
}

function pushMetricHistory(metrics, now) {
  state.metricHistory.push({
    now,
    drawLength: metrics.drawLength,
    drawWristFaceDistance: metrics.drawWristFaceDistance,
    drawWristShoulderHeight: metrics.drawWristShoulderHeight,
    drawElbowY: metrics.drawElbowY,
    drawWristY: metrics.drawWristY,
    bowElbowY: metrics.bowElbowY,
    bowWristY: metrics.bowWristY,
    drawWristSpeed: metrics.drawWristSpeed
  });
  while (state.metricHistory.length && now - state.metricHistory[0].now > 1800) {
    state.metricHistory.shift();
  }
}

function windowSamples(ms, now, key) {
  return state.metricHistory
    .filter((sample) => now - sample.now <= ms && finite(sample[key]))
    .map((sample) => ({ t: sample.now, v: sample[key] }));
}

function deltaOver(ms, now, key) {
  const samples = windowSamples(ms, now, key);
  if (samples.length < 2) return 0;
  return samples[samples.length - 1].v - samples[0].v;
}

function rangeOver(ms, now, key) {
  const values = windowSamples(ms, now, key).map((sample) => sample.v);
  if (values.length < 2) return Infinity;
  return Math.max(...values) - Math.min(...values);
}

function averageOver(ms, now, key) {
  const values = windowSamples(ms, now, key).map((sample) => sample.v);
  if (!values.length) return NaN;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function slopeOver(ms, now, key) {
  const samples = windowSamples(ms, now, key);
  if (samples.length < 2) return 0;
  const t0 = samples[0].t;
  let sumT = 0;
  let sumV = 0;
  let sumTV = 0;
  let sumTT = 0;
  for (const sample of samples) {
    const t = (sample.t - t0) / 1000;
    sumT += t;
    sumV += sample.v;
    sumTV += t * sample.v;
    sumTT += t * t;
  }
  const n = samples.length;
  const denominator = n * sumTT - sumT * sumT;
  if (Math.abs(denominator) < 1e-9) return 0;
  return (n * sumTV - sumT * sumV) / denominator;
}

function cvOver(ms, now, key) {
  const values = windowSamples(ms, now, key).map((sample) => sample.v);
  if (values.length < 2) return Infinity;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance) / Math.max(Math.abs(mean), 1e-6);
}

function confirmPhase(phase, passed, now, confirmMs = 0) {
  if (!passed) {
    if (state.candidate.phase === phase) state.candidate = { phase: null, since: 0, count: 0 };
    return false;
  }
  if (state.candidate.phase === phase) {
    state.candidate.count += 1;
  } else {
    state.candidate = { phase, since: now, count: 1 };
  }
  return now - state.candidate.since >= confirmMs || state.candidate.count >= 3;
}

function confirmStablePhase(phase, passed, now, confirmMs, minCount = 4) {
  if (!passed) {
    if (state.candidate.phase === phase) state.candidate = { phase: null, since: 0, count: 0 };
    return false;
  }
  if (state.candidate.phase === phase) {
    state.candidate.count += 1;
  } else {
    state.candidate = { phase, since: now, count: 1 };
  }
  return now - state.candidate.since >= confirmMs && state.candidate.count >= minCount;
}

function setPhase(phase, confidence, now) {
  const previous = state.phase;
  const allowedTransitions = {
    INIT: ["DRAW"],
    DRAW: ["ANCHOR", "INIT"],
    ANCHOR: ["RELEASE", "INIT"],
    RELEASE: ["INIT"]
  };
  if (previous !== phase && !allowedTransitions[previous]?.includes(phase)) return false;
  state.phase = phase;
  state.confidence = confidence;
  if (previous !== phase) {
    state.lastPhaseAt = now;
    state.candidate = { phase: null, since: 0, count: 0 };
    if (phase === "ANCHOR") {
      state.lastAnchorAt = now;
    }
    if (phase === "RELEASE") {
      state.lastReleaseAt = now;
      state.releaseResetSeen = false;
      state.anchorSnapshot = null;
    }
    logEvent(`階段切換：${previous} → ${phase}`);
  }
  return true;
}

function updatePhase(metrics, now) {
  pushMetricHistory(metrics, now);

  const hasPose = finite(metrics.drawLength) && finite(metrics.drawWristFaceDistance);
  const timeInPhase = now - state.lastPhaseAt;
  const movementWindowMs = 300;
  const trendWindowMs = 420;
  const anchorWindowMs = 260;
  const restPoseSeen = metrics.drawLength < 0.68;

  if (restPoseSeen) {
    state.releaseResetSeen = true;
    state.anchorSnapshot = null;
  }

  if (!hasPose) {
    setPhase("INIT", 0, now);
    return;
  }

  if (state.phase === "RELEASE") {
    if (timeInPhase < 1300) {
      state.confidence = 0.72;
      return;
    }
    if (restPoseSeen || timeInPhase > 2200) setPhase("INIT", 0.28, now);
    return;
  }

  const releaseSpeed = metrics.drawWristSpeed;
  const wristFaceSlope = slopeOver(trendWindowMs, now, "drawWristFaceDistance");
  const releaseCandidate =
    timeInPhase >= MIN_ANCHOR_BEFORE_RELEASE_MS &&
    releaseSpeed >= 0.3 &&
    wristFaceSlope >= 0.06;
  if (state.phase === "ANCHOR") {
    if (confirmStablePhase("RELEASE", releaseCandidate, now, ANCHOR_TO_RELEASE_CONFIRM_MS, 4)) {
      setPhase("RELEASE", 0.92, now);
      return;
    }
    const anchorResetCandidate = timeInPhase > 900 && restPoseSeen;
    if (confirmStablePhase("INIT", anchorResetCandidate, now, 360, 8)) setPhase("INIT", 0.3, now);
    else state.confidence = 0.78;
    return;
  }

  const drawElbowUp = -slopeOver(movementWindowMs, now, "drawElbowY");
  const drawWristUp = -slopeOver(movementWindowMs, now, "drawWristY");
  const bowElbowUp = -slopeOver(movementWindowMs, now, "bowElbowY");
  const bowWristUp = -slopeOver(movementWindowMs, now, "bowWristY");
  const bowRaised =
    metrics.bowWristY <= 0.42 &&
    metrics.bowElbowY <= 0.55 &&
    metrics.bowArmAngle >= 132;
  const drawCandidate =
    bowRaised &&
    drawElbowUp >= 0.08 &&
    drawWristUp >= 0.08 &&
    bowElbowUp >= 0.05 &&
    bowWristUp >= 0.05 &&
    metrics.drawLength >= 0.72;

  const drawLengthSlope = slopeOver(anchorWindowMs, now, "drawLength");
  const faceCv = cvOver(anchorWindowMs, now, "drawWristFaceDistance");
  const lengthCv = cvOver(anchorWindowMs, now, "drawLength");
  const anchorCandidate =
    Math.abs(drawLengthSlope) <= 0.2 &&
    metrics.drawWristShoulderHeight < 0.68 &&
    faceCv <= 0.58 &&
    lengthCv <= 0.42 &&
    metrics.bowArmAngle >= 118 &&
    metrics.drawElbowAngle <= 160 &&
    metrics.drawLength >= 0.76;

  if (state.phase === "INIT") {
    if (confirmPhase("DRAW", drawCandidate, now, 80)) setPhase("DRAW", 0.68, now);
    else state.confidence = 0.24;
    return;
  }

  if (state.phase === "DRAW") {
    const anchorReady = timeInPhase >= MIN_DRAW_BEFORE_ANCHOR_MS && anchorCandidate;
    if (confirmStablePhase("ANCHOR", anchorReady, now, DRAW_TO_ANCHOR_CONFIRM_MS, 4)) {
      state.anchorSnapshot = {
        drawLength: metrics.drawLength,
        drawWristFaceDistance: metrics.drawWristFaceDistance
      };
      setPhase("ANCHOR", 0.86, now);
      return;
    }
    state.confidence = anchorCandidate ? 0.64 : 0.58;
  }
}

function drawPose(canvasCtx, landmarks, width, height, drawRect = null) {
  if (!state.poseEnabled || !landmarks || !DrawingUtils || !PoseLandmarker) return;
  const renderLandmarks = projectLandmarksToRect(landmarks, drawRect, width, height);
  const utils = new DrawingUtils(canvasCtx);
  const bodyConnections = PoseLandmarker.POSE_CONNECTIONS.filter((connection) => {
    const start = connection.start ?? connection[0];
    const end = connection.end ?? connection[1];
    return start > FACE_LANDMARK_INDEX_MAX && end > FACE_LANDMARK_INDEX_MAX;
  });
  const bodyLandmarks = renderLandmarks.filter((_, index) => index > FACE_LANDMARK_INDEX_MAX);
  utils.drawConnectors(renderLandmarks, bodyConnections, {
    color: "#7dd3fc",
    lineWidth: Math.max(2, width / 420)
  });
  utils.drawLandmarks(bodyLandmarks, {
    color: "#fbbf24",
    fillColor: "#22c55e",
    radius: Math.max(2, width / 260)
  });
  const leftShoulder = point(renderLandmarks, LANDMARKS.leftShoulder);
  const rightShoulder = point(renderLandmarks, LANDMARKS.rightShoulder);
  const shoulderWidth = distance(leftShoulder, rightShoulder) || 1;
  const faceCenter = estimateFaceCenter(renderLandmarks, leftShoulder, rightShoulder, shoulderWidth);
  if (faceCenter) {
    const radius = Math.max(4, width / 180);
    canvasCtx.save();
    canvasCtx.beginPath();
    canvasCtx.arc(faceCenter.x * width, faceCenter.y * height, radius, 0, Math.PI * 2);
    canvasCtx.fillStyle = "#fb7185";
    canvasCtx.strokeStyle = "#ffffff";
    canvasCtx.lineWidth = Math.max(1.5, width / 520);
    canvasCtx.fill();
    canvasCtx.stroke();
    canvasCtx.restore();
  }
}

function drawWaitingFrame(targetCtx, width, height, label = "等待相機畫面") {
  targetCtx.clearRect(0, 0, width, height);
  targetCtx.fillStyle = "#020617";
  targetCtx.fillRect(0, 0, width, height);
  targetCtx.fillStyle = "#94a3b8";
  targetCtx.textAlign = "center";
  targetCtx.font = `${Math.max(16, width / 34)}px system-ui`;
  targetCtx.fillText(label, width / 2, height / 2);
}

function drawVideoFrame(targetCtx, video, width, height, landmarks = null) {
  targetCtx.clearRect(0, 0, width, height);
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawWaitingFrame(targetCtx, width, height, "正在累積延遲回看");
    return;
  }
  const drawRect = drawImageCover(targetCtx, video, width, height);
  drawPose(targetCtx, landmarks, width, height, drawRect);
}

function drawDecodedFrame(targetCtx, decoded, width, height, landmarks = null, sourcePoseRect = null) {
  targetCtx.clearRect(0, 0, width, height);
  if (!decoded?.image) {
    drawWaitingFrame(targetCtx, width, height, "正在累積延遲回看");
    return;
  }
  const drawRect = drawImageCover(targetCtx, decoded.image, width, height);
  const poseRect = projectRectToTarget(sourcePoseRect, decoded.image.width, decoded.image.height, drawRect);
  drawPose(targetCtx, landmarks, width, height, poseRect);
}

function drawLiveFrame(targetCtx, width, height, landmarks = null) {
  targetCtx.clearRect(0, 0, width, height);
  const source = els.video;
  if (source.width || source.videoWidth) {
    const drawRect = drawImageCover(targetCtx, source, width, height);
    drawPose(targetCtx, landmarks, width, height, drawRect);
    return;
  }
  drawWaitingFrame(targetCtx, width, height);
}

async function loop(now) {
  if (!state.running) return;
  requestAnimationFrame(loop);
  if (!els.video.videoWidth) return;

  resizeCanvases();
  const interval = 1000 / state.targetFps;
  const renderTime = performance.now();
  if (now - state.lastAnalysisAt < interval) {
    startDelayRecorderSegment();
    const replay = currentDelayReplay(renderTime);
    if (replay) {
      drawVideoFrame(ctx, replay.video, els.mainCanvas.width, els.mainCanvas.height, replay.sample?.landmarks);
      state.hasReplayFrame = true;
    } else {
      const fallbackReplay = await currentFallbackReplay(renderTime);
      if (fallbackReplay) {
        drawDecodedFrame(
          ctx,
          fallbackReplay.decoded,
          els.mainCanvas.width,
          els.mainCanvas.height,
          fallbackReplay.sample?.landmarks,
          fallbackReplay.frame?.drawRect
        );
        state.hasReplayFrame = true;
      } else if (!shouldHoldReplayFrame()) {
        drawLiveFrame(ctx, els.mainCanvas.width, els.mainCanvas.height, state.lastLandmarks);
      }
    }
    if (state.pipEnabled) drawLiveFrame(pipCtx, els.pipCanvas.width, els.pipCanvas.height, state.lastLandmarks);
    return;
  }
  state.lastAnalysisAt = now;

  state.captureDrawRect = drawImageCover(captureCtx, els.video, capture.width, capture.height);
  let landmarks = null;
  if (state.poseLandmarker) {
    const result = state.poseLandmarker.detectForVideo(els.video, performance.now());
    landmarks = result.landmarks?.[0] || null;
  }
  const sampleTime = performance.now();
  const metrics = landmarks ? computeMetrics(landmarks, sampleTime) : {};
  if (landmarks) updatePhase(metrics, sampleTime);
  state.lastMetrics = { ...metrics, phase: state.phase };
  state.lastLandmarks = compactLandmarks(landmarks);
  pushPoseSample(sampleTime, landmarks, state.lastMetrics);
  captureFallbackFrame(sampleTime, landmarks, state.lastMetrics);
  startDelayRecorderSegment();
  pruneDelayBuffer(sampleTime);

  const replay = currentDelayReplay(sampleTime);
  if (replay) {
    drawVideoFrame(ctx, replay.video, els.mainCanvas.width, els.mainCanvas.height, replay.sample?.landmarks);
    state.hasReplayFrame = true;
  } else {
    const fallbackReplay = await currentFallbackReplay(sampleTime);
    if (fallbackReplay) {
      drawDecodedFrame(
        ctx,
        fallbackReplay.decoded,
        els.mainCanvas.width,
        els.mainCanvas.height,
        fallbackReplay.sample?.landmarks,
        fallbackReplay.frame?.drawRect
      );
      state.hasReplayFrame = true;
    } else if (!shouldHoldReplayFrame()) {
      drawLiveFrame(ctx, els.mainCanvas.width, els.mainCanvas.height, state.lastLandmarks);
    }
  }
  if (state.pipEnabled) drawLiveFrame(pipCtx, els.pipCanvas.width, els.pipCanvas.height, state.lastLandmarks);
  const heldSample = shouldHoldReplayFrame() ? nearestPoseSample(sampleTime - state.delayMs) : null;
  const fallbackSample = state.delayFallbackActive ? nearestFallbackFrame(sampleTime - state.delayMs) : null;
  renderMetrics(replay?.sample?.metrics || fallbackSample?.metrics || heldSample?.metrics || state.lastMetrics);
}

function formatMetric(metric, value) {
  if (metric.id === "phase") return state.phase;
  if (!Number.isFinite(value)) return "--";
  if (metric.unit === "deg") return `${value.toFixed(1)} deg`;
  if (metric.unit === "/s") return value.toFixed(2);
  return value.toFixed(2);
}

function renderMetrics(metrics = {}) {
  els.phaseValue.textContent = state.phase;
  els.phaseCard.dataset.phase = state.phase;
  els.phaseConfidence.value = state.confidence;
  els.metricBoard.replaceChildren(
    ...metricDefs.map((metric) => {
      const value = metric.id === "phase" ? state.phase : metrics[metric.id];
      const card = document.createElement("div");
      card.className = "metric-card";
      if (metric.good && Number.isFinite(value)) card.classList.add(metric.good(value) ? "good" : "warn");
      card.innerHTML = `<span class="metric-name"></span><strong class="metric-value"></strong>`;
      card.querySelector(".metric-name").textContent = metric.label;
      card.querySelector(".metric-value").textContent = formatMetric(metric, value);
      return card;
    })
  );
}

function installPwa() {
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
}

function bindDrag() {
  els.pipCanvas.addEventListener("pointerdown", (event) => {
    els.pipCanvas.setPointerCapture(event.pointerId);
    const rect = els.pipCanvas.getBoundingClientRect();
    state.drag = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  });
  els.pipCanvas.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    const parent = els.pipCanvas.parentElement.getBoundingClientRect();
    const width = els.pipCanvas.offsetWidth;
    const height = els.pipCanvas.offsetHeight;
    const left = Math.min(parent.width - width - 6, Math.max(6, event.clientX - parent.left - state.drag.x));
    const top = Math.min(parent.height - height - 6, Math.max(6, event.clientY - parent.top - state.drag.y));
    els.pipCanvas.style.left = `${left}px`;
    els.pipCanvas.style.top = `${top}px`;
    els.pipCanvas.style.right = "auto";
  });
  els.pipCanvas.addEventListener("pointerup", () => {
    state.drag = null;
  });
}

els.cameraBtn.addEventListener("click", () => startCamera().catch((error) => {
  const message = cameraErrorMessage(error);
  els.cameraBtn.disabled = false;
  els.cameraBtn.textContent = "啟動相機";
  els.cameraStatus.textContent = message;
  els.cameraStatus.title = message;
  logEvent(`相機錯誤：${message}`);
}));
els.switchBtn.addEventListener("click", () => switchCamera().catch((error) => logEvent(`切換鏡頭失敗：${error.message}`)));
els.delayBtn.addEventListener("click", () => {
  state.delayEnabled = !state.delayEnabled;
  if (!state.delayEnabled) {
    stopDelayRecorder(false);
    clearDelayBuffer();
    state.delayFallbackActive = false;
  } else if (state.running) {
    state.delayRecorderSupported = !!window.MediaRecorder && !isIOSDevice();
    state.delayFallbackActive = (!state.delayRecorderSupported || isIOSDevice()) && state.delayFallbackSupported;
    state.delaySupportWarningShown = false;
    state.delayFallbackWarningShown = false;
    state.delayRecorderGeneration += 1;
    if (state.delayFallbackActive) {
      logEvent(isIOSDevice() ? "iPhone 使用相容延遲回看" : "壓縮錄影不可用，使用相容延遲回看");
      state.delayFallbackWarningShown = true;
    } else if (!state.delayRecorderSupported) {
      logEvent("此瀏覽器不支援延遲回看，改顯示即時畫面");
      state.delaySupportWarningShown = true;
    } else {
      startDelayRecorderSegment();
    }
  }
  setStatus();
});
els.pipBtn.addEventListener("click", () => {
  state.pipEnabled = !state.pipEnabled;
  setStatus();
});
els.poseBtn.addEventListener("click", () => {
  state.poseEnabled = !state.poseEnabled;
  setStatus();
});
els.delaySlider.addEventListener("input", () => {
  state.delayMs = Number(els.delaySlider.value) * 1000;
  if (state.delayMs < DELAY_SEGMENT_MS) {
    stopDelayRecorder(false);
    clearDelayBuffer();
  } else {
    pruneDelayBuffer(performance.now());
    if (state.running && state.delayEnabled) startDelayRecorderSegment();
  }
  setStatus();
});
els.rotateBtn.addEventListener("click", () => els.metricBoard.classList.toggle("rotated"));
window.addEventListener("resize", syncDisplayFrameAspect);
window.addEventListener("orientationchange", syncDisplayFrameAspect);

bindDrag();
installPwa();
renderMetrics();
syncDisplayFrameAspect();
setStatus();
clearCanvas();
