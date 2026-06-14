const CACHE_NAME = "archer-pose-studio-v41";
const ASSETS = [
  "./",
  "./index.html",
  "./styles.css",
  "./app.js",
  "./manifest.webmanifest",
  "./icon.svg",
  "./models/pose_landmarker_full.task",
  "./vendor/mediapipe/tasks-vision/vision_bundle.mjs",
  "./vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.js",
  "./vendor/mediapipe/tasks-vision/wasm/vision_wasm_internal.wasm",
  "./vendor/mediapipe/tasks-vision/wasm/vision_wasm_module_internal.js",
  "./vendor/mediapipe/tasks-vision/wasm/vision_wasm_module_internal.wasm",
  "./vendor/mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.js",
  "./vendor/mediapipe/tasks-vision/wasm/vision_wasm_nosimd_internal.wasm"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

async function cacheFirst(request) {
  const cached = await caches.match(request, { ignoreSearch: true });
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
  }
  return response;
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    return (await caches.match(request, { ignoreSearch: true })) || caches.match("./index.html");
  }
}

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  const isAppShell =
    event.request.mode === "navigate" ||
    url.pathname.endsWith("/app.js") ||
    url.pathname.endsWith("/styles.css") ||
    url.pathname.endsWith("/index.html");
  event.respondWith(isAppShell ? networkFirst(event.request) : cacheFirst(event.request));
});
