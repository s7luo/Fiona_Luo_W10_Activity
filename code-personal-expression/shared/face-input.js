/* FaceMesh is pretrained. We never change its weights.
 * This class owns camera permissions, fresh results, validity, and shutdown.
 * A sequence number lets the collector avoid recording a stale result twice.
 * The token invalidates in-flight camera/model work after Stop or a restart.
 */
class FaceInput {
  constructor(settings) {
    this.settings = settings;
    this.video = document.createElement("video");
    this.video.muted = true; this.video.playsInline = true;
    this.video.setAttribute("aria-hidden", "true");
    this.video.style.display = "none";
    document.body.append(this.video);
    this.mesh = null; this.stream = null; this.faces = [];
    this.sequence = 0; this.receivedAt = 0; this.running = false; this.token = 0;
    this.detectionFrame = 0; this.trackingError = null;
  }
  async start() {
    const token = ++this.token;
    this.trackingError = null;
    try {
      // Browsers require localhost/HTTPS for camera access; never request audio.
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Use localhost/Live Server or an HTTPS page with camera permission.");
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
      // A stop/restart may happen while the permission prompt is still open.
      if (token !== this.token) { this.stream.getTracks().forEach(t => t.stop()); return; }
      this.video.srcObject = this.stream;
      await this.video.play();
      if (token !== this.token) return;
      // play() can resolve before the browser has decoded the first camera frame.
      await this.waitForVideoFrame(token);
      if (token !== this.token) return;
      // maxFaces: 2 lets us reject two-person input rather than silently choosing.
      if (!this.mesh) {
        // The pinned ml5 build exposes a thenable factory; await it to get the
        // initialized model instance before using detect() or getConnections().
        this.mesh = await ml5.faceMesh({
          runtime: "mediapipe",
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
          maxFaces: 2,
          refineLandmarks: false,
          flipped: false
        });
      }
      if (token !== this.token) return;
      this.running = true;
      // The loop is owned here so detector failures can be caught and displayed.
      this.detectFrame(token);
    } catch (error) { this.stop(); throw error; }
  }
  async detectFrame(token) {
    try {
      // Await each inference before scheduling the next frame to avoid piling
      // up work when detection is slower than the display refresh rate.
      const faces = await this.mesh.detect(this.video);
      if (!this.running || token !== this.token) return;
      this.faces = faces; this.receivedAt = performance.now(); this.sequence++;
      this.detectionFrame = requestAnimationFrame(() => {
        this.detectionFrame = 0;
        this.detectFrame(token);
      });
    } catch (error) {
      if (!this.running || token !== this.token) return;
      // A rejected inference must not leave the UI waiting forever without a
      // fresh-result callback. Stop the stream and expose the reason to latest().
      this.stop();
      this.trackingError = error?.message || String(error);
    }
  }
  waitForVideoFrame(token) {
    // HAVE_CURRENT_DATA is numeric state 2; without it, inference may receive
    // a video element with dimensions but no usable decoded image yet.
    if (this.video.readyState >= 2) return Promise.resolve();
    return new Promise((resolve, reject) => {
      let timeout;
      // Remove all listeners/timers on every completion path to avoid leaking
      // handlers across camera restarts.
      const cleanup = () => {
        clearTimeout(timeout);
        this.video.removeEventListener("loadeddata", onLoaded);
        this.video.removeEventListener("error", onError);
        this.video.removeEventListener("emptied", onEmptied);
      };
      const finish = error => {
        cleanup();
        if (error) reject(error); else resolve();
      };
      const onLoaded = () => finish();
      const onError = () => finish(new Error("The camera did not provide a usable video frame."));
      // Stop clears srcObject and emits "emptied"; resolve only if this start
      // was cancelled, otherwise report that the active stream ended early.
      const onEmptied = () => token !== this.token
        ? finish()
        : finish(new Error("The camera stopped before its first video frame."));
      this.video.addEventListener("loadeddata", onLoaded);
      this.video.addEventListener("error", onError);
      this.video.addEventListener("emptied", onEmptied);
      timeout = setTimeout(() => finish(new Error("Timed out waiting for the first camera frame.")), 10000);
      // Recheck after listeners are attached in case readiness changed meanwhile.
      if (this.video.readyState >= 2) finish();
    });
  }
  latest() {
    const now = performance.now();
    // Tracking failures are distinct from an absent face or a stopped camera.
    if (this.trackingError) return { valid: false, reason: `Face tracking failed: ${this.trackingError}` };
    if (!this.running) return { valid: false, reason: "Camera stopped" };
    if (document.hidden) return { valid: false, reason: "Tab hidden" };
    if (now - this.receivedAt > this.settings.staleMs) return { valid: false, reason: "Waiting for fresh tracking" };
    // Reject both zero and multiple faces; never choose one person silently.
    if (this.faces.length !== 1) return { valid: false, reason: this.faces.length ? "One person at a time" : "No face detected" };
    const measured = FaceFeatures.extract(this.faces[0]);
    if (!measured || measured.faceWidth < this.settings.minimumFaceWidth) return { valid: false, reason: "Move closer and face the camera" };
    return { valid: true, ...measured, face: this.faces[0], sequence: this.sequence, at: this.receivedAt, source: "camera" };
  }
  stop() {
    this.running = false; this.token++;
    // Cancel a queued frame; an already-running detect() is ignored by its token.
    if (this.detectionFrame) cancelAnimationFrame(this.detectionFrame);
    this.detectionFrame = 0;
    // Always release the device stream when leaving camera mode or the page.
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null; this.video.srcObject = null;
    this.faces = []; this.receivedAt = 0; this.trackingError = null;
  }
}
window.FaceInput = FaceInput;
