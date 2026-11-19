/* Prediction is not yet a designed response.
 * Separate entry/exit thresholds reduce rapid switching ("hysteresis").
 * holdMs requires the proposed new state to last before it is accepted.
 * This class does not train the neural network or measure happiness.
 * `scores.smile` is expected to be a finite value between zero and one.
 */
class SmileGate {
  constructor(settings) { this.settings = settings; this.reset(); }
  // Start without a settled state; the first valid score initializes smoothing.
  reset() { this.smoothed = null; this.state = "uncertain"; this.candidate = null; this.since = 0; }
  update(scores, now = performance.now()) {
    const value = scores?.smile;
    if (!Number.isFinite(value)) { this.reset(); return this.state; }
    const { smoothing, enter, exit, holdMs } = this.settings;
    // Exponential moving average: larger smoothing reacts faster, smaller
    // smoothing gives recent fluctuations less influence.
    this.smoothed = this.smoothed === null ? value : this.smoothed + smoothing * (value - this.smoothed);
    let desired = this.state;
    if (this.smoothed >= enter) desired = "smiling";
    else if (this.smoothed <= exit) desired = "not smiling";
    // Between thresholds, retain a previously settled state, or uncertainty.
    // If the desired state changes, restart its hold timer rather than reusing
    // time accumulated for a different candidate.
    if (desired === this.state) { this.candidate = null; return this.state; }
    if (desired !== this.candidate) { this.candidate = desired; this.since = now; }
    if (now - this.since >= holdMs) { this.state = desired; this.candidate = null; }
    return this.state;
  }
}
window.SmileGate = SmileGate;
