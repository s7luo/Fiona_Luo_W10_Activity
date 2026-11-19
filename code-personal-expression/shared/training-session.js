/* Each burst is one intentional recording round, not 24 independent people.
 * Samples are staged in a temporary buffer, then committed together.
 * Cancelling a burst never leaves a half-recorded class in the dataset.
 * The sequence and time checks limit duplicate frames and cap the effective
 * sampling rate even when the application render loop is faster.
 */
class TrainingSession {
  constructor(settings) { this.settings = settings; this.bursts = []; this.active = null; this.serial = 0; }
  begin(label, source, now = performance.now()) {
    if (this.active) throw new Error("Finish or cancel the current burst first.");
    // Start time includes the countdown; samples cannot enter the buffer early.
    this.active = { label, source, startsAt: now + this.settings.countdownSeconds * 1000,
      samples: [], lastSequence: -1, lastSampleAt: -Infinity, id: ++this.serial };
  }
  tick(input, now = performance.now()) {
    const b = this.active;
    if (!b) return null;
    if (now < b.startsAt) return { kind: "countdown", seconds: Math.ceil((b.startsAt - now) / 1000) };
    // A burst that cannot reach its quota within the limit is discarded whole.
    if (now - b.startsAt > this.settings.maxBurstSeconds * 1000) {
      this.active = null; return { kind: "timeout", message: "Burst cancelled: not enough valid input. Reposition and try again." };
    }
    // Require a new tracker sequence and enforce the configured sampling interval.
    if (input.valid && input.source === b.source && input.sequence !== b.lastSequence &&
        now - b.lastSampleAt >= 1000 / this.settings.samplesPerSecond) {
      b.samples.push([...input.values]); b.lastSequence = input.sequence; b.lastSampleAt = now;
    }
    if (b.samples.length >= this.settings.samplesPerBurst) {
      // Commit only complete bursts so counts and undo always refer to atomic units.
      this.bursts.push({ id: b.id, label: b.label, source: b.source, samples: b.samples });
      this.active = null; return { kind: "complete", count: b.samples.length };
    }
    return { kind: "recording", count: b.samples.length, reason: input.valid ? "" : input.reason };
  }
  cancel() { this.active = null; }
  // Undo removes a whole committed burst, not an arbitrary number of samples.
  undo() { return this.bursts.pop(); }
  reset() { this.bursts = []; this.active = null; }
  // Counts are observations, while the interface separately reports burst count.
  counts(labels) { return Object.fromEntries(labels.map(l => [l, this.bursts.filter(b => b.label === l).reduce((n, b) => n + b.samples.length, 0)])); }
  // Flatten only at training time; stored bursts retain their boundaries.
  examples() { return this.bursts.flatMap(b => b.samples.map(values => ({ values, label: b.label }))); }
}
window.TrainingSession = TrainingSession;
