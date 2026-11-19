/* Application glue. Read the small shared classes before this interface file.
 * Three modes stay visibly distinct:
 *  - camera: numerical features extracted from actual tracked landmarks;
 *  - synthetic: authored feature vectors, with a genuinely trained network;
 *  - design test: authored class scores, bypassing the classifier entirely.
 * This class wires those independent components to the DOM and owns the
 * session-level rules that prevent mixing data sources or stale predictions.
 */
class WorkshopApp {
  constructor() {
    this.config = WorkshopConfig;
    this.labels = this.config.labels.map(l => l.id);
    if (new Set(this.labels).size !== this.labels.length || !this.labels.includes("smile")) throw new Error("Use unique label IDs, including smile.");
    this.face = new FaceInput(this.config.tracking);
    this.collector = new TrainingSession(this.config.collection);
    this.model = new ExpressionModel(this.labels, this.config.training);
    this.gate = new SmileGate({ ...this.config.gate });
    // `source` is fixed by the first completed burst/model load and guards
    // against training or testing with a different kind of input by accident.
    this.mode = "camera"; this.step = 1; this.source = null;
    this.input = { valid: false, reason: "No input" };
    // Generation/sequence values invalidate asynchronous inference results
    // after a reset, source change, or newer camera observation.
    this.scores = null; this.design = null; this.inferenceBusy = false;
    this.lastSequence = -1; this.lastPredictionAt = 0; this.generation = 0;
    this.syntheticSequence = 0; this.syntheticAt = 0; this.syntheticPaused = false;
    this.lastRender = 0; this.cameraLoading = false;
  }
  el(id) { return document.getElementById(id); }
  status(text, error = false) { this.el("status").textContent = text; this.el("status").classList.toggle("error", error); }
  start() {
    // Build repeated options, counts, and score rows from configuration. Adding
    // a label then changes data requirements, not a fixed number of UI elements.
    for (const label of this.config.labels) {
      this.el("label").add(new Option(label.name, label.id));
      this.el("synthetic-profile").add(new Option(label.name, label.id));
      const row = document.createElement("tr"); row.id = `count-${label.id}`;
      row.innerHTML = `<td></td><td>0</td><td>0</td>`; row.cells[0].textContent = label.name;
      this.el("counts").append(row);
      const score = document.createElement("div"); score.className = "score-row"; score.id = `score-${label.id}`;
      score.innerHTML = `<span></span><div class="score-track"><div></div></div><output>—</output>`;
      score.firstElementChild.textContent = label.name; this.el("scores").append(score);
    }
    for (const name of FaceFeatures.schema.names) {
      const row = document.createElement("tr"); row.innerHTML = `<td></td><td>—</td>`;
      row.cells[0].textContent = name; this.el("features").append(row);
    }
    // Step controls only change visible panels; session data stays in memory.
    document.querySelectorAll("[data-step]").forEach(b => b.addEventListener("click", () => this.setStep(Number(b.dataset.step))));
    this.bind("camera-start", async () => {
      this.cameraLoading = true; this.render(); this.status("Requesting camera access and loading pretrained FaceMesh…");
      try { await this.face.start(); if (this.face.running) this.status("Camera started. Wait for fresh landmarks."); }
      finally { this.cameraLoading = false; this.render(); }
    });
    this.bind("camera-stop", () => { this.face.stop(); this.invalidatePredictions(); this.status("Camera stopped. Existing examples stay in this page."); });
    this.bind("input-mode", () => this.changeMode(), "change");
    this.bind("label", () => this.updatePrompt(), "change");
    this.bind("record", () => this.beginBurst());
    this.bind("cancel-burst", () => { this.collector.cancel(); this.el("burst-progress").textContent = ""; this.status("Burst cancelled. Its temporary examples were not added."); });
    this.bind("undo", () => { this.collector.undo(); this.invalidateModel(); if (!this.collector.bursts.length) this.source = null; this.status("Removed the most recent burst. Retrain after dataset changes."); });
    this.bind("train", () => this.train());
    this.bind("dataset-save", () => this.download(this.dataset(), "dataset.json"));
    this.bind("dataset-load", async e => {
      // File parsing and schema validation happen before replacing session data.
      const file = e.target.files[0]; if (file) await this.loadDataset(JSON.parse(await file.text()));
      e.target.value = "";
    }, "change");
    this.bind("load-practice", async () => {
      // Practice examples are authored numbers, not camera observations.
      if (this.collector.bursts.length || this.model.ready) throw new Error("Reset the session before loading the practice dataset.");
      const response = await fetch("data/synthetic-practice.json");
      if (!response.ok) throw new Error("Practice dataset could not be loaded.");
      await this.loadDataset(await response.json());
      this.status("Synthetic practice dataset loaded. Train a real model on these authored numerical examples.");
    });
    this.bind("synthetic-variation", () => this.el("variation-value").value = Number(this.el("synthetic-variation").value).toFixed(2), "input");
    this.bind("synthetic-missing", () => {
      // This control tests how missing input affects predictions and collection.
      this.syntheticPaused = !this.syntheticPaused;
      this.el("synthetic-missing").textContent = this.syntheticPaused ? "Resume synthetic input" : "Simulate missing input";
      this.invalidatePredictions();
    });
    this.bind("hold-ms", () => {
      this.gate.settings.holdMs = Number(this.el("hold-ms").value);
      this.el("hold-value").value = `${this.gate.settings.holdMs} ms`; this.gate.reset();
    }, "input");
    document.querySelectorAll("[data-design]").forEach(b => this.bindElement(b, () => {
      // Design-test scores bypass both feature extraction and the classifier.
      this.generation++; this.design = b.dataset.design; this.gate.reset(); this.status("DESIGN TEST / authored scores. No inference or training.");
    }));
    this.bind("design-live", () => {
      this.design = null; this.invalidatePredictions(); this.status("Returned to model inference. Waiting for fresh input.");
    });
    this.bind("model-save", async () => {
      await this.model.save();
      // The session manifest complements ml5's three network/metadata files.
      this.download(this.manifest(), "session.json");
      this.status("Requested 4 model-set downloads. Check that all 4 arrived; allow multiple downloads if your browser asks.");
    });
    this.bind("model-load", async e => {
      // All four files must come from one export; the manifest is the source
      // of truth for feature schema, labels, source, and response settings.
      const files = [...e.target.files]; e.target.value = "";
      const manifestFile = files.find(f => f.name === "session.json");
      if (!manifestFile) throw new Error("Select session.json and all three model files together.");
      const manifest = JSON.parse(await manifestFile.text());
      if (manifest.format !== "personal-expression-model-v1") throw new Error("Not a workshop session manifest.");
      if (!manifest.gate || !["enter","exit","holdMs","smoothing"].every(k => Number.isFinite(manifest.gate[k])) ||
          manifest.gate.enter > 1 || manifest.gate.exit < 0 || manifest.gate.enter <= manifest.gate.exit ||
          manifest.gate.holdMs < 0 || manifest.gate.holdMs > 1500 ||
          manifest.gate.smoothing <= 0 || manifest.gate.smoothing > 1) throw new Error("Invalid response settings in session.json.");
      if (this.collector.bursts.length) throw new Error("Reset before loading a model so old dataset examples are not confused with its training data.");
      this.invalidatePredictions(); this.render();
      await this.model.load(files, manifest);
      this.source = manifest.source; this.changeModeTo(this.source); this.design = null;
      this.gate.settings = { ...this.config.gate, ...manifest.gate };
      this.el("hold-ms").value = this.gate.settings.holdMs; this.el("hold-value").value = `${this.gate.settings.holdMs} ms`;
      this.setStep(3); this.status("Saved model and normalization metadata loaded. No training examples were loaded.");
    }, "change");
    this.bind("reset", () => this.el("reset-confirm").hidden = false);
    this.bind("reset-no", () => this.el("reset-confirm").hidden = true);
    this.bind("reset-yes", () => this.reset());
    document.addEventListener("visibilitychange", () => {
      // Hidden-tab time must not count as fresh inference or finish a burst.
      this.invalidatePredictions();
      if (document.hidden && this.collector.active) { this.collector.cancel(); this.el("burst-progress").textContent = ""; }
    });
    window.addEventListener("pagehide", () => this.face.stop());
    this.updatePrompt(); this.setStep(1); this.render();
  }
  bind(id, fn, event = "click") { this.bindElement(this.el(id), fn, event); }
  bindElement(element, fn, event = "click") {
    element.addEventListener(event, async e => {
      // Route rejected async event handlers to the visible status area rather
      // than leaving a failed operation as an unhandled promise rejection.
      try { await fn(e); } catch (error) { this.status(error.message || String(error), true); }
      this.render();
    });
  }
  setStep(step) {
    this.step = step;
    document.querySelectorAll("[data-view]").forEach(v => v.hidden = Number(v.dataset.view) !== step);
    document.querySelectorAll(".examples [data-step]").forEach(b => {
      if (Number(b.dataset.step) === step) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    });
    const titles = ["Coordinates are not feelings.", "Our labels become training examples.", "A prediction becomes a designed response."];
    const descriptions = [
      "Inspect the six measurements we designed from FaceMesh landmarks. Nothing is being recorded yet.",
      "Record short, varied performances under labels you choose. Then train a separate, personal classifier.",
      "Test fresh input, inspect raw scores, and decide how stable the response should be. Save your model for later reuse."
    ];
    this.el("step-title").textContent = titles[step-1]; this.el("step-description").textContent = descriptions[step-1];
    // Design scores exist only in the testing view; clear them when leaving it.
    if (step !== 3) { this.design = null; this.invalidatePredictions(); }
    this.render();
  }
  updatePrompt() { this.el("expression-prompt").textContent = this.config.labels.find(l => l.id === this.el("label").value)?.prompt || ""; }
  changeMode() {
    const next = this.el("input-mode").value;
    // A data source cannot be changed after collection/model use without reset.
    if (this.collector.active || this.model.busy || this.source && next !== this.source) {
      this.el("input-mode").value = this.mode;
      throw new Error("Do not mix camera and synthetic examples. Finish the current operation, then reset before changing source.");
    }
    this.changeModeTo(next); this.status(next === "synthetic" ? "Synthetic input selected. No webcam or FaceMesh is used." : "Webcam input selected. Start the camera when ready.");
  }
  changeModeTo(mode) {
    this.face.stop(); this.mode = mode; this.el("input-mode").value = mode;
    this.el("synthetic-controls").hidden = mode !== "synthetic";
    this.syntheticPaused = false; this.el("synthetic-missing").textContent = "Simulate missing input";
    this.invalidatePredictions();
  }
  syntheticInput(now) {
    if (this.syntheticPaused || document.hidden) return { valid: false, reason: "Synthetic input paused" };
    // These six-number profiles are authored, not captured or representative
    // people. Deterministic jitter makes the practice route reproducible.
    const profiles = {
      resting: [.34,.035,0,.27,.27,.12], smile: [.44,.12,.07,.21,.21,.12],
      frown: [.30,.025,-.06,.23,.23,.085], wide_eyes: [.34,.08,0,.42,.42,.19],
      mouth_open: [.34,.65,0,.28,.28,.13]
    };
    if (now - this.syntheticAt > 90 || !this.syntheticValues) {
      this.syntheticAt = now; this.syntheticSequence++;
      const base = profiles[this.el("synthetic-profile").value] || profiles.resting;
      const variation = Number(this.el("synthetic-variation").value);
      this.syntheticValues = base.map((v,i) => v + Math.sin(this.syntheticSequence * 1.73 + i * 2.31) * .035 * variation);
    }
    return { valid: true, values: this.syntheticValues, sequence: this.syntheticSequence, at: this.syntheticAt, source: "synthetic" };
  }
  beginBurst() {
    // Validate the live input and data provenance before starting the countdown.
    if (!this.input.valid) throw new Error("Wait for valid input before recording.");
    if (this.model.busy) throw new Error("Wait for training to finish.");
    if (this.source && this.source !== this.mode) throw new Error("Reset before changing dataset source.");
    this.design = null;
    this.collector.begin(this.el("label").value, this.mode);
    this.status("Get ready. Only fresh valid measurements will be recorded.");
  }
  invalidatePredictions() {
    // Bump generation so an already-running classify() cannot restore stale scores.
    this.generation++; this.scores = null; this.lastPredictionAt = 0;
    this.lastSequence = -1; this.gate.reset();
  }
  invalidateModel() {
    this.invalidatePredictions(); this.model.reset();
    this.el("training-progress").value = 0; this.el("training-text").textContent = "Dataset changed. Train a new model.";
  }
  async train() {
    // Training is enabled only after every configured class has enough examples.
    if (this.collector.active) throw new Error("Finish or cancel the recording burst first.");
    const counts = this.collector.counts(this.labels);
    if (!this.labels.every(l => counts[l] >= this.config.collection.minPerLabel)) throw new Error("Record at least one full burst for every label.");
    this.invalidatePredictions(); this.design = null; this.status("Training the custom classifier. FaceMesh is unchanged."); this.render();
    const job = this.model.train(this.collector.examples(), this.source, (epoch, logs) => {
      this.el("training-progress").value = epoch + 1;
      const loss = typeof logs === "number" ? logs : logs?.loss;
      this.el("training-text").textContent = `Epoch ${epoch+1}/${this.config.training.epochs}${Number.isFinite(loss) ? ` · training loss ${loss.toFixed(4)}` : ""}`;
    });
    this.render(); await job;
    this.status("Training complete. Test fresh performances in Example 03; training loss is not proof of generalisation.");
    this.setStep(3);
  }
  update() {
    const now = performance.now();
    this.input = this.mode === "camera" ? this.face.latest() : this.syntheticInput(now);
    // Tick collection independently of drawing; only valid, new observations
    // can advance the active burst.
    const event = this.collector.tick(this.input, now);
    if (event?.kind === "countdown") this.el("burst-progress").textContent = `Get ready: ${event.seconds}`;
    if (event?.kind === "recording") this.el("burst-progress").textContent = `Recording ${event.count}/${this.config.collection.samplesPerBurst}${event.reason ? ` · ${event.reason}` : ""}`;
    if (event?.kind === "complete") {
      this.source = this.mode; this.invalidateModel();
      this.el("burst-progress").textContent = `Added ${event.count} examples.`;
      this.status("Burst added. Vary the performance before the next burst."); this.render();
    }
    if (event?.kind === "timeout") { this.el("burst-progress").textContent = ""; this.status(event.message, true); this.render(); }
    if (this.step === 3) {
      if (this.design) {
        // The design test supplies fixed scores to the gate and never calls ml5.
        const smile = this.design === "smile" ? .95 : this.design === "resting" ? .05 : .57;
        this.scores = Object.fromEntries(this.labels.map(l => [l, l === "smile" ? smile : (1-smile)/(this.labels.length-1)]));
        this.gate.update(this.scores, now);
      } else if (!this.input.valid || this.model.source !== this.mode || !this.model.ready) {
        if (this.scores) this.invalidatePredictions();
      } else if (this.model.ready && !this.model.busy && !this.inferenceBusy && this.input.sequence !== this.lastSequence) {
        // Classify each fresh input at most once. `generation` and capture time
        // guard against results arriving after source/state changes or staleness.
        this.lastSequence = this.input.sequence;
        const token = this.generation, capturedAt = this.input.at;
        this.inferenceBusy = true;
        this.model.classify(this.input.values).then(scores => {
          if (token !== this.generation || !scores || this.design || !this.input.valid || document.hidden ||
              performance.now() - capturedAt > this.config.tracking.staleMs) return;
          this.scores = scores; this.lastPredictionAt = performance.now(); this.gate.update(scores);
        }).catch(error => { if (token === this.generation) { this.invalidatePredictions(); this.status(`Prediction failed: ${error.message}`, true); } })
          .finally(() => this.inferenceBusy = false);
      }
      if (!this.design && this.scores && now - this.lastPredictionAt > this.config.tracking.staleMs) this.invalidatePredictions();
    }
    if (now - this.lastRender > 120) { this.render(); this.lastRender = now; }
  }
  render() {
    // Keep controls and readouts derived from current state instead of updating
    // them ad hoc in every event handler.
    const active = !!this.collector.active, busy = this.model.busy, locked = active || busy;
    this.el("camera-start").disabled = this.mode !== "camera" || this.face.running || this.cameraLoading || busy;
    this.el("camera-stop").disabled = !this.face.running && !this.cameraLoading;
    this.el("record").disabled = locked || !this.input.valid;
    this.el("label").disabled = locked; this.el("input-mode").disabled = locked || this.cameraLoading;
    this.el("cancel-burst").disabled = !active;
    this.el("undo").disabled = locked || !this.collector.bursts.length;
    this.el("dataset-save").disabled = locked || !this.collector.bursts.length;
    this.el("dataset-load").disabled = locked;
    this.el("model-load").disabled = locked;
    this.el("load-practice").disabled = locked;
    this.el("reset").disabled = busy || this.cameraLoading;
    this.el("reset-yes").disabled = busy || this.cameraLoading;
    this.el("model-save").disabled = busy || !this.model.ready;
    const counts = this.collector.counts(this.labels);
    this.el("train").disabled = locked || !this.labels.every(l => counts[l] >= this.config.collection.minPerLabel);
    for (const label of this.labels) {
      const row = this.el(`count-${label}`);
      row.cells[1].textContent = this.collector.bursts.filter(b => b.label === label).length;
      row.cells[2].textContent = counts[label];
      const score = this.el(`score-${label}`), value = this.scores?.[label];
      score.querySelector(".score-track div").style.width = `${Number.isFinite(value) ? value*100 : 0}%`;
      score.querySelector("output").textContent = Number.isFinite(value) ? `${(value*100).toFixed(0)}%` : "—";
    }
    this.el("features").querySelectorAll("tr").forEach((row,i) => row.cells[1].textContent = this.input.valid ? this.input.values[i].toFixed(4) : "—");
    this.el("source-badge").textContent = this.mode === "camera" ? "WEBCAM / measured features" : "SYNTHETIC / authored features";
    this.el("tracking").textContent = this.input.valid ? "Valid input" : this.input.reason;
    const sourceName = this.model.source === "synthetic" ? "synthetic examples (not faces)" : "this visitor’s recorded facial measurements";
    this.el("model-provenance").textContent = this.model.ready ? `Model ready · trained from ${sourceName}. Dataset changes require retraining.` : "Train a model in Example 02, or load a complete saved model set.";
    const prefix = this.design ? "DESIGN TEST / " : this.mode === "synthetic" ? "SYNTHETIC / " : "";
    this.el("gate-state").textContent = prefix + (this.scores ? this.gate.state : this.model.ready ? "Waiting for valid input" : "Waiting for a model");
    this.el("smile-score").textContent = this.scores ? `Raw smile score ${this.scores.smile.toFixed(3)} · smoothed ${this.gate.smoothed?.toFixed(3) ?? "—"}` : "No current prediction. Scores clear when tracking is missing or stale.";
    this.el("training-progress").max = this.config.training.epochs;
  }
  drawPreview() {
    push();
    if (this.mode === "camera" && this.face.running && this.face.video.readyState >= 2) {
      // Mirror only the display. Feature extraction always uses unflipped data,
      // so the measurement values do not depend on presentation orientation.
      translate(width,0); scale(-1,1);
      // Canvas accepts a native HTMLVideoElement directly. This avoids making
      // a p5.MediaElement just to display the video owned by FaceInput.
      drawingContext.drawImage(this.face.video,0,0,width,height);
      if (this.el("landmarks").checked && this.face.faces.length) {
        const sx = width/this.face.video.videoWidth, sy = height/this.face.video.videoHeight;
        const context = drawingContext;
        // ml5 returns landmark-index pairs for the wireframe; scale each point
        // from camera pixels to the canvas before drawing its connecting edge.
        const connections = this.face.mesh.getConnections();
        context.beginPath();
        for (const face of this.face.faces) {
          for (const [a, b] of connections) {
            const pointA = face.keypoints[a], pointB = face.keypoints[b];
            if (!pointA || !pointB) continue;
            context.moveTo(pointA.x*sx, pointA.y*sy);
            context.lineTo(pointB.x*sx, pointB.y*sy);
          }
        }
        context.strokeStyle = "#c4e8fb";
        context.lineWidth = 0.75;
        context.stroke();
      }
    } else {
      fill("#eff5f8"); noStroke(); textAlign(CENTER,CENTER); textSize(24);
      text(this.mode === "synthetic" ? "SYNTHETIC FEATURES" : "Your camera is off",width/2,height/2-20);
      textSize(16); text(this.mode === "synthetic" ? "No face · no video · real custom training" : "Start only when you are ready.",width/2,height/2+20);
    }
    pop();
  }
  manifest() {
    // Keep workshop-level metadata separate from ml5's network export format.
    return { format: "personal-expression-model-v1", schema: FaceFeatures.schema.id,
      features: [...FaceFeatures.schema.names], labels: [...this.labels], source: this.model.source,
      gate: { ...this.gate.settings }, versions: { ...this.config.versions } };
  }
  dataset() {
    // Preserve bursts and provenance so imported data retains its sampling units.
    return { format: "personal-expression-dataset-v1", schema: FaceFeatures.schema.id,
      features: [...FaceFeatures.schema.names], labels: [...this.labels], source: this.source, bursts: this.collector.bursts };
  }
  async loadDataset(data) {
    // Validate the complete file before assigning it; loading never merges sets.
    if (this.collector.bursts.length || this.model.ready) throw new Error("Reset before loading a dataset. Loading does not merge datasets.");
    if (data.format !== "personal-expression-dataset-v1" || data.schema !== FaceFeatures.schema.id ||
        JSON.stringify(data.features) !== JSON.stringify(FaceFeatures.schema.names) ||
        JSON.stringify(data.labels) !== JSON.stringify(this.labels) || !["camera","synthetic"].includes(data.source) ||
        !Array.isArray(data.bursts) || !data.bursts.length || data.bursts.length > 500)
      throw new Error("Dataset schema, labels, or source do not match this example.");
    for (const b of data.bursts) {
      if (!this.labels.includes(b.label) || b.source !== data.source || !Array.isArray(b.samples) || !b.samples.length || b.samples.length > 1000)
        throw new Error("Dataset contains an invalid burst.");
      b.samples.forEach(v => this.model.inputs(v));
    }
    this.invalidateModel(); this.collector.bursts = data.bursts;
    this.source = data.source; this.changeModeTo(this.source);
    this.setStep(2); this.status("Dataset loaded. It contains labelled measurements, not a trained model."); this.render();
  }
  download(data, name) {
    // Exports are explicit user-initiated downloads; the page stores no files.
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = name; document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }
  reset() {
    // Stop hardware and clear transient state; downloaded files are unaffected.
    this.face.stop(); this.collector.reset(); this.model.reset(); this.source = null;
    this.design = null; this.invalidatePredictions(); this.syntheticValues = null;
    this.gate.settings = { ...this.config.gate }; this.el("hold-ms").value = this.config.gate.holdMs;
    this.el("hold-value").value = `${this.config.gate.holdMs} ms`;
    this.el("training-progress").value = 0; this.el("training-text").textContent = "No custom model trained.";
    this.el("burst-progress").textContent = ""; this.el("reset-confirm").hidden = true;
    this.changeModeTo("camera"); this.setStep(1); this.status("Session cleared and camera stopped. Downloaded files are not deleted.");
  }
}
window.WorkshopApp = WorkshopApp;
