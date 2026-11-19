/* A small custom neural network, separate from pretrained FaceMesh.
 * This wrapper uses ml5's public API for training, inference, save, and load.
 * Keep feature names/order and labels identical when reusing a saved model.
 * `ready` means the classifier can infer; `busy` prevents overlapping
 * training/loading/saving operations from changing its state unexpectedly.
 */
class ExpressionModel {
  constructor(labels, settings) {
    this.labels = [...labels]; this.settings = settings;
    this.network = null; this.ready = false; this.busy = false; this.source = null;
  }
  create() {
    // Named input fields make normalization metadata portable between sessions.
    return ml5.neuralNetwork({
      inputs: [...FaceFeatures.schema.names], outputs: ["expression"],
      task: "classification", debug: false,
      learningRate: this.settings.learningRate, hiddenUnits: this.settings.hiddenUnits
    });
  }
  inputs(values) {
    // Persisted arrays are positional, so reject wrong lengths and non-numbers
    // before mapping each position to the schema's stable feature name.
    if (!Array.isArray(values) || values.length !== FaceFeatures.schema.names.length || !values.every(Number.isFinite))
      throw new Error("Feature vector does not match the six-ratio schema.");
    return Object.fromEntries(FaceFeatures.schema.names.map((name, i) => [name, values[i]]));
  }
  async train(examples, source, onEpoch = () => {}) {
    if (this.busy) throw new Error("Model is busy.");
    this.busy = true; this.ready = false;
    this.disposeNetwork();
    const nn = this.create(); this.network = nn;
    try {
      // Keep labels categorical and normalize using the training examples only.
      for (const example of examples) {
        if (!this.labels.includes(example.label)) throw new Error("Unexpected training label.");
        nn.addData(this.inputs(example.values), { expression: example.label });
      }
      nn.normalizeData();
      // Passing both callbacks exposes progress; the returned Promise also
      // lets this wrapper handle errors in ml5 1.4.0.
      await nn.train({ epochs: this.settings.epochs, batchSize: this.settings.batchSize,
        validationSplit: 0, shuffle: true }, (epoch, logs) => onEpoch(epoch, logs), () => {});
      this.ready = true; this.source = source;
    } catch (error) { this.disposeNetwork(); throw error; }
    finally { this.busy = false; }
  }
  async classify(values) {
    if (!this.ready || this.busy) return null;
    const results = await this.network.classify(this.inputs(values));
    // ml5 sorts predictions by score. Match by label, never by result position.
    return Object.fromEntries(this.labels.map(label => [label, results.find(r => r.label === label)?.confidence || 0]));
  }
  async save() {
    if (!this.ready) throw new Error("Train or load a model first.");
    // ml5 writes the network, weights, and its own normalization/label metadata.
    // Workshop-specific provenance and response settings are saved separately.
    this.busy = true;
    try { await this.network.save("model"); }
    finally { this.busy = false; }
  }
  async load(files, manifest) {
    const expected = ["model.json", "model_meta.json", "model.weights.bin"];
    for (const name of expected) if (!files.some(f => f.name === name)) throw new Error(`Missing ${name}. Keep the original filenames.`);
    if (manifest.schema !== FaceFeatures.schema.id ||
        JSON.stringify(manifest.features) !== JSON.stringify(FaceFeatures.schema.names) ||
        JSON.stringify(manifest.labels) !== JSON.stringify(this.labels) ||
        !["camera", "synthetic"].includes(manifest.source))
      throw new Error("This model has incompatible labels, features, or source information.");
    // Validate ml5 metadata before loading; matching filenames alone do not
    // establish that an export has this workshop's feature and label schema.
    const meta = JSON.parse(await files.find(f => f.name === "model_meta.json").text());
    if (JSON.stringify(Object.keys(meta.inputs || {})) !== JSON.stringify(FaceFeatures.schema.names) ||
        !meta.outputs?.expression?.legend ||
        JSON.stringify(Object.keys(meta.outputs.expression.legend).sort()) !== JSON.stringify([...this.labels].sort()))
      throw new Error("Model metadata does not match this workshop's inputs and labels.");
    this.busy = true; this.ready = false; this.disposeNetwork();
    // ml5.load expects URLs. Revoke each temporary URL even when loading fails.
    const urls = Object.fromEntries(files.map(f => [f.name, URL.createObjectURL(f)]));
    try {
      this.network = this.create();
      await this.network.load({ model: urls["model.json"], metadata: urls["model_meta.json"], weights: urls["model.weights.bin"] });
      this.ready = true; this.source = manifest.source;
    } catch (error) { this.disposeNetwork(); throw error; }
    finally { this.busy = false; Object.values(urls).forEach(URL.revokeObjectURL); }
  }
  disposeNetwork() {
    // ml5 has no public dispose method. Limit this version-specific cleanup to
    // this wrapper; global TensorFlow disposal would also affect FaceMesh.
    const nn = this.network;
    if (nn) {
      nn.neuralNetwork?.model?.optimizer?.dispose?.();
      nn.neuralNetwork?.model?.dispose?.();
      const tensors = nn.data?.training;
      if (tensors) ml5.tf.dispose(tensors);
    }
    this.network = null; this.ready = false;
  }
  // Reset only the custom classifier; FaceMesh is pretrained and kept separate.
  reset() { this.disposeNetwork(); this.source = null; }
}
window.ExpressionModel = ExpressionModel;
