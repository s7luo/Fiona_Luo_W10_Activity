/* EDIT HERE FIRST.
 * These are design choices, not scientific definitions of emotion.
 * Keep each label's id unique. All controls, counts, and bars use this list.
 */
window.WorkshopConfig = {
  // Keep these pinned versions aligned with the CDN URLs and header in index.html.
  versions: { p5: "2.4.0", ml5: "1.4.0" },

  // IDs are the stable values stored in datasets and model files; names and
  // prompts are only the labels shown to the person using the workshop.
  labels: [
    { id: "resting", name: "Resting", prompt: "Let your face rest comfortably." },
    { id: "smile", name: "Smile", prompt: "Perform a comfortable smile. Try small and broad versions across bursts." },
    { id: "frown", name: "Frown", prompt: "Perform a frown. Try small and broad versions across bursts." },
    { id: "wide_eyes", name: "Wide eyes", prompt: "Open your eyes wider, if comfortable." },
  ],

  // A burst takes a short pause, samples only new tracker results, and stops
  // after maxBurstSeconds if it cannot collect the requested number.
  collection: { countdownSeconds: 2, samplesPerBurst: 24, samplesPerSecond: 8, maxBurstSeconds: 20, minPerLabel: 24 },

  // Settings for the custom classifier trained on the six FaceFeatures ratios.
  training: { epochs: 60, batchSize: 16, learningRate: 0.02, hiddenUnits: 24 },

  // The response gate smooths model scores and requires a sustained threshold
  // crossing; these are interaction rules, not calibrated emotion values.
  gate: { enter: 0.7, exit: 0.45, holdMs: 300, smoothing: 0.25 },

  // Old camera results are rejected, and very small faces are not measured.
  tracking: { staleMs: 650, minimumFaceWidth: 110 },
};
