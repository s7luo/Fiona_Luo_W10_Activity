# Holding a Smile

Part one of the Week 10 GBDA 414 workshop: train a personal expression model. Open this whole folder in VS Code, start Live Server on `index.html`, and use current Chrome or another current Chromium browser with hardware acceleration enabled if available.

## Run

No package installation, build step, API key, or account is required. The page loads pinned p5.js 2.4.0 and ml5.js 1.4.0 builds from jsDelivr, so an internet connection is required to open the workshop. FaceMesh also fetches pretrained model assets when first started.

1. Open `index.html` through Live Server, not by double-clicking the file.
2. Example 01: start the camera, allow access, and wait for FaceMesh to load. Inspect its wireframe and the measurements. Nothing is collected automatically.
3. Example 02: select a label and record one burst. Repeat for each label. A burst contains 24 fresh valid observations at up to eight per second.
4. Train after at least one complete burst for every label. Three varied bursts per label are recommended, but examples remain correlated performances, not independent people.
5. Example 03: perform fresh expressions, inspect the scores, and test the response. New predictions do not enter the training set.

The three numbered buttons change views within one session. Reloading the browser clears the session unless you deliberately saved files.

## What is trained?

`FaceInput` runs pretrained FaceMesh with ml5's MediaPipe runtime; `FaceFeatures` derives six ratios from its landmarks. `ExpressionModel` trains a separate small ml5 neural network to associate those ratios with the labels in `config.js`. The labels describe performed facial configurations, not verified emotions.

The model's classification scores are not calibrated measures of emotional truth or smile intensity. Unknown inputs can still receive high scores. The response layer's uncertainty and hysteresis are design rules, not a robust out-of-distribution detector.

## Camera-free route

Choose **Synthetic feature simulator / not a face** before collecting examples. Select a feature pattern as well as its intended training label, or use **Load synthetic practice dataset** in Example 02. Select **Train custom model**, then test with the synthetic feature patterns in Example 03.

This route really trains an ml5 neural network on authored numbers. It does not run FaceMesh, use a face, validate facial recognition, or produce an emotion detector. The interface and exports retain the synthetic source label.

After resetting, you may load the supplied prepared synthetic model using the four files in `data/prepared-synthetic-model/`. It is a training-workflow fallback, not a personal model for a visitor. If you add a fifth class, the bundled four-class practice data/model will no longer match your label configuration.

## Files worth opening

| File | Read it for |
|---|---|
| `config.js` | Labels, recording settings, training settings, smile-response settings. |
| `sketch.js` | Thin p5 setup/draw functions. |
| `shared/face-input.js` | Camera ownership, pretrained FaceMesh, fresh-result validity. |
| `shared/face-features.js` | The six measurements and fixed feature schema. |
| `shared/training-session.js` | Countdown, fresh sample collection, burst boundaries. |
| `shared/expression-model.js` | ml5 custom training, inference, save/load, cleanup. |
| `shared/smile-gate.js` | Smoothing, entry/exit thresholds, sustained-state timing. |
| `shared/workshop-app.js` | Controls, view changes, drawing, and session coordination. |

The six inputs are mouth width / face width, mouth opening / mouth width, signed mouth-corner lift / mouth width, two eye-opening / eye-width ratios, and mean brow-eye spacing / face width. Distance ratios reduce sensitivity to screen position and size, but do not remove perspective, pose, lighting, identity, or tracking differences. The signed corner measurement uses an eye-aligned vertical direction; this is not full 3D pose correction.

## Save the dataset and model separately

**Save dataset** downloads `dataset.json`, with numerical observations, labels, source, burst boundaries, and feature schema. It does not contain network weights. Loading a dataset requires an empty session and requires retraining.

**Save model set (4 files)** requests:

- `model.json`: network structure and weight manifest.
- `model.weights.bin`: trained weights.
- `model_meta.json`: ml5 feature ranges and label metadata.
- `session.json`: this workshop's feature schema, source, labels, and response settings.

Allow multiple downloads if your browser asks. Keep these filenames together in a folder; do not mix exports or accept browser-added suffixes such as `(1)`. To test reloading: save, check all four files exist, reset, select all four through **Load model set**, then test fresh input. Loading the model does not restore training examples.

Downloaded facial measurements and personal models may be sensitive. Saving is optional; do not post them publicly. Reset stops the camera and clears the current page, but cannot delete downloaded files.

## Add a fifth class

Duplicate the complete folder first. In `config.js`, add a unique entry such as:

```js
{ id: "mouth_open", name: "Mouth open", prompt: "Open your mouth comfortably, without smiling or straining." }
```

Reload, record every configured class, and retrain. Counts, selectors, and bars are generated from the configuration. The fifth class is not created merely by adding its name: it needs examples and a newly trained model. The supplied synthetic simulator has an authored `mouth_open` pattern for the camera-free route.

## Test response without claiming inference

Open **Design test / no inference** in Example 03. Its buttons supply high, low, or intermediate smile scores directly to the response layer. Use **Return to inference** before evaluating the trained model.

The hold-time slider changes how long a proposed state must persist. It does not change the network weights. Missing/stale camera input clears predictions; it does not become the resting class.

## Troubleshooting

- **Blank page or disabled startup:** Check the console, confirm the CDN is reachable, and keep `shared` and `data` beside `index.html`. Use Live Server and hardware acceleration.
- **Camera fails:** Check permission, use localhost or HTTPS, close other camera apps, and restart. Camera access inside an embedded hosted preview may be restricted; run the downloaded example locally.
- **No face or two faces:** Move closer, face forward, improve even lighting, and use one person at a time.
- **Slow first start:** FaceMesh is downloading model assets. Have the synthetic route and prepared synthetic model available.
- **Burst incomplete:** Only fresh, valid results count. It cancels after 20 seconds without enough usable observations; reposition and repeat.
- **Classes do not separate:** Try varied recording rounds, review selected features, and report limitations. Training loss alone does not establish performance on fresh input.
- **Save/load error:** Use all four files from one export, with original names and matching labels/schema. A Teachable Machine image model is not this format.
- **Source mismatch:** Reset before switching between facial and synthetic data. The example intentionally prevents mixing them in one session.

## Technical references

- ml5 FaceMesh: https://docs.ml5js.org/reference/facemesh
- ml5 NeuralNetwork: https://docs.ml5js.org/reference/neural-network
- ml5 with p5 2.x: https://ml5js.org/blog/using-ml5-with-p5-2/
- p5.js: https://p5js.org/
- ml5 source: https://github.com/ml5js/ml5-next-gen/tree/v1.4.0

