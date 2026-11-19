/* p5's lifecycle stays short. The reusable classes do the work.
 * The three example buttons change views without discarding this session.
 * Nothing is persisted automatically across a browser reload.
 */
let workshop;
let startupError = null;

// p5 calls setup once after the deferred scripts have loaded. Configure
// TensorFlow first so both FaceMesh and the custom network use the same backend.
async function setup() {
  const canvas = createCanvas(640, 480);
  canvas.parent("canvas-holder");
  pixelDensity(1); frameRate(30);
  textFont("sans-serif"); textSize(18);
  try {
    // Choose the WebGL backend before constructing either model.
    await ml5.tf.setBackend("webgl"); await ml5.tf.ready();
    workshop = new WorkshopApp();
    window.workshop = workshop; // Expose the session for optional console inspection.
    workshop.start();
  } catch (error) {
    // setup errors otherwise leave the canvas blank with no useful explanation.
    startupError = error.message;
    document.getElementById("status").textContent = `Could not initialise: ${error.message}. Try current Chrome with hardware acceleration.`;
  }
}

// Keep p5's frame loop thin: the application updates state and draws the
// camera/simulator preview, while its reusable classes own the actual logic.
function draw() {
  background("#202b34");
  if (workshop) { workshop.update(); workshop.drawPreview(); }
  else {
    fill("#ffffff"); textAlign(CENTER,CENTER);
    text(startupError ? "Could not start. See the status message." : "Loading the workshop…", width/2,height/2);
  }
}
