/* Geometry is not an emotion.
 * FaceMesh gives us coordinates. This class makes OUR selected measurements.
 * It is pure JavaScript: the same function is used in collection and inference.
 * The schema's array order is also the neural network's input order and is
 * embedded in saved datasets/models, so changing it requires a schema version.
 */
class FaceFeatures {
  static schema = {
    id: "personal-expression-six-ratios-v1",
    names: ["mouthWidth", "mouthOpen", "cornerLift", "leftEyeOpen", "rightEyeOpen", "browGap"],
    description: "Six 2D ratios; distances remove translation/scale; signed corner lift uses an eye-aligned vertical axis."
  };
  static distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
  static midpoint(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
  static extract(face) {
    // FaceMesh normally returns 468 points without refined iris landmarks.
    // Return null, rather than partial numbers, whenever required geometry is absent.
    const p = face?.keypoints;
    if (!p || p.length < 468) return null;
    // Indices used below: eye corners/lids, mouth corners/center, face sides,
    // and inner brows. Keep this list in sync with every indexed measurement.
    const required = [33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14, 234, 454, 105, 334];
    if (!required.every(i => p[i] && Number.isFinite(p[i].x) && Number.isFinite(p[i].y))) return null;
    const distance = (i, j) => this.distance(p[i], p[j]);
    const faceWidth = distance(234, 454), mouthWidth = distance(61, 291);
    const eyeAWidth = distance(33, 133), eyeBWidth = distance(362, 263);
    // Degenerate distances would make the ratios unstable or divide by zero.
    if (Math.min(faceWidth, mouthWidth, eyeAWidth, eyeBWidth) < 1) return null;
    // Project the mouth-center/corner offset onto an eye-aligned down axis.
    // This reduces sign flips when the head rolls, but is not 3D pose correction.
    const eyeA = this.midpoint(p[33], p[133]), eyeB = this.midpoint(p[362], p[263]);
    const eyeSpan = this.distance(eyeA, eyeB);
    if (eyeSpan < 1) return null;
    let down = { x: -(eyeB.y - eyeA.y) / eyeSpan, y: (eyeB.x - eyeA.x) / eyeSpan };
    if (down.y < 0) down = { x: -down.x, y: -down.y };
    const centre = this.midpoint(p[13], p[14]), corners = this.midpoint(p[61], p[291]);
    const cornerLift = ((centre.x - corners.x) * down.x + (centre.y - corners.y) * down.y) / mouthWidth;
    // Feature order must match `schema.names`: all are ratios, not emotion scores.
    const values = [
      mouthWidth / faceWidth,
      distance(13, 14) / mouthWidth,
      cornerLift,
      distance(159, 145) / eyeAWidth,
      distance(386, 374) / eyeBWidth,
      (this.distance(p[105], eyeA) + this.distance(p[334], eyeB)) / (2 * faceWidth)
    ];
    return values.every(Number.isFinite) ? { values, faceWidth } : null;
  }
}
window.FaceFeatures = FaceFeatures;
