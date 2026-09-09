// Standalone sanity check for computeGuideCropRect - no test framework
// needed since it's a pure function. Run with: node src/lib/cropMath.test.mjs
import { computeGuideCropRect } from './cropMath.js'

let failures = 0
function approxEqual(actual, expected, tolerance = 0.5, label = '') {
  if (Math.abs(actual - expected) > tolerance) {
    console.error(`FAIL ${label}: expected ${expected}, got ${actual}`)
    failures++
  }
}

// Case 1: video and container have the same aspect ratio (16:9-ish square
// test) - no cropping should occur, scale should be a simple ratio.
{
  const r = computeGuideCropRect({
    videoWidth: 1000, videoHeight: 1000,
    containerWidth: 500, containerHeight: 500,
    guideWidthFraction: 0.8, aspectRatio: 1.6,
  })
  // scale = 0.5, no offset since aspect ratios match.
  // guideW=400, guideH=250, guideX=(500-400)/2=50, guideY=(500-250)/2=125
  approxEqual(r.source.width, 400 / 0.5, 0.01, 'case1 srcW')   // 800
  approxEqual(r.source.height, 250 / 0.5, 0.01, 'case1 srcH')  // 500
  approxEqual(r.source.x, 50 / 0.5, 0.01, 'case1 srcX')        // 100
  approxEqual(r.source.y, 125 / 0.5, 0.01, 'case1 srcY')       // 250
}

// Case 2: wide video (1920x1080) in a narrower container (400x600, portrait
// phone-shaped) - cover should scale up significantly and crop the sides.
{
  const r = computeGuideCropRect({
    videoWidth: 1920, videoHeight: 1080,
    containerWidth: 400, containerHeight: 600,
    guideWidthFraction: 0.85, aspectRatio: 1.6,
  })
  // scale = max(400/1920, 600/1080) = max(0.208, 0.556) = 0.556
  const expectedScale = 600 / 1080
  const expectedGuideW = 400 * 0.85
  const expectedGuideH = expectedGuideW / 1.6
  approxEqual(r.guide.width, expectedGuideW, 0.01, 'case2 guideW')
  approxEqual(r.guide.height, expectedGuideH, 0.01, 'case2 guideH')
  // Source rect must be fully within native video bounds
  if (r.source.x < 0 || r.source.y < 0) {
    console.error(`FAIL case2: source rect starts outside video bounds`, r.source)
    failures++
  }
  if (r.source.x + r.source.width > 1920 + 0.5 || r.source.y + r.source.height > 1080 + 0.5) {
    console.error(`FAIL case2: source rect extends past video bounds`, r.source, 'video=1920x1080')
    failures++
  }
  approxEqual(r.source.width / r.source.height, 1.6, 0.01, 'case2 source aspect ratio preserved')
}

// Case 3: tall video (portrait phone camera, 1080x1920) in a wide container
// (800x450, landscape desktop-ish) - the guide must still land inside bounds
// and its own aspect ratio must be preserved after mapping back to native px.
{
  const r = computeGuideCropRect({
    videoWidth: 1080, videoHeight: 1920,
    containerWidth: 800, containerHeight: 450,
    guideWidthFraction: 0.85, aspectRatio: 1.6,
  })
  if (r.source.x < -0.5 || r.source.y < -0.5) {
    console.error(`FAIL case3: source rect starts outside video bounds`, r.source)
    failures++
  }
  if (r.source.x + r.source.width > 1080 + 0.5 || r.source.y + r.source.height > 1920 + 0.5) {
    console.error(`FAIL case3: source rect extends past video bounds`, r.source, 'video=1080x1920')
    failures++
  }
  approxEqual(r.source.width / r.source.height, 1.6, 0.01, 'case3 source aspect ratio preserved')
}

// Case 4: missing dimensions (video not ready yet) should return null, not throw
{
  const r = computeGuideCropRect({ videoWidth: 0, videoHeight: 0, containerWidth: 500, containerHeight: 500 })
  if (r !== null) {
    console.error('FAIL case4: expected null when video dimensions are 0')
    failures++
  }
}

if (failures === 0) {
  console.log('All cropMath checks passed.')
  process.exit(0)
} else {
  console.error(`${failures} check(s) failed.`)
  process.exit(1)
}
