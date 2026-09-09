/**
 * Maps the on-screen guide rectangle (drawn over a `<video>` that's
 * rendered with `object-fit: cover`) to the video's native pixel
 * resolution, so a capture can crop exactly the region the user aligned
 * their card to - not the whole frame.
 *
 * `object-fit: cover` scales the video to fill the container while
 * preserving aspect ratio, cropping whichever dimension overflows and
 * centering the result. To go from a point in container/CSS coordinates
 * to native video pixels, we have to reconstruct that same scale-and-
 * center transform and invert it.
 *
 * Pure function - no DOM/camera access - so it's directly testable
 * (see cropMath.test.mjs) without a real camera or browser.
 */
export function computeGuideCropRect({
  videoWidth,
  videoHeight,
  containerWidth,
  containerHeight,
  guideWidthFraction = 0.85,
  aspectRatio = 1.6,
}) {
  if (!videoWidth || !videoHeight || !containerWidth || !containerHeight) {
    return null
  }

  // object-fit: cover scale factor: the smaller-overflowing dimension
  // sets the scale so the container is fully covered.
  const scale = Math.max(containerWidth / videoWidth, containerHeight / videoHeight)
  const displayedWidth = videoWidth * scale
  const displayedHeight = videoHeight * scale

  // How much of the scaled video is cropped off each side to center it
  // within the container.
  const offsetX = (displayedWidth - containerWidth) / 2
  const offsetY = (displayedHeight - containerHeight) / 2

  // The guide rectangle, in container/CSS coordinates - centered,
  // guideWidthFraction of the container's width, at the target aspect ratio.
  const guideW = containerWidth * guideWidthFraction
  const guideH = guideW / aspectRatio
  const guideX = (containerWidth - guideW) / 2
  const guideY = (containerHeight - guideH) / 2

  // container coords -> displayed-video coords (add back the cropped
  // offset) -> native-video pixel coords (divide out the scale).
  const srcX = (guideX + offsetX) / scale
  const srcY = (guideY + offsetY) / scale
  const srcW = guideW / scale
  const srcH = guideH / scale

  return {
    // CSS rect for rendering the overlay
    guide: { x: guideX, y: guideY, width: guideW, height: guideH },
    // Native-pixel rect for canvas.drawImage's source rectangle
    source: { x: srcX, y: srcY, width: srcW, height: srcH },
  }
}
