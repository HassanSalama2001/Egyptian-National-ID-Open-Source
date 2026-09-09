import { useEffect, useRef, useState } from 'react'
import { computeGuideCropRect } from '../lib/cropMath'

const ASPECT_RATIO = 1.6 // matches the backend's rectified canvas (1200x750) and real ID card proportions
const GUIDE_WIDTH_FRACTION = 0.85

export default function CameraCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null)
  const containerRef = useRef(null)
  const streamRef = useRef(null)
  const [error, setError] = useState(null)
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function start() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Your browser doesn't support camera access.")
        return
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch (err) {
        setError(
          err.name === 'NotAllowedError'
            ? 'Camera access was denied. You can allow it in your browser settings, or use file upload instead.'
            : "Couldn't access a camera on this device. Use file upload instead."
        )
      }
    }

    start()
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const handleCapture = () => {
    const video = videoRef.current
    const container = containerRef.current
    if (!video || !container) return

    const rect = computeGuideCropRect({
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      containerWidth: container.clientWidth,
      containerHeight: container.clientHeight,
      guideWidthFraction: GUIDE_WIDTH_FRACTION,
      aspectRatio: ASPECT_RATIO,
    })
    if (!rect) return

    const canvas = document.createElement('canvas')
    canvas.width = Math.round(rect.source.width)
    canvas.height = Math.round(rect.source.height)
    const ctx = canvas.getContext('2d')
    ctx.drawImage(
      video,
      rect.source.x, rect.source.y, rect.source.width, rect.source.height,
      0, 0, canvas.width, canvas.height
    )

    canvas.toBlob((blob) => {
      if (!blob) return
      // Wrap as a File (not a bare Blob) so it flows through exactly the
      // same path as a picked file - FormData.append needs a filename to
      // behave consistently, and App.jsx's handler doesn't need to know
      // or care which source produced it.
      const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' })
      onCapture(file)
    }, 'image/jpeg', 0.92)
  }

  if (error) {
    return (
      <div className="space-y-4 rounded-2xl border border-red-200 bg-white p-8 text-center">
        <p className="font-medium text-red-700">{error}</p>
        <button onClick={onCancel} className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700">
          Use file upload instead
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div
        ref={containerRef}
        className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl bg-slate-900"
      >
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          onLoadedMetadata={() => setIsReady(true)}
          className="h-full w-full object-cover"
        />

        {isReady && (
          <div
            className="pointer-events-none absolute rounded-xl border-2 border-white/90 shadow-[0_0_0_9999px_rgba(0,0,0,0.45)]"
            style={{
              left: `${(1 - GUIDE_WIDTH_FRACTION) / 2 * 100}%`,
              width: `${GUIDE_WIDTH_FRACTION * 100}%`,
              aspectRatio: `${ASPECT_RATIO}`,
              top: '50%',
              transform: 'translateY(-50%)',
            }}
          >
            {/* corner brackets for a clearer "align here" cue */}
            {['top-0 left-0 border-t-2 border-l-2', 'top-0 right-0 border-t-2 border-r-2',
              'bottom-0 left-0 border-b-2 border-l-2', 'bottom-0 right-0 border-b-2 border-r-2'].map((cls) => (
              <span key={cls} className={`absolute h-5 w-5 border-white ${cls}`} />
            ))}
          </div>
        )}

        {!isReady && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-sm text-white/70">Starting camera…</p>
          </div>
        )}
      </div>

      <p className="text-center text-sm text-slate-500">
        Align the card within the frame, then capture
      </p>

      <div className="flex justify-center gap-3">
        <button
          onClick={onCancel}
          className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50"
        >
          Cancel
        </button>
        <button
          onClick={handleCapture}
          disabled={!isReady}
          className="rounded-lg bg-blue-600 px-6 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
        >
          Capture
        </button>
      </div>
    </div>
  )
}
