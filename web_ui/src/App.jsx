import { useState } from 'react'
import UploadZone from './components/UploadZone'
import CameraCapture from './components/CameraCapture'
import ResultsView from './components/ResultsView'
import { extractIdCard } from './api'

export default function App() {
  const [captureMode, setCaptureMode] = useState('upload') // 'upload' | 'camera'
  const [originalImage, setOriginalImage] = useState(null)
  const [result, setResult] = useState(null)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState(null)

  // Handles a File from either source (picked/dropped, or captured from
  // the camera and wrapped into a File by CameraCapture) identically -
  // the rest of the app doesn't need to know which one produced it.
  const handleFileSelect = async (file) => {
    setCaptureMode('upload')
    setOriginalImage(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setIsUploading(true)
    try {
      const response = await extractIdCard(file)
      setResult(response.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsUploading(false)
    }
  }

  const reset = () => {
    setCaptureMode('upload')
    setOriginalImage(null)
    setResult(null)
    setError(null)
    setIsUploading(false)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-4xl px-4 py-4">
          <h1 className="text-lg font-semibold text-slate-800">Egyptian National ID — OCR Demo</h1>
          <p className="text-sm text-slate-500">
            Local, open-source extraction pipeline — every detection step shown, nothing hidden
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-4 py-8">
        {captureMode === 'camera' ? (
          <CameraCapture onCapture={handleFileSelect} onCancel={() => setCaptureMode('upload')} />
        ) : !originalImage ? (
          <UploadZone
            onFileSelect={handleFileSelect}
            isUploading={isUploading}
            onUseCamera={() => setCaptureMode('camera')}
          />
        ) : isUploading ? (
          <div className="flex flex-col items-center gap-4 rounded-2xl border border-slate-200 bg-white p-16">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-blue-600" />
            <p className="text-sm text-slate-500">Detecting the card and reading each field…</p>
          </div>
        ) : error ? (
          <div className="space-y-4 rounded-2xl border border-red-200 bg-white p-8 text-center">
            <p className="font-medium text-red-700">{error}</p>
            <button onClick={reset} className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700">
              Try again
            </button>
          </div>
        ) : result ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <ResultsView originalImage={originalImage} result={result} onReset={reset} />
          </div>
        ) : null}
      </main>
    </div>
  )
}
