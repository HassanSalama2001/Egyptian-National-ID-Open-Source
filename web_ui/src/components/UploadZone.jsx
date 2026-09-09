import { useRef, useState } from 'react'

export default function UploadZone({ onFileSelect, isUploading }) {
  const [isDragActive, setIsDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFileSelect(file)
  }

  return (
    <div
      onClick={() => !isUploading && inputRef.current?.click()}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
      className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
        px-8 py-16 text-center transition-colors cursor-pointer
        ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-white hover:border-slate-400'}
        ${isUploading ? 'pointer-events-none opacity-60' : ''}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
      />

      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-blue-100 text-blue-600">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="h-7 w-7">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
        </svg>
      </div>

      {isUploading ? (
        <p className="text-sm font-medium text-slate-500">Processing your photo…</p>
      ) : (
        <>
          <p className="text-base font-medium text-slate-700">
            Drop a photo of the ID card here, or click to browse
          </p>
          <p className="text-sm text-slate-400">
            For best results, frame the card so it fills the photo — like the guides in most banking apps
          </p>
          <p className="text-xs uppercase tracking-wide text-slate-400">JPG or PNG</p>
        </>
      )}
    </div>
  )
}
