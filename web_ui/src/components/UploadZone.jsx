import { useRef, useState } from 'react'

export default function UploadZone({ onFileSelect, isUploading, onUseCamera }) {
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
    <div className="space-y-3">
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

      {!isUploading && onUseCamera && (
        <button
          onClick={onUseCamera}
          className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white py-3 text-sm font-medium text-slate-600 hover:bg-slate-50"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="h-5 w-5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z" />
          </svg>
          Use your camera instead
        </button>
      )}
    </div>
  )
}
