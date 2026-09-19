export default function FieldRow({ label, value, cropSrc, rtl }) {
  const hasValue = value && value.trim().length > 0

  return (
    <div className="flex items-center gap-4 rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex h-16 w-40 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-slate-100">
        {cropSrc ? (
          <img src={cropSrc} alt={`${label} — detected region`} className="max-h-full max-w-full object-contain" />
        ) : (
          <span className="text-xs text-slate-400">no region detected</span>
        )}
      </div>

      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</p>
        <p
          dir={rtl ? 'rtl' : 'ltr'}
          title={hasValue ? value : undefined}
          className={`text-base break-words ${hasValue ? 'font-semibold text-slate-800' : 'italic text-slate-400'}`}
        >
          {hasValue ? value : 'not detected'}
        </p>
      </div>
    </div>
  )
}
