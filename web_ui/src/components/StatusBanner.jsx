const STATUS_STYLES = {
  success: {
    bg: 'bg-green-50 border-green-200',
    text: 'text-green-800',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 text-green-600">
        <path strokeLinecap="round" strokeLinejoin="round" d="m4.5 12.75 6 6 9-13.5" />
      </svg>
    ),
    label: 'Verified',
  },
  low_confidence: {
    bg: 'bg-amber-50 border-amber-200',
    text: 'text-amber-800',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 text-amber-600">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-8.99 4.5h.008v.008h-.008v-.008Z" />
      </svg>
    ),
    label: 'Needs review',
  },
  no_card_detected: {
    bg: 'bg-red-50 border-red-200',
    text: 'text-red-800',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 text-red-600">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
      </svg>
    ),
    label: 'No card found',
  },
  unreadable_image: {
    bg: 'bg-red-50 border-red-200',
    text: 'text-red-800',
    icon: (
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-5 w-5 text-red-600">
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
      </svg>
    ),
    label: 'Couldn’t read file',
  },
}

export default function StatusBanner({ status, messages }) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.no_card_detected

  return (
    <div className={`flex items-start gap-3 rounded-xl border px-4 py-3 ${style.bg}`}>
      <div className="mt-0.5 shrink-0">{style.icon}</div>
      <div>
        <p className={`text-sm font-semibold ${style.text}`}>{style.label}</p>
        {messages?.map((m, i) => (
          <p key={i} className={`text-sm ${style.text} opacity-90`}>{m}</p>
        ))}
      </div>
    </div>
  )
}
