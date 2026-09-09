function humanize(value) {
  if (!value) return '—'
  return value
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

export default function DecodedInfo({ decoded }) {
  if (!decoded) return null

  const items = [
    { label: 'Birth date (from ID number)', value: decoded.birth_date },
    { label: 'Governorate of birth', value: humanize(decoded.governorate) },
    { label: 'Gender', value: humanize(decoded.gender) },
    { label: 'Century', value: decoded.century },
  ]

  return (
    <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
      <p className="mb-3 text-sm font-semibold text-blue-900">
        Decoded from the National ID number's checksum
      </p>
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {items.map((item) => (
          <div key={item.label}>
            <dt className="text-xs text-blue-700/70">{item.label}</dt>
            <dd className="text-sm font-medium text-blue-900">{item.value || '—'}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
