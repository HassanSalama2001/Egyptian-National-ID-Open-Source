import { useState } from 'react'
import StatusBanner from './StatusBanner'
import FieldRow from './FieldRow'
import DecodedInfo from './DecodedInfo'

const FIELDS = [
  { cropKey: 'first_name', valueKey: 'first_name', label: 'First Name', rtl: true },
  { cropKey: 'full_name', valueKey: 'full_name', label: 'Full Name (rest)', rtl: true },
  { cropKey: 'address', valueKey: 'address', label: 'Address', rtl: true },
  { cropKey: 'national_id', valueKey: 'national_id', label: 'National ID', rtl: false },
  { cropKey: 'birth_date', valueKey: 'date_of_birth', label: 'Date of Birth', rtl: false },
  { cropKey: 'serial_number', valueKey: 'card_serial_number', label: 'Serial Number', rtl: false },
]

// gender/religion/marital_status are fuzzy-matched to an enum (see
// postprocessing/enum_matcher.py) rather than passed through as raw OCR
// text - "unknown" is the no-match fallback, not a detected value, so
// isEnum fields treat it the same as empty for display purposes.
const BACK_FIELDS = [
  { cropKey: 'national_id', valueKey: 'national_id', label: 'National ID', rtl: false },
  { cropKey: 'issue_date', valueKey: 'issue_date', label: 'Issue Date', rtl: false },
  { cropKey: 'expiry_date', valueKey: 'expiry_date', label: 'Expiry Date', rtl: false },
  { cropKey: 'profession', valueKey: 'profession', label: 'Profession', rtl: true },
  { cropKey: 'gender', valueKey: 'gender', label: 'Gender', rtl: false, isEnum: true },
  { cropKey: 'religion', valueKey: 'religion', label: 'Religion', rtl: false, isEnum: true },
  { cropKey: 'marital_status', valueKey: 'marital_status', label: 'Marital Status', rtl: false, isEnum: true },
]

export default function ResultsView({ originalImage, result, onReset }) {
  const [showJson, setShowJson] = useState(false)
  const front = result.front
  const back = result.back
  const side = front || back
  const sideFields = front ? FIELDS : BACK_FIELDS

  return (
    <div className="flex flex-col gap-6">
      <StatusBanner status={result.status} messages={result.messages} />

      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-3">
          <p className="text-sm font-semibold text-slate-600">Your photo</p>
          <img src={originalImage} alt="Uploaded" className="w-full rounded-xl border border-slate-200 object-contain" />
        </div>
        <div className="space-y-3">
          <p className="text-sm font-semibold text-slate-600">
            Card we detected <span className="font-normal text-slate-400">(after alignment)</span>
          </p>
          {result.card_image ? (
            <img src={result.card_image} alt="Detected card" className="w-full rounded-xl border border-slate-200 object-contain" />
          ) : (
            <div className="flex h-40 items-center justify-center rounded-xl border border-dashed border-slate-300 text-sm text-slate-400">
              No card boundary detected
            </div>
          )}
        </div>
      </div>

      {side && (
        <>
          <div>
            <p className="mb-3 text-sm font-semibold text-slate-600">
              Field-by-field detection
              <span className="ml-2 font-normal text-slate-400">— each region is cropped and read independently</span>
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              {sideFields.map((f) => {
                const rawValue = side[f.valueKey]
                const value = f.isEnum && rawValue === 'unknown' ? '' : rawValue
                return (
                  <FieldRow
                    key={f.cropKey}
                    label={f.label}
                    value={value}
                    cropSrc={side.field_crops?.[f.cropKey]}
                    rtl={f.rtl}
                  />
                )
              })}
            </div>
          </div>

          <DecodedInfo decoded={result.decoded} />
        </>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-slate-200 pt-4 text-sm text-slate-500">
        <div className="flex gap-4">
          <span>Confidence: <strong className="text-slate-700">{Math.round(result.confidence * 100)}%</strong></span>
          <span>Processed in <strong className="text-slate-700">{result.processing_time_ms}ms</strong></span>
        </div>
        <div className="flex gap-3">
          <button onClick={() => setShowJson((v) => !v)} className="text-blue-600 hover:underline">
            {showJson ? 'Hide raw response' : 'Show raw response'}
          </button>
          <button onClick={onReset} className="font-medium text-slate-700 hover:underline">
            Scan another card
          </button>
        </div>
      </div>

      {showJson && (
        <pre className="max-h-80 overflow-auto rounded-xl bg-slate-900 p-4 text-xs text-slate-100">
          {JSON.stringify(result, null, 2)}
        </pre>
      )}
    </div>
  )
}
