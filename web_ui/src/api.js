const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/ocr'

export async function extractIdCard(file, onProgress) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(API_URL, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let detail = 'Something went wrong while processing this image. Please try again.'
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch {
      // response wasn't JSON - keep the generic message
    }
    throw new Error(detail)
  }

  return response.json()
}
