/* ──────────────────────────────────────────────────────────────────────── */
/*  API client for the off-road segmentation backend                       */
/* ──────────────────────────────────────────────────────────────────────── */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export interface ClassDistribution {
  id: number
  name: string
  percentage: number
  color: string
}

export interface SegmentationResult {
  original_b64: string
  mask_b64: string
  overlay_b64: string
  shap_b64?: string
  defog_b64?: string
  class_distribution: ClassDistribution[]
  terrain_grid: number[][]
  inference_ms: number
  image_size: { w: number; h: number }
  pipeline: {
    backbone: string
    head: string
    classes: string[]
    total_classes: number
  }
}

export interface ModelInfo {
  backbone: { name: string; source: string; embedding_dim: number; patch_size: number; frozen: boolean }
  head: { name: string; blocks: number; hidden_dim: number; heads: number; params: string }
  training: { optimizer: string; scheduler: string; loss: string; amp: boolean; epochs: number; batch_size: number }
  classes: { total: number; names: string[] }
  input: { original: string; resized: string }
  dataset: { train: number; val: number; test: number }
}

export async function segmentImage(file: File): Promise<SegmentationResult> {
  const formData = new FormData()
  formData.append("file", file)

  const res = await fetch(`${API_BASE}/api/segment`, {
    method: "POST",
    body: formData,
  })

  if (!res.ok) {
    throw new Error(`Segmentation failed: ${res.status} ${res.statusText}`)
  }

  return res.json()
}

export async function getModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/api/model-info`)
  if (!res.ok) throw new Error("Failed to fetch model info")
  return res.json()
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) })
    return res.ok
  } catch {
    return false
  }
}
