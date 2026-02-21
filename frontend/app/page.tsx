"use client"

import { useState, useCallback, useRef } from "react"
import { NeuralBackground } from "@/components/neural-background"
import { HeroSection } from "@/components/hero-section"
import { UploadSection } from "@/components/upload-section"
import { ProcessingPipeline } from "@/components/processing-pipeline"
import { OutputDashboard } from "@/components/output-dashboard"
import { StatisticsPanel } from "@/components/statistics-panel"
import { RiskGauge } from "@/components/risk-gauge"
import { ModelTransparency } from "@/components/model-transparency"
import { Terrain3D } from "@/components/terrain-3d"
import { SiteFooter } from "@/components/site-footer"
import { segmentImage, type SegmentationResult } from "@/lib/api"

export default function Home() {
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [currentStep, setCurrentStep] = useState(-1)
  const [segResult, setSegResult] = useState<SegmentationResult | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const uploadRef = useRef<HTMLDivElement>(null)

  const handleUploadClick = useCallback(() => {
    uploadRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])

  const handleImageUploaded = useCallback(async (file: File, url: string) => {
    setImageUrl(url)
    setIsProcessing(true)
    setIsComplete(false)
    setSegResult(null)
    setApiError(null)
    setCurrentStep(0)

    // Animate pipeline steps while waiting for API
    const stepTimers: ReturnType<typeof setTimeout>[] = []
    for (let i = 1; i <= 6; i++) {
      stepTimers.push(setTimeout(() => setCurrentStep(i), i * 600))
    }

    try {
      const result = await segmentImage(file)
      setSegResult(result)
      stepTimers.forEach(clearTimeout)
      setCurrentStep(7)
      setTimeout(() => {
        setIsProcessing(false)
        setIsComplete(true)
      }, 400)
    } catch (err: any) {
      console.error("Segmentation API error:", err)
      setApiError(err.message || "Backend not available")
      stepTimers.forEach(clearTimeout)
      setCurrentStep(7)
      setTimeout(() => {
        setIsProcessing(false)
        setIsComplete(true)
      }, 400)
    }
  }, [])

  return (
    <main className="relative min-h-screen bg-background">
      <NeuralBackground />

      <HeroSection onUploadClick={handleUploadClick} />

      <UploadSection
        onImageUploaded={handleImageUploaded}
        uploadRef={uploadRef}
      />

      {(isProcessing || isComplete) && (
        <ProcessingPipeline
          isProcessing={isProcessing || isComplete}
          currentStep={isComplete ? 7 : currentStep}
        />
      )}

      {apiError && isComplete && (
        <div className="relative z-10 mx-auto max-w-3xl px-4 py-2">
          <div className="rounded-lg border border-accent/30 bg-accent/5 px-4 py-3 text-center">
            <p className="text-xs text-accent">
              ⚠ Backend unavailable — showing demo data. Start the API with:{" "}
              <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-[11px]">
                uv run uvicorn api:app --port 8000
              </code>
            </p>
          </div>
        </div>
      )}

      <OutputDashboard originalImage={imageUrl} isComplete={isComplete} segResult={segResult} />

      <StatisticsPanel isComplete={isComplete} segResult={segResult} />

      <RiskGauge isComplete={isComplete} segResult={segResult} />

      <ModelTransparency />

      <Terrain3D segResult={segResult} />

      <SiteFooter />
    </main>
  )
}
