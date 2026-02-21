"use client"

import { motion, AnimatePresence } from "framer-motion"
import {
  Sun,
  Wind,
  Layers,
  Grid3x3,
  ShieldAlert,
  ArrowRight,
  ImageIcon,
  CheckCircle2,
} from "lucide-react"
import type { SegmentationResult } from "@/lib/api"

/* ── Pipeline steps definition ──────────────────────────────────────────── */
interface Step {
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>
  label: string
  sublabel: string
  color: string
  /** Which field from segResult to display as intermediate output */
  previewField?: keyof SegmentationResult
  previewLabel?: string
}

const steps: Step[] = [
  {
    icon: ImageIcon,
    label: "Input Image",
    sublabel: "Original capture",
    color: "#94a3b8",
    previewField: "original_b64",
    previewLabel: "Original 960×540",
  },
  {
    icon: Grid3x3,
    label: "Resize",
    sublabel: "→ 384×384",
    color: "#f59e0b",
    previewField: "original_b64",
    previewLabel: "Resized 384×384",
  },
  {
    icon: Sun,
    label: "Dehazing",
    sublabel: "DCP + CLAHE",
    color: "#06b6d4",
    previewField: "defog_b64",
    previewLabel: "Dehazed Output",
  },
  {
    icon: Layers,
    label: "ConvNeXt Encoder",
    sublabel: "Feature extraction",
    color: "#3b82f6",
    // No image available for internal features — show defog as "preprocessed"
    previewField: "defog_b64",
    previewLabel: "Preprocessed Input",
  },
  {
    icon: Wind,
    label: "Mix-Attention",
    sublabel: "Enc–Dec sync",
    color: "#8b5cf6",
    previewField: "overlay_b64",
    previewLabel: "Attention Regions",
  },
  {
    icon: ShieldAlert,
    label: "U-Net Decode",
    sublabel: "Progressive fusion",
    color: "#ec4899",
    previewField: "mask_b64",
    previewLabel: "Class Logits",
  },
  {
    icon: CheckCircle2,
    label: "Segmentation",
    sublabel: "4-class output",
    color: "#10b981",
    previewField: "mask_b64",
    previewLabel: "Final Mask",
  },
]

interface ProcessingPipelineProps {
  isProcessing: boolean
  currentStep: number
  segResult?: SegmentationResult | null
}

export function ProcessingPipeline({ isProcessing, currentStep, segResult }: ProcessingPipelineProps) {
  // Which step is "expanded" showing intermediate output
  const activeIdx = Math.min(currentStep, steps.length - 1)
  const completedStep = !isProcessing && segResult ? steps.length - 1 : activeIdx

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          AI Processing Pipeline
        </h2>
        <p className="mb-10 text-center text-sm text-muted-foreground">
          Multi-stage perception workflow — step-by-step intermediate outputs
        </p>

        {/* ── Desktop: horizontal strip ─────────────────────────────── */}
        <div className="hidden items-start justify-center gap-1 lg:flex">
          {steps.map((step, i) => {
            const isActive = i <= completedStep
            const isCurrent = isProcessing && i === activeIdx
            const hasPrev = isActive && segResult && step.previewField
            const src = hasPrev ? `data:image/png;base64,${segResult[step.previewField as keyof SegmentationResult] as string}` : null

            return (
              <div key={step.label} className="flex items-start">
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.1 }}
                  className={`relative flex flex-col items-center gap-2 rounded-xl border px-3 py-4 transition-all min-w-[100px] ${isCurrent
                    ? "border-primary/50 bg-primary/10 shadow-lg shadow-primary/10"
                    : isActive
                      ? "border-chart-4/30 bg-chart-4/5"
                      : "border-border bg-card/30"
                    }`}
                >
                  {isCurrent && (
                    <motion.div
                      className="absolute inset-0 rounded-xl border border-primary/30"
                      animate={{ opacity: [0.3, 0.8, 0.3] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                    />
                  )}

                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-lg"
                    style={{ backgroundColor: isActive ? `${step.color}20` : "rgba(148,163,184,0.1)" }}
                  >
                    <step.icon
                      className="h-5 w-5"
                      style={{ color: isActive ? step.color : "#64748b" }}
                    />
                  </div>
                  <span
                    className="text-center text-xs font-medium leading-tight"
                    style={{ color: isActive ? step.color : "#64748b" }}
                  >
                    {step.label}
                  </span>
                  <span className="text-center text-[10px] text-muted-foreground">
                    {step.sublabel}
                  </span>

                  {/* Intermediate image preview */}
                  <AnimatePresence>
                    {src && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="w-full overflow-hidden"
                      >
                        <div className="mt-1 overflow-hidden rounded-md border border-border">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={src}
                            alt={step.previewLabel}
                            className="h-14 w-full object-cover"
                          />
                        </div>
                        <p className="mt-1 text-center text-[9px] text-muted-foreground">
                          {step.previewLabel}
                        </p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>

                {i < steps.length - 1 && (
                  <ArrowRight
                    className={`mx-1 mt-5 h-4 w-4 shrink-0 ${isActive ? "text-primary" : "text-muted-foreground/30"
                      }`}
                  />
                )}
              </div>
            )
          })}
        </div>

        {/* ── Mobile: vertical list ─────────────────────────────────── */}
        <div className="flex flex-col gap-3 lg:hidden">
          {steps.map((step, i) => {
            const isActive = i <= completedStep
            const isCurrent = isProcessing && i === activeIdx
            const hasPrev = isActive && segResult && step.previewField
            const src = hasPrev ? `data:image/png;base64,${segResult[step.previewField as keyof SegmentationResult] as string}` : null

            return (
              <motion.div
                key={step.label}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className={`flex flex-col gap-2 rounded-lg border px-4 py-3 transition-all ${isCurrent
                  ? "border-primary/50 bg-primary/10"
                  : isActive
                    ? "border-chart-4/30 bg-chart-4/5"
                    : "border-border bg-card/30"
                  }`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className="flex h-8 w-8 items-center justify-center rounded-lg"
                    style={{ backgroundColor: isActive ? `${step.color}20` : "rgba(148,163,184,0.1)" }}
                  >
                    <step.icon
                      className="h-4 w-4"
                      style={{ color: isActive ? step.color : "#64748b" }}
                    />
                  </div>
                  <div>
                    <span className="text-sm font-medium" style={{ color: isActive ? step.color : "#64748b" }}>
                      {step.label}
                    </span>
                    <p className="text-xs text-muted-foreground">{step.sublabel}</p>
                  </div>
                  {isCurrent && (
                    <motion.div
                      className="ml-auto h-2 w-2 rounded-full bg-primary"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
                    />
                  )}
                </div>

                {/* Intermediate image on mobile */}
                <AnimatePresence>
                  {src && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                    >
                      <div className="overflow-hidden rounded-md border border-border">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={src} alt={step.previewLabel} className="h-24 w-full object-cover" />
                      </div>
                      <p className="mt-1 text-center text-[10px] text-muted-foreground">
                        {step.previewLabel}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )
          })}
        </div>
      </motion.div>
    </section>
  )
}
