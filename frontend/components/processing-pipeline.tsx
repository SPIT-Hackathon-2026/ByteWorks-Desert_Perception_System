"use client"

import { useEffect, useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Sun, Wind, Layers, Grid3x3,
  ShieldAlert, ImageIcon, CheckCircle2, Sparkles,
} from "lucide-react"
import { HUDScanner } from "./hud-scanner"
import type { SegmentationResult } from "@/lib/api"

/* ── Constants ──────────────────────────────────────────────────────── */
const THUMB_H = 96
const ACTIVE_H = 200

/* ── Steps ─────────────────────────────────────────────────────────── */
const steps = [
  { icon: ImageIcon, label: "Input", sublabel: "Original capture", color: "#64748b", field: "original_b64" as const, caption: "Original" },
  { icon: Grid3x3, label: "Resize", sublabel: "→ 384×384", color: "#f59e0b", field: "original_b64" as const, caption: "Resized 384×384" },
  { icon: Sun, label: "Preprocess", sublabel: "Dehaze + CLAHE", color: "#06b6d4", field: "defog_b64" as const, caption: "Enhanced" },
  { icon: Layers, label: "ConvNeXt", sublabel: "Feature extract", color: "#3b82f6", field: "defog_b64" as const, caption: "Preprocessed" },
  { icon: Wind, label: "Mix-Attn", sublabel: "Enc–Dec sync", color: "#8b5cf6", field: "overlay_b64" as const, caption: "Attention" },
  { icon: ShieldAlert, label: "U-Net Dec", sublabel: "Decode features", color: "#ec4899", field: "mask_b64" as const, caption: "Logits" },
  { icon: CheckCircle2, label: "Output", sublabel: "4-class mask", color: "#10b981", field: "mask_b64" as const, caption: "Final mask" },
]

interface ProcessingPipelineProps {
  isProcessing: boolean
  currentStep: number
  segResult?: SegmentationResult | null
}

function b64Src(seg: SegmentationResult | null | undefined, field: keyof SegmentationResult) {
  if (!seg) return null
  const v = seg[field]
  return typeof v === "string" ? `data:image/png;base64,${v}` : null
}

/* ── Arrow ──────────────────────────────────────────────────────────── */
function Arrow({ live, color }: { live: boolean; color: string }) {
  return (
    <div className="relative flex shrink-0 items-center justify-center" style={{ width: 22 }}>
      <div className="absolute h-0.5 w-full rounded opacity-40" style={{ background: color }} />
      {live && (
        <motion.div className="absolute h-2.5 w-2.5 rounded-full"
          style={{ background: color, boxShadow: `0 0 10px ${color}` }}
          animate={{ x: [-8, 8, -8] }}
          transition={{ duration: 0.95, repeat: Infinity, ease: "easeInOut" }} />
      )}
      <svg width="9" height="13" viewBox="0 0 9 13" className="relative z-10">
        <path d="M1 1 L8 6.5 L1 12" stroke={color} strokeWidth="2" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

/* ── Main component ─────────────────────────────────────────────────── */
export function ProcessingPipeline({ isProcessing, currentStep, segResult }: ProcessingPipelineProps) {
  const activeIdx = Math.min(Math.max(currentStep, 0), steps.length - 1)

  /* ── Sequential image reveal after API results arrive ─────────── */
  const [revealStep, setRevealStep] = useState(-1)
  const revealTimer = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!isProcessing && segResult) {
      setRevealStep(-1)
      let step = 0
      revealTimer.current = setInterval(() => {
        setRevealStep(step)
        step++
        if (step >= steps.length && revealTimer.current) {
          clearInterval(revealTimer.current)
        }
      }, 450)
    }
    if (isProcessing) {
      setRevealStep(-1)
      if (revealTimer.current) clearInterval(revealTimer.current)
    }
    return () => { if (revealTimer.current) clearInterval(revealTimer.current) }
  }, [isProcessing, segResult])

  /* ── Equalise after full reveal ───────────────────────────────── */
  const allRevealed = revealStep >= steps.length - 1
  const [allEqualised, setAllEqualised] = useState(false)
  useEffect(() => {
    if (allRevealed) {
      const t = setTimeout(() => setAllEqualised(true), 500)
      return () => clearTimeout(t)
    }
    setAllEqualised(false)
  }, [allRevealed])
  useEffect(() => { if (isProcessing) setAllEqualised(false) }, [isProcessing])

  return (
    <section id="pipeline" className="relative z-10 mx-auto w-full max-w-7xl px-4 py-16">
      <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }} transition={{ duration: 0.5 }}>

        <div className="mb-2 flex items-center justify-center gap-3">
          <div className="h-px flex-1 max-w-12 bg-linear-to-r from-transparent to-primary/40" />
          <h2 className="text-center text-2xl font-bold tracking-tight">AI Processing Pipeline</h2>
          <div className="h-px flex-1 max-w-12 bg-linear-to-l from-transparent to-primary/40" />
        </div>
        <p className="mb-10 text-center text-sm text-muted-foreground">
          Watch each stage activate as your image passes through the model
        </p>

        {/* Pipeline row */}
        <div className="flex w-full items-start">
          {steps.map((step, i) => {
            /* ── Per-step status flags ────────────────────────── */
            const isLiveActive = isProcessing && i === activeIdx
            const isLiveDone   = isProcessing && i < activeIdx
            const isRevealing  = !isProcessing && segResult != null && i === revealStep && !allEqualised
            const isRevealed   = !isProcessing && segResult != null && (i < revealStep || allEqualised)

            const showActive = isLiveActive || isRevealing
            const showDone   = isLiveDone || isRevealed
            const isWaiting  = !showActive && !showDone

            /* ── Image height ─────────────────────────────────── */
            let imgH = 0
            if (isLiveActive)       imgH = ACTIVE_H
            else if (isLiveDone)    imgH = 48
            else if (isRevealing)   imgH = ACTIVE_H
            else if (isRevealed)    imgH = THUMB_H

            const src = b64Src(segResult, step.field)
            const Icon = step.icon
            const arrowLive = isProcessing && i === activeIdx - 1

            return (
              <div key={step.label} className="flex flex-1 items-start">
                {/* Block card */}
                <motion.div
                  layout
                  className="min-w-0 flex-1 overflow-hidden rounded-xl border-2"
                  animate={{
                    borderColor: showActive ? step.color : showDone ? `${step.color}55` : "var(--border)",
                    opacity: isWaiting ? 0.32 : 1,
                    scale: showActive ? 1.03 : 1,
                    boxShadow: showActive
                      ? `0 0 0 2px ${step.color}25, 0 8px 32px ${step.color}30`
                      : showDone
                      ? `0 2px 8px ${step.color}10`
                      : "none",
                  }}
                  transition={{ duration: 0.5, type: "spring", stiffness: 260, damping: 22 }}
                  style={{ background: "var(--card)" }}
                >
                  {/* Header */}
                  <div className="flex flex-col items-center gap-1 px-1.5 py-2.5"
                    style={{ background: showActive ? `${step.color}18` : showDone ? `${step.color}0c` : "transparent" }}>

                    <div className="flex h-8 w-8 items-center justify-center rounded-xl"
                      style={{ background: showActive ? `${step.color}2c` : showDone ? `${step.color}16` : "rgba(148,163,184,0.1)" }}>
                      {showActive ? (
                        <motion.div animate={{ rotate: 360 }}
                          transition={{ duration: 2.2, repeat: Infinity, ease: "linear" }}>
                          <Icon className="h-3.5 w-3.5" style={{ color: step.color }} />
                        </motion.div>
                      ) : (
                        <Icon className="h-3.5 w-3.5" style={{ color: showDone ? step.color : "#94a3b8" }} />
                      )}
                    </div>

                    <p className="text-center text-[10px] font-semibold"
                      style={{ color: showDone || showActive ? step.color : "#94a3b8" }}>
                      {step.label}
                    </p>

                    {showActive && (
                      <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                        className="text-center text-[8px] text-muted-foreground">
                        {step.sublabel}
                      </motion.p>
                    )}

                    {showActive ? (
                      <div className="flex gap-1 pt-0.5">
                        {[0, 1, 2].map(j => (
                          <motion.div key={j} className="h-1.5 w-1.5 rounded-full"
                            style={{ background: step.color }}
                            animate={{ scale: [1, 1.7, 1], opacity: [0.4, 1, 0.4] }}
                            transition={{ duration: 0.7, repeat: Infinity, delay: j * 0.22 }} />
                        ))}
                      </div>
                    ) : showDone ? (
                      <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
                        transition={{ type: "spring", stiffness: 280 }}>
                        <CheckCircle2 className="h-3 w-3" style={{ color: step.color }} />
                      </motion.div>
                    ) : null}
                  </div>

                  {/* ── Image pane ──────────────────────────────── */}
                  <motion.div
                    className="overflow-hidden bg-muted/20"
                    animate={{ height: imgH }}
                    transition={{ duration: 0.5, ease: "easeInOut" }}
                  >
                    <HUDScanner
                      isActive={showActive}
                      color={step.color}
                      label={`SIG_0x${i.toString(16).toUpperCase()}${activeIdx.toString(16)}`}
                    >
                      {/* Active with no image → scanning animation */}
                      {showActive && !src && (
                        <div className="relative flex h-full flex-col items-center justify-center">
                          <Sparkles className="relative h-5 w-5 opacity-20" style={{ color: step.color }} />
                          <p className="relative mt-1 text-[8px] text-muted-foreground uppercase tracking-tighter">Analyzing Feed</p>
                        </div>
                      )}

                      {/* Active / Revealing with image → pop-in */}
                      {showActive && src && (
                        <motion.img key={`active-img-${i}`} src={src} alt={step.caption}
                          initial={{ opacity: 0, scale: 1.06 }} animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.4 }}
                          className="h-full w-full object-cover" />
                      )}

                      {/* Done with image → thumbnail */}
                      {showDone && src && (
                        <img key={`done-img-${i}`} src={src} alt={step.caption}
                          className="h-full w-full object-cover" />
                      )}

                      {/* Done without image → checkmark placeholder */}
                      {showDone && !src && (
                        <div className="flex h-full w-full items-center justify-center">
                          <CheckCircle2 className="h-5 w-5 opacity-20" style={{ color: step.color }} />
                        </div>
                      )}
                    </HUDScanner>
                  </motion.div>
                </motion.div>

                {i < steps.length - 1 && <Arrow live={arrowLive} color={step.color} />}
              </div>
            )
          })}
        </div>

        {/* Completion bar */}
        <AnimatePresence>
          {allRevealed && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }} transition={{ duration: 0.4 }}
              className="mt-8 flex items-center justify-between rounded-xl border border-emerald-500/25 bg-emerald-500/5 px-5 py-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                <span className="text-sm font-semibold text-emerald-500">All stages complete</span>
              </div>
              {segResult?.inference_ms && (
                <span className="text-xs text-muted-foreground">Inference: {segResult.inference_ms}ms</span>
              )}
              <span className="text-xs text-muted-foreground">{steps.length} stages · 4 classes</span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </section>
  )
}
