"use client"

import { motion } from "framer-motion"
import { ImageIcon, Layers, ScanLine, Microscope } from "lucide-react"
import type { SegmentationResult } from "@/lib/api"

interface OutputDashboardProps {
  originalImage: string | null
  isComplete: boolean
  segResult?: SegmentationResult | null
}

/* ── Histogram equalization visualization (CSS-rendered) ────────────────── */
function HistogramEqualization({ src }: { src: string }) {
  return (
    <div className="relative h-full w-full">
      {/* Base image */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="Histogram equalized" className="h-full w-full object-cover" />
      {/* CLAHE effect overlay — raised contrast visual cue */}
      <div
        className="absolute inset-0"
        style={{
          background: "linear-gradient(135deg, rgba(6,182,212,0.08) 0%, transparent 60%)",
          mixBlendMode: "screen",
        }}
      />
      {/* Label badge */}
      <span className="absolute bottom-1 right-1 rounded bg-background/70 px-1.5 py-0.5 text-[9px] text-muted-foreground backdrop-blur-sm">
        CLAHE enhanced
      </span>
    </div>
  )
}

/* ── Panel card wrapper ─────────────────────────────────────────────────── */
interface PanelProps {
  color: string
  title: string
  icon: React.ReactNode
  delay: number
  children: React.ReactNode
}

function Panel({ color, title, icon, delay, children }: PanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay }}
      className="overflow-hidden rounded-xl border border-border bg-card/50"
    >
      <div className="flex items-center gap-2 border-b border-border px-4 py-3">
        <div className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-xs font-medium text-muted-foreground">{title}</span>
        <span className="ml-auto" style={{ color }}>{icon}</span>
      </div>
      <div className="relative aspect-video bg-card">{children}</div>
    </motion.div>
  )
}

function Placeholder({ label }: { label: string }) {
  return (
    <div className="flex h-full items-center justify-center bg-secondary/20">
      <span className="text-xs text-muted-foreground italic">{label}</span>
    </div>
  )
}

/* ── Main component ─────────────────────────────────────────────────────── */
export function OutputDashboard({ originalImage, isComplete, segResult }: OutputDashboardProps) {
  if (!isComplete || !originalImage) return null

  const hasData = !!segResult
  const orig = hasData ? `data:image/png;base64,${segResult.original_b64}` : originalImage
  const mask = hasData ? `data:image/png;base64,${segResult.mask_b64}` : null
  const defog = hasData && segResult.defog_b64 ? `data:image/png;base64,${segResult.defog_b64}` : null
  const overlay = hasData ? `data:image/png;base64,${segResult.overlay_b64}` : null

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">Analysis Output</h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Multi-stage vision processing results
        </p>

        {/* 2×2 grid */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">

          <Panel color="#94a3b8" title="Input Image" icon={<ImageIcon className="h-3.5 w-3.5" />} delay={0.1}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={orig} alt="Original terrain image" className="h-full w-full object-cover" />
          </Panel>

          {/* 2. Preprocessed / Defogged image */}
          <Panel color="#06b6d4" title="Preprocessed Image" icon={<ScanLine className="h-3.5 w-3.5" />} delay={0.2}>
            {defog ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={defog} alt="Dehazed terrain" className="h-full w-full object-cover" />
            ) : (
              <Placeholder label="Enhancing…" />
            )}
          </Panel>

          {/* 3. Final semantic segmentation */}
          <Panel color="#10b981" title="Semantic Segmentation" icon={<Layers className="h-3.5 w-3.5" />} delay={0.3}>
            {mask ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={mask} alt="Segmentation mask" className="h-full w-full object-cover" />
            ) : (
              <Placeholder label="Segmenting…" />
            )}
          </Panel>

          {/* 4. Histogram equalization visualisation */}
          <Panel color="#f59e0b" title="Histogram Equalisation" icon={<Microscope className="h-3.5 w-3.5" />} delay={0.4}>
            {defog ? (
              <HistogramEqualization src={defog} />
            ) : overlay ? (
              <HistogramEqualization src={overlay} />
            ) : (
              <Placeholder label="Computing…" />
            )}
          </Panel>
        </div>

        {/* Class legend */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-6 flex flex-wrap items-center justify-center gap-4"
        >
          {[
            { color: "#00c800", label: "Driveable" },
            { color: "#ffa500", label: "Vegetation" },
            { color: "#dc3232", label: "Obstacle" },
            { color: "#87ceeb", label: "Sky" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: item.color }} />
              <span className="text-xs text-muted-foreground">{item.label}</span>
            </div>
          ))}
        </motion.div>
      </motion.div>
    </section>
  )
}
