"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Eye, EyeOff, Layers, Thermometer } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import type { SegmentationResult } from "@/lib/api"

interface OutputDashboardProps {
  originalImage: string | null
  isComplete: boolean
  segResult?: SegmentationResult | null
}

export function OutputDashboard({ originalImage, isComplete, segResult }: OutputDashboardProps) {
  const [showOverlay, setShowOverlay] = useState(true)
  const [showConfidence, setShowConfidence] = useState(false)
  const [defogSlider, setDefogSlider] = useState([50])

  if (!isComplete || !originalImage) return null

  const hasRealData = !!segResult
  const maskSrc = hasRealData ? `data:image/png;base64,${segResult.mask_b64}` : undefined
  const overlaySrc = hasRealData ? `data:image/png;base64,${segResult.overlay_b64}` : undefined

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          Analysis Output
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Semantic segmentation results and terrain classification
        </p>

        {/* Toggle controls */}
        <div className="mb-6 flex flex-wrap items-center justify-center gap-3">
          <Button
            variant={showOverlay ? "default" : "outline"}
            size="sm"
            className={`gap-2 ${showOverlay ? "bg-primary text-primary-foreground" : "border-border text-foreground"}`}
            onClick={() => setShowOverlay(!showOverlay)}
          >
            {showOverlay ? (
              <Eye className="h-4 w-4" />
            ) : (
              <EyeOff className="h-4 w-4" />
            )}
            Segmentation Overlay
          </Button>
          <Button
            variant={showConfidence ? "default" : "outline"}
            size="sm"
            className={`gap-2 ${showConfidence ? "bg-primary text-primary-foreground" : "border-border text-foreground"}`}
            onClick={() => setShowConfidence(!showConfidence)}
          >
            <Thermometer className="h-4 w-4" />
            Confidence Map
          </Button>
        </div>

        {/* Image grid */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Original */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="overflow-hidden rounded-xl border border-border bg-card/50"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <div className="h-2 w-2 rounded-full bg-chart-4" />
              <span className="text-xs font-medium text-muted-foreground">
                Original Image
              </span>
            </div>
            <div className="relative aspect-video">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={originalImage}
                alt="Original terrain image"
                className="h-full w-full object-cover"
              />
            </div>
          </motion.div>

          {/* Segmentation */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="overflow-hidden rounded-xl border border-border bg-card/50"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <div className="h-2 w-2 rounded-full bg-primary" />
              <span className="text-xs font-medium text-muted-foreground">
                Semantic Segmentation
              </span>
              <Layers className="ml-auto h-3.5 w-3.5 text-primary" />
            </div>
            <div className="relative aspect-video bg-card">
              {hasRealData ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={showOverlay ? overlaySrc : maskSrc}
                    alt="Segmentation result"
                    className="h-full w-full object-cover"
                  />
                </>
              ) : (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={originalImage}
                    alt="Semantic segmentation overlay"
                    className="h-full w-full object-cover"
                  />
                  {showOverlay && (
                    <div className="absolute inset-0 mix-blend-multiply">
                      <SegmentationOverlay />
                    </div>
                  )}
                </>
              )}
            </div>
          </motion.div>

          {/* Heatmap */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="overflow-hidden rounded-xl border border-border bg-card/50"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <div className="h-2 w-2 rounded-full bg-accent" />
              <span className="text-xs font-medium text-muted-foreground">
                Confidence Heatmap
              </span>
              <Thermometer className="ml-auto h-3.5 w-3.5 text-accent" />
            </div>
            <div className="relative aspect-video bg-card">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={originalImage}
                alt="Confidence heatmap"
                className="h-full w-full object-cover opacity-60"
              />
              <div className="absolute inset-0">
                <HeatmapOverlay />
              </div>
            </div>
          </motion.div>

          {/* Defog comparison */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="overflow-hidden rounded-xl border border-border bg-card/50"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <div className="h-2 w-2 rounded-full bg-chart-2" />
              <span className="text-xs font-medium text-muted-foreground">
                Defog Comparison
              </span>
            </div>
            <div className="relative aspect-video bg-card">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={originalImage}
                alt="Defog comparison"
                className="h-full w-full object-cover"
                style={{
                  filter: `contrast(${1 + defogSlider[0] / 100}) brightness(${1 + defogSlider[0] / 200})`,
                }}
              />
              <div className="absolute bottom-0 left-0 right-0 bg-background/60 p-3 backdrop-blur-sm">
                <div className="flex items-center gap-3">
                  <span className="text-xs text-muted-foreground">
                    Before
                  </span>
                  <Slider
                    value={defogSlider}
                    onValueChange={setDefogSlider}
                    max={100}
                    step={1}
                    className="flex-1"
                    aria-label="Defog intensity slider"
                  />
                  <span className="text-xs text-muted-foreground">After</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Legend */}
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
              <div
                className="h-3 w-3 rounded-sm"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-xs text-muted-foreground">
                {item.label}
              </span>
            </div>
          ))}
        </motion.div>
      </motion.div>
    </section>
  )
}

function SegmentationOverlay() {
  return (
    <svg
      viewBox="0 0 400 225"
      className="h-full w-full"
      aria-hidden="true"
      preserveAspectRatio="none"
    >
      <defs>
        <linearGradient id="seg-sand" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#d97706" stopOpacity="0.6" />
        </linearGradient>
        <linearGradient id="seg-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#0891b2" stopOpacity="0.3" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="400" height="90" fill="url(#seg-sky)" />
      <path
        d="M0 90 Q100 70 200 95 Q300 120 400 85 L400 160 Q300 170 200 155 Q100 140 0 165 Z"
        fill="url(#seg-sand)"
      />
      <ellipse cx="320" cy="180" rx="60" ry="30" fill="#64748b" opacity="0.4" />
      <ellipse cx="80" cy="190" rx="40" ry="20" fill="#10b981" opacity="0.4" />
      <rect
        x="180"
        y="175"
        width="30"
        height="25"
        rx="4"
        fill="#ef4444"
        opacity="0.5"
      />
    </svg>
  )
}

function HeatmapOverlay() {
  return (
    <svg
      viewBox="0 0 400 225"
      className="h-full w-full"
      aria-hidden="true"
      preserveAspectRatio="none"
    >
      <defs>
        <radialGradient id="heat1" cx="30%" cy="70%" r="40%">
          <stop offset="0%" stopColor="#ef4444" stopOpacity="0.7" />
          <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0.1" />
        </radialGradient>
        <radialGradient id="heat2" cx="70%" cy="40%" r="35%">
          <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.6" />
          <stop offset="60%" stopColor="#10b981" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
        </radialGradient>
        <radialGradient id="heat3" cx="50%" cy="85%" r="30%">
          <stop offset="0%" stopColor="#ef4444" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.2" />
        </radialGradient>
      </defs>
      <rect x="0" y="0" width="400" height="225" fill="url(#heat1)" />
      <rect x="0" y="0" width="400" height="225" fill="url(#heat2)" />
      <rect x="0" y="0" width="400" height="225" fill="url(#heat3)" />
    </svg>
  )
}
