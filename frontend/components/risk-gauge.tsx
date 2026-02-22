"use client"

import { motion } from "framer-motion"
import { Shield, AlertTriangle, Zap, MapPin } from "lucide-react"
import { cn } from "@/lib/utils"
import type { SegmentationResult } from "@/lib/api"

function computeRiskFactors(segResult?: SegmentationResult | null) {
  // ── Prefer live risk_assessment from the backend (UGVEnsemble) ──────────
  if (segResult?.risk_assessment) {
    const r = segResult.risk_assessment
    return {
      irReflection: r.sensors?.ir_ratio ?? 0,
      ultrasoundDist: r.sensors?.dist_cm ?? 0,
      camoufProb: r.ensemble?.camouf_ensemble ?? 0,
      terrainHazard: r.risk_score, // Primary hazard from fused model
      fromEnsemble: !!r.ensemble,
    }
  }

  // ── Fallback logic (Synthetic) ──────────────────────────────────────────
  const pct = (name: string) =>
    segResult?.class_distribution?.find(c => c.name === name)?.percentage ?? 0

  const obstacle = pct("Obstacle")
  const driveable = pct("Driveable")
  const rock = pct("Rock") || pct("Rough")

  return {
    irReflection: Math.min((obstacle + rock * 0.5) / 100, 1),
    ultrasoundDist: Math.max(5, (driveable / 100) * 400),
    camoufProb: Math.min((obstacle * 1.2) / 100, 1),
    terrainHazard: Math.min((obstacle * 2) / 100, 1),
    fromEnsemble: false,
  }
}

interface RiskGaugeProps {
  isComplete: boolean
  segResult?: SegmentationResult | null
}

export function RiskGauge({ isComplete, segResult }: RiskGaugeProps) {
  if (!isComplete) return null

  const factors = computeRiskFactors(segResult)

  const riskScore = segResult?.risk_assessment?.risk_score ?? factors.terrainHazard
  const riskLevel = segResult?.risk_assessment?.risk_level
    ?? (riskScore < 0.3 ? "LOW" : riskScore < 0.6 ? "MEDIUM" : "HIGH")

  const riskColor =
    riskLevel === "LOW" ? "#10b981" : riskLevel === "MEDIUM" ? "#f59e0b" : "#ef4444"
  const riskAngle = riskScore * 180

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          Terrain Map
        </h2>
        <p className="mb-2 text-center text-sm text-muted-foreground">
          Sensor-fused obstacle and complexity analysis
        </p>

        {factors.fromEnsemble && (
          <div className="mb-6 flex justify-center">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-500">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              UGV Ensemble Active
            </span>
          </div>
        )}
        {!factors.fromEnsemble && <div className="mb-8" />}

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {/* Circular gauge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col items-center rounded-xl border border-border bg-card/50 p-8 backdrop-blur-sm md:col-span-1"
          >
            <div className="relative mb-6">
              <svg width="200" height="120" viewBox="0 0 200 120">
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="12"
                  strokeLinecap="round"
                />
                <motion.path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke={riskColor}
                  strokeWidth="12"
                  strokeLinecap="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: riskScore }}
                  transition={{ duration: 1.5, delay: 0.3 }}
                />
                <motion.line
                  x1="100" y1="100" x2="100" y2="30"
                  stroke={riskColor}
                  strokeWidth="2"
                  strokeLinecap="round"
                  initial={{ rotate: -90 }}
                  animate={{ rotate: riskAngle - 90 }}
                  transition={{ duration: 1.5, delay: 0.3, type: "spring" }}
                  style={{ transformOrigin: "100px 100px" }}
                />
                <circle cx="100" cy="100" r="4" fill={riskColor} />
              </svg>
            </div>
            <div className="text-center">
              <p className="font-mono text-3xl font-bold" style={{ color: riskColor }}>
                {(riskScore * 100).toFixed(0)}%
              </p>
              <p className="mt-1 text-sm font-semibold tracking-wider" style={{ color: riskColor }}>
                {riskLevel} HAZARD
              </p>
            </div>
          </motion.div>

          {/* Breakdown */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm md:col-span-2"
          >
            <h3 className="mb-5 text-sm font-semibold text-foreground">
              Hardware Dataset Signals
            </h3>
            <div className="space-y-4">
              <RiskFactor
                label="IR Reflection Density"
                value={factors.irReflection}
                color="text-amber-500"
                barColor="bg-amber-500"
                icon={<Zap className="h-4 w-4" />}
                description="Surface reflectivity proxy"
              />
              <RiskFactor
                label="Ultrasonic Distance"
                value={factors.ultrasoundDist / 400}
                rawValue={`${factors.ultrasoundDist.toFixed(1)} cm`}
                color="text-blue-500"
                barColor="bg-blue-500"
                icon={<MapPin className="h-4 w-4" />}
                description="Sonar clearance measurement"
              />
              <RiskFactor
                label="Camouflage Probability"
                value={factors.camoufProb}
                color="text-emerald-500"
                barColor="bg-emerald-500"
                icon={<Shield className="h-4 w-4" />}
                description="Ensemble: Hidden hazard detection"
              />
              <RiskFactor
                label="Terrain Hazard Index"
                value={factors.terrainHazard}
                color="text-rose-500"
                barColor="bg-rose-500"
                icon={<AlertTriangle className="h-4 w-4" />}
                description="Combined sensor-fusion risk"
              />
            </div>

            <div className="mt-6 rounded-lg border border-border bg-secondary/50 p-4">
              <p className="font-mono text-xs text-muted-foreground">
                <span className="text-foreground">Signal Processing:</span>{" "}
                f(IR, Sonar) → UGV-Ensemble → Hazard Score
              </p>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}

interface RiskFactorProps {
  label: string
  value: number
  rawValue?: string
  color: string
  barColor: string
  icon: React.ReactNode
  description: string
}

function RiskFactor({ label, value, rawValue, color, barColor, icon, description }: RiskFactorProps) {
  return (
    <div className="group space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 font-medium text-foreground/90">
          <div className={cn("rounded-md p-1 bg-background/50 border border-border/50", color)}>
            {icon}
          </div>
          <div>
            <p>{label}</p>
            <p className="text-[10px] text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100">
              {description}
            </p>
          </div>
        </div>
        <span className={cn("font-mono font-bold", color)}>
          {rawValue || `${(value * 100).toFixed(0)}%`}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(value, 1) * 100}%` }}
          transition={{ duration: 1, delay: 0.5 }}
          className={cn("h-full rounded-full", barColor)}
        />
      </div>
    </div>
  )
}
