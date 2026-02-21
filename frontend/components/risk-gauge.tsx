"use client"

import { motion } from "framer-motion"
import { ShieldAlert, AlertTriangle, CheckCircle } from "lucide-react"
import type { SegmentationResult } from "@/lib/api"

// Obstacle class names — higher % → more risk
const OBSTACLE_CLASSES = new Set(["Obstacle"])
const OPEN_CLASSES = new Set(["Sky", "Driveable"])

function computeRiskFactors(segResult?: SegmentationResult | null) {
  if (!segResult) {
    return { obstacleDensity: 0.25, uncertainty: 0.38, terrainComplexity: 0.45, visibility: 0.82 }
  }
  const dist = segResult.class_distribution
  const obstacleDensity = dist
    .filter((c) => OBSTACLE_CLASSES.has(c.name))
    .reduce((s, c) => s + c.percentage, 0) / 100

  const openArea = dist
    .filter((c) => OPEN_CLASSES.has(c.name))
    .reduce((s, c) => s + c.percentage, 0) / 100

  // How many distinct classes are present → complexity
  const activeClasses = dist.filter((c) => c.percentage > 0.5).length
  const terrainComplexity = Math.min(activeClasses / 4, 1)

  const visibility = Math.min(openArea + 0.3, 1)
  const uncertainty = 1 - visibility

  return { obstacleDensity, uncertainty, terrainComplexity, visibility }
}

interface RiskGaugeProps {
  isComplete: boolean
  segResult?: SegmentationResult | null
}

export function RiskGauge({ isComplete, segResult }: RiskGaugeProps) {
  if (!isComplete) return null

  const factors = computeRiskFactors(segResult)
  const riskScore = 0.4 * factors.obstacleDensity + 0.3 * factors.uncertainty + 0.2 * factors.terrainComplexity + 0.1 * (1 - factors.visibility)
  const riskLevel =
    riskScore < 0.3 ? "LOW" : riskScore < 0.6 ? "MEDIUM" : "HIGH"
  const riskColor =
    riskScore < 0.3 ? "#10b981" : riskScore < 0.6 ? "#f59e0b" : "#ef4444"
  const riskAngle = riskScore * 180

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          Risk Assessment
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Weighted obstacle and uncertainty analysis
        </p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {/* Circular gauge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="flex flex-col items-center rounded-xl border border-border bg-card/50 p-8 backdrop-blur-sm md:col-span-1"
          >
            <div className="relative mb-6">
              <svg width="200" height="120" viewBox="0 0 200 120" aria-label={`Risk gauge showing ${riskLevel} risk at ${(riskScore * 100).toFixed(0)} percent`}>
                {/* Background arc */}
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="12"
                  strokeLinecap="round"
                />
                {/* Colored arc */}
                <motion.path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke={riskColor}
                  strokeWidth="12"
                  strokeLinecap="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: riskScore }}
                  transition={{ duration: 1.5, delay: 0.3 }}
                  style={{ filter: `drop-shadow(0 0 6px ${riskColor}40)` }}
                />
                {/* Needle */}
                <motion.line
                  x1="100"
                  y1="100"
                  x2="100"
                  y2="30"
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
              <p
                className="font-mono text-3xl font-bold"
                style={{ color: riskColor }}
              >
                {(riskScore * 100).toFixed(0)}%
              </p>
              <p
                className="mt-1 text-sm font-semibold tracking-wider"
                style={{ color: riskColor }}
              >
                {riskLevel} RISK
              </p>
            </div>
          </motion.div>

          {/* Risk breakdown */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm md:col-span-2"
          >
            <h3 className="mb-5 text-sm font-semibold text-foreground">
              Risk Factor Breakdown
            </h3>
            <div className="space-y-5">
              {[
                {
                  label: "Obstacle Density",
                  value: factors.obstacleDensity,
                  weight: 0.4,
                  icon: ShieldAlert,
                  color: "#ef4444",
                },
                {
                  label: "Uncertainty Score",
                  value: factors.uncertainty,
                  weight: 0.3,
                  icon: AlertTriangle,
                  color: "#f59e0b",
                },
                {
                  label: "Terrain Complexity",
                  value: factors.terrainComplexity,
                  weight: 0.2,
                  icon: AlertTriangle,
                  color: "#f59e0b",
                },
                {
                  label: "Visibility Score",
                  value: factors.visibility,
                  weight: 0.1,
                  icon: CheckCircle,
                  color: "#10b981",
                },
              ].map((factor) => (
                <div key={factor.label}>
                  <div className="mb-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <factor.icon
                        className="h-4 w-4"
                        style={{ color: factor.color }}
                      />
                      <span className="text-sm text-foreground">
                        {factor.label}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-muted-foreground">
                        w={factor.weight}
                      </span>
                      <span
                        className="font-mono text-sm font-medium"
                        style={{ color: factor.color }}
                      >
                        {(factor.value * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${factor.value * 100}%` }}
                      transition={{ duration: 1, delay: 0.5 }}
                      className="h-full rounded-full"
                      style={{ backgroundColor: factor.color }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 rounded-lg border border-border bg-secondary/50 p-4">
              <p className="font-mono text-xs text-muted-foreground">
                {"Risk = "}
                <span className="text-foreground">
                  {"sum(w_i * factor_i)"}
                </span>
                {` = 0.4(${factors.obstacleDensity.toFixed(2)}) + 0.3(${factors.uncertainty.toFixed(2)}) + 0.2(${factors.terrainComplexity.toFixed(2)}) + 0.1(${(1 - factors.visibility).toFixed(2)}) = `}
                <span className="font-bold" style={{ color: riskColor }}>
                  {riskScore.toFixed(3)}
                </span>
              </p>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}
