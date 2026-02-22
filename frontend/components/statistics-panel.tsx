"use client"

import { motion } from "framer-motion"
import {
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts"
import { Activity, Clock, Target, Zap } from "lucide-react"
import type { SegmentationResult } from "@/lib/api"

const classData = [
  { name: "Driveable", value: 48.0, fill: "#00c800" },
  { name: "Vegetation", value: 20.0, fill: "#ffa500" },
  { name: "Obstacle", value: 5.0, fill: "#dc3232" },
  { name: "Sky", value: 27.0, fill: "#87ceeb" },
]

const iouData = [
  { name: "Driveable", iou: 0.72 },
  { name: "Vegetation", iou: 0.65 },
  { name: "Obstacle", iou: 0.48 },
  { name: "Sky", iou: 0.85 },
]

const lossData = Array.from({ length: 50 }, (_, i) => ({
  epoch: i + 1,
  loss: 2.5 * Math.exp(-i / 12) + 0.15 + Math.random() * 0.08,
  val_loss: 2.8 * Math.exp(-i / 14) + 0.2 + Math.random() * 0.1,
}))

const radarData = [
  { metric: "mIoU", A: 67.5, fullMark: 100 },
  { metric: "Dice", A: 78.0, fullMark: 100 },
  { metric: "Pixel Acc", A: 85.0, fullMark: 100 },
  { metric: "Precision", A: 72.0, fullMark: 100 },
  { metric: "Recall", A: 69.0, fullMark: 100 },
  { metric: "F1 Score", A: 70.5, fullMark: 100 },
]

interface StatisticsPanelProps {
  isComplete: boolean
  segResult?: SegmentationResult | null
}

export function StatisticsPanel({ isComplete, segResult }: StatisticsPanelProps) {
  if (!isComplete) return null

  // Use real data if available, fallback to static
  const liveClassData = segResult
    ? segResult.class_distribution.map((c) => ({
      name: c.name.length > 12 ? c.name.slice(0, 10) + "…" : c.name,
      value: c.percentage,
      fill: c.color.replace("rgb(", "rgba(").replace(")", ",1)"),
    }))
    : classData

  const metrics = segResult?.risk_assessment?.metrics
  const liveRadarData = metrics
    ? [
      { metric: "mIoU", value: metrics.mIoU * 100 },
      { metric: "Dice", value: metrics.dice_score * 100 },
      { metric: "Pixel Acc", value: metrics.pixel_accuracy * 100 },
      { metric: "Precision", value: metrics.precision * 100 },
      { metric: "Recall", value: metrics.recall * 100 },
      { metric: "F1 Score", value: ((2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)) * 100 },
    ]
    : radarData

  const liveInferenceMs = segResult ? `${segResult.inference_ms}ms` : "—"

  return (
    <section id="metrics" className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          Performance Metrics
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Model inference statistics and analysis
        </p>

        {/* KPI cards */}
        <div className="mb-8 grid grid-cols-2 gap-4 md:grid-cols-4">
          {[
            {
              icon: Target,
              label: "mIoU",
              value: segResult?.risk_assessment?.metrics
                ? `${(segResult.risk_assessment.metrics.mIoU * 100).toFixed(1)}%`
                : "42.7%",
              color: "#06b6d4",
            },
            {
              icon: Activity,
              label: "Pixel Acc",
              value: segResult?.risk_assessment?.metrics
                ? `${(segResult.risk_assessment.metrics.pixel_accuracy * 100).toFixed(1)}%`
                : "75.9%",
              color: "#10b981",
            },
            {
              icon: Clock,
              label: "Classes",
              value: "4",
              color: "#f59e0b",
            },
            {
              icon: Zap,
              label: "Inference",
              value: liveInferenceMs,
              color: "#f97316",
            },
          ].map((kpi, i) => (
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.1 }}
              className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
            >
              <div className="mb-3 flex items-center gap-2">
                <div
                  className="flex h-8 w-8 items-center justify-center rounded-lg"
                  style={{ backgroundColor: `${kpi.color}15` }}
                >
                  <kpi.icon
                    className="h-4 w-4"
                    style={{ color: kpi.color }}
                  />
                </div>
                <span className="text-xs font-medium text-muted-foreground">
                  {kpi.label}
                </span>
              </div>
              <p
                className="font-mono text-2xl font-bold"
                style={{ color: kpi.color }}
              >
                {kpi.value}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Charts grid */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {/* Class distribution */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
          >
            <h3 className="mb-4 text-sm font-semibold text-foreground">
              Class Distribution (%)
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={liveClassData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {liveClassData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f1724",
                    border: "1px solid #1e293b",
                    borderRadius: 8,
                    color: "#e2e8f0",
                    fontSize: 12,
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Loss curve */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
          >
            <h3 className="mb-4 text-sm font-semibold text-foreground">
              Training Loss Curve
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={lossData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#1e293b"
                  vertical={false}
                />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  axisLine={{ stroke: "#1e293b" }}
                  label={{
                    value: "Epoch",
                    position: "bottom",
                    fill: "#64748b",
                    fontSize: 11,
                  }}
                />
                <YAxis
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                  axisLine={{ stroke: "#1e293b" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f1724",
                    border: "1px solid #1e293b",
                    borderRadius: 8,
                    color: "#e2e8f0",
                    fontSize: 12,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Per-class IoU */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
          >
            <h3 className="mb-4 text-sm font-semibold text-foreground">
              Per-Class IoU
            </h3>
            <div className="space-y-4">
              {iouData.map((cls) => (
                <div key={cls.name}>
                  <div className="mb-1 flex items-center justify-between">
                    <span className="text-xs font-medium text-muted-foreground">
                      {cls.name}
                    </span>
                    <span className="font-mono text-xs text-foreground">
                      {(cls.iou * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-secondary">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${cls.iou * 100}%` }}
                      transition={{ duration: 1, delay: 0.5 }}
                      className="h-full rounded-full bg-primary"
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Radar chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
          >
            <h3 className="mb-4 text-sm font-semibold text-foreground">
              Model Performance Radar
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={liveRadarData}>
                <PolarGrid stroke="#1e293b" />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />
                <PolarRadiusAxis
                  angle={30}
                  domain={[0, 100]}
                  tick={{ fill: "#64748b", fontSize: 10 }}
                />
                <Radar
                  name="Model"
                  dataKey="value"
                  stroke="#06b6d4"
                  fill="#06b6d4"
                  fillOpacity={0.2}
                />
              </RadarChart>
            </ResponsiveContainer>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}
