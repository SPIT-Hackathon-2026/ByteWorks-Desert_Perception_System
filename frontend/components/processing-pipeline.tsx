"use client"

import { motion } from "framer-motion"
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

const steps = [
  { icon: ImageIcon, label: "Input 960×540", color: "#94a3b8" },
  { icon: Grid3x3, label: "Resize 476×266", color: "#f59e0b" },
  { icon: Layers, label: "DINOv2 Backbone", color: "#06b6d4" },
  { icon: Wind, label: "Patch Tokens", color: "#22d3ee" },
  { icon: Sun, label: "SegFormer Head", color: "#8b5cf6" },
  { icon: ShieldAlert, label: "Argmax", color: "#ef4444" },
  { icon: CheckCircle2, label: "4-Class Output", color: "#10b981" },
]

interface ProcessingPipelineProps {
  isProcessing: boolean
  currentStep: number
}

export function ProcessingPipeline({
  isProcessing,
  currentStep,
}: ProcessingPipelineProps) {
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
        <p className="mb-12 text-center text-sm text-muted-foreground">
          Multi-stage perception and analysis workflow
        </p>

        {/* Desktop pipeline */}
        <div className="hidden items-center justify-center gap-1 lg:flex">
          {steps.map((step, i) => {
            const isActive = isProcessing && i <= currentStep
            const isCurrent = isProcessing && i === currentStep

            return (
              <div key={step.label} className="flex items-center">
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.1 }}
                  className={`relative flex flex-col items-center gap-2 rounded-xl border px-4 py-4 transition-all ${
                    isCurrent
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
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        ease: "easeInOut",
                      }}
                    />
                  )}
                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-lg"
                    style={{
                      backgroundColor: isActive
                        ? `${step.color}20`
                        : "rgba(148,163,184,0.1)",
                    }}
                  >
                    <step.icon
                      className="h-5 w-5"
                      style={{
                        color: isActive ? step.color : "#64748b",
                      }}
                    />
                  </div>
                  <span
                    className="text-center text-xs font-medium"
                    style={{
                      color: isActive ? step.color : "#64748b",
                    }}
                  >
                    {step.label}
                  </span>
                </motion.div>
                {i < steps.length - 1 && (
                  <ArrowRight
                    className={`mx-1 h-4 w-4 shrink-0 ${
                      isActive ? "text-primary" : "text-muted-foreground/30"
                    }`}
                  />
                )}
              </div>
            )
          })}
        </div>

        {/* Mobile pipeline */}
        <div className="flex flex-col gap-3 lg:hidden">
          {steps.map((step, i) => {
            const isActive = isProcessing && i <= currentStep
            const isCurrent = isProcessing && i === currentStep

            return (
              <motion.div
                key={step.label}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className={`flex items-center gap-3 rounded-lg border px-4 py-3 transition-all ${
                  isCurrent
                    ? "border-primary/50 bg-primary/10"
                    : isActive
                      ? "border-chart-4/30 bg-chart-4/5"
                      : "border-border bg-card/30"
                }`}
              >
                <div
                  className="flex h-8 w-8 items-center justify-center rounded-lg"
                  style={{
                    backgroundColor: isActive
                      ? `${step.color}20`
                      : "rgba(148,163,184,0.1)",
                  }}
                >
                  <step.icon
                    className="h-4 w-4"
                    style={{
                      color: isActive ? step.color : "#64748b",
                    }}
                  />
                </div>
                <span
                  className="text-sm font-medium"
                  style={{
                    color: isActive ? step.color : "#64748b",
                  }}
                >
                  {step.label}
                </span>
                {isCurrent && (
                  <motion.div
                    className="ml-auto h-2 w-2 rounded-full bg-primary"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  />
                )}
              </motion.div>
            )
          })}
        </div>
      </motion.div>
    </section>
  )
}
