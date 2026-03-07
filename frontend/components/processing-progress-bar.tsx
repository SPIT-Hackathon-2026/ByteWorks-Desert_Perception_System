"use client"

import { motion, AnimatePresence } from "framer-motion"
import { Loader2 } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

const STEPS = [
  "Input Capture",
  "Image Normalization",
  "Dehazing & Enhancement",
  "Feature Extraction",
  "Mix-Attention Sync",
  "U-Net Decoding",
  "Finalizing Output",
]

interface ProcessingProgressBarProps {
  isProcessing: boolean
  currentStep: number
  className?: string
}

export function ProcessingProgressBar({
  isProcessing,
  currentStep,
  className,
}: ProcessingProgressBarProps) {
  const progressPercent = Math.min(
    100,
    Math.max(0, ((currentStep + 1) / (STEPS.length + 1)) * 100)
  )
  const currentLabel = STEPS[Math.min(currentStep, STEPS.length - 1)] ?? "Processing"

  if (!isProcessing && currentStep < 0) return null

  return (
    <AnimatePresence>
      {isProcessing && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.3 }}
          className={cn(
            "fixed left-4 right-4 top-20 z-50 mx-auto max-w-2xl rounded-xl border border-primary/30 bg-card/95 px-4 py-4 shadow-xl backdrop-blur-xl",
            className
          )}
        >
          <div className="flex items-center gap-3">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
            >
              <Loader2 className="h-5 w-5 text-primary" />
            </motion.div>
            <div className="flex-1 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-foreground">
                  {currentLabel}
                </span>
                <span className="font-mono text-xs text-muted-foreground">
                  {Math.round(progressPercent)}%
                </span>
              </div>
              <Progress value={progressPercent} className="h-2" />
            </div>
          </div>
          <div className="mt-2 flex gap-1">
            {STEPS.map((_, i) => (
              <div
                key={i}
                className={cn(
                  "h-1 flex-1 rounded-full transition-all duration-300",
                  i <= currentStep ? "bg-primary" : "bg-muted"
                )}
              />
            ))}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
