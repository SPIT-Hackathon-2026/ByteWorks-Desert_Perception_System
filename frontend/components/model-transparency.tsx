"use client"

import { motion } from "framer-motion"
import {
  Brain,
  Database,
  Layers,
  Settings2,
  ArrowRight,
  GitBranch,
} from "lucide-react"

const architectureLayers = [
  { name: "Input", desc: "960×540 → 476×266 RGB", color: "#94a3b8" },
  { name: "DINOv2 ViT-S/14", desc: "Frozen backbone · 384-dim patch tokens", color: "#06b6d4" },
  { name: "SegFormer Transformer", desc: "4 blocks · 8 heads · efficient self-attention", color: "#22d3ee" },
  { name: "Mix-FFN Decoder", desc: "FC → DWConv → GELU → FC per block", color: "#f59e0b" },
  { name: "Output", desc: "4-class segmentation · 4.4M params", color: "#10b981" },
]

export function ModelTransparency() {
  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          Model Transparency
        </h2>
        <p className="mb-10 text-center text-sm text-muted-foreground">
          Architecture, training, and generalization strategy
        </p>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {/* Architecture diagram */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm lg:col-span-2"
          >
            <div className="mb-5 flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold text-foreground">
                DINOv2 + SegFormer Pipeline
              </h3>
            </div>

            <div className="flex flex-col items-center gap-2">
              {architectureLayers.map((layer, i) => (
                <div key={layer.name} className="flex w-full flex-col items-center">
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: i * 0.1 }}
                    className="flex w-full max-w-md items-center gap-4 rounded-lg border border-border bg-secondary/30 px-5 py-3"
                  >
                    <div
                      className="h-3 w-3 shrink-0 rounded-full"
                      style={{ backgroundColor: layer.color }}
                    />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-foreground">
                        {layer.name}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {layer.desc}
                      </p>
                    </div>
                  </motion.div>
                  {i < architectureLayers.length - 1 && (
                    <ArrowRight className="my-1 h-4 w-4 rotate-90 text-muted-foreground/30" />
                  )}
                </div>
              ))}
            </div>
          </motion.div>

          {/* Details cards */}
          <div className="flex flex-col gap-4">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
            >
              <div className="mb-3 flex items-center gap-2">
                <Settings2 className="h-4 w-4 text-accent" />
                <h3 className="text-sm font-semibold text-foreground">
                  Loss Function
                </h3>
              </div>
              <p className="mb-2 text-xs leading-relaxed text-muted-foreground">
                Class-weighted CrossEntropyLoss with inverse-frequency
                weights across 4 super-classes.
              </p>
              <div className="rounded-lg bg-secondary/50 p-3">
                <p className="font-mono text-xs text-primary">
                  {"L = CE(w_c) — weighted by class frequency"}
                </p>
                <p className="mt-1 font-mono text-xs text-muted-foreground">
                  {"AdamW lr=1e-3, OneCycleLR cosine"}
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
            >
              <div className="mb-3 flex items-center gap-2">
                <Database className="h-4 w-4 text-chart-4" />
                <h3 className="text-sm font-semibold text-foreground">
                  Dataset
                </h3>
              </div>
              <ul className="space-y-2 text-xs text-muted-foreground">
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                  4,176 off-road terrain images
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                  4 super-classes (Driveable, Vegetation, Obstacle, Sky)
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                  2,857 train / 317 val / 1,002 test
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-primary" />
                  NEAREST-mode mask interpolation
                </li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm"
            >
              <div className="mb-3 flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-chart-2" />
                <h3 className="text-sm font-semibold text-foreground">
                  Generalization
                </h3>
              </div>
              <ul className="space-y-2 text-xs text-muted-foreground">
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-chart-2" />
                  DINOv2 self-supervised pre-training
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-chart-2" />
                  4-class super-grouping for robust IoU
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-chart-2" />
                  Mixed precision (AMP) training
                </li>
                <li className="flex items-center gap-2">
                  <div className="h-1.5 w-1.5 rounded-full bg-chart-2" />
                  Inverse-frequency class weighting
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
      </motion.div>
    </section>
  )
}
