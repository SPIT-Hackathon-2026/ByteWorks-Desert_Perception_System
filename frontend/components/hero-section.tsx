"use client"

import { motion } from "framer-motion"
import { Upload, Cpu, Shield, Scan } from "lucide-react"
import { Button } from "@/components/ui/button"

interface HeroSectionProps {
  onUploadClick: () => void
}

export function HeroSection({ onUploadClick }: HeroSectionProps) {
  return (
    <section className="relative flex min-h-[90vh] flex-col items-center justify-center px-4 py-20 text-center">
      {/* Radial glow behind title */}
      <div
        className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
        style={{
          width: 800,
          height: 800,
          background:
            "radial-gradient(circle, rgba(6,182,212,0.08) 0%, rgba(6,182,212,0.02) 40%, transparent 70%)",
        }}
        aria-hidden="true"
      />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-10"
      >
        <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
          </span>
          <span className="text-sm font-medium text-primary">
            System Online
          </span>
        </div>

        <h1 className="mx-auto max-w-4xl text-balance text-5xl font-bold tracking-tight text-foreground md:text-7xl">
          Autonomous Desert{" "}
          <span className="bg-gradient-to-r from-primary to-[#22d3ee] bg-clip-text text-transparent">
            Perception System
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-pretty text-lg leading-relaxed text-muted-foreground md:text-xl">
          Robust Semantic Segmentation for Unseen Environments. AI-powered
          terrain analysis for Unmanned Ground Vehicle navigation in hostile
          desert conditions.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
        className="relative z-10 mt-10 flex flex-wrap items-center justify-center gap-4"
      >
        <Button
          size="lg"
          className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
          onClick={onUploadClick}
        >
          <Upload className="h-5 w-5" />
          Upload Image
        </Button>
        <Button
          size="lg"
          variant="outline"
          className="gap-2 border-border text-foreground hover:bg-secondary"
        >
          <Scan className="h-5 w-5" />
          View Demo
        </Button>
      </motion.div>

      {/* Feature cards */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
        className="relative z-10 mt-20 grid w-full max-w-5xl grid-cols-1 gap-4 md:grid-cols-3"
      >
        {[
          {
            icon: Cpu,
            title: "DINOv2 + SegFormer",
            description:
              "Frozen DINOv2 ViT-S/14 backbone with SegFormer transformer head for 4-class segmentation",
          },
          {
            icon: Target,
            title: "4 Super-Classes",
            description:
              "Driveable, Vegetation, Obstacle, Sky — optimised for UGV navigation with high mIoU",
          },
          {
            icon: Zap,
            title: "Real-Time Terrain Analysis",
            description:
              "Trees, bushes, grass, flowers, logs, rocks, landscape & sky — full off-road segmentation",
          },
        ].map((feature, i) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 + i * 0.15 }}
            className="group rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm transition-colors hover:border-primary/30 hover:bg-card/80"
          >
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
              <feature.icon className="h-5 w-5 text-primary" />
            </div>
            <h3 className="mb-2 text-sm font-semibold text-foreground">
              {feature.title}
            </h3>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {feature.description}
            </p>
          </motion.div>
        ))}
      </motion.div>
    </section>
  )
}
