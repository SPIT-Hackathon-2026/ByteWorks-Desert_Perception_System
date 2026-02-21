"use client"

import { motion } from "framer-motion"
import { Upload, Cpu, Shield, Scan, Target, Zap } from "lucide-react"
import { Button } from "@/components/ui/button"

interface HeroSectionProps {
  onUploadClick: () => void
}

export function HeroSection({ onUploadClick }: HeroSectionProps) {
  return (
    <section className="relative flex min-h-[90vh] flex-col items-center justify-center px-4 py-20 text-center">
      {/* Background Image with subtle animation */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        <motion.div
          initial={{ scale: 1.1, opacity: 0 }}
          animate={{ scale: 1, opacity: 0.7 }}
          transition={{ duration: 2, ease: "easeOut" }}
          className="h-full w-full"
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/desert-bg.png"
            alt="Desert Background"
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-background/20 via-background/60 to-background" />
        </motion.div>
      </div>

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
          asChild
        >
          <a href="/demo.gif" target="_blank" rel="noopener noreferrer">
            <Scan className="h-5 w-5" />
            View Demo
          </a>
        </Button>
      </motion.div>


    </section>
  )
}
