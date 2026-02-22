import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, Cpu, Shield, Scan, Target, Zap, Activity, Eye, Compass } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

function useTypewriter(text: string, speed: number = 25) {
  const [displayedText, setDisplayedText] = useState("")
  useEffect(() => {
    let i = 0
    const timer = setInterval(() => {
      setDisplayedText(text.slice(0, i))
      i++
      if (i > text.length) clearInterval(timer)
    }, speed)
    return () => clearInterval(timer)
  }, [text, speed])
  return displayedText
}

interface HeroSectionProps {
  onUploadClick: () => void
}

export function HeroSection({ onUploadClick }: HeroSectionProps) {
  const typingText = useTypewriter("Robust Semantic Segmentation for Unseen Environments. AI-powered terrain analysis for Unmanned Ground Vehicle navigation in hostile desert conditions.", 20)

  return (
    <section className="relative flex min-h-[90vh] flex-col items-center justify-center overflow-hidden px-4 py-20 text-center">
      {/* Background with floating HUD elements */}
      <div className="absolute inset-0 z-0">
        <motion.div
          initial={{ scale: 1.1, opacity: 0 }}
          animate={{ scale: 1, opacity: 0.92 }}
          transition={{ duration: 2, ease: "easeOut" }}
          className="h-full w-full"
        >
          <img
            src="/desert-bg.png"
            alt="Desert"
            className="h-full w-full object-cover"
          />
          <div className="absolute inset-0 bg-linear-to-b from-background/5 via-background/30 to-background dark:from-background/10 dark:via-background/40 dark:to-background" />
        </motion.div>

        {/* Floating HUD Elements */}
        <FloatingIcon icon={Activity} top="20%" left="15%" delay={0} color="text-primary/40" />
        <FloatingIcon icon={Eye} top="15%" right="20%" delay={1} color="text-[#22d3ee]/30" />
        <FloatingIcon icon={Target} bottom="30%" left="10%" delay={2} color="text-primary/30" />
        <FloatingIcon icon={Cpu} bottom="25%" right="15%" delay={0.5} color="text-[#22d3ee]/40" />
        <FloatingIcon icon={Compass} top="40%" right="5%" delay={1.5} color="text-primary/20" />
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
          <span className="text-sm font-medium uppercase tracking-tighter text-primary">
            UGV Perception Active
          </span>
        </div>

        <h1 className="mx-auto max-w-4xl text-balance text-5xl font-bold tracking-tight text-foreground md:text-7xl">
          Autonomous Desert{" "}
          <span className="bg-linear-to-r from-primary to-[#22d3ee] bg-clip-text text-transparent">
            Perception System
          </span>
        </h1>

        <div className="mx-auto mt-6 max-w-2xl">
          <p className="min-h-16 text-pretty font-mono text-sm leading-relaxed text-muted-foreground md:text-lg">
            {typingText}
            <span className="ml-1 inline-block h-4 w-1 animate-pulse bg-primary" />
          </p>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
        className="relative z-10 mt-10 flex flex-wrap items-center justify-center gap-4"
      >
        <Button
          size="lg"
          className="group gap-2 bg-primary px-8 text-primary-foreground hover:bg-primary/90"
          onClick={onUploadClick}
        >
          <Upload className="h-5 w-5 transition-transform group-hover:-translate-y-1" />
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

function FloatingIcon({ icon: Icon, top, left, right, bottom, delay, color }: any) {
  return (
    <motion.div
      className={cn("absolute z-10", color)}
      style={{ top, left, right, bottom }}
      animate={{
        y: [0, -20, 0],
        opacity: [0.2, 0.5, 0.2],
        scale: [1, 1.1, 1],
      }}
      transition={{
        duration: 4 + Math.random() * 2,
        repeat: Infinity,
        delay,
        ease: "easeInOut",
      }}
    >
      <Icon className="h-8 w-8 stroke-[1px] md:h-12 md:w-12" />
    </motion.div>
  )
}
