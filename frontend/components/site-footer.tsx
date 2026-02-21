"use client"

import { Cpu } from "lucide-react"

export function SiteFooter() {
  return (
    <footer className="relative z-10 border-t border-border bg-card/30 px-4 py-10 backdrop-blur-sm">
      <div className="mx-auto flex max-w-6xl flex-col items-center gap-4 text-center">
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-primary" />
          <span className="text-sm font-semibold text-foreground">
            Desert Perception System
          </span>
        </div>
        <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
          DINOv2 + SegFormer pipeline for off-road terrain
          segmentation. Built with Next.js, PyTorch, and FastAPI.
        </p>
        <div className="flex items-center gap-6">
          {["Architecture", "API Docs", "Dataset", "Research"].map(
            (link) => (
              <a
                key={link}
                href="#"
                className="text-xs text-muted-foreground transition-colors hover:text-primary"
              >
                {link}
              </a>
            )
          )}
        </div>
        <p className="text-xs text-muted-foreground/60">
          Off-Road Perception System v3.0 — 4-Class Segmentation
        </p>
      </div>
    </footer>
  )
}
