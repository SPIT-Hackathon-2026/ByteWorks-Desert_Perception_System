"use client"

import { useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { Box, Route } from "lucide-react"

export function Placeholder3D() {
  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          3D Visualization
        </h2>
        <p className="mb-10 text-center text-sm text-muted-foreground">
          Interactive terrain and navigation preview
        </p>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Neural network 3D placeholder */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="overflow-hidden rounded-xl border border-border bg-card/30"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <Box className="h-3.5 w-3.5 text-primary" />
              <span className="text-xs font-medium text-muted-foreground">
                Neural Network Visualization
              </span>
            </div>
            <div className="relative flex h-[400px] items-center justify-center">
              <NeuralNet3DPlaceholder />
              <div className="absolute bottom-4 left-4 right-4 rounded-lg bg-background/60 p-3 text-center backdrop-blur-sm">
                <p className="text-xs text-muted-foreground">
                  React Three Fiber neural network layer visualization
                </p>
              </div>
            </div>
          </motion.div>

          {/* UGV Path placeholder */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="overflow-hidden rounded-xl border border-border bg-card/30"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <Route className="h-3.5 w-3.5 text-accent" />
              <span className="text-xs font-medium text-muted-foreground">
                UGV Path Simulation
              </span>
            </div>
            <div className="relative flex h-[400px] items-center justify-center">
              <TerrainPlaceholder />
              <div className="absolute bottom-4 left-4 right-4 rounded-lg bg-background/60 p-3 text-center backdrop-blur-sm">
                <p className="text-xs text-muted-foreground">
                  3D desert terrain with UGV navigation path
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}

function NeuralNet3DPlaceholder() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)

    const w = canvas.offsetWidth
    const h = canvas.offsetHeight

    let frame = 0
    let animId: number

    const layers = [4, 8, 12, 8, 6, 4]
    const layerX = layers.map((_, i) => 60 + (i * (w - 120)) / (layers.length - 1))

    function draw() {
      if (!ctx) return
      ctx.clearRect(0, 0, w, h)
      frame += 0.02

      // Draw connections
      for (let l = 0; l < layers.length - 1; l++) {
        for (let i = 0; i < layers[l]; i++) {
          for (let j = 0; j < layers[l + 1]; j++) {
            const y1 = h / 2 + (i - (layers[l] - 1) / 2) * 22
            const y2 = h / 2 + (j - (layers[l + 1] - 1) / 2) * 22
            const pulse =
              Math.sin(frame + l * 0.5 + i * 0.3 + j * 0.2) * 0.5 + 0.5
            ctx.beginPath()
            ctx.moveTo(layerX[l], y1)
            ctx.lineTo(layerX[l + 1], y2)
            ctx.strokeStyle = `rgba(6, 182, 212, ${0.05 + pulse * 0.1})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }

      // Draw nodes
      for (let l = 0; l < layers.length; l++) {
        for (let i = 0; i < layers[l]; i++) {
          const y = h / 2 + (i - (layers[l] - 1) / 2) * 22
          const pulse =
            Math.sin(frame * 2 + l * 0.8 + i * 0.4) * 0.5 + 0.5

          ctx.beginPath()
          ctx.arc(layerX[l], y, 4 + pulse * 2, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(6, 182, 212, ${0.3 + pulse * 0.5})`
          ctx.fill()

          ctx.beginPath()
          ctx.arc(layerX[l], y, 2, 0, Math.PI * 2)
          ctx.fillStyle = `rgba(6, 182, 212, ${0.6 + pulse * 0.4})`
          ctx.fill()
        }
      }

      animId = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animId)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full"
      style={{ imageRendering: "auto" }}
      aria-label="Animated neural network visualization placeholder"
    />
  )
}

function TerrainPlaceholder() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)

    const w = canvas.offsetWidth
    const h = canvas.offsetHeight

    let frame = 0
    let animId: number

    function draw() {
      if (!ctx) return
      ctx.clearRect(0, 0, w, h)
      frame += 0.01

      // Draw perspective grid (terrain)
      const horizon = h * 0.35
      ctx.strokeStyle = "rgba(245, 158, 11, 0.1)"
      ctx.lineWidth = 0.5

      // Horizontal lines
      for (let i = 0; i < 20; i++) {
        const t = i / 20
        const y = horizon + (h - horizon) * t * t
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      // Vertical lines (perspective)
      for (let i = -10; i <= 10; i++) {
        const topX = w / 2 + i * 8
        const bottomX = w / 2 + i * (w / 4)
        ctx.beginPath()
        ctx.moveTo(topX, horizon)
        ctx.lineTo(bottomX, h)
        ctx.stroke()
      }

      // Draw dunes (sine waves)
      for (let d = 0; d < 3; d++) {
        const baseY = horizon + 30 + d * 40
        ctx.beginPath()
        ctx.moveTo(0, baseY)
        for (let x = 0; x <= w; x++) {
          const y =
            baseY +
            Math.sin((x / w) * Math.PI * 3 + frame + d * 1.5) *
              (15 - d * 3) +
            Math.sin((x / w) * Math.PI * 7 + frame * 0.5) * 5
          ctx.lineTo(x, y)
        }
        ctx.strokeStyle = `rgba(245, 158, 11, ${0.15 + d * 0.05})`
        ctx.lineWidth = 1
        ctx.stroke()
      }

      // Draw UGV path
      ctx.beginPath()
      ctx.setLineDash([6, 4])
      const pathPoints: [number, number][] = []
      for (let t = 0; t <= 1; t += 0.01) {
        const x = w / 2 + Math.sin(t * Math.PI * 2 + frame) * (40 + t * 60)
        const y = horizon + t * (h - horizon - 20)
        pathPoints.push([x, y])
      }
      if (pathPoints.length > 0) {
        ctx.moveTo(pathPoints[0][0], pathPoints[0][1])
        for (const [x, y] of pathPoints) {
          ctx.lineTo(x, y)
        }
      }
      ctx.strokeStyle = "rgba(6, 182, 212, 0.6)"
      ctx.lineWidth = 2
      ctx.stroke()
      ctx.setLineDash([])

      // Draw UGV dot
      const ugvIdx = Math.floor(
        ((Math.sin(frame * 2) + 1) / 2) * (pathPoints.length - 1)
      )
      if (pathPoints[ugvIdx]) {
        const [ux, uy] = pathPoints[ugvIdx]
        ctx.beginPath()
        ctx.arc(ux, uy, 6, 0, Math.PI * 2)
        ctx.fillStyle = "rgba(6, 182, 212, 0.8)"
        ctx.fill()
        ctx.beginPath()
        ctx.arc(ux, uy, 12, 0, Math.PI * 2)
        ctx.fillStyle = "rgba(6, 182, 212, 0.15)"
        ctx.fill()
      }

      animId = requestAnimationFrame(draw)
    }

    draw()
    return () => cancelAnimationFrame(animId)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full"
      style={{ imageRendering: "auto" }}
      aria-label="Animated UGV path simulation placeholder"
    />
  )
}
