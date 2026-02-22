"use client"

import { useEffect, useRef } from "react"

interface Node {
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  opacity: number
  pulse: number
  pulseSpeed: number
  hue: number
}

interface NeuralBackgroundProps {
  isProcessing?: boolean
}

export function NeuralBackground({ isProcessing = false }: NeuralBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    let animationId: number
    const nodes: Node[] = []
    const nodeCount = 72
    const connectionDistance = 175

    function isDark() {
      return document.documentElement.classList.contains("dark")
    }

    function resize() {
      if (!canvas) return
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }

    resize()
    window.addEventListener("resize", resize)

    // Each node gets its own hue offset for variety
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.55,
        vy: (Math.random() - 0.5) * 0.55,
        radius: Math.random() * 2.2 + 1.2,
        opacity: Math.random() * 0.55 + 0.3,
        pulse: Math.random() * Math.PI * 2,
        pulseSpeed: 0.018 + Math.random() * 0.025,
        hue: Math.random() * 40,
      })
    }

    function animate() {
      if (!ctx || !canvas) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      const dark = isDark()

      // When processing: Shift hues slightly towards cyan/teal
      const baseHue = dark
        ? (isProcessing ? 180 : 188)
        : (isProcessing ? 190 : 210) // More visible blues in light mode

      const speedMult = isProcessing ? 2.5 : 1
      const pulseMult = isProcessing ? 1.8 : 1

      for (const node of nodes) {
        node.x += node.vx * speedMult
        node.y += node.vy * speedMult
        node.pulse += node.pulseSpeed * pulseMult

        if (node.x < 0) node.x = canvas.width
        if (node.x > canvas.width) node.x = 0
        if (node.y < 0) node.y = canvas.height
        if (node.y > canvas.height) node.y = 0

        const pulsedOpacity = Math.min(node.opacity + Math.sin(node.pulse) * 0.2, 1)
        const pulsedRadius = node.radius + Math.sin(node.pulse * 1.3) * 0.6
        const hue = baseHue + node.hue

        // Outer glow
        const grd = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, pulsedRadius * (isProcessing ? 7 : 5)
        )
        const sat = dark ? (isProcessing ? "100%" : "90%") : (isProcessing ? "100%" : "95%")
        const light = dark ? "60%" : "45%" // Slightly darker in light mode for contrast
        grd.addColorStop(0, `hsla(${hue}, ${sat}, ${light}, ${pulsedOpacity * (dark ? 0.85 : 0.6)})`)
        grd.addColorStop(0.4, `hsla(${hue}, ${sat}, ${light}, ${pulsedOpacity * 0.2})`)
        grd.addColorStop(1, `hsla(${hue}, ${sat}, ${light}, 0)`)
        ctx.beginPath()
        ctx.arc(node.x, node.y, pulsedRadius * (isProcessing ? 7 : 5), 0, Math.PI * 2)
        ctx.fillStyle = grd
        ctx.fill()

        // Core dot
        ctx.beginPath()
        ctx.arc(node.x, node.y, pulsedRadius, 0, Math.PI * 2)
        ctx.fillStyle = `hsla(${hue}, ${sat}, ${light}, ${Math.min(pulsedOpacity, 0.95)})`
        ctx.fill()
      }

      // Connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)

          if (dist < connectionDistance) {
            const fade = 1 - dist / connectionDistance
            const lineOpacity = fade * (dark ? 0.20 : 0.30) * (isProcessing ? 1.5 : 1)
            const hue = baseHue + (nodes[i].hue + nodes[j].hue) / 2
            const sat = dark ? "85%" : "75%"
            const light = dark ? "58%" : "48%"
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.strokeStyle = `hsla(${hue}, ${sat}, ${light}, ${lineOpacity})`
            ctx.lineWidth = fade * (isProcessing ? 2.2 : 1.4)
            ctx.stroke()
          }
        }
      }

      animationId = requestAnimationFrame(animate)
    }

    animate()

    const observer = new MutationObserver(() => { })
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })

    return () => {
      window.removeEventListener("resize", resize)
      cancelAnimationFrame(animationId)
      observer.disconnect()
    }
  }, [isProcessing])

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0 z-0"
      style={{ opacity: 1 }}
      aria-hidden="true"
    />
  )
}
