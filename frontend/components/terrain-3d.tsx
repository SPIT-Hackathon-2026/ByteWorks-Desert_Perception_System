"use client"

import { useRef, useMemo, useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Box, Route, Eye, Play, Pause, Film, RotateCcw, Maximize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls, Text } from "@react-three/drei"
import * as THREE from "three"
import type { SegmentationResult } from "@/lib/api"

const CLASS_NAMES = ["Driveable", "Vegetation", "Obstacle", "Sky"]
const DESERT_COLORS = [
  new THREE.Color("#c29d7c"), // Driveable (Sand)
  new THREE.Color("#4ade80"), // Vegetation (Green)
  new THREE.Color("#451a03"), // Obstacle (Rocks)
  new THREE.Color("#38bdf8"), // Sky (Blue)
]

function generateTerrainClasses(): number[][] {
  const gW = 34, gH = 19
  const grid: number[][] = []
  for (let z = 0; z < gH; z++) {
    const row: number[] = []
    for (let x = 0; x < gW; x++) {
      const nz = z / gH
      if (nz < 0.25) row.push(3) // Sky
      else if (nz < 0.35) row.push(Math.random() < 0.6 ? 1 : 0)
      else if (nz < 0.55) {
        const r = Math.random()
        row.push(r < 0.35 ? 0 : r < 0.65 ? 1 : r < 0.85 ? 2 : 0)
      } else {
        const r = Math.random()
        row.push(r < 0.5 ? 0 : r < 0.75 ? 1 : 2)
      }
    }
    grid.push(row)
  }
  return grid
}

function TerrainMesh({ segResult }: { segResult?: SegmentationResult | null }) {
  const meshRef = useRef<THREE.Mesh>(null)

  // ── Texture Loader ─────────────────────────────────────────────────────
  const texture = useMemo(() => {
    if (!segResult?.overlay_b64) return null
    const loader = new THREE.TextureLoader()
    return loader.load(`data:image/png;base64,${segResult.overlay_b64}`)
  }, [segResult?.overlay_b64])

  const { geometry } = useMemo(() => {
    const grid = segResult?.terrain_grid || generateTerrainClasses()
    const gW = grid[0].length
    const gH = grid.length

    // Create geometry larger than the grid points
    const geo = new THREE.PlaneGeometry(gW * 0.4, gH * 0.4, gW - 1, gH - 1)
    const positions = geo.attributes.position
    const colorArr = new Float32Array(positions.count * 3)

    for (let i = 0; i < positions.count; i++) {
      const ix = i % gW
      const iz = Math.floor(i / gW)
      const cls = grid[iz][ix] ?? 0

      // Calculate height based on class
      // Obstacles and Vegetation are elevated. Sky is a distant background.
      let h = 0
      if (cls === 1) h = 0.5 + Math.random() * 0.3 // Vegetation
      else if (cls === 2) h = 0.8 + Math.random() * 0.5 // Obstacle
      else if (cls === 3) h = 0.05 // Sky (flat at the back)
      else h = Math.sin(ix * 0.3) * Math.cos(iz * 0.3) * 0.1 // Driveable (rough sand)

      positions.setZ(i, h)

      const c = DESERT_COLORS[cls] || DESERT_COLORS[0]
      colorArr[i * 3] = c.r
      colorArr[i * 3 + 1] = c.g
      colorArr[i * 3 + 2] = c.b
    }

    geo.computeVertexNormals()
    return { geometry: geo }
  }, [segResult?.terrain_grid])

  useFrame((s) => {
    if (meshRef.current && !segResult) {
      meshRef.current.rotation.z = Math.sin(s.clock.elapsedTime * 0.1) * 0.02
    }
  })

  return (
    <group position={[0, -0.5, 0]} rotation={[-Math.PI / 2.2, 0, 0]}>
      <mesh ref={meshRef} geometry={geometry}>
        {texture ? (
          <meshStandardMaterial map={texture} side={THREE.DoubleSide} roughness={0.8} metalness={0.1} />
        ) : (
          <meshStandardMaterial vertexColors side={THREE.DoubleSide} flatShading />
        )}
      </mesh>
      <DesertDecorations segResult={segResult} />
    </group>
  )
}

/* ── Desert Decorations (Cactus, etc.) ────────────────────────────────── */
function Cactus({ position, scale = 1 }: { position: [number, number, number], scale?: number }) {
  return (
    <group position={position} scale={scale}>
      {/* Main stem */}
      <mesh position={[0, 0.4, 0]}>
        <cylinderGeometry args={[0.08, 0.1, 0.8, 8]} />
        <meshStandardMaterial color="#166534" roughness={0.8} />
      </mesh>
      {/* Arm 1 */}
      <group position={[0, 0.4, 0]} rotation={[0, 0, 0.8]}>
        <mesh position={[0, 0.2, 0]}>
          <cylinderGeometry args={[0.06, 0.06, 0.4, 8]} />
          <meshStandardMaterial color="#166534" />
        </mesh>
        <mesh position={[0, 0.4, 0]} rotation={[0, 0, -0.8]}>
          <cylinderGeometry args={[0.06, 0.06, 0.3, 8]} />
          <meshStandardMaterial color="#166534" />
        </mesh>
      </group>
      {/* Arm 2 */}
      <group position={[0, 0.25, 0]} rotation={[0, 0, -0.8]}>
        <mesh position={[0, 0.15, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.3, 8]} />
          <meshStandardMaterial color="#166534" />
        </mesh>
        <mesh position={[0, 0.3, 0]} rotation={[0, 0, 0.8]}>
          <cylinderGeometry args={[0.05, 0.05, 0.2, 8]} />
          <meshStandardMaterial color="#166534" />
        </mesh>
      </group>
    </group>
  )
}

function DesertDecorations({ segResult }: { segResult?: SegmentationResult | null }) {
  const decorations = useMemo(() => {
    const grid = segResult?.terrain_grid || []
    if (grid.length === 0) return []

    const items: React.ReactNode[] = []
    const gH = grid.length
    const gW = grid[0].length

    // Scatter some cacti in vegetation or driveable areas
    for (let i = 0; i < 15; i++) {
      const rx = Math.floor(Math.random() * gW)
      const rz = Math.floor(Math.random() * gH)
      const cls = grid[rz][rx]

      if (cls === 1 || (cls === 0 && Math.random() < 0.1)) {
        // Map grid to terrain coordinates
        const x = (rx - gW / 2) * 0.4
        const y = (rz - gH / 2) * 0.4
        items.push(
          <Cactus
            key={i}
            position={[x, -y, 0]}
            scale={0.6 + Math.random() * 0.8}
          />
        )
      }
    }
    return items
  }, [segResult?.terrain_grid])

  return <group rotation={[Math.PI / 2, 0, 0]}>{decorations}</group>
}

/* ── Realistic UGV Model ─────────────────────────────────────────────── */
function UGVModel({ color = "#06b6d4" }: { color?: string }) {
  return (
    <group>
      {/* Main Chassis */}
      <mesh position={[0, 0.15, 0]}>
        <boxGeometry args={[0.5, 0.2, 0.4]} />
        <meshStandardMaterial color="#334155" metalness={0.8} roughness={0.2} />
      </mesh>
      {/* Top Deck (Solar/Sensors) */}
      <mesh position={[0, 0.26, 0]}>
        <boxGeometry args={[0.45, 0.04, 0.35]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.2} />
      </mesh>
      {/* Wheels */}
      {[-0.2, 0.2].map(x => [-0.18, 0.18].map(z => (
        <mesh key={`${x}-${z}`} position={[x, 0.08, z]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.1, 0.1, 0.08, 16]} />
          <meshStandardMaterial color="#1e293b" />
        </mesh>
      )))}
      {/* Sensor Tower (Lidar) */}
      <mesh position={[0.15, 0.35, 0]}>
        <cylinderGeometry args={[0.04, 0.04, 0.15, 12]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
      {/* Lidar Head */}
      <mesh position={[0.15, 0.45, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 0.05, 12]} />
        <meshStandardMaterial color="#1e293b" emissive={color} emissiveIntensity={0.5} />
      </mesh>
    </group>
  )
}

function UGVPath() {
  const groupRef = useRef<THREE.Group>(null)
  const curve = useMemo(() => new THREE.CatmullRomCurve3([
    new THREE.Vector3(-4.5, 0.4, 2.5), new THREE.Vector3(-2.5, 0.5, 1.2),
    new THREE.Vector3(-0.5, 0.3, 0.5), new THREE.Vector3(1.5, 0.6, -0.8),
    new THREE.Vector3(3, 0.4, -1.5), new THREE.Vector3(4.5, 0.5, -3),
    new THREE.Vector3(5.5, 0.3, -4),
  ]), [])

  const lineObj = useMemo(() => {
    const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(80))
    const mat = new THREE.LineDashedMaterial({ color: "#06b6d4", dashSize: 0.3, gapSize: 0.15 })
    const l = new THREE.Line(geom, mat)
    l.computeLineDistances()
    return l
  }, [curve])

  useFrame((s) => {
    if (groupRef.current) {
      const t = (s.clock.elapsedTime * 0.15) % 1
      const pos = curve.getPointAt(t)
      const tangent = curve.getTangentAt(t)
      groupRef.current.position.copy(pos)
      groupRef.current.lookAt(pos.clone().add(tangent))
    }
  })

  return (
    <group>
      <primitive object={lineObj} />
      <group ref={groupRef}>
        <UGVModel />
      </group>
    </group>
  )
}

/* ── Glowing layer block ─────────────────────────────────────────────────
   A floating box with inner glow/emissive and a text label below          */

interface GlowBlockProps {
  position: [number, number, number]
  size: [number, number, number]
  color: string
  label: string
  sublabel?: string
  delay?: number
  shape?: "box" | "cylinder" | "sphere"
}

function GlowBlock({ position, size, color, label, sublabel, delay = 0, shape = "box" }: GlowBlockProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.Mesh>(null)

  useFrame((s) => {
    const t = s.clock.elapsedTime
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(t * 1.4 + delay) * 0.08
    }
    if (glowRef.current) {
      const mat = glowRef.current.material as THREE.MeshStandardMaterial
      mat.emissiveIntensity = 0.4 + Math.sin(t * 2.5 + delay) * 0.25
    }
  })

  const [w, h, d] = size

  return (
    <group>
      {/* Outer glow shell */}
      <mesh ref={glowRef} position={[position[0], position[1], position[2]]}>
        {shape === "cylinder"
          ? <cylinderGeometry args={[w * 0.7, w * 0.7, h * 1.15, 32]} />
          : <boxGeometry args={[w * 1.18, h * 1.18, d * 1.18]} />
        }
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.13}
          emissive={color}
          emissiveIntensity={0.5}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Main block */}
      <mesh ref={meshRef} position={position}>
        {shape === "cylinder"
          ? <cylinderGeometry args={[w * 0.6, w * 0.6, h, 32]} />
          : <boxGeometry args={[w, h, d]} />
        }
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.82}
          roughness={0.15}
          metalness={0.6}
          emissive={color}
          emissiveIntensity={0.15}
        />
      </mesh>

      {/* Label */}
      <Text
        position={[position[0], position[1] - h / 2 - 0.28, position[2]]}
        fontSize={0.14}
        color="#e2e8f0"
        anchorX="center"
        anchorY="top"
        maxWidth={1.6}
      >
        {label}
      </Text>
      {sublabel && (
        <Text
          position={[position[0], position[1] - h / 2 - 0.48, position[2]]}
          fontSize={0.10}
          color="#64748b"
          anchorX="center"
          anchorY="top"
          maxWidth={1.6}
        >
          {sublabel}
        </Text>
      )}
    </group>
  )
}

/* ── Animated data flow beam ────────────────────────────────────────────── */
interface DataBeamProps {
  from: [number, number, number]
  to: [number, number, number]
  color: string
  delay?: number
}

function DataBeam({ from, to, color, delay = 0 }: DataBeamProps) {
  const particleRef = useRef<THREE.Mesh>(null)

  useFrame((s) => {
    if (particleRef.current) {
      const t = ((s.clock.elapsedTime * 0.7 + delay) % 1.0)
      particleRef.current.position.lerpVectors(
        new THREE.Vector3(...from),
        new THREE.Vector3(...to),
        t
      )
      particleRef.current.visible = t < 0.95
    }
  })

  const mid = new THREE.Vector3(
    (from[0] + to[0]) / 2,
    (from[1] + to[1]) / 2,
    (from[2] + to[2]) / 2,
  )
  const dir = new THREE.Vector3(to[0] - from[0], to[1] - from[1], to[2] - from[2])
  const len = dir.length()
  const rot = new THREE.Euler(0, 0, Math.atan2(dir.y, dir.x))

  return (
    <group>
      {/* Static dim line */}
      <mesh position={[mid.x, mid.y, mid.z]} rotation={rot}>
        <cylinderGeometry args={[0.006, 0.006, len, 4]} />
        <meshStandardMaterial color={color} transparent opacity={0.18} emissive={color} emissiveIntensity={0.1} />
      </mesh>
      {/* Flying particle */}
      <mesh ref={particleRef}>
        <sphereGeometry args={[0.06, 8, 8]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} transparent opacity={0.9} />
      </mesh>
    </group>
  )
}

/* ── Ambient particles ──────────────────────────────────────────────────── */
function AmbientParticles() {
  const ref = useRef<THREE.Points>(null)
  const count = 120
  const { positions, colors } = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const col = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 10
      pos[i * 3 + 1] = (Math.random() - 0.5) * 3
      pos[i * 3 + 2] = (Math.random() - 0.5) * 2
      const c = new THREE.Color().setHSL(0.52 + Math.random() * 0.15, 0.8, 0.5 + Math.random() * 0.3)
      col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b
    }
    return { positions: pos, colors: col }
  }, [])

  useFrame((s) => {
    if (!ref.current) return
    const pos = ref.current.geometry.attributes.position.array as Float32Array
    for (let i = 0; i < count; i++) {
      pos[i * 3] += Math.sin(s.clock.elapsedTime + i * 0.15) * 0.003
      pos[i * 3 + 1] += Math.cos(s.clock.elapsedTime * 0.7 + i * 0.2) * 0.002
    }
    ref.current.geometry.attributes.position.needsUpdate = true
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} count={count} itemSize={3} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} count={count} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.04} vertexColors transparent opacity={0.55} />
    </points>
  )
}

/* ── U-MixFormer Architecture Scene ────────────────────────────────────── */
/*   Layout (top view):
     Input → CNX-S1 → CNX-S2 → CNX-S3 → CNX-S4
                                              ↓ (Mix-Att)
     Output ← Fuse ← Dec-3 ← Dec-2 ← Dec-1 ←┘
*/

const ARCH_LAYERS = [
  // Encoder row (y=1)
  { id: "input", x: -4.5, y: 0, label: "Input", sublabel: "384×384 RGB", color: "#64748b", size: [0.55, 1.0, 0.35] as [number, number, number] },
  { id: "enc1", x: -3.0, y: 0, label: "ConvNeXt S1", sublabel: "96ch · 96×96", color: "#06b6d4", size: [0.6, 1.3, 0.4] as [number, number, number] },
  { id: "enc2", x: -1.5, y: 0, label: "ConvNeXt S2", sublabel: "192ch · 48×48", color: "#0ea5e9", size: [0.6, 1.5, 0.4] as [number, number, number] },
  { id: "enc3", x: 0.0, y: 0, label: "ConvNeXt S3", sublabel: "384ch · 24×24", color: "#3b82f6", size: [0.6, 1.8, 0.4] as [number, number, number] },
  { id: "enc4", x: 1.5, y: 0, label: "ConvNeXt S4", sublabel: "768ch · 12×12", color: "#6366f1", size: [0.6, 2.0, 0.4] as [number, number, number] },
  // Mix-Attention bridge
  { id: "mix", x: 1.5, y: -2.0, label: "Mix-Attention", sublabel: "Sync · 4 heads", color: "#a855f7", size: [0.7, 0.5, 0.7] as [number, number, number], shape: "cylinder" as const },
  // Decoder row (y=-2.0)
  { id: "dec1", x: 0.0, y: -2.0, label: "U-Net Dec 3", sublabel: "384ch fused", color: "#8b5cf6", size: [0.6, 1.3, 0.4] as [number, number, number] },
  { id: "dec2", x: -1.5, y: -2.0, label: "U-Net Dec 2", sublabel: "192ch fused", color: "#ec4899", size: [0.6, 1.1, 0.4] as [number, number, number] },
  { id: "dec3", x: -3.0, y: -2.0, label: "U-Net Dec 1", sublabel: "96ch fused", color: "#f59e0b", size: [0.6, 0.9, 0.4] as [number, number, number] },
  { id: "out", x: -4.5, y: -2.0, label: "Seg Output", sublabel: "4 classes", color: "#10b981", size: [0.65, 1.0, 0.45] as [number, number, number] },
]

const ARCH_BEAMS = [
  // Encoder forward
  { from: "input", to: "enc1", color: "#06b6d4", delay: 0 },
  { from: "enc1", to: "enc2", color: "#0ea5e9", delay: 0.3 },
  { from: "enc2", to: "enc3", color: "#3b82f6", delay: 0.6 },
  { from: "enc3", to: "enc4", color: "#6366f1", delay: 0.9 },
  // Bridge down
  { from: "enc4", to: "mix", color: "#a855f7", delay: 1.2 },
  // Decoder backward
  { from: "mix", to: "dec1", color: "#8b5cf6", delay: 1.5 },
  { from: "dec1", to: "dec2", color: "#ec4899", delay: 1.8 },
  { from: "dec2", to: "dec3", color: "#f59e0b", delay: 2.1 },
  { from: "dec3", to: "out", color: "#10b981", delay: 2.4 },
  // Skip connections (encoder→decoder, offset in Z to be visible)
  { from: "enc3", to: "dec1", color: "#6366f180", delay: 0.8 },
  { from: "enc2", to: "dec2", color: "#3b82f680", delay: 0.5 },
  { from: "enc1", to: "dec3", color: "#0ea5e980", delay: 0.2 },
]

function ArchitectureScene() {
  const posMap = useMemo(() => {
    const m: Record<string, [number, number, number]> = {}
    for (const l of ARCH_LAYERS) m[l.id] = [l.x, l.y, 0]
    return m
  }, [])

  return (
    <group position={[0, 1, 0]}>
      <AmbientParticles />

      {/* Data beams */}
      {ARCH_BEAMS.map(({ from, to, color, delay }) => (
        <DataBeam
          key={`${from}-${to}`}
          from={posMap[from]}
          to={posMap[to]}
          color={color}
          delay={delay}
        />
      ))}

      {/* Layer blocks */}
      {ARCH_LAYERS.map((l, i) => (
        <GlowBlock
          key={l.id}
          position={[l.x, l.y, 0]}
          size={l.size}
          color={l.color}
          label={l.label}
          sublabel={l.sublabel}
          delay={i * 0.4}
          shape={l.shape ?? "box"}
        />
      ))}

      {/* Section labels */}
      <Text position={[-1.5, 1.7, 0]} fontSize={0.16} color="#94a3b8" anchorX="center">
        ── ConvNeXt Encoder ──
      </Text>
      <Text position={[-1.5, -3.4, 0]} fontSize={0.16} color="#94a3b8" anchorX="center">
        ── U-Net Decoder ──
      </Text>
    </group>
  )
}

/* ── Legend cards ───────────────────────────────────────────────────────── */
const LEGEND_ARCH = [
  { label: "ConvNeXt Encoder", detail: "4-stage hierarchy", color: "#06b6d4" },
  { label: "Mix-Attention", detail: "Enc-Dec alignment", color: "#a855f7" },
  { label: "U-Net Decoder", detail: "Progressive refinement", color: "#ec4899" },
  { label: "Seg Output", detail: "4-class mask", color: "#10b981" },
]

/* ── Pipeline Animation Video Player ──────────────────────────────────── */
function PipelineVideo() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isPlaying, setIsPlaying] = useState(true)
  const [progress, setProgress] = useState(0)
  const [currentHead, setCurrentHead] = useState("")

  const HEAD_NAMES = [
    "Linear Head", "MLP Head", "ConvNeXt Head",
    "ConvNeXt Deep Head", "MultiScale Head",
    "Hybrid Head", "SegFormer Head",
  ]
  const HEAD_COLORS_CSS = [
    "#ff006e", "#fb5607", "#ffbe0b",
    "#8338ec", "#00f5d4", "#f72585", "#4cc9f0",
  ]

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    const onTime = () => {
      const p = v.currentTime / (v.duration || 1)
      setProgress(p)
      const idx = Math.min(Math.floor(p * HEAD_NAMES.length), HEAD_NAMES.length - 1)
      setCurrentHead(HEAD_NAMES[idx])
    }
    v.addEventListener("timeupdate", onTime)
    return () => v.removeEventListener("timeupdate", onTime)
  }, [])

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) { v.play(); setIsPlaying(true) }
    else { v.pause(); setIsPlaying(false) }
  }

  const restart = () => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = 0
    v.play()
    setIsPlaying(true)
  }

  const headIdx = Math.min(
    Math.floor(progress * HEAD_NAMES.length),
    HEAD_NAMES.length - 1
  )

  return (
    <div className="relative h-130 overflow-hidden bg-[#07080f]">
      {/* Video */}
      <video
        ref={videoRef}
        src="/segheads.mp4"
        autoPlay
        loop
        muted
        playsInline
        className="h-full w-full object-contain"
      />

      {/* Overlay controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/80 via-black/40 to-transparent px-4 pb-4 pt-12">
        {/* Progress bar */}
        <div className="mb-3 h-1 overflow-hidden rounded-full bg-white/10">
          <motion.div
            className="h-full rounded-full"
            style={{ backgroundColor: HEAD_COLORS_CSS[headIdx] }}
            animate={{ width: `${progress * 100}%` }}
            transition={{ duration: 0.2 }}
          />
        </div>

        {/* Head indicator dots */}
        <div className="mb-3 flex items-center justify-center gap-2">
          {HEAD_NAMES.map((name, i) => (
            <div key={name} className="flex flex-col items-center gap-1">
              <div
                className="h-2 w-2 rounded-full transition-all duration-300"
                style={{
                  backgroundColor: HEAD_COLORS_CSS[i],
                  opacity: i === headIdx ? 1 : 0.3,
                  transform: i === headIdx ? "scale(1.5)" : "scale(1)",
                  boxShadow: i === headIdx ? `0 0 8px ${HEAD_COLORS_CSS[i]}` : "none",
                }}
              />
            </div>
          ))}
        </div>

        {/* Controls row */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={togglePlay}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
            >
              {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
            </button>
            <button
              onClick={restart}
              className="flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <span
              className="rounded-full px-3 py-1 text-xs font-semibold"
              style={{
                backgroundColor: `${HEAD_COLORS_CSS[headIdx]}20`,
                color: HEAD_COLORS_CSS[headIdx],
                border: `1px solid ${HEAD_COLORS_CSS[headIdx]}40`,
              }}
            >
              {currentHead || HEAD_NAMES[0]}
            </span>
          </div>

          <span className="font-mono text-[10px] text-white/40">
            {Math.floor(headIdx + 1)}/{HEAD_NAMES.length} heads
          </span>
        </div>
      </div>
    </div>
  )
}

/* ── Main component ─────────────────────────────────────────────────────── */
export function Terrain3D({ segResult }: { segResult?: SegmentationResult | null }) {
  const [activeView, setActiveView] = useState<"animation" | "architecture" | "terrain">("animation")

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
        <p className="mb-6 text-center text-sm text-muted-foreground">
          Interactive U-MixFormer architecture &amp; segmented terrain preview
        </p>

        {/* View toggle */}
        <div className="mb-6 flex justify-center gap-2">
          <Button
            size="sm"
            variant={activeView === "animation" ? "default" : "outline"}
            className={`gap-2 ${activeView === "animation" ? "bg-primary text-primary-foreground" : "border-border text-foreground"}`}
            onClick={() => setActiveView("animation")}
          >
            <Film className="h-4 w-4" />
            Segmentation Heads
          </Button>
          <Button
            size="sm"
            variant={activeView === "architecture" ? "default" : "outline"}
            className={`gap-2 ${activeView === "architecture" ? "bg-primary text-primary-foreground" : "border-border text-foreground"}`}
            onClick={() => setActiveView("architecture")}
          >
            <Box className="h-4 w-4" />
            Pipeline Architecture
          </Button>
          <Button
            size="sm"
            variant={activeView === "terrain" ? "default" : "outline"}
            className={`gap-2 ${activeView === "terrain" ? "bg-primary text-primary-foreground" : "border-border text-foreground"}`}
            onClick={() => setActiveView("terrain")}
          >
            <Route className="h-4 w-4" />
            Terrain Map
          </Button>
        </div>

        <div className="grid grid-cols-1 gap-4">
          {/* 3D Canvas / Video */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="overflow-hidden rounded-xl border border-border bg-card/30"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              {activeView === "animation" ? (
                <>
                  <Film className="h-3.5 w-3.5 text-primary" />
                  <span className="text-xs font-medium text-muted-foreground">
                    Segmentation Head Architectures · 3D Animated
                  </span>
                </>
              ) : activeView === "architecture" ? (
                <>
                  <Box className="h-3.5 w-3.5 text-primary" />
                  <span className="text-xs font-medium text-muted-foreground">
                    U-MixFormer Segmentation Pipeline
                  </span>
                </>
              ) : (
                <>
                  <Route className="h-3.5 w-3.5 text-accent" />
                  <span className="text-xs font-medium text-muted-foreground">
                    Segmented Terrain with UGV Navigation Path
                  </span>
                </>
              )}
              <div className="ml-auto flex items-center gap-1">
                {activeView !== "animation" && (
                  <>
                    <Eye className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] text-muted-foreground">
                      Click &amp; drag to orbit
                    </span>
                  </>
                )}
              </div>
            </div>

            <AnimatePresence mode="wait">
              {activeView === "animation" && (
                <motion.div
                  key="animation"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <PipelineVideo />
                </motion.div>
              )}

              {activeView === "architecture" && (
                <motion.div
                  key="architecture"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="relative h-130"
                >
                  <Canvas
                    camera={{ position: [0, 0, 9], fov: 50 }}
                    gl={{ antialias: true, alpha: true }}
                    style={{ background: "transparent" }}
                  >
                    <ambientLight intensity={0.4} />
                    <directionalLight position={[5, 8, 5]} intensity={0.9} />
                    <pointLight position={[-4, 3, 2]} intensity={0.8} color="#06b6d4" />
                    <pointLight position={[4, -2, 2]} intensity={0.6} color="#a855f7" />
                    <pointLight position={[0, 0, 4]} intensity={0.4} color="#10b981" />
                    <ArchitectureScene />
                    <OrbitControls enablePan enableZoom enableRotate minDistance={3} maxDistance={16} autoRotate autoRotateSpeed={0.4} />
                  </Canvas>
                </motion.div>
              )}

              {activeView === "terrain" && (
                <motion.div
                  key="terrain"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="relative h-130"
                >
                  <Canvas
                    camera={{ position: [5, 4, 5], fov: 50 }}
                    gl={{ antialias: true, alpha: true }}
                    style={{ background: "transparent" }}
                  >
                    <ambientLight intensity={0.4} />
                    <directionalLight position={[5, 8, 5]} intensity={0.9} />
                    <pointLight position={[-4, 3, 2]} intensity={0.8} color="#06b6d4" />
                    <pointLight position={[4, -2, 2]} intensity={0.6} color="#a855f7" />
                    <pointLight position={[0, 0, 4]} intensity={0.4} color="#10b981" />
                    <TerrainMesh segResult={segResult} />
                    <UGVPath />
                    <OrbitControls enablePan enableZoom enableRotate minDistance={3} maxDistance={16} autoRotate autoRotateSpeed={0.6} />
                  </Canvas>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Animation legend */}
          {activeView === "animation" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7"
            >
              {[
                { label: "Linear", color: "#ff006e" },
                { label: "MLP", color: "#fb5607" },
                { label: "ConvNeXt", color: "#ffbe0b" },
                { label: "ConvNeXt Deep", color: "#8338ec" },
                { label: "MultiScale", color: "#00f5d4" },
                { label: "Hybrid", color: "#f72585" },
                { label: "SegFormer", color: "#4cc9f0" },
              ].map((h) => (
                <div
                  key={h.label}
                  className="flex items-center gap-2 rounded-lg border border-border bg-card/40 px-3 py-2"
                >
                  <div
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: h.color }}
                  />
                  <span className="text-[11px] font-medium text-foreground">{h.label}</span>
                </div>
              ))}
            </motion.div>
          )}

          {/* Terrain legend */}
          {activeView === "terrain" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-wrap justify-center gap-3"
            >
              {CLASS_NAMES.map((name, i) => {
                const c = DESERT_COLORS[i]
                const hex = `#${new THREE.Color(c.r, c.g, c.b).getHexString()}`
                return (
                  <div key={name} className="flex items-center gap-1.5">
                    <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: hex }} />
                    <span className="text-sm text-muted-foreground">{name}</span>
                  </div>
                )
              })}
            </motion.div>
          )}

          {/* Architecture legend */}
          {activeView === "architecture" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="grid grid-cols-2 gap-3 sm:grid-cols-4"
            >
              {LEGEND_ARCH.map((item) => (
                <div
                  key={item.label}
                  className="flex items-center gap-2 rounded-lg border border-border bg-card/40 p-3"
                >
                  <div
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <div>
                    <p className="text-xs font-medium text-foreground">{item.label}</p>
                    <p className="text-[10px] text-muted-foreground">{item.detail}</p>
                  </div>
                </div>
              ))}
            </motion.div>
          )}
        </div>
      </motion.div>
    </section>
  )
}
