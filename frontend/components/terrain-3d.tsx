"use client"

import { useRef, useMemo, useState, useCallback } from "react"
import { motion } from "framer-motion"
import { Box, Route, RotateCcw, Eye } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Canvas, useFrame, type ThreeElements } from "@react-three/fiber"
import { OrbitControls, Text, Float, Environment } from "@react-three/drei"
import * as THREE from "three"
import type { SegmentationResult } from "@/lib/api"

/* ──────────────────────────────────────────────────────────────────────── */
/*  Off-road class palette (matches config.py COLOR_PALETTE – 4 classes)   */
/* ──────────────────────────────────────────────────────────────────────── */
const CLASS_COLORS = [
  new THREE.Color(0 / 255, 200 / 255, 0 / 255),     // 0 Driveable
  new THREE.Color(255 / 255, 165 / 255, 0 / 255),    // 1 Vegetation
  new THREE.Color(220 / 255, 50 / 255, 50 / 255),    // 2 Obstacle
  new THREE.Color(135 / 255, 206 / 255, 235 / 255),  // 3 Sky
]

const CLASS_NAMES = [
  "Driveable", "Vegetation", "Obstacle", "Sky",
]

const CLASS_HEIGHTS = [
  0.15, // Driveable – flat ground
  0.8,  // Vegetation – medium
  0.6,  // Obstacle – rocks/logs
  3.0,  // Sky – highest
]

/* ──────────────────────────────────────────────────────────────────────── */
/*  Procedural terrain mesh with class-colored cells                       */
/* ──────────────────────────────────────────────────────────────────────── */
const GRID_W = 34
const GRID_H = 19

function generateTerrainClasses(): number[][] {
  const grid: number[][] = []
  for (let z = 0; z < GRID_H; z++) {
    const row: number[] = []
    for (let x = 0; x < GRID_W; x++) {
      const nz = z / GRID_H
      const nx = x / GRID_W
      if (nz < 0.25) {
        // Sky band at the top
        row.push(3)
      } else if (nz < 0.35) {
        // Transition: vegetation / driveable
        row.push(Math.random() < 0.6 ? 1 : 0)
      } else if (nz < 0.55) {
        // Middle: mix of classes
        const r = Math.random()
        if (r < 0.35) row.push(0)      // Driveable
        else if (r < 0.65) row.push(1) // Vegetation
        else if (r < 0.85) row.push(2) // Obstacle
        else row.push(0)               // Driveable
      } else {
        // Bottom: ground-dominated
        const r = Math.random()
        if (r < 0.5) row.push(0)       // Driveable
        else if (r < 0.75) row.push(1) // Vegetation
        else row.push(2)               // Obstacle
      }
    }
    grid.push(row)
  }
  return grid
}

function TerrainMesh({ terrainGrid }: { terrainGrid?: number[][] }) {
  const meshRef = useRef<THREE.Mesh>(null)

  const { geometry, colors } = useMemo(() => {
    const terrainClasses = terrainGrid && terrainGrid.length > 0
      ? terrainGrid
      : generateTerrainClasses()

    const gW = terrainClasses[0].length
    const gH = terrainClasses.length

    const geo = new THREE.PlaneGeometry(
      gW * 0.3, gH * 0.3,
      gW - 1, gH - 1,
    )

    const positions = geo.attributes.position
    const colorArr = new Float32Array(positions.count * 3)

    for (let i = 0; i < positions.count; i++) {
      const ix = i % gW
      const iz = Math.floor(i / gW)

      const cls = terrainClasses[Math.min(iz, gH - 1)][Math.min(ix, gW - 1)]

      // Height from class + noise
      const baseHeight = CLASS_HEIGHTS[cls]
      const noise = (Math.sin(ix * 0.5) * Math.cos(iz * 0.7)) * 0.15
      positions.setZ(i, baseHeight + noise)

      // Color from class
      const c = CLASS_COLORS[cls]
      colorArr[i * 3] = c.r
      colorArr[i * 3 + 1] = c.g
      colorArr[i * 3 + 2] = c.b
    }

    geo.setAttribute("color", new THREE.BufferAttribute(colorArr, 3))
    geo.computeVertexNormals()

    return { geometry: geo, colors: colorArr }
  }, [terrainGrid])

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.z = Math.sin(state.clock.elapsedTime * 0.1) * 0.02
    }
  })

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[-Math.PI / 2.5, 0, 0]} position={[0, -0.5, 0]}>
      <meshStandardMaterial vertexColors side={THREE.DoubleSide} flatShading />
    </mesh>
  )
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  3D Neural Network Architecture Visualization                          */
/* ──────────────────────────────────────────────────────────────────────── */

interface LayerNodeProps {
  position: [number, number, number]
  color: string
  size: [number, number, number]
  label: string
  delay: number
}

function LayerNode({ position, color, size, label, delay }: LayerNodeProps) {
  const meshRef = useRef<THREE.Mesh>(null)

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y =
        position[1] + Math.sin(state.clock.elapsedTime * 1.5 + delay) * 0.06
    }
  })

  return (
    <group>
      <mesh ref={meshRef} position={position}>
        <boxGeometry args={size} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={0.75}
          roughness={0.3}
          metalness={0.4}
        />
      </mesh>
      <Text
        position={[position[0], position[1] - size[1] / 2 - 0.2, position[2]]}
        fontSize={0.12}
        color="#94a3b8"
        anchorX="center"
        anchorY="top"
      >
        {label}
      </Text>
    </group>
  )
}

function ConnectionBeam({ start, end, color }: {
  start: [number, number, number]
  end: [number, number, number]
  color: string
}) {
  const ref = useRef<THREE.Mesh>(null)

  const { midPoint, length, rotation } = useMemo(() => {
    const s = new THREE.Vector3(...start)
    const e = new THREE.Vector3(...end)
    const mid = new THREE.Vector3().addVectors(s, e).multiplyScalar(0.5)
    const dir = new THREE.Vector3().subVectors(e, s)
    const len = dir.length()
    const rot = new THREE.Euler(0, 0, Math.atan2(dir.y, dir.x))
    return { midPoint: mid, length: len, rotation: rot }
  }, [start, end])

  useFrame((state) => {
    if (ref.current) {
      const mat = ref.current.material as THREE.MeshStandardMaterial
      mat.opacity = 0.2 + Math.sin(state.clock.elapsedTime * 2) * 0.1
    }
  })

  return (
    <mesh ref={ref} position={[midPoint.x, midPoint.y, midPoint.z]} rotation={rotation}>
      <cylinderGeometry args={[0.008, 0.008, length, 4]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.3}
        emissive={color}
        emissiveIntensity={0.3}
      />
    </mesh>
  )
}

function ArchitectureScene() {
  const layers = [
    { x: -2.5, label: "Input\n476×266", color: "#94a3b8", size: [0.3, 1.2, 0.3] as [number, number, number] },
    { x: -1.2, label: "DINOv2\nViT-S/14", color: "#06b6d4", size: [0.5, 1.6, 0.5] as [number, number, number] },
    { x: 0.0, label: "Patch\nTokens", color: "#22d3ee", size: [0.35, 1.0, 0.35] as [number, number, number] },
    { x: 1.2, label: "SegFormer\n4 Blocks", color: "#8b5cf6", size: [0.5, 1.4, 0.5] as [number, number, number] },
    { x: 2.5, label: "Output\n4 Classes", color: "#10b981", size: [0.4, 1.0, 0.4] as [number, number, number] },
  ]

  const positions: [number, number, number][] = layers.map((l) =>
    [l.x, 0, 0]
  )

  return (
    <group position={[0, 0, 0]}>
      {/* Connections */}
      {[
        { from: 0, to: 1, c: "#94a3b8" },
        { from: 1, to: 2, c: "#06b6d4" },
        { from: 2, to: 3, c: "#22d3ee" },
        { from: 3, to: 4, c: "#8b5cf6" },
      ].map(({ from, to, c }, i) => (
        <ConnectionBeam
          key={i}
          start={positions[from]}
          end={positions[to]}
          color={c}
        />
      ))}

      {/* Layer nodes */}
      {layers.map((l, i) => (
        <LayerNode
          key={i}
          position={positions[i]}
          color={l.color}
          size={l.size}
          label={l.label}
          delay={i * 0.5}
        />
      ))}

      {/* Floating data particles */}
      <DataParticles />
    </group>
  )
}

function DataParticles() {
  const ref = useRef<THREE.Points>(null)
  const count = 80

  const { positions, colors } = useMemo(() => {
    const pos = new Float32Array(count * 3)
    const col = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 7
      pos[i * 3 + 1] = (Math.random() - 0.5) * 2.5
      pos[i * 3 + 2] = (Math.random() - 0.5) * 1.5
      const c = new THREE.Color().setHSL(0.52, 0.8, 0.5 + Math.random() * 0.3)
      col[i * 3] = c.r
      col[i * 3 + 1] = c.g
      col[i * 3 + 2] = c.b
    }
    return { positions: pos, colors: col }
  }, [])

  useFrame((state) => {
    if (ref.current) {
      const pos = ref.current.geometry.attributes.position.array as Float32Array
      for (let i = 0; i < count; i++) {
        pos[i * 3] += Math.sin(state.clock.elapsedTime + i * 0.1) * 0.003
        pos[i * 3 + 1] += Math.cos(state.clock.elapsedTime * 0.7 + i * 0.2) * 0.002
      }
      ref.current.geometry.attributes.position.needsUpdate = true
    }
  })

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
          count={count}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
          count={count}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.04} vertexColors transparent opacity={0.6} />
    </points>
  )
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  UGV Path on Terrain                                                    */
/* ──────────────────────────────────────────────────────────────────────── */

function UGVPath() {
  const lineRef = useRef<THREE.Line>(null)
  const dotRef = useRef<THREE.Mesh>(null)

  const curve = useMemo(() => {
    return new THREE.CatmullRomCurve3([
      new THREE.Vector3(-4, 0.4, 2),
      new THREE.Vector3(-2.5, 0.5, 1),
      new THREE.Vector3(-1, 0.3, 0.5),
      new THREE.Vector3(0.5, 0.6, -0.5),
      new THREE.Vector3(2, 0.4, -1),
      new THREE.Vector3(3.5, 0.5, -2),
      new THREE.Vector3(4.5, 0.3, -2.5),
    ])
  }, [])

  const lineObj = useMemo(() => {
    const points = curve.getPoints(60)
    const geom = new THREE.BufferGeometry().setFromPoints(points)
    const mat = new THREE.LineDashedMaterial({ color: "#06b6d4", dashSize: 0.2, gapSize: 0.1 })
    const l = new THREE.Line(geom, mat)
    l.computeLineDistances()
    return l
  }, [curve])

  useFrame((state) => {
    if (dotRef.current) {
      const t = ((Math.sin(state.clock.elapsedTime * 0.5) + 1) / 2)
      const pos = curve.getPointAt(t)
      dotRef.current.position.copy(pos)
    }
  })

  return (
    <group>
      <primitive ref={lineRef} object={lineObj} />
      <mesh ref={dotRef}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial color="#06b6d4" emissive="#06b6d4" emissiveIntensity={0.5} />
      </mesh>
    </group>
  )
}

/* ──────────────────────────────────────────────────────────────────────── */
/*  Main exported component                                                */
/* ──────────────────────────────────────────────────────────────────────── */
export function Terrain3D({ segResult }: { segResult?: SegmentationResult | null }) {
  const [activeView, setActiveView] = useState<"architecture" | "terrain">("architecture")

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
          Interactive architecture &amp; segmented terrain preview
        </p>

        {/* View toggle */}
        <div className="mb-6 flex justify-center gap-2">
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
          {/* Main 3D Canvas */}
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="overflow-hidden rounded-xl border border-border bg-card/30"
          >
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              {activeView === "architecture" ? (
                <>
                  <Box className="h-3.5 w-3.5 text-primary" />
                  <span className="text-xs font-medium text-muted-foreground">
                    DINOv2 → SegFormer Pipeline
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
                <Eye className="h-3 w-3 text-muted-foreground" />
                <span className="text-[10px] text-muted-foreground">
                  Click &amp; drag to orbit
                </span>
              </div>
            </div>
            <div className="relative h-[480px]">
              <Canvas
                camera={{
                  position: activeView === "architecture"
                    ? [0, 1.5, 5]
                    : [5, 4, 5],
                  fov: 50,
                }}
                gl={{ antialias: true, alpha: true }}
                style={{ background: "transparent" }}
              >
                <ambientLight intensity={0.5} />
                <directionalLight position={[5, 5, 5]} intensity={0.8} />
                <pointLight position={[-3, 3, -3]} intensity={0.4} color="#06b6d4" />

                {activeView === "architecture" ? (
                  <ArchitectureScene />
                ) : (
                  <group>
                    <TerrainMesh terrainGrid={segResult?.terrain_grid} />
                    <UGVPath />
                  </group>
                )}

                <OrbitControls
                  enablePan={true}
                  enableZoom={true}
                  enableRotate={true}
                  minDistance={2}
                  maxDistance={12}
                  autoRotate
                  autoRotateSpeed={0.5}
                />
              </Canvas>
            </div>
          </motion.div>

          {/* Class legend for terrain view */}
          {activeView === "terrain" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-wrap justify-center gap-3"
            >
              {CLASS_NAMES.map((name, i) => {
                const c = CLASS_COLORS[i]
                const hex = `#${new THREE.Color(c.r, c.g, c.b).getHexString()}`
                return (
                  <div key={name} className="flex items-center gap-1.5">
                    <div
                      className="h-3 w-3 rounded-sm"
                      style={{ backgroundColor: hex }}
                    />
                    <span className="text-xs text-muted-foreground">{name}</span>
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
              {[
                { label: "DINOv2 ViT-S/14", detail: "Frozen · 384-dim", color: "#06b6d4" },
                { label: "SegFormer Head", detail: "4.4M params · 4 blocks", color: "#8b5cf6" },
                { label: "Output", detail: "4-class segmentation", color: "#10b981" },
              ].map((item) => (
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
