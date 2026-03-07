"use client"

import { useRef, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { Shield, ShieldAlert, Route, Play, Pause } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import * as THREE from "three"
import type { SegmentationResult } from "@/lib/api"

const CLASS_NAMES = ["Driveable", "Vegetation", "Obstacle", "Sky"]
const SAFE_COLOR = new THREE.Color("#10b981")
const UNSAFE_COLOR = new THREE.Color("#ef4444")
const CAUTION_COLOR = new THREE.Color("#f59e0b")

function generateTerrainFromGrid(grid: number[][]): number[][] {
  if (grid.length > 0) return grid
  const gW = 24
  const gH = 14
  const out: number[][] = []
  for (let z = 0; z < gH; z++) {
    const row: number[] = []
    for (let x = 0; x < gW; x++) {
      const nz = z / gH
      if (nz < 0.2) row.push(3)
      else if (nz < 0.4) row.push(Math.random() < 0.7 ? 0 : 1)
      else row.push(Math.random() < 0.6 ? 0 : Math.random() < 0.8 ? 1 : 2)
    }
    out.push(row)
  }
  return out
}

function getSafetyColor(cls: number): THREE.Color {
  if (cls === 0) return SAFE_COLOR
  if (cls === 1) return CAUTION_COLOR
  return UNSAFE_COLOR
}

function computePathPoints(terrainGrid: number[][], scale: number): THREE.Vector3[] {
  const gH = terrainGrid.length
  const gW = terrainGrid[0]?.length ?? 0
  const points: THREE.Vector3[] = []

  for (let z = 2; z < gH - 2; z += 2) {
    let bestX = Math.floor(gW / 2)
    let bestScore = -1
    for (let x = 1; x < gW - 1; x++) {
      const cls = terrainGrid[z]?.[x] ?? 0
      const score = cls === 0 ? 10 : cls === 1 ? 5 : -10
      if (score > bestScore) {
        bestScore = score
        bestX = x
      }
    }
    const px = (bestX - gW / 2) * scale
    const pz = (z - gH / 2) * scale
    points.push(new THREE.Vector3(px, 0.2, -pz))
  }

  return points
}

function SafetyTerrainMesh({ terrainGrid }: { terrainGrid: number[][] }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const { geometry } = useMemo(() => {
    const gW = terrainGrid[0].length
    const gH = terrainGrid.length
    const geo = new THREE.PlaneGeometry(gW * 0.35, gH * 0.35, gW - 1, gH - 1)
    const positions = geo.attributes.position
    const colorArr = new Float32Array(positions.count * 3)

    for (let i = 0; i < positions.count; i++) {
      const ix = i % gW
      const iz = Math.floor(i / gW)
      const cls = terrainGrid[iz]?.[ix] ?? 0
      let h = 0
      if (cls === 1) h = 0.4 + Math.random() * 0.2
      else if (cls === 2) h = 0.7 + Math.random() * 0.4
      else if (cls === 3) h = 0.02
      else h = Math.sin(ix * 0.2) * 0.08

      positions.setZ(i, h)
      const c = getSafetyColor(cls)
      colorArr[i * 3] = c.r
      colorArr[i * 3 + 1] = c.g
      colorArr[i * 3 + 2] = c.b
    }
    geo.computeVertexNormals()
    return { geometry: geo }
  }, [terrainGrid])

  return (
    <group position={[0, -0.4, 0]} rotation={[-Math.PI / 2.2, 0, 0]}>
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          flatShading
          transparent
          opacity={0.9}
        />
      </mesh>
    </group>
  )
}

function SafetyPathLine({ terrainGrid }: { terrainGrid: number[][] }) {
  const lineObj = useMemo(() => {
    const scale = 0.35
    const points = computePathPoints(terrainGrid, scale)
    if (points.length < 2) return null
    const geom = new THREE.BufferGeometry().setFromPoints(
      points.map((p) => new THREE.Vector3(p.x, 0.1, p.z))
    )
    const mat = new THREE.LineDashedMaterial({
      color: "#06b6d4",
      dashSize: 0.25,
      gapSize: 0.12,
    })
    const line = new THREE.Line(geom, mat)
    line.computeLineDistances()
    return line
  }, [terrainGrid])

  if (!lineObj) return null
  return <primitive object={lineObj} />
}

function UGVAgent({ isPlaying, terrainGrid }: { isPlaying: boolean; terrainGrid: number[][] }) {
  const groupRef = useRef<THREE.Group>(null)
  const scale = 0.35

  const pathPoints = useMemo(
    () => computePathPoints(terrainGrid, scale),
    [terrainGrid]
  )

  const curve = useMemo(
    () => new THREE.CatmullRomCurve3(pathPoints),
    [pathPoints]
  )

  useFrame((s) => {
    if (!groupRef.current || !isPlaying) return
    const t = (s.clock.elapsedTime * 0.12) % 1
    const pos = curve.getPointAt(t)
    const tangent = curve.getTangentAt(t)
    groupRef.current.position.copy(pos)
    groupRef.current.lookAt(pos.clone().add(tangent))
  })

  return (
    <group ref={groupRef}>
      <mesh position={[0, 0.15, 0]}>
        <boxGeometry args={[0.35, 0.15, 0.25]} />
        <meshStandardMaterial color="#334155" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.22, 0]}>
        <boxGeometry args={[0.3, 0.04, 0.2]} />
        <meshStandardMaterial
          color="#06b6d4"
          emissive="#06b6d4"
          emissiveIntensity={0.3}
        />
      </mesh>
      {[-0.12, 0.12].map((x) =>
        [-0.1, 0.1].map((z) => (
          <mesh
            key={`${x}-${z}`}
            position={[x, 0.06, z]}
            rotation={[Math.PI / 2, 0, 0]}
          >
            <cylinderGeometry args={[0.06, 0.06, 0.05, 12]} />
            <meshStandardMaterial color="#1e293b" />
          </mesh>
        ))
      )}
    </group>
  )
}

export function SafetyNavigator3D({ segResult }: { segResult?: SegmentationResult | null }) {
  const [isPlaying, setIsPlaying] = useState(true)
  const terrainGrid = useMemo(
    () => generateTerrainFromGrid(segResult?.terrain_grid || []),
    [segResult?.terrain_grid]
  )

  return (
    <section className="relative z-10 mx-auto w-full max-w-6xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-foreground">
              Safety Navigator Path (3D)
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Safe (green) vs obstacle (red) zones with planned path
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setIsPlaying(!isPlaying)}
              className="gap-2"
            >
              {isPlaying ? (
                <Pause className="h-4 w-4" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              {isPlaying ? "Pause" : "Play"}
            </Button>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-border bg-card/30">
          <div className="flex items-center gap-2 border-b border-border px-4 py-3">
            <Shield className="h-4 w-4 text-emerald-500" />
            <span className="text-xs font-medium text-muted-foreground">
              Driveable (Safe)
            </span>
            <ShieldAlert className="ml-4 h-4 w-4 text-amber-500" />
            <span className="text-xs font-medium text-muted-foreground">
              Vegetation (Caution)
            </span>
            <ShieldAlert className="ml-4 h-4 w-4 text-red-500" />
            <span className="text-xs font-medium text-muted-foreground">
              Obstacle (Avoid)
            </span>
            <Route className="ml-auto h-4 w-4 text-primary" />
            <span className="text-[10px] text-muted-foreground">
              Planned path
            </span>
          </div>
          <div className="relative h-[400px]">
            <Canvas
              camera={{ position: [4, 3, 4], fov: 50 }}
              gl={{ antialias: true, alpha: true }}
              style={{ background: "transparent" }}
            >
              <ambientLight intensity={0.5} />
              <directionalLight position={[5, 8, 5]} intensity={0.9} />
              <pointLight position={[-3, 4, 2]} intensity={0.6} color="#10b981" />
              <pointLight position={[3, -2, 2]} intensity={0.5} color="#06b6d4" />
              <SafetyTerrainMesh terrainGrid={terrainGrid} />
              <SafetyPathLine terrainGrid={terrainGrid} />
              <UGVAgent isPlaying={isPlaying} terrainGrid={terrainGrid} />
              <OrbitControls
                enablePan
                enableZoom
                enableRotate
                minDistance={2}
                maxDistance={12}
                autoRotate
                autoRotateSpeed={0.3}
              />
            </Canvas>
          </div>
        </div>

        <div className="mt-4 flex flex-wrap justify-center gap-4">
          {CLASS_NAMES.map((name, i) => {
            const colors = ["#10b981", "#f59e0b", "#ef4444", "#38bdf8"]
            return (
              <div key={name} className="flex items-center gap-2">
                <div
                  className="h-3 w-3 rounded-sm"
                  style={{ backgroundColor: colors[i] }}
                />
                <span className="text-xs text-muted-foreground">{name}</span>
              </div>
            )
          })}
        </div>
      </motion.div>
    </section>
  )
}
