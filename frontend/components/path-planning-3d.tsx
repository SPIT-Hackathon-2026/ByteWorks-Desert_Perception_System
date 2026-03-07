"use client"

import { useRef, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { MapPin, Navigation, Target } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import * as THREE from "three"
import type { SegmentationResult } from "@/lib/api"

function generateGrid(segResult?: SegmentationResult | null): number[][] {
  const grid = segResult?.terrain_grid
  if (grid && grid.length > 0) return grid
  const gW = 20
  const gH = 12
  const out: number[][] = []
  for (let z = 0; z < gH; z++) {
    const row: number[] = []
    for (let x = 0; x < gW; x++) {
      const r = Math.random()
      row.push(r < 0.6 ? 0 : r < 0.85 ? 1 : 2)
    }
    out.push(row)
  }
  return out
}

function aStarPath(
  grid: number[][],
  start: [number, number],
  end: [number, number]
): [number, number][] {
  const gH = grid.length
  const gW = grid[0]?.length ?? 0
  const getCost = (x: number, z: number) => {
    const cls = grid[z]?.[x] ?? 2
    return cls === 0 ? 1 : cls === 1 ? 3 : 999
  }
  const h = (x: number, z: number) =>
    Math.abs(x - end[0]) + Math.abs(z - end[1])

  const open = new Map<string, { x: number; z: number; g: number; f: number }>()
  const closed = new Set<string>()
  const parent = new Map<string, [number, number]>()
  const key = (x: number, z: number) => `${x},${z}`

  open.set(key(start[0], start[1]), {
    x: start[0],
    z: start[1],
    g: 0,
    f: h(start[0], start[1]),
  })

  while (open.size > 0) {
    let best: { x: number; z: number; g: number; f: number } | null = null
    let bestKey = ""
    for (const [k, v] of open) {
      if (!best || v.f < best.f) {
        best = v
        bestKey = k
      }
    }
    if (!best) break
    open.delete(bestKey)
    closed.add(bestKey)

    if (best.x === end[0] && best.z === end[1]) {
      const path: [number, number][] = []
      let cur: [number, number] | undefined = [best.x, best.z]
      while (cur) {
        path.unshift(cur)
        cur = parent.get(key(cur[0], cur[1]))
      }
      return path
    }

    const neighbors = [
      [best.x - 1, best.z],
      [best.x + 1, best.z],
      [best.x, best.z - 1],
      [best.x, best.z + 1],
      [best.x - 1, best.z - 1],
      [best.x + 1, best.z + 1],
      [best.x - 1, best.z + 1],
      [best.x + 1, best.z - 1],
    ]

    for (const [nx, nz] of neighbors) {
      if (nz < 0 || nz >= gH || nx < 0 || nx >= gW) continue
      if (closed.has(key(nx, nz))) continue
      const cost = getCost(nx, nz)
      if (cost >= 999) continue

      const g = best.g + cost
      const f = g + h(nx, nz)
      const nk = key(nx, nz)
      const existing = open.get(nk)
      if (!existing || g < existing.g) {
        open.set(nk, { x: nx, z: nz, g, f })
        parent.set(nk, [best.x, best.z])
      }
    }
  }
  return [start, end]
}

function PathPlanningScene({ segResult }: { segResult?: SegmentationResult | null }) {
  const grid = useMemo(() => generateGrid(segResult), [segResult])
  const gW = grid[0]?.length ?? 20
  const gH = grid.length ?? 12
  const scale = 0.4

  const { pathPoints, terrainGeo } = useMemo(() => {
    const start: [number, number] = [1, 1]
    const end: [number, number] = [gW - 2, gH - 2]
    const path = aStarPath(grid, start, end)

    const points = path.map(
      ([x, z]) =>
        new THREE.Vector3(
          (x - gW / 2) * scale,
          0.15,
          -(z - gH / 2) * scale
        )
    )

    const geo = new THREE.PlaneGeometry(gW * scale, gH * scale, gW - 1, gH - 1)
    const positions = geo.attributes.position
    const colors = new Float32Array(positions.count * 3)
    const safeColor = new THREE.Color("#10b981")
    const obstacleColor = new THREE.Color("#ef4444")
    const vegColor = new THREE.Color("#f59e0b")

    for (let i = 0; i < positions.count; i++) {
      const ix = i % gW
      const iz = Math.floor(i / gW)
      const cls = grid[iz]?.[ix] ?? 0
      const c = cls === 0 ? safeColor : cls === 1 ? vegColor : obstacleColor
      colors[i * 3] = c.r
      colors[i * 3 + 1] = c.g
      colors[i * 3 + 2] = c.b
      let h = cls === 2 ? 0.5 : cls === 1 ? 0.2 : 0.05
      positions.setZ(i, h)
    }
    geo.computeVertexNormals()
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3))

    return { pathPoints: points, terrainGeo: geo }
  }, [grid, gW, gH])

  const lineObj = useMemo(() => {
    if (pathPoints.length < 2) return null
    const geom = new THREE.BufferGeometry().setFromPoints(pathPoints)
    const mat = new THREE.LineBasicMaterial({
      color: "#06b6d4",
      linewidth: 2,
    })
    return new THREE.Line(geom, mat)
  }, [pathPoints])

  return (
    <group position={[0, -0.3, 0]} rotation={[-Math.PI / 2.2, 0, 0]}>
      <mesh geometry={terrainGeo}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          flatShading
        />
      </mesh>
      {lineObj && <primitive object={lineObj} />}
      <mesh position={pathPoints[0]}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial color="#10b981" emissive="#10b981" emissiveIntensity={0.3} />
      </mesh>
      <mesh position={pathPoints[pathPoints.length - 1]}>
        <sphereGeometry args={[0.12, 16, 16]} />
        <meshStandardMaterial color="#f59e0b" emissive="#f59e0b" emissiveIntensity={0.3} />
      </mesh>
      <AnimatedUGV pathPoints={pathPoints} />
    </group>
  )
}

function AnimatedUGV({ pathPoints }: { pathPoints: THREE.Vector3[] }) {
  const ref = useRef<THREE.Group>(null)
  const curve = useMemo(
    () => new THREE.CatmullRomCurve3(pathPoints),
    [pathPoints]
  )

  useFrame((s) => {
    if (!ref.current || pathPoints.length < 2) return
    const t = (s.clock.elapsedTime * 0.08) % 1
    const pos = curve.getPointAt(t)
    const tangent = curve.getTangentAt(t)
    ref.current.position.copy(pos)
    ref.current.lookAt(pos.clone().add(tangent))
  })

  return (
    <group ref={ref}>
      <mesh position={[0, 0.1, 0]}>
        <boxGeometry args={[0.25, 0.1, 0.2]} />
        <meshStandardMaterial color="#334155" metalness={0.8} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.15, 0]}>
        <boxGeometry args={[0.2, 0.03, 0.15]} />
        <meshStandardMaterial color="#06b6d4" emissive="#06b6d4" emissiveIntensity={0.2} />
      </mesh>
    </group>
  )
}

export function PathPlanning3D({ segResult }: { segResult?: SegmentationResult | null }) {
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
              Path Planning (A*)
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Optimal path from start (green) to goal (orange) avoiding obstacles
            </p>
          </div>
        </div>

        <div className="overflow-hidden rounded-xl border border-border bg-card/30">
          <div className="flex items-center gap-4 border-b border-border px-4 py-3">
            <div className="flex items-center gap-2">
              <MapPin className="h-4 w-4 text-emerald-500" />
              <span className="text-xs text-muted-foreground">Start</span>
            </div>
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-amber-500" />
              <span className="text-xs text-muted-foreground">Goal</span>
            </div>
            <div className="flex items-center gap-2">
              <Navigation className="h-4 w-4 text-primary" />
              <span className="text-xs text-muted-foreground">Planned path</span>
            </div>
          </div>
          <div className="relative h-[380px]">
            <Canvas
              camera={{ position: [5, 4, 5], fov: 50 }}
              gl={{ antialias: true, alpha: true }}
              style={{ background: "transparent" }}
            >
              <ambientLight intensity={0.5} />
              <directionalLight position={[5, 8, 5]} intensity={0.9} />
              <pointLight position={[-3, 4, 2]} intensity={0.5} color="#10b981" />
              <pointLight position={[3, -2, 2]} intensity={0.5} color="#06b6d4" />
              <PathPlanningScene segResult={segResult} />
              <OrbitControls
                enablePan
                enableZoom
                enableRotate
                minDistance={2}
                maxDistance={14}
                autoRotate
                autoRotateSpeed={0.35}
              />
            </Canvas>
          </div>
        </div>
      </motion.div>
    </section>
  )
}
