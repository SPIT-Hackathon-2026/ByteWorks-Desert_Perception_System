"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  LayoutDashboard,
  BarChart3,
  Box,
  Shield,
  Navigation,
  ChevronRight,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { InteractivePieChart } from "@/components/interactive-pie-chart"
import { Terrain3D } from "@/components/terrain-3d"
import { SafetyNavigator3D } from "@/components/safety-navigator-3d"
import { PathPlanning3D } from "@/components/path-planning-3d"
import { RiskGauge } from "@/components/risk-gauge"
import type { SegmentationResult } from "@/lib/api"

type DashboardTab = "overview" | "3d-terrain" | "safety-path" | "path-planning"

const TAB_CONFIG: { id: DashboardTab; label: string; icon: React.ElementType }[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "3d-terrain", label: "3D Terrain", icon: Box },
  { id: "safety-path", label: "Safety Navigator", icon: Shield },
  { id: "path-planning", label: "Path Planning", icon: Navigation },
]

interface UnifiedDashboardProps {
  isComplete: boolean
  segResult?: SegmentationResult | null
}

export function UnifiedDashboard({ isComplete, segResult }: UnifiedDashboardProps) {
  const [activeTab, setActiveTab] = useState<DashboardTab>("overview")

  if (!isComplete) return null

  const liveClassData = segResult
    ? segResult.class_distribution.map((c) => ({
        name: c.name.length > 12 ? c.name.slice(0, 10) + "…" : c.name,
        value: c.percentage,
        fill: c.color.replace("rgb(", "rgba(").replace(")", ",1)"),
      }))
    : [
        { name: "Driveable", value: 52, fill: "#10b981" },
        { name: "Vegetation", value: 22, fill: "#f59e0b" },
        { name: "Obstacle", value: 8, fill: "#ef4444" },
        { name: "Sky", value: 18, fill: "#38bdf8" },
      ]

  return (
    <section
      id="dashboard"
      className="relative z-10 mx-auto w-full max-w-7xl px-4 py-16"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-3xl font-bold tracking-tight text-foreground">
              Unified Dashboard
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Metrics, 3D visualizations, and navigation insights
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {TAB_CONFIG.map((tab) => {
              const Icon = tab.icon
              return (
                <Button
                  key={tab.id}
                  size="sm"
                  variant={activeTab === tab.id ? "default" : "outline"}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    "gap-2 transition-all",
                    activeTab === tab.id && "shadow-lg shadow-primary/20"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                  <ChevronRight
                    className={cn(
                      "h-3.5 w-3.5 transition-transform",
                      activeTab === tab.id && "rotate-90"
                    )}
                  />
                </Button>
              )
            })}
          </div>
        </div>

        <div className="overflow-hidden rounded-2xl border border-border bg-card/40 shadow-xl backdrop-blur-sm">
          <AnimatePresence mode="wait">
            {activeTab === "overview" && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="grid grid-cols-1 gap-6 p-6 lg:grid-cols-2"
              >
                <div className="space-y-6">
                  <InteractivePieChart
                    data={liveClassData}
                    title="Class Distribution"
                    showPercentages
                  />
                  <RiskGauge isComplete={isComplete} segResult={segResult} />
                </div>
                <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-muted/20 p-8">
                  <BarChart3 className="mb-4 h-16 w-16 text-muted-foreground/50" />
                  <p className="text-center text-sm text-muted-foreground">
                    Upload an image to see full analysis metrics, risk assessment,
                    and 3D terrain visualization
                  </p>
                  <div className="mt-4 flex gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <a href="#pipeline">View Pipeline</a>
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setActiveTab("3d-terrain")}
                    >
                      <Box className="mr-2 h-4 w-4" />
                      3D View
                    </Button>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === "3d-terrain" && (
              <motion.div
                key="3d-terrain"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="p-4"
              >
                <Terrain3D segResult={segResult} />
              </motion.div>
            )}

            {activeTab === "safety-path" && (
              <motion.div
                key="safety-path"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="p-4"
              >
                <SafetyNavigator3D segResult={segResult} />
              </motion.div>
            )}

            {activeTab === "path-planning" && (
              <motion.div
                key="path-planning"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3 }}
                className="p-4"
              >
                <PathPlanning3D segResult={segResult} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </section>
  )
}
