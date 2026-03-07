"use client"

import { useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Sector,
  Tooltip,
} from "recharts"
import { cn } from "@/lib/utils"

export interface PieChartDataItem {
  name: string
  value: number
  fill: string
}

interface InteractivePieChartProps {
  data: PieChartDataItem[]
  title?: string
  className?: string
  showPercentages?: boolean
}

const renderActiveShape = (props: any) => {
  const {
    cx,
    cy,
    innerRadius,
    outerRadius,
    startAngle,
    endAngle,
    fill,
    payload,
  } = props

  return (
    <g>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 6}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
        style={{
          filter: "drop-shadow(0 0 8px rgba(0,0,0,0.3))",
          cursor: "pointer",
        }}
      />
    </g>
  )
}

export function InteractivePieChart({
  data,
  title = "Class Distribution",
  className,
  showPercentages = true,
}: InteractivePieChartProps) {
  const [activeIndex, setActiveIndex] = useState<number | null>(null)
  const total = data.reduce((sum, d) => sum + d.value, 0)

  const onPieEnter = useCallback((_: unknown, index: number) => {
    setActiveIndex(index)
  }, [])

  const onPieLeave = useCallback(() => {
    setActiveIndex(null)
  }, [])

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload?.length) return null
    const item = payload[0].payload
    const pct = total > 0 ? ((item.value / total) * 100).toFixed(1) : "0"
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-lg">
        <p className="font-medium text-foreground">{item.name}</p>
        <p className="font-mono text-sm" style={{ color: item.fill }}>
          {item.value.toFixed(1)}% ({pct} of total)
        </p>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("rounded-xl border border-border bg-card/50 p-5 backdrop-blur-sm", className)}
    >
      <h3 className="mb-4 text-sm font-semibold text-foreground">{title}</h3>

      <div className="relative">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={70}
              outerRadius={95}
              paddingAngle={4}
              dataKey="value"
              activeIndex={activeIndex ?? undefined}
              activeShape={renderActiveShape}
              onMouseEnter={onPieEnter}
              onMouseLeave={onPieLeave}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.fill}
                  stroke={activeIndex === index ? "rgba(255,255,255,0.5)" : "transparent"}
                  strokeWidth={activeIndex === index ? 2 : 0}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>

        {showPercentages && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="text-center">
              {activeIndex !== null ? (
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeIndex}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    className="rounded-full bg-card/90 px-4 py-2 shadow-lg"
                  >
                    <p className="text-xs text-muted-foreground">
                      {data[activeIndex]?.name}
                    </p>
                    <p
                      className="font-mono text-lg font-bold"
                      style={{ color: data[activeIndex]?.fill }}
                    >
                      {total > 0
                        ? ((data[activeIndex]?.value ?? 0) / total * 100).toFixed(1)
                        : "0"}
                      %
                    </p>
                  </motion.div>
                </AnimatePresence>
              ) : (
                <div className="rounded-full bg-card/90 px-4 py-2 shadow-lg">
                  <p className="text-xs text-muted-foreground">Total</p>
                  <p className="font-mono text-lg font-bold text-primary">100%</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 flex flex-wrap justify-center gap-3">
        {data.map((item, i) => (
          <button
            key={item.name}
            onClick={() => setActiveIndex(activeIndex === i ? null : i)}
            className={cn(
              "flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
              activeIndex === i
                ? "ring-2 ring-offset-2 ring-offset-background"
                : "hover:bg-muted/50"
            )}
            style={{
              backgroundColor: activeIndex === i ? `${item.fill}20` : "transparent",
              color: item.fill,
              ringColor: item.fill,
            }}
          >
            <div
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: item.fill }}
            />
            {item.name} (
            {total > 0 ? ((item.value / total) * 100).toFixed(1) : "0"}%)
          </button>
        ))}
      </div>
    </motion.div>
  )
}
