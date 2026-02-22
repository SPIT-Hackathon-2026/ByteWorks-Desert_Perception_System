"use client"

import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

interface HUDScannerProps {
    isActive: boolean
    color: string
    children: React.ReactNode
    label?: string
}

export function HUDScanner({ isActive, color, children, label }: HUDScannerProps) {
    return (
        <div className="group relative h-full w-full overflow-hidden">
            {/* Corner Brackets */}
            <div className="pointer-events-none absolute inset-0 z-20">
                <div className="absolute left-2 top-2 h-3 w-3 border-l-2 border-t-2 transition-colors duration-500" style={{ borderColor: isActive ? color : "transparent" }} />
                <div className="absolute right-2 top-2 h-3 w-3 border-r-2 border-t-2 transition-colors duration-500" style={{ borderColor: isActive ? color : "transparent" }} />
                <div className="absolute bottom-2 left-2 h-3 w-3 border-b-2 border-l-2 transition-colors duration-500" style={{ borderColor: isActive ? color : "transparent" }} />
                <div className="absolute bottom-2 right-2 h-3 w-3 border-b-2 border-r-2 transition-colors duration-500" style={{ borderColor: isActive ? color : "transparent" }} />
            </div>

            {/* Targeting Text */}
            {isActive && (
                <div className="pointer-events-none absolute left-3 top-3 z-30 flex flex-col gap-0.5">
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0, 1, 0.5, 1] }}
                        transition={{ duration: 0.1, repeat: Infinity, repeatDelay: 2 }}
                        className="font-mono text-[7px] font-bold uppercase tracking-wider"
                        style={{ color }}
                    >
                        Targeting...
                    </motion.p>
                    <p className="font-mono text-[6px] opacity-50 transition-colors" style={{ color }}>
                        {label || "PROC_0x7F"}
                    </p>
                </div>
            )}

            {/* Main Content */}
            <div className={cn(
                "relative h-full w-full transition-transform duration-700",
                isActive ? "scale-[1.02]" : "scale-100"
            )}>
                {children}

                {/* Digital Grain / Noise Overlay */}
                <div className="pointer-events-none absolute inset-0 opacity-[0.03] mix-blend-overlay"
                    style={{ backgroundImage: "url('https://grainy-gradients.vercel.app/noise.svg')" }} />
            </div>

            {/* Scanline Animation */}
            {isActive && (
                <>
                    <motion.div
                        className="pointer-events-none absolute inset-0 z-10"
                        animate={{ backgroundPosition: ["0 0", "0 100%"] }}
                        transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
                        style={{
                            backgroundImage: `linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06))`,
                            backgroundSize: "100% 4px, 3px 100%",
                            opacity: 0.15
                        }}
                    />
                    <motion.div
                        className="pointer-events-none absolute left-0 right-0 z-20 h-[1px] opacity-50"
                        style={{ background: color, boxShadow: `0 0 8px ${color}` }}
                        animate={{ top: ["0%", "100%", "0%"] }}
                        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                    />
                </>
            )}

            {/* Vignette */}
            <div className="pointer-events-none absolute inset-0 z-10 bg-[radial-gradient(circle,transparent_60%,rgba(0,0,0,0.3)_100%)]" />
        </div>
    )
}
