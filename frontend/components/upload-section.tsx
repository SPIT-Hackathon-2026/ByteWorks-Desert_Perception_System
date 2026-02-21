"use client"

import { useCallback, useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Upload, ImageIcon, X, FileImage } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"

interface UploadSectionProps {
  onImageUploaded: (file: File, url: string) => void
  uploadRef: React.RefObject<HTMLDivElement | null>
}

export function UploadSection({ onImageUploaded, uploadRef }: UploadSectionProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return
      setFileName(file.name)
      const url = URL.createObjectURL(file)
      setPreview(url)
      setUploading(true)
      setProgress(0)

      // Simulate upload progress
      let p = 0
      const interval = setInterval(() => {
        p += Math.random() * 25 + 10
        if (p >= 100) {
          p = 100
          clearInterval(interval)
          setUploading(false)
          setProgress(100)
          onImageUploaded(file, url)
        }
        setProgress(Math.min(p, 100))
      }, 300)
    },
    [onImageUploaded]
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFile(file)
    },
    [handleFile]
  )

  const clearPreview = () => {
    setPreview(null)
    setFileName(null)
    setProgress(0)
    setUploading(false)
  }

  return (
    <section ref={uploadRef} className="relative z-10 mx-auto w-full max-w-3xl px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="mb-2 text-center text-2xl font-bold text-foreground">
          Upload Terrain Image
        </h2>
        <p className="mb-8 text-center text-sm text-muted-foreground">
          Supported formats: PNG, JPG, TIFF, BMP
        </p>

        <div
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`relative flex min-h-[280px] flex-col items-center justify-center rounded-xl border-2 border-dashed transition-all ${
            isDragging
              ? "border-primary bg-primary/5"
              : "border-border bg-card/30 hover:border-primary/40"
          } ${preview ? "p-4" : "p-10"}`}
        >
          <AnimatePresence mode="wait">
            {preview ? (
              <motion.div
                key="preview"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="flex w-full flex-col items-center gap-4"
              >
                <div className="relative overflow-hidden rounded-lg">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={preview}
                    alt="Uploaded terrain preview"
                    className="max-h-[300px] rounded-lg object-contain"
                  />
                  <button
                    onClick={clearPreview}
                    className="absolute right-2 top-2 rounded-full bg-background/80 p-1 backdrop-blur-sm transition-colors hover:bg-destructive hover:text-destructive-foreground"
                    aria-label="Remove image"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <div className="flex w-full items-center gap-3">
                  <FileImage className="h-5 w-5 shrink-0 text-primary" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-foreground">
                      {fileName}
                    </p>
                    {uploading ? (
                      <Progress value={progress} className="mt-2 h-1.5" />
                    ) : (
                      <p className="text-xs text-chart-4">
                        Ready for analysis
                      </p>
                    )}
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center gap-4"
              >
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10">
                  <ImageIcon className="h-8 w-8 text-primary" />
                </div>
                <div className="text-center">
                  <p className="text-base font-medium text-foreground">
                    Drop your terrain image here
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    or click to browse files
                  </p>
                </div>
                <Button
                  variant="outline"
                  className="gap-2 border-border text-foreground"
                  onClick={() => inputRef.current?.click()}
                >
                  <Upload className="h-4 w-4" />
                  Browse Files
                </Button>
              </motion.div>
            )}
          </AnimatePresence>
          {!preview && (
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="absolute inset-0 cursor-pointer opacity-0"
              onChange={handleInputChange}
              aria-label="Upload image file"
            />
          )}
        </div>
      </motion.div>
    </section>
  )
}
