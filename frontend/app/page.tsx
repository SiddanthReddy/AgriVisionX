"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";

/* -------------------------------------------------------------------------- */
/*  Config & types                                                            */
/* -------------------------------------------------------------------------- */

const API_BASE = "http://127.0.0.1:8000";
const MAX_SIZE_MB = 10;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

interface PredictionResult {
  disease: string;
  confidence: number;
  heatmap_url: string;
}

type RiskKey = "low" | "medium" | "high";

const RISK_STYLES: Record<
  RiskKey,
  { label: string; range: string; badge: string; dot: string; bar: string }
> = {
  low: {
    label: "Low Risk",
    range: "0–40% confidence",
    badge: "bg-emerald-50 text-emerald-800 ring-emerald-600/25",
    dot: "bg-emerald-500",
    bar: "from-emerald-400 to-emerald-600",
  },
  medium: {
    label: "Medium Risk",
    range: "41–75% confidence",
    badge: "bg-amber-50 text-amber-800 ring-amber-600/25",
    dot: "bg-amber-500",
    bar: "from-amber-300 to-amber-500",
  },
  high: {
    label: "High Risk",
    range: "76–100% confidence",
    badge: "bg-red-50 text-red-800 ring-red-600/25",
    dot: "bg-red-500",
    bar: "from-red-400 to-red-600",
  },
};

const TECH_DETAILS: ReadonlyArray<readonly [string, string]> = [
  ["Model", "Hybrid CNN + Swin Transformer"],
  ["Number of Classes", "29"],
  ["Input Resolution", "224 × 224"],
  ["Framework", "PyTorch"],
  ["Backend", "FastAPI"],
  ["Frontend", "Next.js"],
  ["Explainability", "Grad-CAM"],
];

const CARD =
  "rounded-2xl border border-white/70 bg-white/70 shadow-[0_8px_30px_rgba(6,78,59,0.06)] ring-1 ring-emerald-900/5 backdrop-blur-xl";
const CARD_HOVER =
  "transition-all duration-300 ease-out hover:-translate-y-1 hover:shadow-[0_18px_45px_rgba(6,78,59,0.14)]";

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                   */
/* -------------------------------------------------------------------------- */

function getRiskKey(confidence: number): RiskKey {
  if (confidence < 41) return "low";
  if (confidence < 76) return "medium";
  return "high";
}

function parseDisease(raw: string): { crop: string; condition: string } {
  const clean = (s: string) => s.replace(/_/g, " ").replace(/\s+/g, " ").trim();
  const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

  let crop = "";
  let condition = raw;

  const triple = raw.indexOf("___");
  if (triple > -1) {
    crop = raw.slice(0, triple);
    condition = raw.slice(triple + 3);
  } else {
    const single = raw.indexOf("_");
    if (single > -1) {
      crop = raw.slice(0, single);
      condition = raw.slice(single + 1);
    }
  }

  return { crop: cap(clean(crop)), condition: cap(clean(condition)) };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/* -------------------------------------------------------------------------- */
/*  Small presentational components                                           */
/* -------------------------------------------------------------------------- */

const ICONS = {
  upload: [
    "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4",
    "M17 8l-5-5-5 5",
    "M12 3v12",
  ],
  layers: ["M12 2 2 7l10 5 10-5-10-5z", "M2 17l10 5 10-5", "M2 12l10 5 10-5"],
  eye: [
    "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z",
    "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  ],
  server: [
    "M4 4h16a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z",
    "M4 13h16a1 1 0 0 1 1 1v5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-5a1 1 0 0 1 1-1z",
    "M7 7.5h.01",
    "M7 16.5h.01",
  ],
  leaf: [
    "M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z",
    "M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12",
  ],
  image: [
    "M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z",
    "M8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3z",
    "M21 15l-5-5L5 21",
  ],
  sparkles: [
    "M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z",
    "M19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16z",
  ],
  x: ["M18 6 6 18", "M6 6l12 12"],
  alert: [
    "M12 9v4",
    "M12 17h.01",
    "M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z",
  ],
} as const;

function Icon({
  name,
  className = "h-5 w-5",
}: {
  name: keyof typeof ICONS;
  className?: string;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {ICONS[name].map((d) => (
        <path key={d} d={d} />
      ))}
    </svg>
  );
}

function StatCard({
  icon,
  label,
  value,
  valueClassName = "text-xl",
  live = false,
}: {
  icon: keyof typeof ICONS;
  label: string;
  value: string;
  valueClassName?: string;
  live?: boolean;
}) {
  return (
    <div className={`${CARD} ${CARD_HOVER} flex flex-col justify-between p-5`}>
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-500">{label}</p>
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-50 text-emerald-700">
          <Icon name={icon} className="h-[18px] w-[18px]" />
        </span>
      </div>
      <p
        className={`mt-5 flex items-center gap-2 font-semibold leading-snug tracking-tight text-emerald-950 ${valueClassName}`}
      >
        {live && (
          <span className="relative flex h-2.5 w-2.5" aria-hidden="true">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-70" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
          </span>
        )}
        {value}
      </p>
    </div>
  );
}

function CardTitle({
  id,
  title,
  subtitle,
}: {
  id?: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="mb-5">
      <h2
        id={id}
        className="text-lg font-semibold tracking-tight text-emerald-950"
      >
        {title}
      </h2>
      {subtitle && <p className="mt-1 text-sm text-slate-500">{subtitle}</p>}
    </div>
  );
}

function Spinner() {
  return (
    <svg
      className="h-4 w-4 animate-spin"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeOpacity="0.25"
        strokeWidth="4"
      />
      <path
        d="M22 12a10 10 0 0 0-10-10"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/*  Page                                                                      */
/* -------------------------------------------------------------------------- */

export default function AgriVisionXPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [barWidth, setBarWidth] = useState(0);
  const [originalLoaded, setOriginalLoaded] = useState(false);
  const [heatmapLoaded, setHeatmapLoaded] = useState(false);
  const [heatmapFailed, setHeatmapFailed] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // Release object URLs when the preview changes or the page unmounts.
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  // Animate the confidence bar and reset image fade-ins for every new result.
  useEffect(() => {
    setBarWidth(0);
    setHeatmapLoaded(false);
    setHeatmapFailed(false);
    if (!result) return;

    const timer = setTimeout(
      () => setBarWidth(Math.min(100, Math.max(0, result.confidence))),
      80
    );
    resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    return () => clearTimeout(timer);
  }, [result]);

  const selectFile = (selected: File | undefined) => {
    if (!selected) return;

    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError("Unsupported file type. Upload a JPG, PNG or WebP image.");
      return;
    }
    if (selected.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`This file is larger than ${MAX_SIZE_MB} MB. Choose a smaller image.`);
      return;
    }

    setError(null);
    setResult(null);
    setOriginalLoaded(false);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    selectFile(e.target.files?.[0]);
    e.target.value = "";
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const handlePredict = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(
          `The server returned an error (status ${response.status}). Try again with a different image.`
        );
      }

      const data = await response.json();

      if (
        typeof data?.disease !== "string" ||
        typeof data?.confidence !== "number" ||
        typeof data?.heatmap_url !== "string"
      ) {
        throw new Error("The server response was not in the expected format.");
      }

      setResult(data as PredictionResult);
    } catch (err) {
      if (err instanceof TypeError) {
        setError(
          `Cannot reach the backend at ${API_BASE}. Start the FastAPI server and try again.`
        );
      } else {
        setError(
          err instanceof Error ? err.message : "Something went wrong. Try again."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const risk = result ? RISK_STYLES[getRiskKey(result.confidence)] : null;
  const parsed = result ? parseDisease(result.disease) : null;

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#f3f6f4] text-slate-800 antialiased">
      <style>{`
        @keyframes avx-rise {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: none; }
        }
        @keyframes avx-shine {
          from { background-position: -200% 0; }
          to { background-position: 200% 0; }
        }
        .avx-rise { animation: avx-rise 0.6s ease-out both; }
        .avx-shine {
          background-image: linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.45) 50%, transparent 70%);
          background-size: 200% 100%;
          animation: avx-shine 2.4s linear infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .avx-rise, .avx-shine { animation: none; }
          * { transition-duration: 0.01ms !important; }
        }
      `}</style>

      {/* Ambient background */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 -z-0 h-[520px] bg-[radial-gradient(60%_60%_at_50%_0%,rgba(16,185,129,0.14),transparent)]"
      />

      <main className="relative mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-6 sm:px-6 sm:py-10 lg:px-8">
        {/* ------------------------------------------------------------------ */}
        {/* 1. Hero                                                            */}
        {/* ------------------------------------------------------------------ */}
        <header className="relative overflow-hidden rounded-3xl border border-emerald-900/30 bg-gradient-to-br from-emerald-950 via-emerald-900 to-emerald-700 px-6 py-14 text-white shadow-xl shadow-emerald-950/20 sm:px-12 sm:py-20">
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -right-24 -top-24 h-80 w-80 rounded-full bg-emerald-400/25 blur-3xl"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -bottom-32 -left-16 h-72 w-72 rounded-full bg-lime-300/10 blur-3xl"
          />
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-0 opacity-[0.07] [background-image:linear-gradient(to_right,white_1px,transparent_1px),linear-gradient(to_bottom,white_1px,transparent_1px)] [background-size:44px_44px]"
          />

          <div className="relative max-w-3xl">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-3.5 py-1.5 text-sm font-medium text-emerald-50 backdrop-blur-md">
              <span className="relative flex h-2.5 w-2.5" aria-hidden="true">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-300 opacity-70" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-400" />
              </span>
              🟢 Model Online
            </span>

            <h1 className="mt-6 text-5xl font-semibold tracking-tight sm:text-6xl lg:text-7xl">
              🌿 AgriVisionX
            </h1>

            <p className="mt-4 text-xl font-medium text-emerald-50 sm:text-2xl">
              Explainable AI Platform for Multi-Crop Disease Diagnosis
            </p>

            <p className="mt-5 inline-block rounded-lg border border-white/15 bg-white/10 px-3 py-1.5 text-sm font-medium text-emerald-100 backdrop-blur-md sm:text-base">
              Hybrid CNN + Swin Transformer + Grad-CAM
            </p>

            <p className="mt-6 max-w-xl text-base leading-relaxed text-emerald-100/90 sm:text-lg">
              Real-time disease prediction and visual explanation for multiple
              crop species.
            </p>

            <p className="mt-8 max-w-2xl border-t border-white/15 pt-5 text-sm leading-relaxed text-emerald-100/70">
              An Explainable Hybrid Vision Transformer Framework for Multi-Crop
              Leaf Disease Classification and Interpretation
            </p>
          </div>
        </header>

        {/* ------------------------------------------------------------------ */}
        {/* 2. Stats                                                           */}
        {/* ------------------------------------------------------------------ */}
        <section
          aria-label="Platform overview"
          className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
        >
          <StatCard
            icon="leaf"
            label="Supported Diseases"
            value="29"
            valueClassName="text-4xl"
          />
          <StatCard
            icon="layers"
            label="Architecture"
            value="Hybrid CNN + Swin Transformer"
            valueClassName="text-lg"
          />
          <StatCard
            icon="eye"
            label="Explainability"
            value="Grad-CAM Enabled"
            valueClassName="text-xl"
          />
          <StatCard
            icon="server"
            label="Backend Status"
            value="Active"
            valueClassName="text-xl"
            live
          />
        </section>

        {/* ------------------------------------------------------------------ */}
        {/* 3. Upload                                                          */}
        {/* ------------------------------------------------------------------ */}
        <section
          aria-labelledby="upload-title"
          className={`${CARD} p-6 sm:p-8`}
        >
          <CardTitle
            id="upload-title"
            title="Upload leaf image"
            subtitle={`JPG, PNG or WebP, up to ${MAX_SIZE_MB} MB. Use a clear, well-lit photo of a single leaf.`}
          />

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_TYPES.join(",")}
            onChange={handleInputChange}
            className="sr-only"
            aria-label="Choose a leaf image"
            tabIndex={-1}
          />

          <div
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`rounded-xl border-2 border-dashed p-6 transition-colors duration-200 sm:p-8 ${
              dragActive
                ? "border-emerald-500 bg-emerald-50/80"
                : "border-emerald-900/15 bg-white/50 hover:border-emerald-500/60 hover:bg-emerald-50/40"
            }`}
          >
            {preview && file ? (
              <div className="flex flex-col items-center gap-6 sm:flex-row">
                <div className="h-44 w-44 shrink-0 overflow-hidden rounded-xl bg-slate-100 ring-1 ring-emerald-900/10">
                  <img
                    src={preview}
                    alt="Selected leaf preview"
                    onLoad={() => setOriginalLoaded(true)}
                    className={`h-full w-full object-cover transition-opacity duration-700 ${
                      originalLoaded ? "opacity-100" : "opacity-0"
                    }`}
                  />
                </div>

                <div className="min-w-0 flex-1 text-center sm:text-left">
                  <p className="text-sm font-medium text-slate-500">
                    Selected file
                  </p>
                  <p
                    title={file.name}
                    className="mt-1 truncate text-base font-semibold text-emerald-950"
                  >
                    {file.name}
                  </p>
                  <p className="mt-1 text-sm text-slate-500">
                    {formatBytes(file.size)}
                  </p>

                  <div className="mt-5 flex flex-wrap justify-center gap-3 sm:justify-start">
                    <button
                      type="button"
                      onClick={() => inputRef.current?.click()}
                      className="rounded-lg border border-emerald-900/15 bg-white px-4 py-2 text-sm font-medium text-emerald-900 transition-colors hover:bg-emerald-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                    >
                      Change image
                    </button>
                    <button
                      type="button"
                      onClick={clearFile}
                      className="inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                    >
                      <Icon name="x" className="h-4 w-4" />
                      Remove
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center py-6 text-center">
                <span
                  className={`flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700 transition-transform duration-300 ${
                    dragActive ? "scale-110" : ""
                  }`}
                >
                  <Icon name="upload" className="h-7 w-7" />
                </span>
                <p className="mt-4 text-base font-semibold text-emerald-950">
                  {dragActive
                    ? "Drop the image to select it"
                    : "Drag and drop a leaf image here"}
                </p>
                <p className="mt-1 text-sm text-slate-500">or</p>
                <button
                  type="button"
                  onClick={() => inputRef.current?.click()}
                  className="mt-3 rounded-lg border border-emerald-900/15 bg-white px-4 py-2 text-sm font-medium text-emerald-900 shadow-sm transition-all hover:-translate-y-0.5 hover:bg-emerald-50 hover:shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
                >
                  Choose file
                </button>
              </div>
            )}
          </div>

          {error && (
            <div
              role="alert"
              className="mt-5 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800"
            >
              <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
              <p>{error}</p>
            </div>
          )}

          <div className="mt-6 flex flex-col items-stretch gap-3 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm text-slate-500" aria-live="polite">
              {loading
                ? "Analyzing leaf. This usually takes a few seconds."
                : file
                ? "Ready when you are."
                : "Select an image to enable prediction."}
            </p>
            <button
              type="button"
              onClick={handlePredict}
              disabled={!file || loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-600 to-emerald-800 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-900/20 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-emerald-900/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 disabled:translate-y-0 disabled:cursor-not-allowed disabled:from-slate-300 disabled:to-slate-400 disabled:shadow-none"
            >
              {loading ? (
                <>
                  <Spinner />
                  Analyzing…
                </>
              ) : (
                <>
                  <Icon name="sparkles" className="h-4 w-4" />
                  Predict Disease
                </>
              )}
            </button>
          </div>
        </section>

        {/* ------------------------------------------------------------------ */}
        {/* 4 + 5. Results and visualization (after prediction only)          */}
        {/* ------------------------------------------------------------------ */}
        {result && risk && parsed && (
          <div ref={resultsRef} className="flex scroll-mt-6 flex-col gap-8">
            <section
              aria-labelledby="results-title"
              className={`${CARD} ${CARD_HOVER} avx-rise p-6 sm:p-8`}
            >
              <CardTitle id="results-title" title="Prediction results" />

              <div className="grid gap-8 md:grid-cols-3">
                <div>
                  <p className="text-sm font-medium text-slate-500">
                    Disease name
                  </p>
                  {parsed.crop && (
                    <span className="mt-3 inline-block rounded-md bg-emerald-50 px-2 py-0.5 text-xs font-medium text-emerald-800 ring-1 ring-inset ring-emerald-600/20">
                      {parsed.crop}
                    </span>
                  )}
                  <p className="mt-2 text-2xl font-semibold leading-tight tracking-tight text-emerald-950 sm:text-3xl">
                    {parsed.condition}
                  </p>
                </div>

                <div>
                  <p className="text-sm font-medium text-slate-500">
                    Confidence score
                  </p>
                  <p className="mt-3 text-5xl font-semibold tabular-nums tracking-tight text-emerald-950">
                    {result.confidence.toFixed(2)}
                    <span className="ml-0.5 text-2xl text-slate-400">%</span>
                  </p>
                </div>

                <div>
                  <p className="text-sm font-medium text-slate-500">
                    Risk level
                  </p>
                  <span
                    className={`mt-3 inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-sm font-semibold ring-1 ring-inset ${risk.badge}`}
                  >
                    <span className={`h-2 w-2 rounded-full ${risk.dot}`} />
                    {risk.label}
                  </span>
                  <p className="mt-2 text-xs text-slate-500">{risk.range}</p>
                </div>
              </div>

              <div className="mt-8">
                <div
                  role="progressbar"
                  aria-label="Prediction confidence"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(result.confidence)}
                  className="h-3 w-full overflow-hidden rounded-full bg-emerald-100"
                >
                  <div
                    className={`relative h-full overflow-hidden rounded-full bg-gradient-to-r transition-[width] duration-1000 ease-out ${risk.bar}`}
                    style={{ width: `${barWidth}%` }}
                  >
                    <div className="avx-shine absolute inset-0" />
                  </div>
                </div>
                <div className="mt-2 flex justify-between text-xs text-slate-400">
                  <span>0%</span>
                  <span>100%</span>
                </div>
              </div>
            </section>

            <section
              aria-label="Image visualization"
              className="avx-rise grid gap-6 md:grid-cols-2"
            >
              <div className={`${CARD} ${CARD_HOVER} p-5 sm:p-6`}>
                <h2 className="mb-4 text-lg font-semibold tracking-tight text-emerald-950">
                  Original Leaf Image
                </h2>
                <div className="aspect-square w-full overflow-hidden rounded-xl bg-slate-100 ring-1 ring-emerald-900/10">
                  {preview && (
                    <img
                      src={preview}
                      alt="Original leaf uploaded for analysis"
                      onLoad={() => setOriginalLoaded(true)}
                      className={`h-full w-full object-cover transition-opacity duration-700 ${
                        originalLoaded ? "opacity-100" : "opacity-0"
                      }`}
                    />
                  )}
                </div>
              </div>

              <div className={`${CARD} ${CARD_HOVER} p-5 sm:p-6`}>
                <h2 className="mb-4 text-lg font-semibold tracking-tight text-emerald-950">
                  Grad-CAM Heatmap
                </h2>
                <div className="relative aspect-square w-full overflow-hidden rounded-xl bg-slate-100 ring-1 ring-emerald-900/10">
                  {!heatmapLoaded && !heatmapFailed && (
                    <div className="absolute inset-0 animate-pulse bg-gradient-to-br from-slate-100 to-emerald-100" />
                  )}
                  {heatmapFailed ? (
                    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center text-sm text-slate-500">
                      <Icon name="image" className="h-8 w-8 text-slate-400" />
                      <p>
                        The heatmap could not be loaded. Check that the backend
                        is serving <code>/heatmaps</code>.
                      </p>
                    </div>
                  ) : (
                    <img
                      src={`http://127.0.0.1:8000${result.heatmap_url}`}
                      alt="Grad-CAM heatmap highlighting the regions that influenced the prediction"
                      onLoad={() => setHeatmapLoaded(true)}
                      onError={() => setHeatmapFailed(true)}
                      className={`h-full w-full object-cover transition-opacity duration-700 ${
                        heatmapLoaded ? "opacity-100" : "opacity-0"
                      }`}
                    />
                  )}
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ------------------------------------------------------------------ */}
        {/* 6 + 7. Explanation and technical details                           */}
        {/* ------------------------------------------------------------------ */}
        <div className="grid gap-6 lg:grid-cols-5">
          <section
            aria-labelledby="explanation-title"
            className={`${CARD} ${CARD_HOVER} p-6 sm:p-8 lg:col-span-2`}
          >
            <CardTitle id="explanation-title" title="AI Explanation" />
            <p className="text-[15px] leading-relaxed text-slate-600">
              Grad-CAM visualization highlights the image regions that
              contributed most strongly to the model&apos;s prediction. Warmer
              regions indicate higher importance and provide visual
              interpretability for disease classification.
            </p>

            <div className="mt-6" aria-hidden="true">
              <div className="h-2.5 w-full rounded-full bg-gradient-to-r from-blue-500 via-yellow-300 to-red-500" />
              <div className="mt-2 flex justify-between text-xs text-slate-500">
                <span>Lower importance</span>
                <span>Higher importance</span>
              </div>
            </div>
          </section>

          <section
            aria-labelledby="tech-title"
            className={`${CARD} ${CARD_HOVER} p-6 sm:p-8 lg:col-span-3`}
          >
            <CardTitle id="tech-title" title="Technical Details" />
            <dl className="divide-y divide-emerald-900/10">
              {TECH_DETAILS.map(([label, value]) => (
                <div
                  key={label}
                  className="flex flex-col gap-1 py-3 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between"
                >
                  <dt className="text-sm text-slate-500">{label}</dt>
                  <dd className="text-sm font-semibold text-emerald-950">
                    {value}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        </div>

        {/* ------------------------------------------------------------------ */}
        {/* 8. Footer                                                          */}
        {/* ------------------------------------------------------------------ */}
        <footer className="border-t border-emerald-900/10 pb-4 pt-8 text-center">
          <p className="text-sm font-semibold text-emerald-950">
            AgriVisionX © 2026
          </p>
          <p className="mt-1 text-sm text-slate-500">
            Explainable AI for Smart Agriculture
          </p>
        </footer>
      </main>
    </div>
  );
}
