"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ArrowDown, GitCompareArrows, LoaderCircle } from "lucide-react";
import { runComparison } from "@/services/api";
import { UploadZone } from "@/components/upload-zone";
import { MetricCard } from "@/components/metric-card";
import { SectionHeader } from "@/components/section-header";
import { formatMetric } from "@/lib/utils";

const optical = { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/tiff": [".tif", ".tiff"] };

export function ComparisonWorkspace() {
  const [pix, setPix] = useState<File>(); const [structure, setStructure] = useState<File>(); const [groundTruth, setGroundTruth] = useState<File>();
  const compare = useMutation({ mutationFn: () => pix && structure && groundTruth ? runComparison(pix, structure, groundTruth) : Promise.reject(new Error("Upload two predictions and a common ground truth.")) });
  return <><SectionHeader eyebrow="Evaluation" title="Model Comparison">Compare visual realism against structure-preserving reconstruction on the same target.</SectionHeader>
    <section className="panel p-5 sm:p-7"><div className="grid gap-5 lg:grid-cols-3"><UploadZone label="Pix2Pix prediction" hint="Generated PNG/JPEG/TIFF" accept={optical} file={pix} onFile={setPix}/><UploadZone label="SARFusionFormer prediction" hint="Generated PNG/JPEG/TIFF" accept={optical} file={structure} onFile={setStructure}/><UploadZone label="Common ground truth" hint="Required for comparison" accept={optical} file={groundTruth} onFile={setGroundTruth}/></div><button onClick={() => compare.mutate()} disabled={compare.isPending || !(pix && structure && groundTruth)} className="mt-7 inline-flex items-center gap-2 rounded-xl bg-sky-300 px-5 py-3 text-sm font-semibold text-zinc-950 disabled:opacity-40">{compare.isPending ? <LoaderCircle className="animate-spin" size={17}/> : <GitCompareArrows size={17}/>}Compare models</button>{compare.error && <p className="mt-3 text-sm text-rose-300">{compare.error.message}</p>}</section>
    {!compare.data ? <div className="mt-10 grid place-items-center rounded-3xl border border-dashed border-white/[.1] py-16 text-center text-zinc-500"><ArrowDown className="mb-3"/><p className="text-sm">Upload generated outputs to evaluate both models on one ground truth.</p></div> : <section className="mt-10"><div className="grid gap-4 md:grid-cols-2"><article className="panel p-6"><p className="eyebrow">Pix2Pix</p><h2 className="mt-2 text-2xl font-semibold">Visual realism</h2><ul className="mt-5 space-y-2 text-sm text-zinc-400"><li>Natural colour and texture</li><li>Strong presentation quality</li><li>May invent plausible details</li></ul></article><article className="panel p-6"><p className="eyebrow">SARFusionFormer</p><h2 className="mt-2 text-2xl font-semibold">Scientific structure</h2><ul className="mt-5 space-y-2 text-sm text-zinc-400"><li>Uses VV/VH directly</li><li>Structure preserving</li><li>Less visually colourful</li></ul></article></div><div className="mt-5 grid gap-3 sm:grid-cols-3"><MetricCard label="Pix2Pix PSNR" value={`${formatMetric(compare.data.pix2pix.psnr, 2)} dB`}/><MetricCard label="SARFusionFormer PSNR" value={`${formatMetric(compare.data.sarfusionformer.psnr, 2)} dB`}/><MetricCard label="SSIM · Pix / SAR" value={`${formatMetric(compare.data.pix2pix.ssim, 3)} / ${formatMetric(compare.data.sarfusionformer.ssim, 3)}`}/></div></section>}</>;
}
