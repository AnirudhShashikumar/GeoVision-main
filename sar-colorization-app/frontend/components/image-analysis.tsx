"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Brain, LoaderCircle, X } from "lucide-react";
import { getProviderSettings, runImageAnalysis } from "@/services/api";
import { AIProviderModal } from "@/components/ai-provider-modal";

type Props = {
  image: string;
  analysisType: string;
  title: string;
  metadata?: Record<string, unknown>;
};

const sections = [
  ["Terrain", "terrain"],
  ["Structural and human features", "structural_and_human_features"],
  ["Vegetation and water", "vegetation_and_water"],
  ["Image quality", "image_quality"],
  ["Possible artifacts", "possible_artifacts"],
  ["Notes", "notes"],
  ["Limitations", "limitations"],
  ["Recommended actions", "recommended_actions"],
] as const;

export function ImageAnalysis({ image, analysisType, title, metadata }: Props) {
  const [open, setOpen] = useState(false);
  const [configurationOpen, setConfigurationOpen] = useState(false);
  const [configurationMessage, setConfigurationMessage] = useState("");
  const analysis = useMutation({ mutationFn: () => runImageAnalysis(image, analysisType, metadata) });
  const beginAnalysis = async () => {
    try {
      const provider = await getProviderSettings();
      if (!provider.configured || !provider.supported) {
        setConfigurationMessage("AI analysis requires an API key.");
        setConfigurationOpen(true);
        return;
      }
      setOpen(true);
      analysis.mutate();
    } catch (reason) {
      setConfigurationMessage(reason instanceof Error ? reason.message : "AI analysis requires an API key.");
      setConfigurationOpen(true);
    }
  };
  const continueAfterConfiguration = () => {
    setConfigurationOpen(false);
    setConfigurationMessage("Configuration saved successfully.");
    setOpen(true);
    analysis.reset();
    analysis.mutate();
  };
  const report = analysis.data?.report;
  return <>
    <button onClick={beginAnalysis} aria-label="Analyze image" className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs text-sky-200 hover:bg-sky-300/10"><Brain size={15}/>{"AI Analysis"}</button>
    {open && <div className="fixed inset-0 z-[60] overflow-y-auto bg-black/80 p-4 backdrop-blur-sm"><div className="mx-auto my-8 max-w-3xl rounded-2xl border border-white/10 bg-zinc-950 p-5 shadow-2xl sm:p-7"><div className="flex items-start justify-between gap-5"><div><p className="text-xs uppercase tracking-[.16em] text-sky-300">AI Image Analysis</p><h2 className="mt-1 text-lg font-semibold">{title}</h2></div><button onClick={() => setOpen(false)} className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="Close analysis"><X size={18}/></button></div>
      {configurationMessage && <p className="mt-5 rounded-xl border border-emerald-300/20 bg-emerald-300/[.07] p-3 text-sm text-emerald-100">{configurationMessage}</p>}
      {analysis.isPending && <div className="flex items-center gap-2 py-12 text-sm text-zinc-300"><LoaderCircle className="animate-spin" size={18}/> Reviewing the rendered image…</div>}
      {analysis.error && <p className="mt-6 rounded-xl border border-rose-400/20 bg-rose-400/10 p-4 text-sm text-rose-100">{analysis.error.message}</p>}
      {report && <div className="mt-6 space-y-6 text-sm leading-6 text-zinc-300"><div className="rounded-xl border border-sky-400/15 bg-sky-400/[.06] p-4"><p className="text-xs font-semibold uppercase tracking-[.12em] text-sky-300">Executive summary · {report.confidence} confidence</p><p className="mt-2">{report.executive_summary}</p></div><div className="grid gap-5 sm:grid-cols-2">{sections.map(([label, key]) => <section key={key}><h3 className="font-medium text-zinc-100">{label}</h3><ul className="mt-2 space-y-1 text-zinc-400">{report[key].map((item, index) => <li key={index}>• {item}</li>)}</ul></section>)}</div><div className="rounded-xl border border-amber-300/20 bg-amber-300/[.06] p-4 text-amber-50/90"><span className="font-medium">Scientific caution: </span>{report.disclaimer}</div><p className="text-xs text-zinc-600">Generated analysis</p></div>}
    </div></div>}
    <AIProviderModal open={configurationOpen} onClose={() => setConfigurationOpen(false)} onSaved={continueAfterConfiguration} notice={configurationMessage}/>{configurationMessage && !configurationOpen && !open && <p className="sr-only" role="status">{configurationMessage}</p>}
  </>;
}
