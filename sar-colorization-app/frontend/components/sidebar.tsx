"use client";

import { useQuery } from "@tanstack/react-query";
import { BarChart3, Boxes, FileText, Github, GitCompareArrows, Info, Moon, Orbit, ScanSearch, Settings2, Sparkles } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { getHealth } from "@/services/api";
import { cn } from "@/lib/utils";

const navigation = [
  { href: "/pix2pix", label: "Pix2Pix", icon: Sparkles },
  { href: "/structure", label: "SARFusionFormer", icon: ScanSearch },
  { href: "/comparison", label: "Model Comparison", icon: GitCompareArrows },
  { href: "/architecture", label: "Architecture", icon: Boxes },
  { href: "/benchmark", label: "Benchmark", icon: BarChart3 },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/about", label: "About", icon: Info },
  { href: "/settings", label: "Settings · AI Providers", icon: Settings2 },
];

export function Sidebar() {
  const path = usePathname(); const { resolvedTheme, setTheme } = useTheme();
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: false, refetchInterval: 30_000 });
  return <aside className="sticky top-0 flex h-dvh w-full shrink-0 flex-col border-b border-white/[.08] bg-[#0c0c0e]/90 p-5 backdrop-blur-xl lg:w-72 lg:border-b-0 lg:border-r">
    <Link href="/pix2pix" className="flex items-center gap-3 px-2"><span className="grid h-9 w-9 place-items-center rounded-xl bg-sky-400 text-zinc-950 shadow-glow"><Orbit size={21}/></span><span><b className="block text-sm">GeoVision</b><span className="text-xs text-zinc-500">Advanced SAR reconstruction</span></span></Link>
    <p className="mt-7 px-2 text-xs leading-5 text-zinc-500">AI-powered SAR to Optical Reconstruction Platform</p>
    <nav className="mt-6 space-y-1">{navigation.map(({ href, label, icon: Icon }) => <Link key={href} href={href} className={cn("flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition", path === href ? "bg-sky-400/10 text-sky-200" : "text-zinc-400 hover:bg-white/[.05] hover:text-white")}><Icon size={17}/>{label}</Link>)}</nav>
    <div className="my-7 border-t border-white/[.08]"/><div className="px-2"><p className="eyebrow">Model status</p><div className="mt-3 space-y-2">{["pix2pix", "sarfusionformer", "color_corrector"].map(name => { const item = health.data?.models[name]; const ready = Boolean(item?.available); return <div key={name} className={cn("flex items-center gap-2 rounded-xl border px-3 py-2.5 text-xs transition", ready ? "border-[#39ff74]/60 bg-[#39ff74]/[.08] shadow-[0_0_20px_rgba(57,255,116,.20)]" : "glass")}><span className={cn("h-2.5 w-2.5 shrink-0 rounded-full", ready ? "status-ready-dot bg-[#39ff74]" : "bg-zinc-600")}/><span className="capitalize text-zinc-300">{name.replace("_", " ")}</span><span className={cn("ml-auto", ready ? "text-[#76ff9e]" : "text-zinc-500")}>{ready ? "Ready" : health.isLoading ? "Checking" : "Offline"}</span></div>; })}</div></div>
    <div className="mt-auto flex items-center justify-between border-t border-white/[.08] pt-4"><span className="text-xs text-zinc-500">v1.0 · Research edition</span><div className="flex"><a href="https://github.com" target="_blank" className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="GitHub"><Github size={16}/></a><button onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")} className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="Toggle theme"><Moon size={16}/></button></div></div>
  </aside>;
}
