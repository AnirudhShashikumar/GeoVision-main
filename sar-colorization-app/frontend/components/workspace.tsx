"use client";
import { motion } from "framer-motion";
import { useParams } from "next/navigation";
import { Sidebar } from "@/components/sidebar";
import { Pix2PixWorkspace } from "@/components/pix2pix-workspace";
import { StructureWorkspace } from "@/components/structure-workspace";
import { ComparisonWorkspace } from "@/components/comparison-workspace";
import { ProviderSettings } from "@/components/provider-settings";
import { ArchitectureView } from "@/components/architecture-view";
import { BenchmarkView } from "@/components/benchmark-view";
import { ReportsCenter } from "@/components/reports-center";

const applications = ["Flood Monitoring", "Disaster Response", "Agriculture", "Urban Mapping", "Military Intelligence", "Forest Monitoring", "Climate Research", "Infrastructure Planning", "Coastal Monitoring"];

function ResearchView({ page }: { page: string }) {
 if (page === "architecture") return <ArchitectureView/>;
 if (page === "benchmark") return <BenchmarkView/>;
 if (page === "reports") return <ReportsCenter/>;
 return <section><p className="eyebrow">SAR intelligence</p><h1 className="mt-3 text-5xl font-semibold">GeoVision</h1><p className="muted mt-5 max-w-3xl">See Beyond.</p><h2 className="mt-10 text-xl font-semibold">Applications</h2><div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{applications.map(x => <div key={x} className="panel p-5 transition hover:-translate-y-1"><h3 className="font-medium">{x}</h3><p className="mt-2 text-sm text-zinc-500">SAR-informed optical reconstruction for mission-critical analysis.</p></div>)}</div></section>;
}
export function Workspace() { const params = useParams<{ workspace: string }>(); const content = params.workspace === "structure" ? <StructureWorkspace/> : params.workspace === "comparison" ? <ComparisonWorkspace/> : params.workspace === "pix2pix" ? <Pix2PixWorkspace/> : params.workspace === "settings" ? <ProviderSettings/> : <ResearchView page={params.workspace}/>; return <main className="lg:flex"><Sidebar/><motion.section initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .28 }} className="min-w-0 flex-1 px-5 py-10 sm:px-10 lg:px-14">{content}</motion.section></main>; }
