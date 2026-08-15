import type { Metadata } from "next";
import { Workspace } from "@/components/workspace";

const routeTitles: Record<string, string> = {
  pix2pix: "Pix2Pix",
  structure: "SARFusionFormer",
  comparison: "Model Comparison",
  architecture: "Architecture",
  benchmark: "Benchmark",
  reports: "Reports",
  settings: "AI Providers",
};

export async function generateMetadata({ params }: { params: Promise<{ workspace: string }> }): Promise<Metadata> {
  const { workspace } = await params;
  const title = routeTitles[workspace];
  return title ? { title } : { title: { absolute: "GeoVision" } };
}

export default function WorkspacePage() { return <Workspace />; }
