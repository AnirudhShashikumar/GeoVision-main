import type { BenchmarkData, Health, ImageAnalysisResult, Pix2PixResult, ProviderId, ProviderModels, ProviderSettings, ProviderTestResult, SARFusionResult } from "@/types/api";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010").replace(/\/$/, "");

async function request<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? "Inference request failed.");
  }
  return response.json() as Promise<T>;
}

export const getHealth = async () => {
  const response = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!response.ok) throw new Error("Backend is unavailable.");
  return response.json() as Promise<Health>;
};

export function runPix2Pix(file: File, groundTruth?: File) {
  const form = new FormData(); form.append("file", file);
  if (groundTruth) form.append("ground_truth", groundTruth);
  return request<Pix2PixResult>("/api/pix2pix/infer", form);
}

export function runSARFusionFormer(input: {
  combined?: File; vv?: File; vh?: File; groundTruth?: File; applyColorCorrection: boolean;
}) {
  const form = new FormData();
  if (input.combined) form.append("combined_file", input.combined);
  if (input.vv) form.append("vv_file", input.vv);
  if (input.vh) form.append("vh_file", input.vh);
  if (input.groundTruth) form.append("ground_truth", input.groundTruth);
  form.append("apply_color_correction", String(input.applyColorCorrection));
  return request<SARFusionResult>("/api/sarfusionformer/infer", form);
}

export function runComparison(pix2pix: File, sarfusionformer: File, groundTruth: File) {
  const form = new FormData(); form.append("pix2pix_output", pix2pix); form.append("sarfusionformer_output", sarfusionformer); form.append("ground_truth", groundTruth);
  return request<{ pix2pix: { psnr: number; ssim: number; rgb_l1: number }; sarfusionformer: { psnr: number; ssim: number; rgb_l1: number } }>("/api/compare", form);
}


export async function runImageAnalysis(imageSource: string, analysisType: string, metadata?: Record<string, unknown>) {
  const image = await fetch(imageSource).then(async response => {
    if (!response.ok) throw new Error("The selected image is no longer available for analysis.");
    return response.blob();
  });
  const form = new FormData();
  form.append("image", new File([image], "geovision-render.png", { type: image.type || "image/png" }));
  form.append("analysis_type", analysisType);
  if (metadata) form.append("metadata", JSON.stringify(metadata));
  return request<ImageAnalysisResult>("/api/analysis/image", form);
}


async function settingsRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...options, headers: { "Content-Type": "application/json", ...(options.headers ?? {}) } });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.detail ?? body?.message ?? "Provider settings request failed.");
  return body as T;
}

export function getProviderSettings() {
  return settingsRequest<ProviderSettings>("/api/settings/ai-provider", { method: "GET" });
}

export function getProviderModels() {
  return settingsRequest<ProviderModels>("/api/settings/ai-provider/models?provider=gemini", { method: "GET" });
}

export function saveProviderSettings(provider: ProviderId, apiKey: string, model: string, privacyAcknowledged: boolean) {
  return settingsRequest<ProviderSettings>("/api/settings/ai-provider", { method: "POST", body: JSON.stringify({ provider, api_key: apiKey, model, privacy_acknowledged: privacyAcknowledged }) });
}

export function deleteProviderSettings() {
  return settingsRequest<ProviderSettings>("/api/settings/ai-provider", { method: "DELETE" });
}

export function testProviderSettings() {
  return settingsRequest<ProviderTestResult>("/api/settings/ai-provider/test", { method: "POST", body: "{}" });
}


export async function getBenchmark() {
  const response = await fetch(`${API_URL}/api/benchmark`, { cache: "no-store" });
  if (!response.ok) throw new Error("Benchmark data is unavailable.");
  return response.json() as Promise<BenchmarkData>;
}
