import type { BenchmarkData, Health, ImageAnalysisResult, Pix2PixResult, ProviderId, ProviderModels, ProviderSettings, ProviderTestResult, SARFusionResult } from "@/types/api";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010").replace(/\/$/, "");

export class ApiRequestError extends Error {
  constructor(message: string, public readonly code: string, public readonly status: number) { super(message); this.name = "ApiRequestError"; }
}

function requestError(body: unknown, status: number, fallback: string) {
  const detail = typeof body === "object" && body ? (body as { detail?: unknown; message?: unknown }).detail ?? (body as { message?: unknown }).message : null;
  if (typeof detail === "object" && detail) {
    const error = detail as { code?: unknown; message?: unknown };
    return new ApiRequestError(typeof error.message === "string" ? error.message : fallback, typeof error.code === "string" ? error.code : "UNKNOWN_PROVIDER_ERROR", status);
  }
  return new ApiRequestError(typeof detail === "string" ? detail : fallback, status >= 500 ? "BACKEND_UNAVAILABLE" : "UNKNOWN_PROVIDER_ERROR", status);
}

async function request<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { method: "POST", body: form });
  if (!response.ok) throw requestError(await response.json().catch(() => null), response.status, "Inference request failed.");
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


export type AnalyzeImageRequest = {
  image: Blob;
  analysisType: string;
  modelName: string;
  checkpointName?: string;
  displayMode?: string;
  metrics?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

export async function analyzeImage(input: AnalyzeImageRequest) {
  const form = new FormData();
  const mime = ["image/png", "image/jpeg", "image/webp"].includes(input.image.type) ? input.image.type : "image/png";
  form.append("image", new File([input.image], "geovision-render." + mime.split("/")[1], { type: mime }));
  form.append("analysis_type", input.analysisType);
  form.append("model_name", input.modelName);
  if (input.checkpointName) form.append("checkpoint_name", input.checkpointName);
  if (input.displayMode) form.append("display_mode", input.displayMode);
  if (input.metrics) form.append("metrics_json", JSON.stringify(input.metrics));
  if (input.metadata) form.append("metadata_json", JSON.stringify(input.metadata));
  return request<ImageAnalysisResult>("/api/analysis/image", form);
}

/** @deprecated Use analyzeImage with a stable image Blob. */
export async function runImageAnalysis(imageSource: string, analysisType: string, metadata?: Record<string, unknown>) {
  const image = await fetch(imageSource).then(async response => {
    if (!response.ok) throw new ApiRequestError("The selected image is no longer available for analysis.", "INVALID_IMAGE", response.status);
    return response.blob();
  });
  return analyzeImage({ image, analysisType, modelName: typeof metadata?.model === "string" ? metadata.model : "", metadata });
}


async function settingsRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...options, headers: { "Content-Type": "application/json", ...(options.headers ?? {}) } });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw requestError(body, response.status, "Provider settings request failed.");
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
