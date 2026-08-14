export type Metrics = { psnr: number | null; ssim: number; rgb_l1: number };
export type ModelStatus = { available: boolean; checkpoint: string; error: string | null };
export type Health = { status: string; device: string; models: Record<string, ModelStatus> };

export type Pix2PixResult = {
  input_preview: string; output: string; metrics: Metrics | null;
  inference_time_ms: number; checkpoint: string;
};

export type Diagnostics = {
  raw_rgb_min: number; raw_rgb_max: number; raw_rgb_mean: number; raw_rgb_std: number;
  display_rgb_min: number; display_rgb_max: number; display_rgb_mean: number; display_rgb_std: number;
  stretch_low: number; stretch_high: number; prediction_shape?: number[]; prediction_dtype?: string;
  lab_finite?: boolean; rgb_finite?: boolean; raw_prediction_preserved?: boolean;
  [key: string]: unknown;
};

export type SARFusionResult = {
  raw_output: string; display_output: string;
  corrected_raw_output: string | null; corrected_display_output: string | null;
  vv_preview: string; vh_preview: string; sar_preview: string;
  metrics_raw: Metrics | null; metrics_corrected: Metrics | null;
  inference_time_ms: number; detected_shape: number[]; channel_layout: string;
  checkpoint: string; color_checkpoint: string | null; warning: string | null;
  diagnostics: Diagnostics; corrected_diagnostics: Diagnostics | null;
};


export type ImageAnalysisReport = {
  executive_summary: string;
  terrain: string[];
  structural_and_human_features: string[];
  vegetation_and_water: string[];
  image_quality: string[];
  possible_artifacts: string[];
  notes: string[];
  limitations: string[];
  recommended_actions: string[];
  confidence: "low" | "medium" | "high";
  disclaimer: string;
};

export type ImageAnalysisResult = {
  report: ImageAnalysisReport;
  provider: string;
  model: string;
  cached: boolean;
};


export type ProviderId = "openai" | "gemini" | "anthropic";
export type ProviderSettings = {
  configured: boolean;
  provider: string | null;
  provider_id: ProviderId | null;
  model: string | null;
  masked_key: string | null;
  source: "environment" | "keychain" | null;
  connection_status: "connected" | "configured" | "invalid" | "not_configured";
  supported: boolean;
  managed_by_environment: boolean;
  message?: string;
};

export type ProviderTestResult = {
  success: boolean;
  provider?: string;
  model?: string;
  message: string;
};
