export type InputMode = "file" | "link";

export type SelectedAsset = {
  uri: string;
  name: string;
  mimeType: string;
  size?: number;
  file?: Blob;
};

export type ModalitySummary = {
  modality: string;
  verdict: string;
  confidence: number;
  ai_probability?: number;
  threshold?: number;
  trust_status?: string;
};

export type RawDetectorResult = {
  modality?: string;
  label?: string;
  confidence?: number;
  ai_probability?: number;
  reasons?: string[];
  source?: string;
  model?: string;
  device?: string;
  threshold?: number;
  model_probability?: number;
  manipulation_probability?: number;
  duration_sec?: number;
  frames_analyzed?: number;
  [key: string]: unknown;
};

export type AnalysisReport = {
  overall_verdict: "ALTO RIESGO" | "RIESGO MEDIO" | "BAJO RIESGO";
  risk_score: number;
  ai_percentage: number;
  modalities_analyzed: number;
  recommendation: string;
  trust_status?: "NO CONFIABLE" | "REVISAR" | "CONFIABLE";
  warning: boolean;
  per_modality_summary: ModalitySummary[];
  raw_results?: RawDetectorResult[];
};
