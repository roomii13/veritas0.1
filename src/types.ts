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
};

export type AnalysisReport = {
  overall_verdict: "ALTO RIESGO" | "RIESGO MEDIO" | "BAJO RIESGO";
  risk_score: number;
  ai_percentage: number;
  modalities_analyzed: number;
  recommendation: string;
  warning: boolean;
  per_modality_summary: ModalitySummary[];
  raw_results?: unknown[];
};
