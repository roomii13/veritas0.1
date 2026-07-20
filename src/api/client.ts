import type { AnalysisReport, SelectedAsset } from "../types";
import { Platform } from "react-native";

const runtime = globalThis as typeof globalThis & {
  process?: { env?: Record<string, string | undefined> };
};

export const API_URL =
  runtime.process?.env?.EXPO_PUBLIC_API_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8000";

type ApiError = {
  detail?: string | Array<{ msg?: string; loc?: string[]; type?: string }>;
  message?: string;
};

function formatApiError(error: ApiError | null): string {
  if (!error) {
    return "No se pudo completar el analisis.";
  }

  if (typeof error.detail === "string") {
    return error.detail;
  }

  if (Array.isArray(error.detail)) {
    return error.detail
      .map((item) => item.msg ?? item.type ?? "Error de validacion")
      .join(". ");
  }

  return error.message ?? "No se pudo completar el analisis.";
}

async function parseResponse(response: Response): Promise<AnalysisReport> {
  const payload = (await response.json().catch(() => null)) as
    | AnalysisReport
    | ApiError
    | null;

  if (!response.ok) {
    throw new Error(formatApiError(payload as ApiError | null));
  }

  return payload as AnalysisReport;
}

export async function analyzeFile(asset: SelectedAsset): Promise<AnalysisReport> {
  const form = new FormData();

  if (Platform.OS === "web") {
    let blob = asset.file;
    if (!blob) {
      const fileResponse = await fetch(asset.uri);
      blob = await fileResponse.blob();
    }
    form.append("file", blob, asset.name);
  } else {
    form.append("file", {
      uri: asset.uri,
      name: asset.name,
      type: asset.mimeType,
    } as unknown as Blob);
  }

  const response = await fetch(`${API_URL}/analyze-file`, {
    method: "POST",
    body: form,
  });

  return parseResponse(response);
}

export async function analyzeUrl(url: string): Promise<AnalysisReport> {
  const response = await fetch(`${API_URL}/analyze-url`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });

  return parseResponse(response);
}

export async function checkBackend(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
