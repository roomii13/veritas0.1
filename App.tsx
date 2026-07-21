import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { API_URL, analyzeFile, analyzeUrl, checkBackend } from "./src/api/client";
import type { AnalysisReport, InputMode, RawDetectorResult, SelectedAsset } from "./src/types";

const acceptedMime = ["image/*", "audio/*", "video/*"];

function inferMimeType(name?: string, fallback?: string | null): string {
  if (fallback) {
    return fallback;
  }

  const ext = name?.split(".").pop()?.toLowerCase();
  if (!ext) {
    return "application/octet-stream";
  }

  if (["jpg", "jpeg"].includes(ext)) return "image/jpeg";
  if (ext === "png") return "image/png";
  if (ext === "webp") return "image/webp";
  if (ext === "mp4") return "video/mp4";
  if (ext === "mov") return "video/quicktime";
  if (ext === "avi") return "video/x-msvideo";
  if (ext === "wav") return "audio/wav";
  if (ext === "mp3") return "audio/mpeg";
  if (ext === "m4a") return "audio/mp4";

  return "application/octet-stream";
}

function formatPercent(value: number): string {
  return `${Math.round(value)}%`;
}

function riskColor(report?: AnalysisReport): string {
  if (!report) return "#0F766E";
  if (report.ai_percentage >= 70) return "#C62828";
  if (isPreventiveReport(report)) return "#B26A00";
  return "#16805F";
}

function trustStatus(report?: AnalysisReport): string {
  if (!report) return "LISTO";
  if (report.trust_status) return report.trust_status;
  if (report.ai_percentage >= 70) return "NO CONFIABLE";
  if (isPreventiveReport(report)) return "PREVENCION";
  return "CONFIABLE";
}

function trustIcon(report?: AnalysisReport): keyof typeof MaterialCommunityIcons.glyphMap {
  const status = trustStatus(report);
  if (status === "NO CONFIABLE") return "shield-alert";
  if (status === "PREVENCION" || status === "REVISAR") return "shield-half-full";
  return "shield-check";
}

function isPreventiveReport(report: AnalysisReport): boolean {
  const hasSuspiciousModality = report.per_modality_summary.some(
    (item) =>
      item.verdict === "SOSPECHOSO" ||
      item.trust_status === "PREVENCION" ||
      (typeof item.ai_probability === "number" && item.ai_probability >= 30),
  );
  const usedFallback = report.raw_results?.some(
    (item) => item.source === "local_heuristic_fallback",
  );

  return report.warning || report.prevention === true || report.ai_percentage >= 30 || hasSuspiciousModality || Boolean(usedFallback);
}

function modalityLabel(modality: string): string {
  const labels: Record<string, string> = {
    image: "Imagen",
    audio: "Audio",
    video: "Video",
    link: "Enlace",
  };

  return labels[modality] ?? modality;
}

export default function App() {
  const [mode, setMode] = useState<InputMode>("file");
  const [asset, setAsset] = useState<SelectedAsset | null>(null);
  const [url, setUrl] = useState("");
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkBackend().then(setBackendOnline);
  }, []);

  const canAnalyze = useMemo(() => {
    if (isLoading) return false;
    if (mode === "file") return Boolean(asset);
    return url.trim().length > 8;
  }, [asset, isLoading, mode, url]);

  async function pickDocument() {
    setError(null);
    setReport(null);

    const result = await DocumentPicker.getDocumentAsync({
      type: acceptedMime,
      copyToCacheDirectory: true,
      multiple: false,
    });

    if (result.canceled || !result.assets?.[0]) {
      return;
    }

    const picked = result.assets[0];
    const webFile = "file" in picked ? (picked.file as Blob | undefined) : undefined;
    setAsset({
      uri: picked.uri,
      name: picked.name ?? `veritas-${Date.now()}`,
      mimeType: inferMimeType(picked.name, picked.mimeType ?? webFile?.type),
      size: picked.size,
      file: webFile,
    });
  }

  async function pickFromGallery() {
    setError(null);
    setReport(null);

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images", "videos"] as ImagePicker.MediaType[],
      quality: 1,
    });

    if (result.canceled || !result.assets?.[0]) {
      return;
    }

    const picked = result.assets[0];
    const webFile = "file" in picked ? (picked.file as Blob | undefined) : undefined;
    const name = picked.fileName ?? `veritas-media-${Date.now()}`;
    setAsset({
      uri: picked.uri,
      name,
      mimeType: inferMimeType(name, picked.mimeType ?? webFile?.type),
      size: picked.fileSize,
      file: webFile,
    });
  }

  async function captureWithCamera() {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert("Permiso requerido", "Veritas necesita acceso a la camara.");
      return;
    }

    setError(null);
    setReport(null);

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ["images", "videos"] as ImagePicker.MediaType[],
      quality: 0.9,
    });

    if (result.canceled || !result.assets?.[0]) {
      return;
    }

    const picked = result.assets[0];
    const webFile = "file" in picked ? (picked.file as Blob | undefined) : undefined;
    const name = picked.fileName ?? `captura-veritas-${Date.now()}`;
    setAsset({
      uri: picked.uri,
      name,
      mimeType: inferMimeType(name, picked.mimeType ?? webFile?.type),
      size: picked.fileSize,
      file: webFile,
    });
  }

  async function runAnalysis() {
    if (!canAnalyze) return;

    setIsLoading(true);
    setError(null);
    setReport(null);

    try {
      const nextReport =
        mode === "file" && asset
          ? await analyzeFile(asset)
          : await analyzeUrl(url.trim());
      setReport(nextReport);
      setBackendOnline(true);
    } catch (caught) {
      setBackendOnline(false);
      setError(
        caught instanceof Error
          ? caught.message
          : "No se pudo contactar el backend de Veritas.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  const percentage = report?.ai_percentage ?? 0;
  const barWidth = `${Math.min(Math.max(percentage, 3), 100)}%`;
  const color = riskColor(report ?? undefined);
  const status = trustStatus(report ?? undefined);
  const primaryRaw: RawDetectorResult | undefined = report?.raw_results?.[0];
  const reasons = primaryRaw?.reasons?.slice(0, 4) ?? [];
  const modelDetails = [
    primaryRaw?.device ? `device ${primaryRaw.device}` : null,
    primaryRaw?.source ? String(primaryRaw.source).replaceAll("_", " ") : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : undefined}
        style={styles.keyboard}
      >
        <ScrollView contentContainerStyle={styles.container}>
          <View style={styles.header}>
            <View style={styles.brandMark}>
              <MaterialCommunityIcons name="shield-search" size={31} color="#F7FAF6" />
            </View>
            <View style={styles.brandText}>
              <Text style={styles.appName}>Veritas</Text>
              <Text style={styles.subtitle}>Detector antifraude de IA</Text>
            </View>
          </View>

          <View style={styles.statusRow}>
            <MaterialCommunityIcons
              name={backendOnline ? "server-security" : "server-off"}
              size={18}
              color={backendOnline ? "#24735C" : "#9A3412"}
            />
            <Text style={styles.statusText}>
              Backend: {backendOnline ? "conectado" : "configurar"} · {API_URL}
            </Text>
          </View>

          <View style={styles.segmented}>
            <Pressable
              accessibilityRole="button"
              onPress={() => setMode("file")}
              style={[styles.segment, mode === "file" && styles.segmentActive]}
            >
              <MaterialCommunityIcons
                name="file-upload-outline"
                size={19}
                color={mode === "file" ? "#F7FAF6" : "#263C3A"}
              />
              <Text style={[styles.segmentText, mode === "file" && styles.segmentTextActive]}>
                Archivo
              </Text>
            </Pressable>
            <Pressable
              accessibilityRole="button"
              onPress={() => setMode("link")}
              style={[styles.segment, mode === "link" && styles.segmentActive]}
            >
              <MaterialCommunityIcons
                name="link-variant"
                size={19}
                color={mode === "link" ? "#F7FAF6" : "#263C3A"}
              />
              <Text style={[styles.segmentText, mode === "link" && styles.segmentTextActive]}>
                Enlace
              </Text>
            </Pressable>
          </View>

          {mode === "file" ? (
            <View style={styles.panel}>
              <Text style={styles.panelTitle}>Subir evidencia</Text>
              <Text style={styles.panelCopy}>
                Acepta imagenes, audios y videos desde el movil.
              </Text>

              <View style={styles.actionsGrid}>
                <Pressable style={styles.actionButton} onPress={pickDocument}>
                  <MaterialCommunityIcons name="folder-open-outline" size={22} color="#263C3A" />
                  <Text style={styles.actionText}>Archivo</Text>
                </Pressable>
                <Pressable style={styles.actionButton} onPress={pickFromGallery}>
                  <MaterialCommunityIcons name="image-multiple-outline" size={22} color="#263C3A" />
                  <Text style={styles.actionText}>Galeria</Text>
                </Pressable>
                <Pressable style={styles.actionButton} onPress={captureWithCamera}>
                  <MaterialCommunityIcons name="camera-outline" size={22} color="#263C3A" />
                  <Text style={styles.actionText}>Camara</Text>
                </Pressable>
              </View>

              {asset ? (
                <View style={styles.selectedBox}>
                  <MaterialCommunityIcons name="paperclip" size={20} color="#2F6B63" />
                  <View style={styles.selectedTextWrap}>
                    <Text style={styles.selectedName} numberOfLines={1}>
                      {asset.name}
                    </Text>
                    <Text style={styles.selectedMeta}>{asset.mimeType}</Text>
                  </View>
                </View>
              ) : null}
            </View>
          ) : (
            <View style={styles.panel}>
              <Text style={styles.panelTitle}>Analizar enlace</Text>
              <Text style={styles.panelCopy}>
                Pega una URL directa a imagen, audio o video. Si es una pagina, se evalua como
                enlace sospechoso.
              </Text>
              <TextInput
                value={url}
                onChangeText={(text) => {
                  setUrl(text);
                  setError(null);
                  setReport(null);
                }}
                placeholder="https://ejemplo.com/video.mp4"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                style={styles.input}
                placeholderTextColor="#7B8783"
              />
            </View>
          )}

          <Pressable
            accessibilityRole="button"
            disabled={!canAnalyze}
            onPress={runAnalysis}
            style={[styles.primaryButton, !canAnalyze && styles.primaryButtonDisabled]}
          >
            {isLoading ? (
              <ActivityIndicator color="#F7FAF6" />
            ) : (
              <MaterialCommunityIcons name="radar" size={22} color="#F7FAF6" />
            )}
            <Text style={styles.primaryButtonText}>
              {isLoading ? "Analizando..." : "Analizar con Veritas"}
            </Text>
          </Pressable>

          {error ? (
            <View style={styles.errorBox}>
              <MaterialCommunityIcons name="alert-circle-outline" size={21} color="#B42318" />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          {report ? (
            <View style={styles.resultPanel}>
              <View style={styles.resultTop}>
                <View style={[styles.resultIconWrap, { backgroundColor: color }]}>
                  <MaterialCommunityIcons name={trustIcon(report)} size={31} color="#FFFFFF" />
                </View>
                <View style={styles.resultTitleWrap}>
                  <Text style={styles.resultLabel}>Estado Veritas</Text>
                  <Text style={[styles.statusTitle, { color }]}>{status}</Text>
                  <Text style={styles.statusMeta}>{report.overall_verdict}</Text>
                </View>
                <Text style={[styles.percentText, { color }]}>{formatPercent(percentage)}</Text>
              </View>

              <View style={styles.progressTrack}>
                <View style={[styles.progressFill, { width: (barWidth as any), backgroundColor: color }]} />
              </View>

              <View style={styles.metricStrip}>
                <View style={styles.metricCell}>
                  <Text style={styles.metricLabel}>IA estimada</Text>
                  <Text style={styles.metricValue}>{formatPercent(percentage)}</Text>
                </View>
                <View style={styles.metricDivider} />
                <View style={styles.metricCell}>
                  <Text style={styles.metricLabel}>Modalidades</Text>
                  <Text style={styles.metricValue}>{report.modalities_analyzed}</Text>
                </View>
                <View style={styles.metricDivider} />
                <View style={styles.metricCell}>
                  <Text style={styles.metricLabel}>Decision</Text>
                  <Text style={styles.metricValueSmall}>{status}</Text>
                </View>
              </View>

              {isPreventiveReport(report) ? (
                <View style={[styles.warningBox, status === "PREVENCION" && styles.preventionBox]}>
                  <MaterialCommunityIcons
                    name={status === "NO CONFIABLE" ? "alert-octagon" : "alert"}
                    size={24}
                    color={status === "NO CONFIABLE" ? "#B42318" : "#B26A00"}
                  />
                  <Text style={styles.warningText}>
                    {status === "NO CONFIABLE"
                      ? "NO CONFIAR sin una segunda verificacion."
                      : "PREVENCION: casi 40% o senales moderadas no deben mostrarse como confiables."}
                  </Text>
                </View>
              ) : null}

              <Text style={styles.recommendation}>{report.recommendation}</Text>

              {reasons.length ? (
                <View style={styles.reasonsBlock}>
                  <Text style={styles.sectionLabel}>Senales detectadas</Text>
                  {reasons.map((reason) => (
                    <View key={reason} style={styles.reasonRow}>
                      <MaterialCommunityIcons name="circle-medium" size={18} color="#3B5BDB" />
                      <Text style={styles.reasonText}>{reason}</Text>
                    </View>
                  ))}
                </View>
              ) : null}

              {modelDetails ? <Text style={styles.engineText}>{modelDetails}</Text> : null}

              <View style={styles.modalityList}>
                {report.per_modality_summary.map((item) => (
                  <View key={`${item.modality}-${item.verdict}`} style={styles.modalityItem}>
                    <MaterialCommunityIcons
                      name={item.verdict === "SOSPECHOSO" ? "alert-circle" : "check-circle"}
                      size={22}
                      color={item.verdict === "SOSPECHOSO" ? "#B26A00" : "#24735C"}
                    />
                    <View style={styles.modalityTextWrap}>
                      <Text style={styles.modalityTitle}>{modalityLabel(item.modality)}</Text>
                      <Text style={styles.modalityMeta}>
                        {item.verdict} · confianza {Math.round(item.confidence * 100)}%
                        {typeof item.ai_probability === "number"
                          ? ` · IA ${Math.round(item.ai_probability)}%`
                          : ""}
                        {typeof item.threshold === "number"
                          ? ` · umbral ${Math.round(item.threshold)}%`
                          : ""}
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            </View>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#F4F7FB",
  },
  keyboard: {
    flex: 1,
  },
  container: {
    paddingHorizontal: 20,
    paddingBottom: 32,
    paddingTop: 18,
    gap: 18,
  },
  header: {
    alignItems: "center",
    flexDirection: "row",
    gap: 14,
  },
  brandMark: {
    alignItems: "center",
    backgroundColor: "#1D4ED8",
    borderRadius: 8,
    height: 54,
    justifyContent: "center",
    width: 54,
  },
  brandText: {
    flex: 1,
  },
  appName: {
    color: "#102A43",
    fontSize: 33,
    fontWeight: "800",
    letterSpacing: 0,
  },
  subtitle: {
    color: "#52677A",
    fontSize: 15,
    marginTop: 2,
  },
  statusRow: {
    alignItems: "center",
    flexDirection: "row",
    gap: 8,
    minHeight: 26,
  },
  statusText: {
    color: "#52677A",
    flex: 1,
    fontSize: 12,
  },
  segmented: {
    backgroundColor: "#E6EEF8",
    borderRadius: 8,
    flexDirection: "row",
    padding: 4,
  },
  segment: {
    alignItems: "center",
    borderRadius: 6,
    flex: 1,
    flexDirection: "row",
    gap: 8,
    justifyContent: "center",
    minHeight: 44,
  },
  segmentActive: {
    backgroundColor: "#0F766E",
  },
  segmentText: {
    color: "#233B53",
    fontSize: 15,
    fontWeight: "700",
  },
  segmentTextActive: {
    color: "#F7FAF6",
  },
  panel: {
    backgroundColor: "#FFFFFF",
    borderColor: "#D8E2ED",
    borderRadius: 8,
    borderWidth: 1,
    padding: 16,
    gap: 13,
  },
  panelTitle: {
    color: "#102A43",
    fontSize: 20,
    fontWeight: "800",
  },
  panelCopy: {
    color: "#52677A",
    fontSize: 14,
    lineHeight: 20,
  },
  actionsGrid: {
    flexDirection: "row",
    gap: 10,
  },
  actionButton: {
    alignItems: "center",
    backgroundColor: "#EEF6FF",
    borderColor: "#CCE1FF",
    borderRadius: 8,
    borderWidth: 1,
    flex: 1,
    gap: 7,
    justifyContent: "center",
    minHeight: 76,
    paddingHorizontal: 6,
  },
  actionText: {
    color: "#233B53",
    fontSize: 13,
    fontWeight: "700",
  },
  selectedBox: {
    alignItems: "center",
    backgroundColor: "#F5FAFF",
    borderColor: "#CFE1F5",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 10,
    minHeight: 58,
    paddingHorizontal: 12,
  },
  selectedTextWrap: {
    flex: 1,
  },
  selectedName: {
    color: "#102A43",
    fontSize: 14,
    fontWeight: "800",
  },
  selectedMeta: {
    color: "#52677A",
    fontSize: 12,
    marginTop: 3,
  },
  input: {
    backgroundColor: "#F9FBFF",
    borderColor: "#CFE1F5",
    borderRadius: 8,
    borderWidth: 1,
    color: "#102A43",
    fontSize: 15,
    minHeight: 52,
    paddingHorizontal: 14,
  },
  primaryButton: {
    alignItems: "center",
    backgroundColor: "#0F766E",
    borderRadius: 8,
    flexDirection: "row",
    gap: 10,
    justifyContent: "center",
    minHeight: 54,
  },
  primaryButtonDisabled: {
    backgroundColor: "#91A4B7",
  },
  primaryButtonText: {
    color: "#F7FAF6",
    fontSize: 16,
    fontWeight: "800",
  },
  errorBox: {
    alignItems: "flex-start",
    backgroundColor: "#FFF1F0",
    borderColor: "#F3B4AF",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 9,
    padding: 12,
  },
  errorText: {
    color: "#8E1F15",
    flex: 1,
    fontSize: 13,
    lineHeight: 19,
  },
  resultPanel: {
    backgroundColor: "#FFFFFF",
    borderColor: "#D8E2ED",
    borderRadius: 8,
    borderWidth: 1,
    gap: 15,
    padding: 16,
  },
  resultTop: {
    alignItems: "center",
    flexDirection: "row",
    gap: 12,
    justifyContent: "space-between",
  },
  resultLabel: {
    color: "#52677A",
    fontSize: 13,
    fontWeight: "700",
  },
  resultIconWrap: {
    alignItems: "center",
    borderRadius: 8,
    height: 56,
    justifyContent: "center",
    width: 56,
  },
  resultTitleWrap: {
    flex: 1,
    gap: 2,
  },
  statusTitle: {
    fontSize: 22,
    fontWeight: "900",
    letterSpacing: 0,
  },
  statusMeta: {
    color: "#52677A",
    fontSize: 12,
    fontWeight: "800",
  },
  percentText: {
    fontSize: 36,
    fontWeight: "900",
    letterSpacing: 0,
  },
  verdictBadge: {
    borderRadius: 8,
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  verdictText: {
    fontSize: 12,
    fontWeight: "900",
  },
  progressTrack: {
    backgroundColor: "#E5EAF1",
    borderRadius: 8,
    height: 12,
    overflow: "hidden",
  },
  progressFill: {
    borderRadius: 8,
    height: "100%",
  },
  metricStrip: {
    alignItems: "center",
    backgroundColor: "#F6F9FD",
    borderColor: "#E1E8F0",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    minHeight: 72,
  },
  metricCell: {
    flex: 1,
    gap: 4,
    paddingHorizontal: 10,
  },
  metricDivider: {
    backgroundColor: "#D8E2ED",
    height: 40,
    width: 1,
  },
  metricLabel: {
    color: "#68798B",
    fontSize: 11,
    fontWeight: "800",
  },
  metricValue: {
    color: "#102A43",
    fontSize: 20,
    fontWeight: "900",
  },
  metricValueSmall: {
    color: "#102A43",
    fontSize: 13,
    fontWeight: "900",
  },
  warningBox: {
    alignItems: "center",
    backgroundColor: "#FFF1F0",
    borderColor: "#F3B4AF",
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: "row",
    gap: 10,
    padding: 12,
  },
  preventionBox: {
    backgroundColor: "#FFF7E6",
    borderColor: "#F8D49A",
  },
  warningText: {
    color: "#8E1F15",
    flex: 1,
    fontSize: 14,
    fontWeight: "900",
    lineHeight: 20,
  },
  recommendation: {
    color: "#233B53",
    fontSize: 14,
    lineHeight: 21,
  },
  reasonsBlock: {
    gap: 7,
  },
  sectionLabel: {
    color: "#102A43",
    fontSize: 13,
    fontWeight: "900",
  },
  reasonRow: {
    alignItems: "flex-start",
    flexDirection: "row",
    gap: 4,
  },
  reasonText: {
    color: "#52677A",
    flex: 1,
    fontSize: 13,
    lineHeight: 19,
  },
  engineText: {
    color: "#68798B",
    fontSize: 12,
    lineHeight: 17,
  },
  modalityList: {
    gap: 10,
  },
  modalityItem: {
    alignItems: "center",
    backgroundColor: "#F7FAFC",
    borderRadius: 8,
    flexDirection: "row",
    gap: 10,
    minHeight: 58,
    paddingHorizontal: 12,
  },
  modalityTextWrap: {
    flex: 1,
  },
  modalityTitle: {
    color: "#102A43",
    fontSize: 14,
    fontWeight: "800",
  },
  modalityMeta: {
    color: "#52677A",
    fontSize: 12,
    marginTop: 3,
  },
});
