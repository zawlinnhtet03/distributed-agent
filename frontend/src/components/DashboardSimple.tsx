"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

type AgentStatus = "idle" | "active" | "busy";

// Conversation history entry
type ConversationEntry = {
  id: string;
  sessionId: string;
  query: string;
  response: string;
  timestamp: Date;
  agentActivity: string[];
  trace?: TraceEvent[];
};

type SessionInfo = {
  id: string;
  createdAt: number;
};

type TraceEvent = {
  id: string;
  type: "agent_message" | "tool_call" | "tool_result" | "final" | "error";
  author: string;
  text?: string;
  toolName?: string;
  ts: number;
};

declare global {
  interface Window {
    Plotly?: any;
  }
}

type Artifact =
  | { kind: "table"; title?: string; columns: string[]; rows: Array<Record<string, string>> }
  | { kind: "chart_plotly"; title?: string; spec: any };

type ArtifactDebugState = {
  markersSeen: number;
  parsedOk: number;
  parseFailed: number;
};

function isLikelyMarkdown(text: string) {
  const t = text.trim();
  if (!t) return false;
  if (t.includes("```")) return true;
  if (t.includes("**")) return true;
  if (/^#{1,6}\s/m.test(t)) return true;
  if (/^\s*[-*+]\s+/m.test(t)) return true;
  if (/^\s*[•‣◦]\s+/m.test(t)) return true;
  if (/^\s*\d+\.\s+/m.test(t)) return true;
  if (/\*\*[^*]+\*\*/.test(t)) return true;
  if (/\[[^\]]+\]\([^\)]+\)/.test(t)) return true;
  if (/^\s*\|(.+\|)+\s*$/m.test(t)) return true;
  if (/^\s*[A-Za-z][A-Za-z0-9 _\-]{2,40}:\s*$/m.test(t)) return true;
  return false;
}

function normalizeForMarkdown(text: string) {
  const s = (text ?? "").replace(/\r\n/g, "\n");

  // Convert common bullet glyphs to markdown bullets.
  let out = s.replace(/^\s*[•‣◦]\s+/gm, "- ");

  // Turn standalone "Heading:" lines into bold section headers with spacing.
  // Example: "Key Benefits:" -> "\n\n**Key Benefits**\n"
  out = out.replace(/^(\s*)([A-Za-z][A-Za-z0-9 _\-]{2,60}):\s*$/gm, (_m, ws, title) => {
    const t = String(title).trim();
    return `${ws}\n${ws}**${t}**\n`;
  });

  // Ensure a blank line before ordered list items when stuck to previous text.
  out = out.replace(/([^\n])\n(\s*\d+\.\s+)/g, "$1\n\n$2");

  // De-indent: if the model output has a common left margin, markdown will treat it as a code block.
  // Strip the minimum indentation across non-empty lines.
  {
    const lines = out.split("\n");
    let minIndent = Infinity;
    for (const line of lines) {
      if (!line.trim()) continue;
      const m = line.match(/^([ \t]+)/);
      if (!m) {
        minIndent = 0;
        break;
      }
      minIndent = Math.min(minIndent, m[1].length);
    }
    if (Number.isFinite(minIndent) && minIndent > 0) {
      out = lines
        .map(l => {
          if (!l.trim()) return "";
          return l.slice(minIndent);
        })
        .join("\n");
    }
  }

  return out.trim();
}

function stripPlanSection(text: string) {
  const s = (text ?? "").replace(/\r\n/g, "\n");
  // Remove a leading PLAN block (commonly emitted as **PLAN** or PLAN:)
  // Heuristic: if the response starts with a PLAN heading, drop content until the first blank line.
  if (!/^\s*(\*\*PLAN\*\*|PLAN\s*:)/i.test(s)) return s;
  const idx = s.search(/\n\s*\n/);
  if (idx === -1) return "";
  return s.slice(idx).trim();
}

function cn(...v: Array<string | false | null | undefined>) {
  return v.filter(Boolean).join(" ");
}

const AGENTS = [
  { key: "Planner", name: "Planner", icon: "◈", color: "#38bdf8", description: "Routes & plans tasks" },
  { key: "Aggregator", name: "Aggregator", icon: "◉", color: "#ec4899", description: "Synthesizes results" },
  { key: "RAG", name: "RAG", icon: "◆", color: "#fbbf24", description: "Knowledge base" },
  { key: "Scraper", name: "Scraper", icon: "◎", color: "#10b981", description: "Web research" },
  { key: "Video", name: "Video", icon: "◐", color: "#a855f7", description: "Visual analysis" },
  { key: "Data", name: "Data", icon: "◑", color: "#22c55e", description: "EDA, cleaning, charts" },
  { key: "Guardian", name: "Guardian", icon: "◇", color: "#f43f5e", description: "Validates & verifies" },
];

export default function DashboardSimple() {
  const [query, setQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [finalAnswer, setFinalAnswer] = useState("");
  const [guardianAnswer, setGuardianAnswer] = useState("");
  const [agentStatuses, setAgentStatuses] = useState<Record<string, AgentStatus>>({});
  const [sessionId, setSessionId] = useState("");
  const [sessionReady, setSessionReady] = useState(false);
  const [freshSessionPerRun, setFreshSessionPerRun] = useState(false);
  const [appName] = useState("research_pipeline_agent");
  const [userId] = useState("ui_user");
  const [error, setError] = useState("");
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [artifactDebug, setArtifactDebug] = useState<ArtifactDebugState>({ markersSeen: 0, parsedOk: 0, parseFailed: 0 });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadKind, setUploadKind] = useState<"auto" | "video" | "data">("auto");
  const [uploadedFile, setUploadedFile] = useState<{ kind: string; filename: string; path: string } | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [showSessions, setShowSessions] = useState(false);

  const enableLocalHistory = process.env.NEXT_PUBLIC_PERSIST_SESSIONS === "true";

  // Conversation history stored per-session for multi-turn memory
  const [historyBySession, setHistoryBySession] = useState<Record<string, ConversationEntry[]>>({});
  const [expandedHistoryEntryId, setExpandedHistoryEntryId] = useState<string | null>(null);
  const pendingHistoryRef = useRef<ConversationEntry | null>(null);
  
  const agentBufferRef = useRef<Record<string, string>>({});
  const activeAgentsRef = useRef<Set<string>>(new Set());
  const finalAnswerRef = useRef<string>("");
  const guardianAnswerRef = useRef<string>("");

  const SHARED_STATE_KEY = `adk_shared_state:${appName}:${userId}`;

  const STORAGE_KEYS = {
    sessions: `adk_ui_sessions:${appName}:${userId}`,
    activeSession: `adk_ui_active_session:${appName}:${userId}`,
    historyBySession: `adk_ui_history_by_session:${appName}:${userId}`,
  };

  const ensurePlotly = async () => {
    if (typeof window === "undefined") return;
    if (window.Plotly) return;

    await new Promise<void>((resolve, reject) => {
      const existing = document.querySelector('script[data-plotly-loader="1"]') as HTMLScriptElement | null;
      if (existing) {
        existing.addEventListener("load", () => resolve());
        existing.addEventListener("error", () => reject(new Error("Failed to load Plotly")));
        return;
      }

      const script = document.createElement("script");
      script.src = "https://cdn.plot.ly/plotly-2.30.0.min.js";
      script.async = true;
      script.dataset.plotlyLoader = "1";
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("Failed to load Plotly"));
      document.head.appendChild(script);
    });
  };

  const renderPlotlyArtifacts = async () => {
    if (typeof window === "undefined") return;
    const nodes = Array.from(document.querySelectorAll("[data-plotly='1']")) as HTMLDivElement[];
    if (nodes.length === 0) return;
    await ensurePlotly();
    if (!window.Plotly) {
      console.error("[Plotly] Failed to load Plotly.js library");
      return;
    }

    for (const el of nodes) {
      const specRaw = el.getAttribute("data-plotly-spec");
      if (!specRaw) continue;
      try {
        const spec = JSON.parse(specRaw);
        const data = spec?.data ?? [];
        const layout = {
          ...(spec?.layout ?? {}),
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "rgba(255,255,255,0.85)" },
        };
        const config = { responsive: true, displaylogo: false };
        console.log("[Plotly] Rendering chart with", data.length, "traces");
        await window.Plotly.react(el, data, layout, config);
      } catch (err) {
        console.error("[Plotly] Render failed:", err);
      }
    }
  };

  useEffect(() => {
    void renderPlotlyArtifacts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [artifacts]);

  const extractArtifacts = (
    rawText: string,
  ): { cleanedText: string; found: Artifact[]; markersSeen: number; parseFailed: number } => {
    if (!rawText || !rawText.includes("ADK_ARTIFACT:")) {
      return { cleanedText: rawText, found: [], markersSeen: 0, parseFailed: 0 };
    }

    const lines = rawText.split("\n");
    const found: Artifact[] = [];
    const kept: string[] = [];
    let markersSeen = 0;
    let parseFailed = 0;
    for (const line of lines) {
      const idx = line.indexOf("ADK_ARTIFACT:");
      if (idx === -1) {
        kept.push(line);
        continue;
      }
      markersSeen += 1;
      const jsonPart = line.slice(idx + "ADK_ARTIFACT:".length).trim();
      try {
        const parsed = JSON.parse(jsonPart) as Artifact;
        if (parsed && (parsed as any).kind) {
          found.push(parsed);
        } else {
          parseFailed += 1;
        }
      } catch {
        // keep if parse fails
        parseFailed += 1;
        kept.push(line);
      }
    }
    return { cleanedText: kept.join("\n").trim(), found, markersSeen, parseFailed };
  };

  // Shared state for monitoring - saved to localStorage so Advanced View can access it
  const updateSharedState = (state: {
    agentStatuses?: Record<string, AgentStatus>;
    traceEvents?: TraceEvent[];
    isProcessing?: boolean;
    guardianAnswer?: string;
    sessionId?: string;
    artifacts?: Artifact[];
  }) => {
    try {
      const existing = JSON.parse(localStorage.getItem(SHARED_STATE_KEY) || "{}");
      localStorage.setItem(SHARED_STATE_KEY, JSON.stringify({
        ...existing,
        ...state,
        lastUpdated: Date.now(),
      }));
    } catch {}
  };

  const clearSharedState = () => {
    try {
      localStorage.removeItem(SHARED_STATE_KEY);
    } catch {}
  };

  const loadPersistedState = () => {
    if (!enableLocalHistory) return;
    try {
      const rawSessions = localStorage.getItem(STORAGE_KEYS.sessions);
      const rawActive = localStorage.getItem(STORAGE_KEYS.activeSession);
      const rawHist = localStorage.getItem(STORAGE_KEYS.historyBySession);

      const parsedSessions = rawSessions ? (JSON.parse(rawSessions) as SessionInfo[]) : [];
      const parsedHist = rawHist ? (JSON.parse(rawHist) as Record<string, unknown>) : {};

      if (Array.isArray(parsedSessions)) setSessions(parsedSessions);
      if (rawActive) setSessionId(String(rawActive));
      if (parsedHist && typeof parsedHist === "object") {
        const normalized: Record<string, ConversationEntry[]> = {};
        for (const [sid, entries] of Object.entries(parsedHist)) {
          if (!Array.isArray(entries)) {
            normalized[sid] = [];
            continue;
          }
          normalized[sid] = (entries as unknown[]).map(e => {
            const obj = (e ?? {}) as Record<string, unknown>;
            const ts = obj.timestamp;
            return {
              id: String(obj.id ?? ""),
              sessionId: String(obj.sessionId ?? sid),
              query: String(obj.query ?? ""),
              response: String(obj.response ?? ""),
              timestamp: new Date(typeof ts === "string" || typeof ts === "number" ? ts : Date.now()),
              agentActivity: Array.isArray(obj.agentActivity) ? (obj.agentActivity as unknown[]).map(String) : [],
              trace: Array.isArray(obj.trace) ? (obj.trace as TraceEvent[]) : undefined,
            } as ConversationEntry;
          });
        }
        setHistoryBySession(normalized);
      }
    } catch {
      // ignore
    }
  };

  const persistSessions = (next: SessionInfo[]) => {
    if (!enableLocalHistory) return;
    try {
      localStorage.setItem(STORAGE_KEYS.sessions, JSON.stringify(next));
    } catch {}
  };

  const persistActiveSession = (id: string) => {
    if (!enableLocalHistory) return;
    try {
      localStorage.setItem(STORAGE_KEYS.activeSession, id);
    } catch {}
  };

  const persistHistoryBySession = (next: Record<string, ConversationEntry[]>) => {
    if (!enableLocalHistory) return;
    try {
      localStorage.setItem(
        STORAGE_KEYS.historyBySession,
        JSON.stringify(
          Object.fromEntries(
            Object.entries(next).map(([sid, entries]) => [
              sid,
              (entries ?? []).map(e => ({
                ...e,
                timestamp: e.timestamp instanceof Date ? e.timestamp.toISOString() : e.timestamp,
              })),
            ]),
          ),
        ),
      );
    } catch {}
  };

  const commitPendingHistory = () => {
    const pending = pendingHistoryRef.current;
    if (!pending) return;
    setHistoryBySession(prev => {
      const sid = pending.sessionId;
      const next = {
        ...prev,
        [sid]: [pending, ...(prev[sid] ?? [])],
      };
      persistHistoryBySession(next);
      return next;
    });
    pendingHistoryRef.current = null;
  };

  const createNewSession = async () => {
    const res = await fetch("/api/adk/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ appName, userId }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(detail ? `Failed to create session: ${detail}` : "Failed to create session");
    }
    const data = await res.json();
    const id = String(data?.id ?? "");
    if (!id) throw new Error("Failed to create session");
    setSessionId(id);
    setSessionReady(true);

    setSessions(prev => {
      const next: SessionInfo[] = [{ id, createdAt: Date.now() }, ...prev.filter(s => s.id !== id)];
      persistSessions(next);
      return next;
    });
    persistActiveSession(id);

    return id;
  };

  const ensureBackendSession = async (id: string) => {
    const res = await fetch("/api/adk/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ appName, userId, sessionId: id }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(detail ? `Failed to restore session: ${detail}` : "Failed to restore session");
    }
  };

  const selectSession = async (id: string) => {
    if (!id) return;
    try {
      setSessionReady(false);
      commitPendingHistory();
      await ensureBackendSession(id);
      setSessionId(id);
      setSessionReady(true);
      persistActiveSession(id);
      resetUiForNewRun();
      setShowSessions(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err ?? "Failed to restore session");
      setSessionId(id);
      persistActiveSession(id);
      resetUiForNewRun();
      setShowSessions(false);
      setError(
        `Session selected, but the agent backend is not reachable yet. Start the backend on http://localhost:8001 and click Use again. Details: ${msg}`,
      );
      setSessionReady(false);
    }
  };

  const resetUiForNewRun = () => {
    setFinalAnswer("");
    finalAnswerRef.current = "";
    setGuardianAnswer("");
    guardianAnswerRef.current = "";
    setExpandedHistoryEntryId(null);
    setError("");
    setTraceEvents([]);
    setArtifactDebug({ markersSeen: 0, parsedOk: 0, parseFailed: 0 });
    agentBufferRef.current = {};
    activeAgentsRef.current = new Set();
    AGENTS.forEach(a => updateAgentStatus(a.key, "idle"));
    updateAgentStatus("Planner", "active");
  };

  // Create session on mount
  const sessionCreationStartedRef = useRef(false);
  useEffect(() => {
    if (sessionCreationStartedRef.current) return;
    sessionCreationStartedRef.current = true;
    
    const createSession = async () => {
      try {
        loadPersistedState();

        // If we already have a saved active session, use it.
        const savedActive = (() => {
          if (!enableLocalHistory) return null;
          try {
            return localStorage.getItem(STORAGE_KEYS.activeSession);
          } catch {
            return null;
          }
        })();

        if (savedActive) {
          const sid = String(savedActive);
          setSessionId(sid);
          try {
            await ensureBackendSession(sid);
            setSessionReady(true);
          } catch (err: unknown) {
            const msg = err instanceof Error ? err.message : String(err ?? "Failed to restore session");
            setSessionReady(false);
            setError(
              `Restored sessionId from browser storage, but the agent backend is not reachable. Start the backend on http://localhost:8001 and click Use again. Details: ${msg}`,
            );
          }
          return;
        }

        await createNewSession();
      } catch (err) {
        setError("Failed to connect to agent server");
      }
    };
    createSession();
  }, [appName, userId]);

  // Map author to agent key
  const getAgentKey = (author: string): string | null => {
    const lower = author.toLowerCase();
    if (lower.includes("planner")) return "Planner";
    if (lower.includes("guardian_pre") || lower.includes("pre-guardian") || lower.includes("pre guardian")) return null;
    if (lower.includes("guardian") || lower.includes("valid")) return "Guardian";
    if (lower.includes("scrap") || lower.includes("search") || lower.includes("scraping")) return "Scraper";
    if (lower.includes("data")) return "Data";
    if (lower.includes("video")) return "Video";
    if (lower.includes("rag") || lower.includes("knowledge")) return "RAG";
    // Aggregator has many possible names
    if (lower.includes("aggregat") || lower.includes("root") || lower.includes("research_pipeline") || 
        lower.includes("lead") || lower.includes("analyst") || lower.includes("researchpipeline")) return "Aggregator";
    return null;
  };

  const getAgentMeta = (author: string) => {
    const key = getAgentKey(author);
    const agent = key ? AGENTS.find(a => a.key === key) : undefined;
    return { key, agent };
  };

  const updateAgentStatus = (key: string, status: AgentStatus) => {
    setAgentStatuses(prev => {
      const next = { ...prev, [key]: status };
      updateSharedState({ agentStatuses: next });
      return next;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !sessionReady || isProcessing) return;

    commitPendingHistory();
    setIsProcessing(true);
    resetUiForNewRun();
    clearSharedState();
    updateSharedState({ isProcessing: true, sessionId: freshSessionPerRun ? "" : sessionId });
    setArtifacts([]);
    
    const currentQuery = query.trim();
    const queryWithFile = uploadedFile
      ? `${currentQuery}\n\nAttached file (${uploadedFile.kind}): ${uploadedFile.filename}\nPath: ${uploadedFile.path}`
      : currentQuery;
    
    try {
      const effectiveSessionId = freshSessionPerRun ? await createNewSession() : sessionId;
      const res = await fetch("/api/adk/run-sse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          appName,
          userId,
          sessionId: effectiveSessionId,
          newMessage: { role: "user", parts: [{ text: queryWithFile }] },
          streaming: true,
        }),
      });

      if (!res.ok) {
        const detail = await res.text().catch(() => "");
        setError(detail ? `Failed to process query: ${detail}` : "Failed to process query");
        setIsProcessing(false);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buf = "";
      let latestFinalText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        const events = buf.split("\n\n");
        buf = events.pop() ?? "";

        for (const chunk of events) {
          const dataLines = chunk.split("\n").filter(l => l.startsWith("data:")).map(l => l.slice(5).trim());
          if (!dataLines.length) continue;

          try {
            const evt = JSON.parse(dataLines.join("\n"));
            const evtType = String(evt?.type ?? "agent_message") as TraceEvent["type"];
            const text = evt?.content?.parts?.[0]?.text;
            const extractedText = typeof text === "string" ? extractArtifacts(text) : null;
            const author = String(evt?.author ?? "");
            const agentKey = getAgentKey(author);

            if (evt?.error) {
              const msg = String(evt.error);
              setTraceEvents(prev => [
                ...prev,
                { id: `${Date.now()}-${Math.random().toString(36).slice(2)}`, type: "error", author: author || "server", text: msg, ts: Date.now() },
              ]);
              setError(msg);
              setFinalAnswer(msg);
              setIsProcessing(false);
              AGENTS.forEach(a => updateAgentStatus(a.key, "idle"));
              return;
            }

            if (evtType === "tool_call" || evtType === "tool_result") {
              const toolName = String(evt?.tool_name ?? "");
              const newEvent: TraceEvent = {
                id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
                type: evtType,
                author: author || "agent",
                toolName,
                ts: Date.now(),
              };
              setTraceEvents(prev => {
                const next = [...prev, newEvent];
                updateSharedState({ traceEvents: next });
                return next;
              });
              if (agentKey) {
                updateAgentStatus(agentKey, "busy");
                activeAgentsRef.current.add(agentKey);
              }
              continue;
            }

            if (evtType === "final") {
              const finalText = String(text ?? "");
              if (finalText) {
                latestFinalText = finalText;
                finalAnswerRef.current = finalText;
                setFinalAnswer(finalText);

                // If the server marks the Guardian as the final author, treat it as Task Complete output.
                if (agentKey === "Guardian") {
                  guardianAnswerRef.current = finalText;
                  setGuardianAnswer(finalText);
                  updateSharedState({ guardianAnswer: finalText });
                }
              }
              continue;
            }

            // Debug: log author names to console
            if (author) {
              console.log(`[Agent] ${author} -> ${agentKey || 'unknown'}`);
            }

            if (agentKey) {
              updateAgentStatus(agentKey, "busy");
              activeAgentsRef.current.add(agentKey);
              if (text) {
                const cleanedText = extractedText ? extractedText.cleanedText : String(text);
                agentBufferRef.current[agentKey] = (agentBufferRef.current[agentKey] || "") + cleanedText;
                if (agentKey === "Guardian") {
                  guardianAnswerRef.current = agentBufferRef.current[agentKey];
                  setGuardianAnswer(guardianAnswerRef.current);
                  updateSharedState({ guardianAnswer: guardianAnswerRef.current });
                }
              }
            }

            if (typeof text === "string" && text.trim()) {
              const extracted = extractedText ?? extractArtifacts(text);
              if (extracted.found.length) {
                setArtifacts(prev => {
                  const next = [...prev, ...extracted.found];
                  updateSharedState({ artifacts: next });
                  return next;
                });
              }
              if (extracted.markersSeen > 0) {
                setArtifactDebug(prev => ({
                  markersSeen: prev.markersSeen + extracted.markersSeen,
                  parsedOk: prev.parsedOk + extracted.found.length,
                  parseFailed: prev.parseFailed + extracted.parseFailed,
                }));
              }

              const newEvent: TraceEvent = {
                id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
                type: "agent_message",
                author: author || "agent",
                text: extracted.cleanedText,
                ts: Date.now(),
              };
              setTraceEvents(prev => {
                const next = [...prev, newEvent];
                updateSharedState({ traceEvents: next });
                return next;
              });
            }

            // Check for final/aggregator response
            const isAggregator = author.toLowerCase().includes("aggregat") || 
                                 author.toLowerCase().includes("root") ||
                                 author.toLowerCase().includes("research_pipeline") ||
                                 author.toLowerCase().includes("researchpipeline");
            if (evt?.is_final_response || isAggregator) {
              updateAgentStatus("Aggregator", "busy"); // Ensure Aggregator lights up
              const finalText = agentBufferRef.current["Aggregator"] || text || "";
              if (finalText) {
                latestFinalText = finalText;
                finalAnswerRef.current = finalText;
                setFinalAnswer(finalText);
              }
            }
          } catch {}
        }
      }

      // Finalize and stage for history (commit on next task)
      const finalResponse = latestFinalText || agentBufferRef.current["Aggregator"] || finalAnswerRef.current;
      const taskComplete = guardianAnswerRef.current || guardianAnswer;
      if (finalResponse) {
        setFinalAnswer(finalResponse);
        
        // Save to shared state for Advanced View monitoring
        updateSharedState({
          agentStatuses: Object.fromEntries(AGENTS.map(a => [a.key, agentStatuses[a.key] || "idle"])),
          traceEvents,
          isProcessing: false,
          guardianAnswer: taskComplete ? stripPlanSection(taskComplete) : finalResponse,
          sessionId: effectiveSessionId,
          artifacts,
        });
        
        const newEntry: ConversationEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          sessionId: effectiveSessionId,
          query: queryWithFile,
          response: (taskComplete ? stripPlanSection(taskComplete) : finalResponse) || "",
          timestamp: new Date(),
          agentActivity: Array.from(activeAgentsRef.current),
          trace: traceEvents,
        };
        pendingHistoryRef.current = newEntry;
      }
      
      AGENTS.forEach(a => updateAgentStatus(a.key, "idle"));
      updateSharedState({ isProcessing: false });
      setIsProcessing(false);
      setQuery(""); // Clear input after successful query
    } catch (err) {
      setError("Connection error");
      updateSharedState({ isProcessing: false });
      setIsProcessing(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;
    setIsUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      form.append("kind", uploadKind);
      const res = await fetch("/api/adk/upload", { method: "POST", body: form });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data?.success) {
        setError(String(data?.detail || data?.error || "Upload failed"));
        setIsUploading(false);
        return;
      }
      setUploadedFile({ kind: String(data.kind), filename: String(data.stored_filename), path: String(data.stored_path) });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err ?? "Upload failed");
      setError(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#030508] text-white">
      {/* Background */}
      <div className="pointer-events-none fixed inset-0 bg-gradient-to-br from-cyan-500/[0.07] via-transparent to-purple-500/[0.07]" />
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgba(56,189,248,0.1),transparent_50%)]" />

      {/* Header */}
      <header className="relative border-b border-white/[0.06] bg-black/20 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="relative w-9 h-9">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-lg opacity-20 group-hover:opacity-40 transition-opacity" />
                <div className="absolute inset-[2px] bg-[#030508] rounded-md flex items-center justify-center">
                  <span className="text-cyan-400 text-lg">◈</span>
                </div>
              </div>
              <div>
                <div className="text-lg font-semibold">
                  <span className="text-white">Multi‑Agent</span>
                  <span className="text-cyan-400 ml-1">Workspace</span>
                </div>
              </div>
            </Link>
            
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowSessions(v => !v)}
                className={cn(
                  "rounded-lg px-4 py-2 text-sm font-medium transition",
                  showSessions ? "bg-purple-500/20 text-purple-300" : "text-white/70 hover:text-white hover:bg-white/5"
                )}
              >
                🧠 Sessions ({sessions.length})
              </button>
              <button
                onClick={async () => {
                  if (isProcessing) return;
                  try {
                    setSessionReady(false);
                    await createNewSession();
                    resetUiForNewRun();
                  } catch (err: unknown) {
                    const msg = err instanceof Error ? err.message : String(err ?? "Failed to create new session");
                    setError(msg);
                  }
                }}
                className={cn(
                  "rounded-lg px-4 py-2 text-sm font-medium transition",
                  "text-white/70 hover:text-white hover:bg-white/5"
                )}
              >
                ➕ New Session
              </button>
              {/* <button
                onClick={() => setFreshSessionPerRun(v => !v)}
                className={cn(
                  "rounded-lg px-4 py-2 text-sm font-medium transition",
                  freshSessionPerRun ? "bg-cyan-500/20 text-cyan-200" : "text-white/70 hover:text-white hover:bg-white/5"
                )}
              >
                Fresh per run: {freshSessionPerRun ? "On" : "Off"}
              </button> */}
              {/* <Link 
                href="/dashboard/knowledge"
                className="rounded-lg px-4 py-2 text-sm font-medium text-white/70 hover:text-white hover:bg-white/5 transition"
              >
                📚 Knowledge Base
              </Link>
              <Link 
                href={sessionId ? `/dashboard/advanced?session=${sessionId}` : "/dashboard/advanced"}
                className="rounded-lg px-4 py-2 text-sm font-medium text-white/70 hover:text-white hover:bg-white/5 transition"
              >
                ⚙️ Advanced View
              </Link> */}
            </div>
          </div>

          {showSessions && (
            <div className="mt-4 rounded-xl border border-white/10 bg-white/5 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-white">Sessions</div>
                  <div className="text-xs text-white/60">Pick a session to continue (memory stays inside the selected session).</div>
                </div>
                <button
                  onClick={() => setShowSessions(false)}
                  className="text-xs text-white/60 hover:text-white"
                >
                  Close
                </button>
              </div>

              <div className="mt-3 grid gap-2">
                {sessions.length === 0 ? (
                  <div className="text-sm text-white/60">No saved sessions yet.</div>
                ) : (
                  sessions.map(s => (
                    <div key={s.id} className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-black/20 px-3 py-2">
                      <div className="min-w-0">
                        <div className="truncate text-sm text-white">{s.id}</div>
                        <div className="text-xs text-white/50">Created: {new Date(s.createdAt).toLocaleString()}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        {s.id === sessionId && (
                          <span className="rounded-full bg-emerald-500/20 px-2 py-1 text-xs text-emerald-300">Active</span>
                        )}
                        <button
                          onClick={() => selectSession(s.id)}
                          className="rounded-lg bg-white/10 px-3 py-1.5 text-xs font-medium text-white hover:bg-white/15 transition"
                        >
                          Use
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {sessionId && (
                <div className="mt-3 text-xs text-white/60">
                  Active session: <span className="font-mono text-white/80">{sessionId}</span>
                  <span className="ml-2">History items: {(historyBySession[sessionId] ?? []).length}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </header>

      <main className="relative mx-auto max-w-7xl px-6 py-8">

      {/* Search Section */}
      <section className="mb-8">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold mb-2">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
              Start a Task
            </span>
          </h1>
          <p className="text-white/50 text-sm">
            Your query will be processed by 7 specialized AI agents
          </p>
        </div>

        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex gap-3">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Web search, knowledge base Q&A, data analysis, video analysis with planning, safety review"
                className="flex-1 rounded-xl border border-white/10 bg-white/5 px-5 py-4 text-white placeholder:text-white/40 outline-none focus:border-cyan-400/50 focus:bg-white/[0.07] transition-all"
                disabled={isProcessing}
              />
              <button
                type="submit"
                disabled={isProcessing || !sessionReady}
                className={cn(
                  "rounded-xl px-6 py-4 font-semibold transition-all",
                  isProcessing 
                    ? "bg-white/10 text-white/50 cursor-not-allowed"
                    : "bg-gradient-to-r from-cyan-500 to-purple-500 text-white hover:shadow-lg hover:shadow-cyan-500/25 hover:scale-[1.02]"
                )}
              >
                {isProcessing ? (
                  <span className="flex items-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Processing
                  </span>
                ) : (
                  "Go"
                )}
              </button>
            </div>

            <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.02] p-4">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="flex items-center gap-3">
                  <input
                    type="file"
                    onChange={(e) => {
                      const f = e.target.files?.[0] ?? null;
                      setSelectedFile(f);
                      setUploadedFile(null);
                    }}
                    className="block w-full text-xs text-white/70 file:mr-3 file:rounded-lg file:border-0 file:bg-white/10 file:px-3 file:py-2 file:text-xs file:font-semibold file:text-white hover:file:bg-white/15"
                    disabled={isProcessing || isUploading}
                  />
                  <select
                    value={uploadKind}
                    onChange={(e) => setUploadKind(e.target.value as "auto" | "video" | "data")}
                    className="rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-xs text-white/80 outline-none"
                    disabled={isProcessing || isUploading}
                  >
                    <option value="auto">Auto</option>
                    <option value="video">Video</option>
                    <option value="data">Data</option>
                  </select>
                </div>

                <div className="flex items-center gap-2 justify-end">
                  <button
                    type="button"
                    onClick={handleUpload}
                    disabled={!selectedFile || isUploading || isProcessing}
                    className={cn(
                      "rounded-lg px-4 py-2 text-xs font-semibold transition border",
                      !selectedFile || isUploading || isProcessing
                        ? "border-white/10 bg-white/5 text-white/40 cursor-not-allowed"
                        : "border-cyan-400/30 bg-cyan-400/10 text-cyan-200 hover:bg-cyan-400/15"
                    )}
                  >
                    {isUploading ? "Uploading..." : "Upload"}
                  </button>

                  {uploadedFile && (
                    <div className="text-xs text-white/60">
                      Attached: <span className="text-white/80">{uploadedFile.filename}</span>
                    </div>
                  )}
                </div>
              </div>
              <div className="mt-2 text-[11px] text-white/40">
                Video uploads are saved to the project root (Video agent can find them). Data uploads are saved to <span className="font-mono">datasets/</span> (Data agent can find them).
              </div>
            </div>
          </form>
        </section>

        {/* Agent Status Bar */}
        <section className="mb-8">
          <div className="flex items-center justify-center gap-2 flex-wrap">
            {AGENTS.map((agent) => {
              const status = agentStatuses[agent.key] || "idle";
              return (
                <div
                  key={agent.key}
                  className={cn(
                    "flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium transition-all border",
                    status === "busy" 
                      ? "border-orange-400/50 bg-orange-500/20 text-orange-200" 
                      : status === "active"
                      ? "border-emerald-400/50 bg-emerald-500/20 text-emerald-200"
                      : "border-white/10 bg-white/5 text-white/50"
                  )}
                  title={agent.description}
                >
                  <span style={{ color: status !== "idle" ? agent.color : undefined }}>{agent.icon}</span>
                  <span>{agent.name}</span>
                  {status === "busy" && (
                    <span className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Error Display */}
        {error && (
          <div className="mb-6 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-center text-red-200">
            {error}
          </div>
        )}

        {traceEvents.length > 0 && (
          <section className="mb-6">
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
              <div className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
                <div className="font-semibold">Trace</div>
                <div className="text-xs text-white/40">{traceEvents.length} events</div>
              </div>
              <div className="max-h-[540px] overflow-y-auto">
                {traceEvents.map((ev, idx) => {
                  const { agent } = getAgentMeta(ev.author);
                  const agentColor = agent?.color ?? "#94a3b8";
                  const agentIcon = agent?.icon;
                  const agentName = agent?.name ?? ev.author;

                  const isTool = ev.type === "tool_call" || ev.type === "tool_result";
                  const isError = ev.type === "error";

                  return (
                    <div
                      key={ev.id}
                      className="px-6 py-3 border-b border-white/10 last:border-b-0 hover:bg-white/[0.02] transition-colors"
                    >
                      <div className="grid grid-cols-[40px_28px_1fr_auto] gap-3 items-start">
                        <div className="pt-0.5 text-[11px] text-white/35 tabular-nums">#{idx + 1}</div>

                        <div
                          className="mt-0.5 h-6 w-6 rounded-md flex items-center justify-center border"
                          style={{ backgroundColor: `${agentColor}22`, borderColor: `${agentColor}55` }}
                          title={agentName}
                        >
                          <span style={{ color: agentColor }} className="text-xs leading-none">
                            {agentIcon ?? "◈"}
                          </span>
                        </div>

                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <div className="text-xs text-white/60 truncate max-w-[260px]">{agentName}</div>
                            {ev.type === "agent_message" && (
                              <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-semibold text-white/50">
                                MSG
                              </span>
                            )}
                            {ev.type === "tool_call" && (
                              <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold text-cyan-200 border-cyan-400/30 bg-cyan-400/10">
                                TOOL
                              </span>
                            )}
                            {ev.type === "tool_result" && (
                              <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold text-emerald-200 border-emerald-400/30 bg-emerald-400/10">
                                TOOL DONE
                              </span>
                            )}
                            {ev.type === "error" && (
                              <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold text-red-200 border-red-400/30 bg-red-400/10">
                                ERROR
                              </span>
                            )}
                          </div>

                          {isTool && (
                            <div className="mt-2 inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1 text-xs">
                              <span className={cn("font-semibold", ev.type === "tool_call" ? "text-cyan-200" : "text-emerald-200")}>
                                {ev.type === "tool_call" ? "calling" : "finished"}
                              </span>
                              <span className="font-mono text-white/80">{ev.toolName || "(unknown)"}</span>
                            </div>
                          )}

                          {isError && (
                            <div className="mt-2 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-100 whitespace-pre-wrap">
                              {ev.text}
                            </div>
                          )}

                          {ev.type === "agent_message" && (
                            <div className="mt-2 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                              {(() => {
                                const raw = ev.text ?? "";
                                const text = normalizeForMarkdown(raw);
                                const shouldRenderMarkdown =
                                  isLikelyMarkdown(text) || raw.includes("**") || raw.includes("```") || /[•‣◦]/.test(raw);

                                if (!shouldRenderMarkdown) {
                                  return (
                                    <div className="whitespace-pre-wrap break-words text-sm leading-6 text-white/80">
                                      {raw}
                                    </div>
                                  );
                                }
                                return (
                                  <div
                                    className="prose prose-invert max-w-none prose-sm break-words
                                      prose-headings:text-white prose-headings:font-semibold
                                      prose-p:text-white/80 prose-p:leading-relaxed prose-p:my-2
                                      prose-strong:text-white prose-strong:font-semibold
                                      prose-ul:text-white/80 prose-ol:text-white/80 prose-ul:my-2 prose-ol:my-2
                                      prose-li:marker:text-cyan-400
                                      prose-code:text-purple-300 prose-code:bg-white/10 prose-code:px-1 prose-code:rounded
                                      prose-pre:bg-black/40 prose-pre:border prose-pre:border-white/10 prose-pre:overflow-x-auto
                                      prose-table:block prose-table:overflow-x-auto prose-table:whitespace-nowrap
                                      prose-blockquote:border-cyan-400/40 prose-blockquote:text-white/60"
                                  >
                                    <ReactMarkdown
                                      remarkPlugins={[remarkGfm, remarkBreaks]}
                                      components={{
                                        a: ({ href, children }) => (
                                          <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                                            {children}
                                          </a>
                                        ),
                                        pre: ({ children }) => (
                                          <pre className="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3">
                                            {children}
                                          </pre>
                                        ),
                                        code: ({ children }) => (
                                          <code className="break-words">{children}</code>
                                        ),
                                        table: ({ children }) => (
                                          <div className="overflow-x-auto">
                                            <table>{children}</table>
                                          </div>
                                        ),
                                      }}
                                    >
                                      {text}
                                    </ReactMarkdown>
                                  </div>
                                );
                              })()}
                            </div>
                          )}
                        </div>

                        <div className="pt-0.5 text-[11px] text-white/35 whitespace-nowrap tabular-nums">
                          {new Date(ev.ts).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>
        )}

        {/* Results Section */}
        <section>
          {isProcessing && !finalAnswer && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-8 text-center">
              <div className="flex justify-center mb-4">
                <div className="relative w-16 h-16">
                  <div className="absolute inset-0 border-4 border-cyan-400/30 rounded-full" />
                  <div className="absolute inset-0 border-4 border-transparent border-t-cyan-400 rounded-full animate-spin" />
                  <div className="absolute inset-2 border-4 border-transparent border-t-purple-400 rounded-full animate-spin" style={{ animationDirection: "reverse", animationDuration: "1.5s" }} />
                </div>
              </div>
              <div className="text-white/70 font-medium">Agents are working on your query...</div>
              <div className="text-white/40 text-sm mt-1">This usually takes 10-30 seconds</div>
            </div>
          )}

          {guardianAnswer && (
            <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
              <div className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
                <div className="flex items-center gap-2 text-[12px] font-semibold tracking-wide text-white/70">
                  <span className="text-emerald-400 text-sm">✓</span>
                  <span>Task Complete</span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => navigator.clipboard.writeText(stripPlanSection(guardianAnswer))}
                    className="text-xs text-white/50 hover:text-white/70 px-3 py-1 rounded-lg hover:bg-white/5 transition"
                  >
                    📋 Copy
                  </button>
                  <button
                    onClick={() => {
                      const blob = new Blob([stripPlanSection(guardianAnswer)], { type: "text/markdown" });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement("a");
                      a.href = url;
                      a.download = `result-${new Date().toISOString().slice(0, 10)}.md`;
                      a.click();
                    }}
                    className="text-xs text-white/50 hover:text-white/70 px-3 py-1 rounded-lg hover:bg-white/5 transition"
                  >
                    💾 Export
                  </button>
                </div>
              </div>
              <div className="p-6">
                {(() => {
                  const taskText = stripPlanSection(guardianAnswer);
                  const normalized = normalizeForMarkdown(taskText);
                  const shouldRenderMarkdown =
                    isLikelyMarkdown(normalized) || taskText.includes("**") || taskText.includes("```") || /[•‣◦]/.test(taskText);

                  if (!shouldRenderMarkdown) {
                    return (
                      <div className="whitespace-pre-wrap break-words text-[18px] leading-8 text-white/90">
                        {taskText}
                      </div>
                    );
                  }
                  return (
                    <article className="max-w-none text-[18px] leading-8 text-white/90">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, remarkBreaks]}
                        components={{
                          a: ({ href, children }) => (
                            <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                              {children}
                            </a>
                          ),
                          p: ({ children }) => (
                            <p className="my-2">{children}</p>
                          ),
                          ul: ({ children }) => (
                            <ul className="my-2 list-disc pl-5 marker:text-cyan-400">{children}</ul>
                          ),
                          ol: ({ children }) => (
                            <ol className="my-2 list-decimal pl-5 marker:text-cyan-400">{children}</ol>
                          ),
                          li: ({ children }) => (
                            <li className="my-1">{children}</li>
                          ),
                          h1: ({ children }) => (
                            <h1 className="text-lg font-semibold text-white my-3">{children}</h1>
                          ),
                          h2: ({ children }) => (
                            <h2 className="text-lg font-semibold text-white my-3">{children}</h2>
                          ),
                          h3: ({ children }) => (
                            <h3 className="text-base font-semibold text-white/90 my-3">{children}</h3>
                          ),
                          pre: ({ children }) => (
                            <pre className="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-4">
                              {children}
                            </pre>
                          ),
                          code: ({ children }) => (
                            <code className="break-words">{children}</code>
                          ),
                          table: ({ children }) => (
                            <div className="overflow-x-auto">
                              <table>{children}</table>
                            </div>
                          ),
                        }}
                      >
                        {normalized}
                      </ReactMarkdown>
                    </article>
                  );
                })()}
              </div>
            </div>
          )}

          {!isProcessing && (artifactDebug.markersSeen > 0 || artifactDebug.parseFailed > 0 || artifacts.length > 0) && (
            <div className="mt-4 rounded-xl border border-cyan-400/20 bg-cyan-400/5 px-4 py-3 text-xs text-cyan-100">
              Artifact debug: markers seen {artifactDebug.markersSeen} • parsed {artifactDebug.parsedOk} • failed {artifactDebug.parseFailed} • rendered {artifacts.length}
            </div>
          )}

          {artifacts.length > 0 && (
            <section className="mt-6 rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
              <div className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
                <div className="font-semibold text-white/90">Artifacts</div>
                <div className="text-xs text-white/40">{artifacts.length}</div>
              </div>
              <div className="p-6 space-y-6">
                {(() => {
                  console.log("[Artifacts] Rendering", artifacts.length, "artifacts:", artifacts.map(a => a.kind));
                  return null;
                })()}
                {artifacts.map((a, idx) => {
                  if (a.kind === "table") {
                    return (
                      <div key={idx} className="space-y-2">
                        {a.title && <div className="text-sm font-semibold text-white/80">{a.title}</div>}
                        <div className="overflow-auto rounded-xl border border-white/10 bg-black/20">
                          <table className="min-w-full text-xs text-white/80">
                            <thead className="sticky top-0 bg-black/40">
                              <tr>
                                {a.columns.map((c) => (
                                  <th
                                    key={c}
                                    className="px-3 py-2 text-left font-semibold text-white/70 border-b border-white/10 whitespace-nowrap"
                                  >
                                    {c}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {a.rows.map((r, rIdx) => (
                                <tr key={rIdx} className="border-b border-white/5 last:border-b-0">
                                  {a.columns.map((c) => (
                                    <td key={c} className="px-3 py-2 whitespace-nowrap">
                                      {String((r as any)[c] ?? "")}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    );
                  }

                  if (a.kind === "chart_plotly") {
                    const data = (a.spec && (a.spec as any).data) || [];
                    const layout = (a.spec && (a.spec as any).layout) || {};
                    console.log("[Chart] Rendering artifact", idx, "with spec:", { dataCount: data.length, layout });
                    return (
                      <div key={idx} className="space-y-2">
                        <div className="text-sm font-semibold text-white/80">{a.title || "Chart"}</div>
                        <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                          <div
                            className="w-full overflow-x-auto"
                            data-plotly="1"
                            data-plotly-spec={JSON.stringify({ data, layout })}
                            style={{ height: 520, minHeight: 520 }}
                          />
                        </div>
                      </div>
                    );
                  }

                  const unknownKind = (a as any)?.kind ?? "unknown";
                  console.warn("[Artifacts] Unknown artifact kind:", unknownKind, "at index", idx, a);
                  return (
                    <div key={idx} className="rounded-xl border border-yellow-400/30 bg-yellow-400/10 px-4 py-3 text-sm text-yellow-200">
                      Unknown artifact type: {unknownKind}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {(() => {
            const activeHistory = sessionId ? (historyBySession[sessionId] ?? []) : [];

            if (!isProcessing && !finalAnswer && activeHistory.length === 0) {
              return (
                <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.01] p-12 text-center">
                  <div className="text-4xl mb-4">🔍</div>
                  <div className="text-white/50 font-medium">Enter a query to start researching</div>
                  <div className="text-white/30 text-sm mt-2">
                    Try: &quot;Latest trends in AI agents&quot; or &quot;Research quantum computing&quot;
                  </div>
                </div>
              );
            }

            if (!sessionId || activeHistory.length === 0) return null;

            return (
              <div className="mt-8 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-white/90">💬 Chat History (Active Session)</h2>
                  <button
                    onClick={() => {
                      if (confirm("Clear history for this session?")) {
                        setHistoryBySession(prev => {
                          const next = { ...prev, [sessionId]: [] };
                          persistHistoryBySession(next);
                          return next;
                        });
                        setExpandedHistoryEntryId(null);
                      }
                    }}
                    className="text-xs text-red-400/70 hover:text-red-400 px-3 py-1 rounded-lg hover:bg-red-500/10 transition"
                  >
                    Clear Chat
                  </button>
                </div>

                {activeHistory.map((entry) => (
                  <div key={entry.id} className="rounded-xl border border-white/10 bg-white/[0.02] overflow-hidden">
                    <div className="border-b border-white/10 px-4 py-3 flex items-center justify-between bg-white/[0.02]">
                      <div className="flex items-center gap-3">
                        <span className="text-cyan-400">❓</span>
                        <span className="font-medium text-white/90 truncate max-w-md">{entry.query}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1">
                          {entry.agentActivity.map((agent) => {
                            const agentInfo = AGENTS.find(a => a.key === agent);
                            return agentInfo ? (
                              <span key={agent} className="text-sm" style={{ color: agentInfo.color }} title={agentInfo.name}>
                                {agentInfo.icon}
                              </span>
                            ) : null;
                          })}
                        </div>
                        <span className="text-xs text-white/40">
                          {entry.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                    <div className="p-4 max-h-60 overflow-y-auto">
                      <article className="prose prose-invert prose-sm max-w-none prose-p:text-white/70 prose-headings:text-white/80">
                        <ReactMarkdown
                          components={{
                            a: ({ href, children }) => (
                              <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                                {children}
                              </a>
                            ),
                          }}
                        >
                          {entry.response.length > 500 ? entry.response.slice(0, 500) + "..." : entry.response}
                        </ReactMarkdown>
                      </article>
                    </div>

                    {expandedHistoryEntryId === entry.id && (
                      <div className="border-t border-white/10 bg-black/10 p-4 space-y-4">
                        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
                          <div className="text-xs font-semibold text-white/60 mb-2">You</div>
                          <div className="whitespace-pre-wrap break-words text-sm text-white/80">{entry.query}</div>
                        </div>

                        {entry.trace?.length ? (
                          <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
                            <div className="border-b border-white/10 px-4 py-3 flex items-center justify-between">
                              <div className="font-semibold text-sm">Trace</div>
                              <div className="text-[11px] text-white/40">{entry.trace.length} events</div>
                            </div>
                            <div className="max-h-[520px] overflow-y-auto">
                              {entry.trace.map((ev, idx) => {
                                const { agent } = getAgentMeta(ev.author);
                                const agentColor = agent?.color ?? "#94a3b8";
                                const agentIcon = agent?.icon;
                                const agentName = agent?.name ?? ev.author;

                                const isTool = ev.type === "tool_call" || ev.type === "tool_result";
                                const isError = ev.type === "error";

                                return (
                                  <div
                                    key={ev.id}
                                    className="px-4 py-3 border-b border-white/10 last:border-b-0 hover:bg-white/[0.02] transition-colors"
                                  >
                                    <div className="grid grid-cols-[36px_28px_1fr_auto] gap-3 items-start">
                                      <div className="pt-0.5 text-[11px] text-white/35 tabular-nums">#{idx + 1}</div>

                                      <div
                                        className="mt-0.5 h-6 w-6 rounded-md flex items-center justify-center border"
                                        style={{ backgroundColor: `${agentColor}22`, borderColor: `${agentColor}55` }}
                                        title={agentName}
                                      >
                                        <span style={{ color: agentColor }} className="text-xs leading-none">
                                          {agentIcon ?? "◈"}
                                        </span>
                                      </div>

                                      <div className="min-w-0">
                                        <div className="flex items-center gap-2">
                                          <div className="text-xs text-white/60 truncate max-w-[240px]">{agentName}</div>
                                          {ev.type === "agent_message" && (
                                            <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[10px] font-semibold text-white/50">
                                              MSG
                                            </span>
                                          )}
                                          {ev.type === "tool_call" && (
                                            <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold text-cyan-200 border-cyan-400/30 bg-cyan-400/10">
                                              TOOL
                                            </span>
                                          )}
                                          {ev.type === "tool_result" && (
                                            <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold text-emerald-200 border-emerald-400/30 bg-emerald-400/10">
                                              TOOL DONE
                                            </span>
                                          )}
                                          {ev.type === "error" && (
                                            <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold text-red-200 border-red-400/30 bg-red-400/10">
                                              ERROR
                                            </span>
                                          )}
                                        </div>

                                        {isTool && (
                                          <div className="mt-2 inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1 text-xs">
                                            <span className={cn("font-semibold", ev.type === "tool_call" ? "text-cyan-200" : "text-emerald-200")}>
                                              {ev.type === "tool_call" ? "calling" : "finished"}
                                            </span>
                                            <span className="font-mono text-white/80">{ev.toolName || "(unknown)"}</span>
                                          </div>
                                        )}

                                        {isError && (
                                          <div className="mt-2 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-100 whitespace-pre-wrap">
                                            {ev.text}
                                          </div>
                                        )}

                                        {ev.type === "agent_message" && (
                                          <div className="mt-2 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3">
                                            {(() => {
                                              const raw = ev.text ?? "";
                                              const text = normalizeForMarkdown(raw);
                                              const shouldRenderMarkdown =
                                                isLikelyMarkdown(text) || raw.includes("**") || raw.includes("```") || /[•‣◦]/.test(raw);

                                              if (!shouldRenderMarkdown) {
                                                return (
                                                  <div className="whitespace-pre-wrap break-words text-sm leading-6 text-white/80">
                                                    {raw}
                                                  </div>
                                                );
                                              }
                                              return (
                                                <div
                                                  className="prose prose-invert max-w-none prose-sm break-words
                                                    prose-headings:text-white prose-headings:font-semibold
                                                    prose-p:text-white/80 prose-p:leading-relaxed prose-p:my-2
                                                    prose-strong:text-white prose-strong:font-semibold
                                                    prose-ul:text-white/80 prose-ol:text-white/80 prose-ul:my-2 prose-ol:my-2
                                                    prose-li:marker:text-cyan-400
                                                    prose-code:text-purple-300 prose-code:bg-white/10 prose-code:px-1 prose-code:rounded
                                                    prose-pre:bg-black/40 prose-pre:border prose-pre:border-white/10 prose-pre:overflow-x-auto
                                                    prose-table:block prose-table:overflow-x-auto prose-table:whitespace-nowrap
                                                    prose-blockquote:border-cyan-400/40 prose-blockquote:text-white/60"
                                                >
                                                  <ReactMarkdown
                                                    remarkPlugins={[remarkGfm, remarkBreaks]}
                                                    components={{
                                                      a: ({ href, children }) => (
                                                        <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                                                          {children}
                                                        </a>
                                                      ),
                                                      pre: ({ children }) => (
                                                        <pre className="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3">
                                                          {children}
                                                        </pre>
                                                      ),
                                                      code: ({ children }) => (
                                                        <code className="break-words">{children}</code>
                                                      ),
                                                      table: ({ children }) => (
                                                        <div className="overflow-x-auto">
                                                          <table>{children}</table>
                                                        </div>
                                                      ),
                                                    }}
                                                  >
                                                    {text}
                                                  </ReactMarkdown>
                                                </div>
                                              );
                                            })()}
                                          </div>
                                        )}
                                      </div>

                                      <div className="pt-0.5 text-[11px] text-white/35 whitespace-nowrap tabular-nums">
                                        {new Date(ev.ts ?? Date.now()).toLocaleTimeString()}
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}

                        <div className="rounded-2xl border border-white/10 bg-white/[0.02] overflow-hidden">
                          <div className="border-b border-white/10 px-4 py-3 flex items-center gap-2 text-[12px] font-semibold tracking-wide text-white/70">
                            <span className="text-emerald-400 text-sm">✓</span>
                            <span>Task Complete</span>
                          </div>
                          <div className="p-5">
                            {(() => {
                              const taskText = stripPlanSection(entry.response || "");
                              const normalized = normalizeForMarkdown(taskText);
                              const shouldRenderMarkdown =
                                isLikelyMarkdown(normalized) || taskText.includes("**") || taskText.includes("```") || /[•‣◦]/.test(taskText);

                              if (!shouldRenderMarkdown) {
                                return (
                                  <div className="whitespace-pre-wrap break-words text-[17px] leading-8 text-white/90">
                                    {taskText}
                                  </div>
                                );
                              }
                              return (
                                <article className="max-w-none text-[17px] leading-8 text-white/90">
                                  <ReactMarkdown
                                    remarkPlugins={[remarkGfm, remarkBreaks]}
                                    components={{
                                      a: ({ href, children }) => (
                                        <a href={href} target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">
                                          {children}
                                        </a>
                                      ),
                                      p: ({ children }) => (
                                        <p className="my-2">{children}</p>
                                      ),
                                      ul: ({ children }) => (
                                        <ul className="my-2 list-disc pl-5 marker:text-cyan-400">{children}</ul>
                                      ),
                                      ol: ({ children }) => (
                                        <ol className="my-2 list-decimal pl-5 marker:text-cyan-400">{children}</ol>
                                      ),
                                      li: ({ children }) => (
                                        <li className="my-1">{children}</li>
                                      ),
                                      h1: ({ children }) => (
                                        <h1 className="text-lg font-semibold text-white my-3">{children}</h1>
                                      ),
                                      h2: ({ children }) => (
                                        <h2 className="text-lg font-semibold text-white my-3">{children}</h2>
                                      ),
                                      h3: ({ children }) => (
                                        <h3 className="text-base font-semibold text-white/90 my-3">{children}</h3>
                                      ),
                                      pre: ({ children }) => (
                                        <pre className="overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-4">
                                          {children}
                                        </pre>
                                      ),
                                      code: ({ children }) => (
                                        <code className="break-words">{children}</code>
                                      ),
                                      table: ({ children }) => (
                                        <div className="overflow-x-auto">
                                          <table>{children}</table>
                                        </div>
                                      ),
                                    }}
                                  >
                                    {normalized}
                                  </ReactMarkdown>
                                </article>
                              );
                            })()}
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="border-t border-white/10 px-4 py-2 flex justify-end gap-2">
                      <button
                        onClick={() => {
                          setExpandedHistoryEntryId(prev => (prev === entry.id ? null : entry.id));
                        }}
                        className="text-xs text-white/50 hover:text-white/70 px-3 py-1 rounded-lg hover:bg-white/5 transition"
                      >
                        {expandedHistoryEntryId === entry.id ? "Collapse" : "View Full"}
                      </button>
                      <button
                        onClick={() => {
                          setQuery(entry.query);
                        }}
                        className="text-xs text-cyan-400/70 hover:text-cyan-400 px-3 py-1 rounded-lg hover:bg-cyan-500/10 transition"
                      >
                        Re-run
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            );
          })()}
        </section>
      </main>

      {/* Footer */}
      {/* <footer className="relative border-t border-white/[0.06] mt-12">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <div className="flex items-center justify-between text-xs text-white/40">
            <span>Multi‑Agent Workspace</span>
            <div className="flex items-center gap-4">
              <Link href="/dashboard/knowledge" className="hover:text-white/60 transition">Knowledge Base</Link>
            </div>
          </div>
        </div>
      </footer> */}
    </div>
  );
}
