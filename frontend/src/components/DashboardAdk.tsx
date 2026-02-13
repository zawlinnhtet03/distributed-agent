"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

type ResultItem = {
  id: string;
  title: string;
  source: "Video" | "Web";
  duration: string;
  score: number;
};

type SharedState = {
  agentStatuses?: Record<string, AgentState>;
  traceEvents?: Array<{ id: string; type: string; author: string; text?: string; toolName?: string; ts: number }>;
  isProcessing?: boolean;
  guardianAnswer?: string;
  artifacts?: Artifact[];
  sessionId?: string;
  lastUpdated?: number;
};

type Artifact =
  | { kind: "table"; title?: string; columns: string[]; rows: Array<Record<string, string>> }
  | { kind: "chart_plotly"; title?: string; spec: any };

declare global {
  interface Window {
    Plotly?: any;
  }
}

type AgentState = "active" | "busy" | "idle";

type ShardState = "healthy" | "idle" | "high_load";

type LogLine = {
  ts: string;
  lvl: "info" | "warn" | "error";
  msg: string;
};

type AdkMessage = {
  role: "user" | "model";
  parts: Array<{ text?: string }>;
};

type KbDocument = {
  content: string;
  source: string;
  type: "web" | "video" | "pdf" | "note";
  score: number;
};

type KbStats = {
  totalDocs: number;
  collection: string;
  model: string;
  byType: { [key: string]: number };
};

type KbDocItem = {
  id: string;
  type: string;
  source: string;
  timestamp: string;
  preview: string;
  length: number;
};

type AgentActivity = {
  name: string;
  status: AgentState;
  detail: string;
  lastUpdate: number;
};

function cn(...v: Array<string | false | null | undefined>) {
  return v.filter(Boolean).join(" ");
}

// Format content by shortening long URLs and making them clickable
function formatContent(text: string): React.ReactNode {
  // Split text by URLs
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const parts = text.split(urlRegex);
  
  return parts.map((part, i) => {
    if (part.match(urlRegex)) {
      // It's a URL - shorten and make clickable
      try {
        const url = new URL(part);
        const displayText = url.hostname + (url.pathname.length > 30 ? url.pathname.slice(0, 30) + '...' : url.pathname);
        return (
          <a 
            key={i}
            href={part}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sky-400 hover:text-sky-300 hover:underline break-all"
            title={part}
          >
            {displayText}
          </a>
        );
      } catch {
        // If URL parsing fails, show truncated version
        const shortened = part.length > 60 ? part.slice(0, 60) + '...' : part;
        return <span key={i} className="text-sky-400 break-all" title={part}>{shortened}</span>;
      }
    }
    return <span key={i}>{part}</span>;
  });
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-white/[0.06] bg-white/[0.02] backdrop-blur">
      <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-4">
        <div className="text-sm font-semibold tracking-wide text-white/90">
          {title}
        </div>
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}

function Pill({ status }: { status: AgentState | ShardState }) {
  const cls =
    status === "active" || status === "healthy"
      ? "bg-emerald-500/15 text-emerald-100 ring-emerald-500/30"
      : status === "busy" || status === "high_load"
        ? "bg-orange-500/15 text-orange-100 ring-orange-500/30"
        : "bg-sky-500/15 text-sky-100 ring-sky-500/30";

  return (
    <span className={cn("rounded-full px-2 py-0.5 text-xs ring-1", cls)}>
      {String(status).replaceAll("_", " ")}
    </span>
  );
}

function Bar({ value }: { value: number }) {
  return (
    <div className="h-2 w-full rounded-full bg-white/10">
      <div
        className="h-2 rounded-full bg-gradient-to-r from-sky-400 to-emerald-400"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}

function SourceTypeBadge({ type }: { type: string }) {
  const config: { [k: string]: { bg: string; text: string; icon: string } } = {
    web: { bg: "bg-blue-500/20", text: "text-blue-200", icon: "🌐" },
    video: { bg: "bg-purple-500/20", text: "text-purple-200", icon: "🎬" },
    pdf: { bg: "bg-red-500/20", text: "text-red-200", icon: "📄" },
    note: { bg: "bg-yellow-500/20", text: "text-yellow-200", icon: "📝" },
  };
  const c = config[type] ?? config.web;
  return (
    <span className={cn("rounded-md px-2 py-0.5 text-xs font-medium", c.bg, c.text)}>
      {c.icon} {type.toUpperCase()}
    </span>
  );
}

function RelevanceBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 80 ? "from-emerald-400 to-emerald-500" : pct >= 60 ? "from-sky-400 to-sky-500" : "from-orange-400 to-orange-500";
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 rounded-full bg-white/10">
        <div className={cn("h-1.5 rounded-full bg-gradient-to-r", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs tabular-nums text-white/60">{pct}%</span>
    </div>
  );
}

function SourceCard({ doc, index }: { doc: KbDocument; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const preview = doc.content.length > 200 ? doc.content.slice(0, 200) + "..." : doc.content;
  const hostname = doc.source.startsWith("http") ? new URL(doc.source).hostname : doc.source;
  
  return (
    <div className="rounded-xl border border-white/[0.06] bg-black/30 p-4 transition hover:border-white/20">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-white/10 text-xs font-bold text-white/70">
            {index}
          </span>
          <SourceTypeBadge type={doc.type} />
        </div>
        <RelevanceBar score={doc.score} />
      </div>
      
      <div className="mt-3">
        <a 
          href={doc.source.startsWith("http") ? doc.source : undefined} 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-sm font-medium text-sky-300 hover:text-sky-200 hover:underline"
        >
          {hostname}
        </a>
      </div>
      
      <div className="mt-2 text-sm leading-relaxed text-white/70">
        {expanded ? doc.content : preview}
      </div>
      
      {doc.content.length > 200 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 text-xs font-medium text-white/50 hover:text-white/70"
        >
          {expanded ? "Show less ▲" : "Show more ▼"}
        </button>
      )}
    </div>
  );
}

function KbStatsCard({ stats }: { stats: KbStats | null }) {
  if (!stats) return <div className="text-white/50 text-sm">Loading stats...</div>;
  
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <div className="rounded-xl border border-white/[0.06] bg-black/30 p-4">
        <div className="text-2xl font-bold text-white/90">{stats.totalDocs}</div>
        <div className="text-xs text-white/50">Total Documents</div>
      </div>
      <div className="rounded-xl border border-white/[0.06] bg-black/30 p-4">
        <div className="flex flex-wrap gap-1">
          {Object.entries(stats.byType).map(([type, count]) => (
            <span key={type} className="inline-flex items-center gap-1 rounded-md bg-white/10 px-2 py-1 text-xs text-white/80">
              <SourceTypeBadge type={type} /> {count}
            </span>
          ))}
        </div>
        <div className="mt-1 text-xs text-white/50">By Source Type</div>
      </div>
      <div className="rounded-xl border border-white/[0.06] bg-black/30 p-4">
        <div className="text-sm font-medium text-white/80 truncate">{stats.model}</div>
        <div className="text-xs text-white/50">Embedding Model</div>
      </div>
    </div>
  );
}

function Logs({ lines }: { lines: LogLine[] }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  const lvlColor = (lvl: LogLine["lvl"]) => {
    if (lvl === "error") return "text-red-200";
    if (lvl === "warn") return "text-orange-200";
    return "text-emerald-200";
  };

  return (
    <div
      ref={ref}
      className="h-[220px] overflow-auto rounded-xl border border-white/[0.06] bg-black/30 p-4 font-mono text-xs leading-5 text-white/80"
    >
      {lines.map((l, i) => (
        <div key={`${l.ts}-${i}`} className="whitespace-pre-wrap">
          <span className="text-white/40">[{l.ts}]</span>{" "}
          <span className={cn("font-semibold", lvlColor(l.lvl))}>
            {l.lvl.toUpperCase()}
          </span>
          <span className="text-white/35">:</span> <span>{l.msg}</span>
        </div>
      ))}
    </div>
  );
}

export default function DashboardAdk() {
  const searchParams = useSearchParams();
  const sharedSessionId = searchParams.get("session");
  const isMonitoringMode = !!sharedSessionId;
  
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState("r1");
  const [apps, setApps] = useState<string[]>([]);
  const [appName, setAppName] = useState<string>("research_pipeline_agent");
  const [userId] = useState<string>("ui_user");
  const [sessionId, setSessionId] = useState<string>("");
  const [sessionReady, setSessionReady] = useState(false);
  const [streaming, setStreaming] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [finalAnswer, setFinalAnswer] = useState<string>("");
  const [finalAnswerAuthor, setFinalAnswerAuthor] = useState<string>("");
  const [guardianAnswer, setGuardianAnswer] = useState<string>("");
  const [agentMessages, setAgentMessages] = useState<{[agent: string]: string}>({});
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [activeAgents, setActiveAgents] = useState<{[agent: string]: AgentActivity}>({
    "Planner": { name: "Planner Agent", status: "idle", detail: "Waiting for query", lastUpdate: Date.now() },
    "Guardian": { name: "Guardian Agent", status: "idle", detail: "Pre/Post validation ready", lastUpdate: Date.now() },
    "Scraper": { name: "Scraper Agent", status: "idle", detail: "Ready", lastUpdate: Date.now() },
    "Data": { name: "Data Agent", status: "idle", detail: "Ready", lastUpdate: Date.now() },
    "Video": { name: "Video Agent", status: "idle", detail: "Ready", lastUpdate: Date.now() },
    "RAG": { name: "RAG Agent", status: "idle", detail: "Ready", lastUpdate: Date.now() },
    "Aggregator": { name: "Aggregator Agent", status: "idle", detail: "Waiting", lastUpdate: Date.now() },
  });
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [kbStatsData, setKbStatsData] = useState<KbStats | null>(null);
  const [kbStatsLoading, setKbStatsLoading] = useState(false);
  const [kbQuery, setKbQuery] = useState<string>("");
  const [kbResults, setKbResults] = useState<KbDocument[]>([]);
  const [kbSearching, setKbSearching] = useState(false);
  const [kbError, setKbError] = useState<string>("");
  const [kbDocuments, setKbDocuments] = useState<any[]>([]);
  const [kbDocsLoading, setKbDocsLoading] = useState(false);
  const [showKbDocs, setShowKbDocs] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{loading: boolean; message: string; success?: boolean}>({loading: false, message: ""});
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [isLoadingSharedState, setIsLoadingSharedState] = useState(false);
  const [sharedStateError, setSharedStateError] = useState("");
  
  // Buffer for accumulating streaming tokens per agent
  const agentBufferRef = useRef<{[agent: string]: string}>({});
  const finalAnswerSetRef = useRef<boolean>(false);

  // Shared state key (must match DashboardSimple)
  const SHARED_STATE_KEY = useMemo(() => `adk_shared_state:${appName}:${userId}`, [appName, userId]);

  // Poll for shared state from Simple Dashboard when in monitoring mode
  useEffect(() => {
    if (!isMonitoringMode) return;
    
    const pollSharedState = () => {
      try {
        const raw = localStorage.getItem(SHARED_STATE_KEY);
        if (!raw) {
          setSharedStateError("No shared state found. Run a task in Simple Dashboard first.");
          return;
        }
        
        const state: SharedState = JSON.parse(raw);
        setSharedStateError("");
        
        // Update agent statuses from shared state
        if (state.agentStatuses) {
          Object.entries(state.agentStatuses).forEach(([key, status]) => {
            if (activeAgents[key]) {
              updateAgentStatus(key, status, status === "busy" ? "Processing..." : "Ready");
            }
          });
        }
        
        // Update logs from trace events
        if (state.traceEvents && state.traceEvents.length > 0) {
          const logLines: LogLine[] = state.traceEvents.map((ev, i) => ({
            ts: new Date(ev.ts).toTimeString().slice(0, 8),
            lvl: ev.type === "error" ? "error" : ev.type === "tool_call" ? "warn" : "info",
            msg: ev.text ? `[${ev.author}] ${ev.text.slice(0, 100)}${ev.text.length > 100 ? "..." : ""}` : `[${ev.author}] ${ev.type}`,
          }));
          setLogs(logLines);
        }
        
        // Update final answer
        if (state.guardianAnswer) {
          setGuardianAnswer(state.guardianAnswer);
          setFinalAnswer(state.guardianAnswer);
          setFinalAnswerAuthor("Guardian");
        }

        if (state.artifacts) {
          setArtifacts(state.artifacts);
        }
        
        // Update processing state
        if (state.isProcessing !== undefined) {
          setIsProcessing(state.isProcessing);
        }
      } catch (err) {
        console.error("Failed to read shared state:", err);
      }
    };
    
    // Poll immediately and then every 500ms
    pollSharedState();
    const interval = setInterval(pollSharedState, 500);
    
    return () => clearInterval(interval);
  }, [isMonitoringMode, SHARED_STATE_KEY]);

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
      script.onerror = () => reject(new Error("Failed to load Plotly"))
      document.head.appendChild(script);
    });
  };

  const renderPlotlyArtifacts = async () => {
    if (typeof window === "undefined") return;
    const nodes = Array.from(document.querySelectorAll("[data-plotly='1']")) as HTMLDivElement[];
    if (nodes.length === 0) return;
    await ensurePlotly();
    if (!window.Plotly) return;

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
        await window.Plotly.react(el, data, layout, config);
      } catch {
        // ignore
      }
    }
  };

  useEffect(() => {
    void renderPlotlyArtifacts();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [artifacts]);

  const results: ResultItem[] = useMemo(() => [], []);
  const active = null;

  // Convert activeAgents state to array for rendering
  const agentsList = useMemo(() => Object.values(activeAgents), [activeAgents]);

  const shards = useMemo(() => [] as Array<{ id: string; gpu: number; status: ShardState }>, []);

  // Helper to update agent status
  const updateAgentStatus = (agentKey: string, status: AgentState, detail: string) => {
    setActiveAgents(prev => ({
      ...prev,
      [agentKey]: {
        ...prev[agentKey],
        status,
        detail,
        lastUpdate: Date.now(),
      }
    }));
  };

  // Helper to map author name to agent key
  const getAgentKey = (author: string): string | null => {
    const lower = author.toLowerCase();
    if (lower.includes("planner")) return "Planner";
    if (lower.includes("guardian_pre") || lower.includes("pre-guardian") || lower.includes("pre guardian")) return null; // Skip pre-guardian in UI
    if (lower.includes("guardian") || lower.includes("valid") || lower.includes("verif")) return "Guardian";
    if (lower.includes("scrap") || lower.includes("search")) return "Scraper";
    if (lower.includes("data")) return "Data";
    if (lower.includes("video")) return "Video";
    if (lower.includes("rag") || lower.includes("knowledge")) return "RAG";
    // Aggregator detection
    if (lower.includes("aggregat") || lower.includes("gather") || 
        lower.includes("root_agent") || lower.includes("research_pipeline") ||
        lower.includes("lead") || lower.includes("analyst")) return "Aggregator";
    return null;
  };

  const appendLog = (lvl: LogLine["lvl"], msg: string) => {
    const ts = new Date().toTimeString().slice(0, 8);
    setLogs((prev) => {
      const next = prev.concat([{ ts, lvl, msg }]);
      return next.length > 500 ? next.slice(next.length - 500) : next;
    });
  };

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch("/api/adk/list-apps", { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as unknown;
        if (cancelled) return;
        const list = Array.isArray(data) ? (data as string[]) : [];
        setApps(list);
        if (list.length > 0 && !appName) setAppName(list[0]!);
      } catch {
        // ignore
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [appName]);

  // Create session on component mount or when appName changes
  // If sharedSessionId is provided via URL, use that instead (monitoring mode)
  useEffect(() => {
    let cancelled = false;
    
    const initSession = async () => {
      if (!appName || !userId) return;
      
      // If we have a shared session ID from URL, use it (monitoring mode)
      if (sharedSessionId) {
        appendLog("info", `🔗 Monitoring shared session: ${sharedSessionId}`);
        setSessionId(sharedSessionId);
        setSessionReady(true);
        return;
      }
      
      // Otherwise create a new session
      try {
        appendLog("info", `Creating session for ${appName}...`);
        const res = await fetch("/api/adk/sessions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ appName, userId }),
          cache: "no-store",
        });
        
        if (cancelled) return;
        
        if (res.ok) {
          const data = await res.json();
          const newSessionId = data.id;
          setSessionId(newSessionId);
          setSessionReady(true);
          appendLog("info", `Session created: ${newSessionId}`);
        } else {
          const text = await res.text();
          appendLog("error", `Failed to create session: ${text}`);
        }
      } catch (err: any) {
        if (!cancelled) {
          appendLog("error", `Session error: ${String(err?.message ?? err)}`);
        }
      }
    };
    
    initSession();
    return () => {
      cancelled = true;
    };
  }, [appName, userId, sharedSessionId]);

  // Auto-fetch KB stats on mount
  useEffect(() => {
    const fetchKbStats = async () => {
      try {
        const res = await fetch("/api/adk/kb-stats-direct", { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          setKbStatsData({
            totalDocs: data.totalDocs ?? 0,
            collection: data.collection ?? "research_knowledge_base",
            model: data.model ?? "all-MiniLM-L6-v2",
            byType: data.byType ?? {}
          });
        }
      } catch {
        // Silently fail on initial load - user can click refresh
      }
    };
    fetchKbStats();
  }, []);

  const readSse = async (res: Response) => {
    const reader = res.body?.getReader();
    if (!reader) {
      appendLog("error", "No SSE stream body returned.");
      setIsProcessing(false);
      return;
    }

    const decoder = new TextDecoder();
    let buf = "";
    // Reset buffers for new request
    agentBufferRef.current = {};
    setAgentMessages({});

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const events = buf.split("\n\n");
      buf = events.pop() ?? "";

      for (const chunk of events) {
        const dataLines = chunk
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice("data:".length).trimStart());
        if (dataLines.length === 0) continue;
        const dataStr = dataLines.join("\n");

        try {
          const evt = JSON.parse(dataStr) as any;
          const text: string | undefined = evt?.content?.parts?.[0]?.text;
          const author: string = String(evt?.author ?? "agent");
          const agentKey = getAgentKey(author);

          if (evt?.error) {
            const msg = String(evt.error);
            appendLog("error", `${author}: ${msg}`);
            setFinalAnswer(msg);
            setFinalAnswerAuthor(author);
            setIsProcessing(false);
            return;
          }

          // Update agent status to active/busy
          if (agentKey) {
            updateAgentStatus(agentKey, "busy", "Processing...");
          }

          if (typeof text === "string" && text.trim()) {
            // Buffer tokens per agent instead of logging each one
            if (agentKey) {
              agentBufferRef.current[agentKey] = (agentBufferRef.current[agentKey] || "") + text;
              // Update the accumulated message display
              setAgentMessages(prev => ({
                ...prev,
                [agentKey]: agentBufferRef.current[agentKey] || ""
              }));
            }

            // Check for final response from Guardian (post-validation)
            const isGuardian = agentKey === "Guardian";
            if (evt?.is_final_response || isGuardian) {
              const finalText = isGuardian && agentBufferRef.current["Guardian"] 
                ? agentBufferRef.current["Guardian"] 
                : text;
              setFinalAnswer(finalText);
              setFinalAnswerAuthor(isGuardian ? "Guardian" : author);
              if (isGuardian) {
                setGuardianAnswer(finalText);
              }
              finalAnswerSetRef.current = true;
              if (evt?.is_final_response) {
                appendLog("info", `✅ Final answer from ${isGuardian ? "Guardian (validated)" : author}`);
              }
            }
          } else if (evt?.error) {
            appendLog("error", `${author}: ${String(evt.error)}`);
            if (agentKey) {
              updateAgentStatus(agentKey, "idle", `Error: ${String(evt.error).slice(0, 30)}`);
            }
          }
          // Don't log empty events
        } catch {
          // Only log parse errors if they contain useful info
          if (dataStr.length > 10) {
            appendLog("warn", `Parse: ${dataStr.slice(0, 100)}`);
          }
        }
      }
    }

    // Set aggregator to processing when we start receiving its responses
    if (agentBufferRef.current["Aggregator"]) {
      updateAgentStatus("Aggregator", "busy", "Synthesizing results...");
    }

    // Flush all buffered messages to log at end
    Object.entries(agentBufferRef.current).forEach(([agent, message]) => {
      if (message.trim()) {
        const preview = message.length > 200 ? message.slice(0, 200) + "..." : message;
        appendLog("info", `${agent}: ${preview}`);
      }
    });

    // If no final answer was set, use guardian's response or the longest response
    if (!finalAnswerSetRef.current && Object.keys(agentBufferRef.current).length > 0) {
      // Prefer guardian response (validated)
      if (agentBufferRef.current["Guardian"]) {
        setFinalAnswer(agentBufferRef.current["Guardian"]);
        setFinalAnswerAuthor("Guardian");
        setGuardianAnswer(agentBufferRef.current["Guardian"]);
      } else if (agentBufferRef.current["Aggregator"]) {
        setFinalAnswer(agentBufferRef.current["Aggregator"]);
        setFinalAnswerAuthor("Aggregator");
      } else {
        // Use the longest response as fallback
        let longestAgent = "";
        let longestLen = 0;
        Object.entries(agentBufferRef.current).forEach(([agent, msg]) => {
          if (msg.length > longestLen) {
            longestLen = msg.length;
            longestAgent = agent;
          }
        });
        if (longestAgent && agentBufferRef.current[longestAgent]) {
          setFinalAnswer(agentBufferRef.current[longestAgent]);
          setFinalAnswerAuthor(longestAgent);
        }
      }
    }

    setIsProcessing(false);
    // Reset agents to idle after completion
    Object.keys(activeAgents).forEach(key => {
      updateAgentStatus(key, "idle", "Ready");
    });
    // Clear live agent messages after a short delay to show final answer clearly
    setTimeout(() => {
      setAgentMessages({});
    }, 2000);
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!streaming) {
      appendLog("warn", "Streaming is OFF. Turn Streaming ON to run.");
      return;
    }

    if (!sessionReady) {
      appendLog("warn", "Session not ready. Please wait for session initialization.");
      return;
    }

    // Reset state for new query
    setIsProcessing(true);
    setFinalAnswer("");
    setFinalAnswerAuthor("");
    setGuardianAnswer("");
    setAgentMessages({});
    agentBufferRef.current = {};
    finalAnswerSetRef.current = false;
    
    // Set Planner to active
    updateAgentStatus("Planner", "active", "Planning query...");
    
    appendLog("info", `🔍 Query: "${query}"`);

    const payload = {
      appName,
      userId,
      sessionId,
      newMessage: {
        role: "user",
        parts: [{ text: query }],
      } satisfies AdkMessage,
      streaming: true,
    };

    void (async () => {
      try {
        const res = await fetch("/api/adk/run-sse", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          appendLog("error", `ADK run failed (${res.status})`);
          const t = await res.text();
          if (t) appendLog("error", t);
          setIsProcessing(false);
          return;
        }
        await readSse(res);
      } catch (err: any) {
        appendLog("error", `ADK run error: ${String(err?.message ?? err)}`);
      }
    })();
  };

  return (
    <div className="min-h-screen bg-[#030508] text-white">
      {/* Background gradients matching landing page */}
      <div className="pointer-events-none fixed inset-0 bg-gradient-to-br from-cyan-500/[0.07] via-transparent to-purple-500/[0.07]" />
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgba(56,189,248,0.1),transparent_50%)]" />

      <header className="relative mx-auto w-full max-w-7xl px-6 pt-8 pb-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div className="flex items-center gap-4">
            <a href="/" className="flex items-center gap-3 group">
              <div className="relative w-10 h-10">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-xl opacity-20 group-hover:opacity-30 transition-opacity" />
                <div className="absolute inset-[2px] bg-[#030508] rounded-[10px] flex items-center justify-center">
                  <span className="text-cyan-400 text-lg">◈</span>
                </div>
              </div>
            </a>
            <div>
              <div className="text-[10px] uppercase tracking-[0.3em] text-cyan-400/60">
                {isMonitoringMode ? "🔴 Monitoring Mode" : "Mission Control"}
              </div>
              <h1 className="text-xl font-semibold tracking-tight">
                <span className="text-white">Multi‑Agent</span>
                <span className="text-cyan-400 ml-1">Workspace</span>
              </h1>
              {isMonitoringMode && (
                <div className="text-xs text-white/50 mt-1 font-mono truncate max-w-[300px]">
                  Session: {sharedSessionId}
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Link
              href="/dashboard"
              className="rounded-xl bg-white/[0.02] px-4 py-2 text-sm font-medium text-white/80 ring-1 ring-white/10 transition hover:bg-white/10"
            >
              ← Back to Simple View
            </Link>
            <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-3 py-2">
              <div className="text-[10px] uppercase tracking-widest text-white/50">
                App
              </div>
              <select
                value={appName}
                onChange={(e) => setAppName(e.target.value)}
                className="mt-1 w-[210px] rounded-lg border border-white/[0.06] bg-black/30 px-2 py-1 text-xs text-white/85 outline-none"
              >
                {(apps.length ? apps : [appName]).map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </div>

            <button
              type="button"
              onClick={() => setStreaming((v) => !v)}
              className={cn(
                "rounded-xl px-4 py-2 text-sm font-medium ring-1 ring-white/10 transition",
                streaming
                  ? "bg-emerald-500/15 text-emerald-100 hover:bg-emerald-500/20"
                  : "bg-white/[0.02] text-white/80 hover:bg-white/10",
              )}
            >
              {streaming ? "Streaming: ON" : "Streaming: OFF"}
            </button>
            <button
              type="button"
              onClick={() => setLogs([])}
              className="rounded-xl bg-white/[0.02] px-4 py-2 text-sm font-medium text-white/80 ring-1 ring-white/10 transition hover:bg-white/10"
            >
              Clear Logs
            </button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto w-full max-w-7xl px-6 pb-10">
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-6">
            <Panel title={isMonitoringMode ? "🔴 Monitoring Active Session" : "Query Interface"}>
              {isMonitoringMode ? (
                <div className="space-y-4">
                  <div className="rounded-xl border border-orange-500/20 bg-orange-500/10 p-4">
                    <div className="flex items-center gap-2 text-sm text-orange-200">
                      <span className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                      Monitoring session from Simple Dashboard
                    </div>
                    <div className="mt-2 text-xs text-white/60">
                      Session ID: <span className="font-mono text-white/80">{sessionId}</span>
                    </div>
                  </div>
              {isMonitoringMode && isLoadingSharedState && (
                <div className="flex items-center gap-2 text-xs text-orange-200">
                  <span className="w-2 h-2 bg-orange-400 rounded-full animate-pulse" />
                  Loading shared state...
                </div>
              )}
              {isMonitoringMode && sharedStateError && (
                <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-200">
                  {sharedStateError}
                </div>
              )}
                </div>
              ) : (
                <form onSubmit={onSubmit} className="space-y-4">
                  <div className="flex flex-col gap-3 sm:flex-row">
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Enter your research query..."
                      className="w-full flex-1 rounded-xl border border-white/[0.06] bg-black/30 px-4 py-3 text-sm text-white/90 placeholder:text-white/40 outline-none focus:border-cyan-400/40 transition-colors"
                    />
                    <button
                      type="submit"
                      disabled={isProcessing || !sessionReady}
                      className="rounded-xl bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-cyan-500/20 transition hover:shadow-cyan-500/30 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isProcessing ? "Processing..." : "Run"}
                    </button>
                  </div>
                </form>
              )}
            </Panel>

            <Panel title="Video Transcript (if available)">
              <div className="rounded-xl border border-white/[0.06] bg-black/30 p-4">
                <div className="text-xs font-semibold tracking-wide text-white/70">
                  Transcript
                </div>
                <div className="mt-2 space-y-2 text-sm leading-6 text-white/80">
                  {agentMessages["Video"] ? (
                    <div className="whitespace-pre-wrap">{agentMessages["Video"].slice(0, 500)}{agentMessages["Video"].length > 500 ? "..." : ""}</div>
                  ) : (
                    <div className="text-white/50">Video transcript will appear here when available.</div>
                  )}
                </div>
              </div>
            </Panel>
            
            {/* Final Answer Panel - Prominent display */}
            <Panel title="✅ Task Complete (Guardian Validated)">
              <div className="space-y-4">
                {isProcessing ? (
                  <div className="flex items-center gap-3 rounded-xl border border-sky-500/30 bg-sky-500/10 p-4">
                    <div className="h-5 w-5 animate-spin rounded-full border-2 border-sky-400 border-t-transparent" />
                    <span className="text-sm text-sky-200">Agents are processing your query...</span>
                  </div>
                ) : guardianAnswer ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-xs text-white/50">
                      <span className="rounded-md bg-emerald-500/20 px-2 py-0.5 text-emerald-200">
                        Guardian
                      </span>
                      <span>Validated Final Output</span>
                    </div>
                    <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-sm leading-relaxed text-white/90 whitespace-pre-wrap max-h-[600px] overflow-y-auto">
                      {formatContent(guardianAnswer)}
                    </div>
                  </div>
                ) : finalAnswer ? (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-xs text-white/50">
                      <span className="rounded-md bg-amber-500/20 px-2 py-0.5 text-amber-200">
                        {finalAnswerAuthor || "System"}
                      </span>
                      <span>Response (Pending Validation)</span>
                    </div>
                    <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-sm leading-relaxed text-white/90 whitespace-pre-wrap max-h-[600px] overflow-y-auto">
                      {formatContent(finalAnswer)}
                    </div>
                  </div>
                ) : (
                  <div className="rounded-xl border border-white/[0.06] bg-black/20 p-4 text-sm text-white/50 text-center">
                    Submit a query to see the final answer here
                  </div>
                )}

                {/* Agent Response Previews */}
                {Object.keys(agentMessages).length > 0 && (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold text-white/50">Agent Responses (Live)</div>
                    <div className="grid gap-2">
                      {Object.entries(agentMessages).map(([agent, msg]) => (
                        <div key={agent} className="rounded-lg border border-white/[0.06] bg-black/20 p-3">
                          <div className="text-xs font-medium text-white/70 mb-1">{agent}</div>

            {artifacts.length > 0 && (
              <Panel title="Artifacts">
                <div className="space-y-6">
                  {artifacts.map((a, idx) => {
                    if (a.kind === "table") {
                      return (
                        <div key={idx} className="space-y-2">
                          {a.title && <div className="text-sm font-semibold text-white/80">{a.title}</div>}
                          <div className="overflow-auto rounded-xl border border-white/10 bg-black/20">
                            <table className="min-w-full text-xs text-white/80">
                              <thead className="sticky top-0 bg-black/40">
                                <tr>
                                  {a.columns.map(c => (
                                    <th key={c} className="px-3 py-2 text-left font-semibold text-white/70 border-b border-white/10 whitespace-nowrap">
                                      {c}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {a.rows.map((r, rIdx) => (
                                  <tr key={rIdx} className="border-b border-white/5 last:border-b-0">
                                    {a.columns.map(c => (
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
                    return null;
                  })}
                </div>
              </Panel>
            )}
                          <div className="text-xs text-white/60 line-clamp-3">
                            {msg.slice(0, 300)}{msg.length > 300 ? "..." : ""}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Panel>
          </div>

          <div className="space-y-6">
            <Panel title="System Command Center (Agent Monitoring)">
              <div className="grid gap-6">
                <div>
                  <div className="mb-3 text-xs font-semibold tracking-wide text-white/70">
                    7-Agent Pipeline
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {agentsList.map((a) => (
                      <div
                        key={a.name}
                        className={cn(
                          "rounded-xl border p-4 transition-all",
                          a.status === "busy" 
                            ? "border-orange-500/30 bg-orange-500/10" 
                            : a.status === "active"
                            ? "border-emerald-500/30 bg-emerald-500/10"
                            : "border-white/[0.06] bg-black/20"
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="text-sm font-semibold text-white/90">
                            {a.name}
                          </div>
                          <Pill status={a.status} />
                        </div>
                        <div className="mt-2 text-xs text-white/60">{a.detail}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                    <div className="text-xs font-semibold tracking-wide text-white/70">
                      Live Agent Logs
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setStreaming(true)}
                        className="rounded-lg bg-emerald-500/15 px-3 py-1.5 text-xs font-medium text-emerald-100 ring-1 ring-emerald-500/30 transition hover:bg-emerald-500/20"
                      >
                        Run
                      </button>
                      <button
                        type="button"
                        onClick={() => setStreaming(false)}
                        className="rounded-lg bg-orange-500/15 px-3 py-1.5 text-xs font-medium text-orange-100 ring-1 ring-orange-500/30 transition hover:bg-orange-500/20"
                      >
                        Pause
                      </button>
                    </div>
                  </div>
                  <Logs lines={logs} />
                </div>
              </div>
            </Panel>

            <Panel title="Knowledge Base">
              <div className="space-y-5">
                {/* Stats Section */}
                <div>
                  <div className="mb-3 flex items-center justify-between">
                    <div className="text-xs font-semibold text-white/70">Knowledge Base Stats</div>
                    <button
                      type="button"
                      disabled={kbStatsLoading}
                      className="rounded-lg bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-white/80 ring-1 ring-white/10 transition hover:bg-white/10 disabled:opacity-50"
                      onClick={async () => {
                        setKbStatsLoading(true);
                        setKbError("");
                        try {
                          // Try direct stats endpoint first (faster)
                          const res = await fetch("/api/adk/kb-stats-direct", { cache: "no-store" });
                          if (res.ok) {
                            const data = await res.json();
                            setKbStatsData({
                              totalDocs: data.totalDocs ?? 0,
                              collection: data.collection ?? "research_knowledge_base",
                              model: data.model ?? "all-MiniLM-L6-v2",
                              byType: data.byType ?? {}
                            });
                            if (data.error) {
                              setKbError(data.error);
                            }
                          } else {
                            setKbError("Failed to fetch stats");
                          }
                        } catch (e: any) {
                          setKbError(`Stats error: ${String(e?.message ?? e)}`);
                        } finally {
                          setKbStatsLoading(false);
                        }
                      }}
                    >
                      {kbStatsLoading ? "Loading..." : "Refresh Stats"}
                    </button>
                    <button
                      type="button"
                      className="rounded-lg bg-purple-500/15 px-3 py-1.5 text-xs font-medium text-purple-100 ring-1 ring-purple-500/30 transition hover:bg-purple-500/20"
                      onClick={async () => {
                        setKbDocsLoading(true);
                        try {
                          const res = await fetch("/api/adk/kb-documents", { cache: "no-store" });
                          if (res.ok) {
                            const data = await res.json();
                            setKbDocuments(data.documents || []);
                            setShowKbDocs(true);
                          }
                        } catch (e) {
                          console.error("Failed to fetch documents", e);
                        } finally {
                          setKbDocsLoading(false);
                        }
                      }}
                    >
                      {kbDocsLoading ? "Loading..." : "View All Docs"}
                    </button>
                    <button
                      type="button"
                      className="rounded-lg bg-emerald-500/15 px-3 py-1.5 text-xs font-medium text-emerald-100 ring-1 ring-emerald-500/30 transition hover:bg-emerald-500/20"
                      onClick={() => setShowUploadModal(true)}
                    >
                      📄 Upload Doc
                    </button>
                  </div>
                  <KbStatsCard stats={kbStatsData} />
                </div>

                {/* Search Section */}
                <div>
                  <div className="mb-3 text-xs font-semibold text-white/70">Semantic Search</div>
                  <form 
                    onSubmit={async (e) => {
                      e.preventDefault();
                      if (!kbQuery.trim()) return;
                      setKbSearching(true);
                      setKbError("");
                      try {
                        const res = await fetch("/api/adk/kb-search", {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({ query: kbQuery }),
                        });
                        const text = await res.text();
                        
                        // Parse results from the response - format: **[1] TYPE - source** (Relevance: XX%)
                        const docs: KbDocument[] = [];
                        const docRegex = /\*\*\[(\d+)\]\s*(\w+)\s*-\s*([^\*]+?)\*\*(?:\s*\(Relevance:\s*(\d+)%\))?[^\n]*\n([^]*?)(?=\n\n\*\*\[|$)/g;
                        let match;
                        while ((match = docRegex.exec(text)) !== null) {
                          const relevance = match[4] ? parseInt(match[4]) / 100 : 1 - (parseInt(match[1]) - 1) * 0.1;
                          docs.push({
                            content: match[5].trim(),
                            source: match[3].trim(),
                            type: match[2].toLowerCase() as any,
                            score: relevance
                          });
                        }
                        
                        if (docs.length === 0 && text.includes("empty")) {
                          setKbError("Knowledge base is empty. Run some searches to build it!");
                        } else if (docs.length === 0) {
                          setKbError("No relevant documents found. Try a different query.");
                        }
                        
                        setKbResults(docs);
                      } catch (e: any) {
                        setKbError(`Search error: ${String(e?.message ?? e)}`);
                        setKbResults([]);
                      } finally {
                        setKbSearching(false);
                      }
                    }}
                    className="flex gap-2"
                  >
                    <input
                      value={kbQuery}
                      onChange={(e) => setKbQuery(e.target.value)}
                      placeholder="Search stored knowledge semantically..."
                      className="w-full rounded-xl border border-white/[0.06] bg-black/35 px-4 py-2.5 text-sm text-white/90 placeholder:text-white/40 outline-none focus:border-sky-400/40"
                    />
                    <button
                      type="submit"
                      disabled={kbSearching}
                      className="rounded-xl bg-sky-500/20 px-5 py-2.5 text-sm font-semibold text-sky-100 ring-1 ring-sky-500/30 transition hover:bg-sky-500/25 disabled:opacity-50"
                    >
                      {kbSearching ? "..." : "Search"}
                    </button>
                  </form>
                </div>

                {/* Error Display */}
                {kbError && (
                  <div className="rounded-xl border border-orange-500/30 bg-orange-500/10 px-4 py-3 text-sm text-orange-200">
                    {kbError}
                  </div>
                )}

                {/* Results as Source Cards */}
                {kbResults.length > 0 && (
                  <div>
                    <div className="mb-3 flex items-center justify-between">
                      <div className="text-xs font-semibold text-white/70">
                        Found {kbResults.length} Relevant Documents
                      </div>
                      <button
                        type="button"
                        onClick={() => setKbResults([])}
                        className="text-xs text-white/50 hover:text-white/70"
                      >
                        Clear
                      </button>
                    </div>
                    <div className="space-y-3 max-h-[400px] overflow-auto">
                      {kbResults.map((doc, i) => (
                        <SourceCard key={`${doc.source}-${i}`} doc={doc} index={i + 1} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </Panel>
          </div>
        </div>
      </main>

      {/* KB Documents Modal */}
      {showKbDocs && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-4xl max-h-[80vh] rounded-2xl border border-white/[0.06] bg-[#0a0f1a] shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
              <div>
                <h2 className="text-lg font-semibold text-white/90">📚 Knowledge Base Documents</h2>
                <p className="text-xs text-white/50 mt-1">{kbDocuments.length} documents stored</p>
              </div>
              <button
                onClick={() => setShowKbDocs(false)}
                className="rounded-lg bg-white/10 px-3 py-1.5 text-sm text-white/70 hover:bg-white/20 transition"
              >
                ✕ Close
              </button>
            </div>
            
            <div className="overflow-auto max-h-[calc(80vh-80px)] p-4">
              {kbDocuments.length === 0 ? (
                <div className="text-center py-10 text-white/50">
                  No documents in knowledge base yet.
                </div>
              ) : (
                <div className="space-y-3">
                  {kbDocuments.map((doc, i) => (
                    <div 
                      key={doc.id + i} 
                      className="rounded-xl border border-white/[0.06] bg-black/30 p-4 hover:border-white/20 transition"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-center gap-2">
                          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-white/10 text-xs font-bold text-white/70">
                            {i + 1}
                          </span>
                          <SourceTypeBadge type={doc.type} />
                          <span className="text-xs text-white/40">{doc.length} chars</span>
                        </div>
                        <span className="text-xs text-white/40">{doc.timestamp}</span>
                      </div>
                      
                      <div className="mt-2">
                        <a 
                          href={doc.source.startsWith("http") ? doc.source : undefined}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm font-medium text-sky-300 hover:text-sky-200 hover:underline break-all"
                          title={doc.source}
                        >
                          {doc.source.length > 80 ? doc.source.slice(0, 80) + "..." : doc.source}
                        </a>
                      </div>
                      
                      <div className="mt-2 text-sm text-white/70 whitespace-pre-wrap">
                        {doc.preview}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Upload Document Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-lg rounded-2xl border border-white/[0.06] bg-[#0a0f1a] shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
              <div>
                <h2 className="text-lg font-semibold text-white/90">📄 Upload Document</h2>
                <p className="text-xs text-white/50 mt-1">PDF, TXT, MD, DOCX supported</p>
              </div>
              <button
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadStatus({loading: false, message: ""});
                }}
                className="rounded-lg bg-white/10 px-3 py-1.5 text-sm text-white/70 hover:bg-white/20 transition"
              >
                ✕ Close
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              <div 
                className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center hover:border-white/40 transition cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.md,.markdown,.docx"
                  className="hidden"
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    
                    setUploadStatus({loading: true, message: `Uploading ${file.name}...`});
                    
                    try {
                      const formData = new FormData();
                      formData.append("file", file);
                      formData.append("chunk_size", "500");
                      formData.append("overlap", "100");
                      
                      const res = await fetch("/api/adk/upload-document", {
                        method: "POST",
                        body: formData,
                      });
                      
                      const data = await res.json();
                      
                      if (data.success) {
                        setUploadStatus({
                          loading: false, 
                          message: `✅ Successfully uploaded "${data.filename}"\n${data.chunks_created} chunks created from ${data.total_chars} characters`,
                          success: true
                        });
                        // Refresh stats
                        const statsRes = await fetch("/api/adk/kb-stats-direct", { cache: "no-store" });
                        if (statsRes.ok) {
                          const stats = await statsRes.json();
                          setKbStatsData({
                            totalDocs: stats.totalDocs ?? 0,
                            collection: stats.collection ?? "research_knowledge_base",
                            model: stats.model ?? "all-MiniLM-L6-v2",
                            byType: stats.byType ?? {}
                          });
                        }
                      } else {
                        setUploadStatus({
                          loading: false,
                          message: `❌ Upload failed: ${data.error}`,
                          success: false
                        });
                      }
                    } catch (err: any) {
                      setUploadStatus({
                        loading: false,
                        message: `❌ Error: ${err?.message || "Upload failed"}`,
                        success: false
                      });
                    }
                    
                    // Reset file input
                    e.target.value = "";
                  }}
                />
                
                <div className="text-4xl mb-3">📁</div>
                <div className="text-sm text-white/70">
                  Click to select a file or drag & drop
                </div>
                <div className="text-xs text-white/40 mt-2">
                  Max file size: 10MB
                </div>
              </div>
              
              {uploadStatus.message && (
                <div className={cn(
                  "rounded-xl p-4 text-sm whitespace-pre-wrap",
                  uploadStatus.loading ? "bg-sky-500/10 border border-sky-500/30 text-sky-200" :
                  uploadStatus.success ? "bg-emerald-500/10 border border-emerald-500/30 text-emerald-200" :
                  "bg-red-500/10 border border-red-500/30 text-red-200"
                )}>
                  {uploadStatus.loading && (
                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-sky-400 border-t-transparent mr-2" />
                  )}
                  {uploadStatus.message}
                </div>
              )}
              
              <div className="text-xs text-white/40 space-y-1">
                <div>📋 <strong>How it works:</strong></div>
                <div>1. Document text is extracted</div>
                <div>2. Split into ~500 character chunks with overlap</div>
                <div>3. Each chunk is embedded and stored in ChromaDB</div>
                <div>4. Chunks are searchable via semantic search</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
