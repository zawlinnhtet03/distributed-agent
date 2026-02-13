"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";

type KbStats = {
  totalDocs: number;
  collection: string;
  model: string;
  byType: Record<string, number>;
};

type KbDocument = {
  id: string;
  type: string;
  source: string;
  timestamp: string;
  preview: string;
  length: number;
};

function cn(...v: Array<string | false | null | undefined>) {
  return v.filter(Boolean).join(" ");
}

function TypeBadge({ type }: { type: string }) {
  const config: Record<string, { bg: string; icon: string }> = {
    web: { bg: "bg-blue-500/20 text-blue-300 border-blue-500/30", icon: "🌐" },
    video: { bg: "bg-purple-500/20 text-purple-300 border-purple-500/30", icon: "🎬" },
    pdf: { bg: "bg-red-500/20 text-red-300 border-red-500/30", icon: "📄" },
    note: { bg: "bg-yellow-500/20 text-yellow-300 border-yellow-500/30", icon: "📝" },
  };
  const c = config[type] ?? config.web;
  return (
    <span className={cn("rounded-md px-2 py-0.5 text-xs font-medium border", c.bg)}>
      {c.icon} {type.toUpperCase()}
    </span>
  );
}

export default function KnowledgeBasePage() {
  const [stats, setStats] = useState<KbStats | null>(null);
  const [documents, setDocuments] = useState<KbDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searching, setSearching] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{ loading: boolean; message: string; success?: boolean }>({ loading: false, message: "" });
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch stats and documents
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, docsRes] = await Promise.all([
          fetch("/api/adk/kb-stats-direct"),
          fetch("http://localhost:8003/documents"),
        ]);
        
        if (statsRes.ok) {
          const data = await statsRes.json();
          setStats(data);
        }
        
        if (docsRes.ok) {
          const data = await docsRes.json();
          setDocuments(data.documents || []);
        }
      } catch (err) {
        console.error("Failed to fetch KB data:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setSearching(true);
    try {
      const res = await fetch("/api/adk/kb-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, nResults: 10 }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results || []);
      }
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setSearching(false);
    }
  };

  const handleUpload = async (file: File) => {
    setUploadStatus({ loading: true, message: "Uploading..." });
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("chunk_size", "500");
    formData.append("overlap", "100");

    try {
      const res = await fetch("/api/adk/upload-document", {
        method: "POST",
        body: formData,
      });
      
      const data = await res.json();
      
      if (res.ok && data.success) {
        setUploadStatus({ loading: false, message: `✓ Added ${data.chunks_created} chunks from ${data.filename}`, success: true });
        // Refresh documents
        const docsRes = await fetch("http://localhost:8003/documents");
        if (docsRes.ok) {
          const docsData = await docsRes.json();
          setDocuments(docsData.documents || []);
        }
        // Refresh stats
        const statsRes = await fetch("/api/adk/kb-stats-direct");
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          setStats(statsData);
        }
      } else {
        setUploadStatus({ loading: false, message: data.error || "Upload failed", success: false });
      }
    } catch (err) {
      setUploadStatus({ loading: false, message: "Connection error", success: false });
    }
  };

  return (
    <div className="min-h-screen bg-[#030508] text-white">
      {/* Background */}
      <div className="pointer-events-none fixed inset-0 bg-gradient-to-br from-yellow-500/[0.05] via-transparent to-purple-500/[0.05]" />

      {/* Header */}
      <header className="relative border-b border-white/[0.06] bg-black/20 backdrop-blur-sm">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/dashboard" className="text-white/50 hover:text-white transition">
                ← Back
              </Link>
              <div>
                <h1 className="text-xl font-semibold flex items-center gap-2">
                  <span className="text-2xl">📚</span>
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-400">
                    Knowledge Base
                  </span>
                </h1>
              </div>
            </div>
            
            <button
              onClick={() => setShowUpload(!showUpload)}
              className="rounded-lg bg-gradient-to-r from-yellow-500 to-orange-500 px-4 py-2 text-sm font-medium text-white hover:shadow-lg hover:shadow-yellow-500/25 transition"
            >
              + Upload Document
            </button>
          </div>
        </div>
      </header>

      <main className="relative mx-auto max-w-5xl px-6 py-8">
        {/* Stats Cards */}
        <section className="mb-8 grid gap-4 sm:grid-cols-4">
          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
            <div className="text-3xl font-bold text-white">{stats?.totalDocs ?? 0}</div>
            <div className="text-xs text-white/50 mt-1">Total Documents</div>
          </div>
          {["web", "pdf", "video", "note"].map((type) => (
            <div key={type} className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
              <div className="flex items-center gap-2">
                <TypeBadge type={type} />
                <span className="text-xl font-bold text-white">{stats?.byType?.[type] ?? 0}</span>
              </div>
            </div>
          ))}
        </section>

        {/* Upload Modal */}
        {showUpload && (
          <section className="mb-8 rounded-xl border border-yellow-500/30 bg-yellow-500/5 p-6">
            <h3 className="text-lg font-semibold mb-4">Upload Document</h3>
            <div 
              className="border-2 border-dashed border-white/20 rounded-xl p-8 text-center hover:border-yellow-400/50 transition cursor-pointer"
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.md,.docx"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleUpload(file);
                }}
              />
              <div className="text-4xl mb-3">📁</div>
              <div className="text-white/70 font-medium">Click to upload or drag & drop</div>
              <div className="text-white/40 text-sm mt-1">PDF, TXT, MD, DOCX supported</div>
            </div>
            {uploadStatus.message && (
              <div className={cn(
                "mt-4 rounded-lg p-3 text-sm",
                uploadStatus.success ? "bg-emerald-500/20 text-emerald-300" : uploadStatus.loading ? "bg-yellow-500/20 text-yellow-300" : "bg-red-500/20 text-red-300"
              )}>
                {uploadStatus.message}
              </div>
            )}
          </section>
        )}

        {/* Search */}
        <section className="mb-8">
          <form onSubmit={handleSearch} className="flex gap-3">
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search your knowledge base..."
              className="flex-1 rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-white placeholder:text-white/40 outline-none focus:border-yellow-400/50 transition"
            />
            <button
              type="submit"
              disabled={searching}
              className="rounded-xl bg-white/10 px-6 py-3 font-medium text-white hover:bg-white/15 transition"
            >
              {searching ? "Searching..." : "Search"}
            </button>
          </form>
        </section>

        {/* Search Results */}
        {searchResults.length > 0 && (
          <section className="mb-8">
            <h3 className="text-lg font-semibold mb-4">Search Results</h3>
            <div className="space-y-3">
              {searchResults.map((result, i) => (
                <div key={i} className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <TypeBadge type={result.type || "web"} />
                    <span className="text-xs text-white/50">{Math.round((result.score || 0) * 100)}% match</span>
                  </div>
                  <div className="text-sm text-white/80 line-clamp-3">{result.content}</div>
                  <div className="mt-2 text-xs text-white/40 truncate">{result.source}</div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Documents List */}
        <section>
          <h3 className="text-lg font-semibold mb-4">All Documents ({documents.length})</h3>
          
          {loading ? (
            <div className="text-center py-12 text-white/50">Loading...</div>
          ) : documents.length === 0 ? (
            <div className="rounded-xl border border-dashed border-white/10 p-12 text-center">
              <div className="text-4xl mb-4">📭</div>
              <div className="text-white/50 font-medium">No documents yet</div>
              <div className="text-white/30 text-sm mt-1">Upload documents or run research queries to build your knowledge base</div>
            </div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc) => (
                <div key={doc.id} className="rounded-xl border border-white/10 bg-white/[0.02] p-4 hover:bg-white/[0.04] transition">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <TypeBadge type={doc.type} />
                        <span className="text-xs text-white/40">{doc.length} chars</span>
                      </div>
                      <div className="text-sm text-white/80 line-clamp-2">{doc.preview}</div>
                      <div className="mt-2 text-xs text-white/40 truncate">{doc.source}</div>
                    </div>
                    <div className="text-xs text-white/30">{doc.timestamp}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
