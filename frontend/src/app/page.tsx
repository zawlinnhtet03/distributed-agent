"use client";

import Link from "next/link";
import { useEffect, useState, useRef } from "react";

const agents = [
  {
    name: "Planner Agent",
    role: "Orchestrates missions, decomposes queries, routes tasks.",
    icon: "◈",
    color: "#38bdf8",
    capabilities: ["Task Planning", "Query Routing", "Pipeline Orchestration"],
  },
  {
    name: "Aggregator Agent",
    role: "Synthesizes final report with cross‑source consensus.",
    icon: "◉",
    color: "#ec4899",
    capabilities: ["Report Synthesis", "Consensus Building", "Final Output"],
  },
  {
    name: "Guardian Agent",
    role: "Pre and post-validation of inputs, verifies sources, detects issues.",
    icon: "◇",
    color: "#f43f5e",
    capabilities: ["Input Validation", "Source Verification", "Safety Review"],
  },
  {
    name: "Scraper Agent",
    role: "Live web retrieval, signals, trends, and citations.",
    icon: "◎",
    color: "#10b981",
    capabilities: ["Web Search", "Trend Analysis", "Citation Tracking"],
  },
  {
    name: "RAG Agent",
    role: "Vector memory, semantic recall, long‑term knowledge.",
    icon: "◆",
    color: "#fbbf24",
    capabilities: ["Vector Search", "Semantic Recall", "Knowledge Retrieval"],
  },
  {
    name: "Video Agent",
    role: "Frame intelligence, visual evidence, timeline insights.",
    icon: "◐",
    color: "#a855f7",
    capabilities: ["Video Analysis", "Frame Extraction", "Timeline Parsing"],
  },
  {
    name: "Data Agent",
    role: "EDA, cleaning, and visualization over local datasets.",
    icon: "◑",
    color: "#22c55e",
    capabilities: ["Data Analysis", "Visualization", "CSV Processing"],
  },

 
];

// Neural Particle Network
function NeuralParticles() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    const colors = ["#38bdf8", "#10b981", "#a855f7", "#fbbf24", "#ec4899"];
    const particles = Array.from({ length: 30 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      size: Math.random() * 2.5 + 1,
      color: colors[Math.floor(Math.random() * colors.length)],
    }));

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    window.addEventListener("mousemove", handleMouseMove);

    let animationId: number;
    const animate = () => {
      // Clear canvas completely to remove trails/shadows
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const mouse = mouseRef.current;

      particles.forEach((p, i) => {
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 200) {
          p.vx += dx * 0.00003;
          p.vy += dy * 0.00003;
        }

        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = canvas.width;
        if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height;
        if (p.y > canvas.height) p.y = 0;

        p.vx *= 0.99;
        p.vy *= 0.99;

        particles.forEach((p2, j) => {
          if (i >= j) return;
          const d = Math.hypot(p.x - p2.x, p.y - p2.y);
          if (d < 200) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(56, 189, 248, ${0.3 * (1 - d / 200)})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowBlur = 6;
        ctx.shadowColor = p.color;
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      animationId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouseMove);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return <canvas ref={canvasRef} className="fixed inset-0 z-0 pointer-events-none" />;
}

// Live Stats Counter
function LiveStats() {
  const [stats, setStats] = useState({ queries: 15847, latency: 23, accuracy: 99.7 });

  useEffect(() => {
    const interval = setInterval(() => {
      setStats((prev) => ({
        queries: prev.queries + Math.floor(Math.random() * 3),
        latency: Math.max(15, Math.min(40, prev.latency + (Math.random() - 0.5) * 4)),
        accuracy: Math.max(99.2, Math.min(99.9, prev.accuracy + (Math.random() - 0.5) * 0.05)),
      }));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex gap-8 mt-8">
      <div className="text-center">
        <div className="text-2xl font-bold font-mono text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
          {stats.queries.toLocaleString()}
        </div>
        <div className="text-xs text-white/50 uppercase tracking-wider mt-1">Queries</div>
      </div>
      <div className="w-px bg-white/10" />
      <div className="text-center">
        <div className="text-2xl font-bold font-mono text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-purple-400">
          {stats.latency.toFixed(0)}ms
        </div>
        <div className="text-xs text-white/50 uppercase tracking-wider mt-1">Latency</div>
      </div>
      <div className="w-px bg-white/10" />
      <div className="text-center">
        <div className="text-2xl font-bold font-mono text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
          {stats.accuracy.toFixed(1)}%
        </div>
        <div className="text-xs text-white/50 uppercase tracking-wider mt-1">Accuracy</div>
      </div>
    </div>
  );
}

// Orbiting Agents Visualization - Pure CSS animation to avoid hydration errors
function OrbitingAgents() {
  return (
    <div className="relative w-[400px] h-[400px] flex items-center justify-center">
      {/* Core */}
      <div className="absolute w-24 h-24 rounded-full bg-gradient-to-br from-cyan-500/30 to-purple-500/30 flex items-center justify-center border border-cyan-400/30 shadow-[0_0_80px_rgba(56,189,248,0.5)]">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-cyan-400/40 to-purple-400/40 flex items-center justify-center animate-pulse">
          <span className="text-4xl text-cyan-300 drop-shadow-[0_0_15px_rgba(56,189,248,0.8)]">∞</span>
        </div>
      </div>
      
      {/* Orbital Rings - 2 main paths matching the agent orbits */}
      {/* Inner orbit path (radius 115, scaled to ellipse ~230x104) */}
      <div 
        className="absolute border border-cyan-500/30 animate-spin-slowest" 
        style={{ 
          width: 230, 
          height: 104, 
          borderRadius: '50%',
          animationDuration: '28s',
          boxShadow: '0 0 20px rgba(56,189,248,0.15)'
        }} 
      />
      {/* Outer orbit path (radius 165, scaled to ellipse ~330x149) */}
      <div 
        className="absolute border border-purple-500/30 animate-spin-slower" 
        style={{ 
          width: 330, 
          height: 149, 
          borderRadius: '50%',
          animationDuration: '35s',
          animationDirection: 'reverse',
          boxShadow: '0 0 20px rgba(168,85,247,0.15)'
        }} 
      />

      {/* Inner orbit - 3 agents */}
      <div className="absolute w-full h-full animate-spin-slowest" style={{ animationDuration: '28s' }}>
        {agents.slice(0, 3).map((agent, i) => {
          const angle = (i / 3) * 360;
          const radius = 115;
          const radian = (angle * Math.PI) / 180;
          const x = Math.round(Math.cos(radian) * radius);
          const y = Math.round(Math.sin(radian) * radius * 0.45);

          return (
            <div
              key={agent.name}
              className="absolute left-1/2 top-1/2"
              style={{ transform: `translate(${x - 50}px, ${y - 20}px)`, zIndex: 10 }}
            >
              <div
                className="px-3 py-1.5 rounded-full backdrop-blur-md border-2 flex items-center gap-2 whitespace-nowrap animate-spin-slower"
                style={{
                  background: `${agent.color}30`,
                  borderColor: `${agent.color}80`,
                  boxShadow: `0 0 25px ${agent.color}60`,
                  animationDirection: 'reverse',
                  animationDuration: '28s',
                }}
              >
                <span className="text-base" style={{ color: agent.color, filter: `drop-shadow(0 0 6px ${agent.color})` }}>{agent.icon}</span>
                <span className="text-xs font-semibold text-white">{agent.name.split(" ")[0]}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Outer orbit - 4 agents (opposite direction) */}
      <div className="absolute w-full h-full animate-spin-slower" style={{ animationDuration: '35s', animationDirection: 'reverse' }}>
        {agents.slice(3, 7).map((agent, i) => {
          const angle = (i / 4) * 360 + 60;
          const radius = 165;
          const radian = (angle * Math.PI) / 180;
          const x = Math.round(Math.cos(radian) * radius);
          const y = Math.round(Math.sin(radian) * radius * 0.45);

          return (
            <div
              key={agent.name}
              className="absolute left-1/2 top-1/2"
              style={{ transform: `translate(${x - 50}px, ${y - 20}px)`, zIndex: 10 }}
            >
              <div
                className="px-3 py-1.5 rounded-full backdrop-blur-md border-2 flex items-center gap-2 whitespace-nowrap animate-spin-slow"
                style={{
                  background: `${agent.color}30`,
                  borderColor: `${agent.color}80`,
                  boxShadow: `0 0 25px ${agent.color}60`,
                  animationDuration: '35s',
                }}
              >
                <span className="text-base" style={{ color: agent.color, filter: `drop-shadow(0 0 6px ${agent.color})` }}>{agent.icon}</span>
                <span className="text-xs font-semibold text-white">{agent.name.split(" ")[0]}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Terminal Typing Effect - Animated system info
function TerminalTyping() {
  const [text, setText] = useState("");
  const [lineIndex, setLineIndex] = useState(0);
  const lines = [
    "$ python -m app.api_server",
    "$ uvicorn running on http://localhost:8001",
    "$ 7 agents loaded: Planner, Guardian, Aggregator...",
    "$ Tavily API: Connected",
    "$ ChromaDB: Initialized",
    "$ Ready for queries",
  ];

  useEffect(() => {
    if (lineIndex >= lines.length) {
      setTimeout(() => { setText(""); setLineIndex(0); }, 3000);
      return;
    }

    const currentLine = lines[lineIndex];
    let charIndex = 0;

    const typeInterval = setInterval(() => {
      if (charIndex <= currentLine.length) {
        setText(lines.slice(0, lineIndex).join("\n") + (lineIndex > 0 ? "\n" : "") + currentLine.slice(0, charIndex));
        charIndex++;
      } else {
        clearInterval(typeInterval);
        setTimeout(() => setLineIndex((prev) => prev + 1), 400);
      }
    }, 35);

    return () => clearInterval(typeInterval);
  }, [lineIndex]);

  return (
    <div className="rounded-xl bg-black/60 border border-white/10 overflow-hidden h-full">
      <div className="flex items-center gap-2 px-4 py-2 bg-white/5 border-b border-white/5">
        <div className="w-3 h-3 rounded-full bg-red-500/80" />
        <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
        <div className="w-3 h-3 rounded-full bg-green-500/80" />
        <span className="ml-2 text-xs text-white/40 font-mono">api_server.log</span>
      </div>
      <pre className="p-4 text-sm font-mono text-emerald-400 min-h-[200px]">
        {text}
        <span className="animate-pulse">▋</span>
      </pre>
    </div>
  );
}

// Telemetry Cards - Static capabilities display
function TelemetryCards() {
  const data = [
    { label: "Agent Pipeline", value: "7 Agents", desc: "Planner → Guardian → Aggregator", color: "#38bdf8" },
    { label: "Knowledge Store", value: "ChromaDB", desc: "Vector memory for RAG retrieval", color: "#10b981" },
    { label: "Safety Layer", value: "Pre/Post", desc: "Dual Guardian validation", color: "#f43f5e" },
    { label: "Session Memory", value: "Persistent", desc: "Conversation history per session", color: "#fbbf24" },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {data.map((item) => (
        <div
          key={item.label}
          className="relative p-4 rounded-xl bg-white/[0.02] border border-white/[0.06] overflow-hidden group hover:border-white/10 transition-colors"
        >
          <div className="absolute top-3 right-3 w-2 h-2 rounded-full" style={{ background: item.color }} />
          <div className="text-[10px] uppercase tracking-wider text-white/40 mb-1">{item.label}</div>
          <div className="text-lg font-bold" style={{ color: item.color }}>
            {item.value}
          </div>
          <div className="mt-1 text-[10px] text-white/40">{item.desc}</div>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  return (
    <div className="relative min-h-screen bg-[#030508] text-white overflow-x-hidden">
      {/* Background Effects */}
      <NeuralParticles />
      <div className="fixed inset-0 bg-gradient-to-br from-cyan-500/[0.07] via-transparent to-purple-500/[0.07] pointer-events-none z-[1]" />
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,rgba(56,189,248,0.1),transparent_50%)] pointer-events-none z-[1]" />
      
      {/* Scanlines */}
      <div className="fixed inset-0 pointer-events-none z-[52] opacity-[0.02]" style={{
        backgroundImage: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)"
      }} />

      {/* Header */}
      <header className="relative z-20 mx-auto max-w-7xl px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-400 to-purple-500 rounded-xl opacity-20 animate-pulse" />
            <div className="absolute inset-[2px] bg-[#030508] rounded-[10px] flex items-center justify-center">
              <span className="text-cyan-400 text-lg">◈</span>
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.3em] text-cyan-400/60">Distributed Intelligence</div>
            <div className="text-sm font-semibold">
              <span className="text-white">Multi‑Agent</span>
              <span className="text-cyan-400 ml-1">Workspace</span>
            </div>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-8 text-sm text-white/50">
          <a href="#agents" className="hover:text-white transition-colors">Agents</a>
          <a href="#telemetry" className="hover:text-white transition-colors">Telemetry</a>
          <a href="#architecture" className="hover:text-white transition-colors">Architecture</a>
        </nav>

        <Link
          href="/dashboard"
          className="group flex items-center gap-2 px-5 py-2.5 rounded-full bg-white/5 border border-white/10 text-sm font-medium hover:bg-white/10 hover:border-white/20 transition-all"
        >
          Enter Dashboard
          <svg className="w-4 h-4 transition-transform group-hover:translate-x-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </Link>
      </header>

      {/* Hero Section */}
      <main className="relative z-10">
        <section className="mx-auto max-w-7xl px-6 pt-12 pb-24">
          <div className="grid lg:grid-cols-2 gap-12 items-center min-h-[60vh]">
            {/* Left Column */}
            <div className="space-y-6">
              {/* Status Badge */}
              <div className="inline-flex items-center gap-3 px-4 py-2 rounded-full bg-white/[0.03] border border-white/[0.06]">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400" />
                </span>
                <span className="text-xs font-mono tracking-wide text-white/60">MULTI-AGENT RETRIEVAL STACK</span>
                <span className="text-white/20">•</span>
                <span className="text-xs font-semibold text-emerald-400">ONLINE</span>
              </div>

              {/* Headline */}
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-[1.1] tracking-tight">
                <span className="text-white">The next generation of</span>
                <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-emerald-400 to-purple-400">
                  autonomous agent
                </span>
                <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-400 to-amber-400">
                  infrastructure
                </span>
              </h1>

              {/* Description */}
              <p className="text-lg text-white/60 leading-relaxed max-w-xl">
                Orchestrate web, video, and memory agents into a single research swarm. 
                Real-time evidence streams, semantic recall, and synthesized reports delivered with <span className="text-cyan-400">neural‑grade clarity</span>.
              </p>

              {/* Live Stats */}
              {/* Replaced with capability highlights */}
              <div className="flex flex-wrap gap-6 mt-6">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-sm text-white/70">Pre/Post Guardian Safety</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  <span className="text-sm text-white/70">Multi-Modal Analysis</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-400" />
                  <span className="text-sm text-white/70">Session Memory</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400" />
                  <span className="text-sm text-white/70">Knowledge Base RAG</span>
                </div>
              </div>

              {/* CTA Buttons */}
              <div className="flex flex-wrap items-center gap-4 pt-4">
                <Link
                  href="/dashboard"
                  className="group relative px-6 py-3 rounded-full bg-gradient-to-r from-cyan-500 via-emerald-500 to-purple-500 text-sm font-semibold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all hover:scale-[1.02]"
                >
                  <span className="relative z-10 flex items-center gap-2">
                    <span>◈</span>
                    Launch Mission Control
                  </span>
                </Link>
                <button className="flex items-center gap-3 px-5 py-3 rounded-full border border-white/10 text-sm text-white/70 hover:bg-white/5 hover:text-white transition-all">
                  <span className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center text-xs">▶</span>
                  Watch Demo
                </button>
              </div>

              {/* Feature Pills */}
              <div className="flex flex-wrap gap-3 pt-2">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] text-xs text-white/60">
                  <span className="text-rose-400">◇</span>
                  Pre/Post Guardian
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] text-xs text-white/60">
                  <span className="text-cyan-400">◎</span>
                  Web Research
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] text-xs text-white/60">
                  <span className="text-purple-400">◐</span>
                  Video Analysis
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] text-xs text-white/60">
                  <span className="text-emerald-400">◑</span>
                  Data Analytics
                </div>
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.06] text-xs text-white/60">
                  <span className="text-amber-400">◆</span>
                  Vector Memory
                </div>
              </div>
            </div>

            {/* Right Column - Orbiting Visualization */}
            <div className="flex items-center justify-center">
              <OrbitingAgents />
            </div>
          </div>
        </section>

        {/* Agents Section */}
        <section id="agents" className="mx-auto max-w-7xl px-6 py-20">
          <div className="mb-12">
            <div className="text-xs font-mono text-cyan-400 tracking-wider mb-3">◈ AGENT SWARM</div>
            <h2 className="text-3xl font-bold">
              Meet the <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">collective</span>
            </h2>
            <p className="text-white/50 mt-2">Seven specialized agents working in orchestrated harmony</p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map((agent, i) => (
              <div
                key={agent.name}
                className="group relative p-5 rounded-2xl bg-white/[0.02] border border-white/[0.06] hover:border-white/10 transition-all duration-300 hover:-translate-y-1"
              >
                {/* Glow on hover */}
                <div
                  className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
                  style={{ background: `radial-gradient(circle at 50% 0%, ${agent.color}15, transparent 70%)` }}
                />

                <div className="relative">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-4">
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center text-lg"
                      style={{ background: `${agent.color}15`, color: agent.color }}
                    >
                      {agent.icon}
                    </div>
                    <div className="flex items-center gap-2 text-[10px] uppercase text-white/40">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                      Active
                    </div>
                  </div>

                  {/* Content */}
                  <h3 className="font-semibold text-white/90 mb-2">{agent.name}</h3>
                  <p className="text-sm text-white/50 mb-4 leading-relaxed">{agent.role}</p>

                  {/* Capabilities */}
                  <div className="flex flex-wrap gap-1.5">
                    {agent.capabilities.map((cap) => (
                      <span
                        key={cap}
                        className="px-2 py-0.5 rounded text-[10px] bg-white/[0.05] text-white/60 border border-white/[0.08]"
                      >
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Telemetry Section */}
        <section id="telemetry" className="mx-auto max-w-7xl px-6 py-20">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Telemetry Panel */}
            <div className="p-6 rounded-2xl bg-white/[0.02] border border-white/[0.06]">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center text-cyan-400">
                  ◎
                </div>
                <div>
                  <div className="text-[10px] font-mono text-cyan-400 tracking-wider">LIVE TELEMETRY</div>
                  <div className="font-semibold">System Metrics</div>
                </div>
              </div>
              <TelemetryCards />
            </div>

            {/* Terminal Panel */}
            <div>
              <TerminalTyping />
            </div>
          </div>
        </section>

        {/* Architecture Section */}
        <section id="architecture" className="mx-auto max-w-7xl px-6 py-20">
          <div className="text-center mb-12">
            <div className="text-xs font-mono text-cyan-400 tracking-wider mb-3">◆ ARCHITECTURE</div>
            <h2 className="text-3xl font-bold">
              From intent to <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">insight</span>
            </h2>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { step: "01", title: "Query Intake", desc: "Natural language parsing & intent classification", color: "#38bdf8" },
              { step: "02", title: "Task Decomposition", desc: "Planner breaks down into parallel subtasks", color: "#10b981" },
              { step: "03", title: "Swarm Execution", desc: "Agents retrieve, analyze, and validate", color: "#a855f7" },
              { step: "04", title: "Synthesis", desc: "Aggregator compiles consensus report", color: "#ec4899" },
            ].map((item, i) => (
              <div key={item.step} className="relative p-5 rounded-2xl bg-white/[0.02] border border-white/[0.06] text-center">
                <div
                  className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full text-xs font-bold text-white"
                  style={{ background: item.color }}
                >
                  {item.step}
                </div>
                <div className="mt-4 text-2xl mb-3" style={{ color: item.color }}>⬢</div>
                <h4 className="font-semibold text-white/90 mb-2">{item.title}</h4>
                <p className="text-xs text-white/50 leading-relaxed">{item.desc}</p>
                
                {/* Connector */}
                {i < 3 && (
                  <div className="hidden lg:block absolute top-1/2 -right-2 w-4 h-0.5" style={{ background: `linear-gradient(90deg, ${item.color}, ${["#10b981", "#a855f7", "#ec4899"][i]})` }} />
                )}
              </div>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="mx-auto max-w-7xl px-6 py-20">
          <div className="relative p-12 rounded-3xl overflow-hidden text-center">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-purple-500/10 to-pink-500/10" />
            <div className="absolute inset-0 border border-white/[0.06] rounded-3xl" />
            <div className="relative">
              <h2 className="text-3xl font-bold mb-4">
                Ready to test your <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">agent swarm</span>?
              </h2>
              <p className="text-white/50 mb-8">Start orchestrating intelligent research missions in minutes.</p>
              <Link
                href="/dashboard"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-white text-[#030508] font-semibold hover:scale-105 transition-transform"
              >
                Initialize Swarm
                <span>→</span>
              </Link>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/[0.06] bg-black/20 backdrop-blur-xl">
        <div className="mx-auto max-w-7xl px-6 py-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="text-sm text-white/40">© 2026 Multi‑Agent Workspace.</div>
            <div className="flex items-center gap-2 text-sm text-white/40">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              All systems operational
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
