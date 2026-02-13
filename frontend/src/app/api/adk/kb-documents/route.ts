import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  try {
    // Fetch documents from RAG agent
    const res = await fetch("http://localhost:8003/documents", {
      cache: "no-store",
    });
    
    if (!res.ok) {
      return NextResponse.json(
        { documents: [], error: `RAG agent returned ${res.status}` },
        { status: 200 }
      );
    }
    
    const data = await res.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { documents: [], error: error?.message || "Failed to fetch documents" },
      { status: 200 }
    );
  }
}
