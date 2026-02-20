import { NextResponse } from "next/server";

const ADK_API_URL = process.env.ADK_API_URL || "http://localhost:8001";

export async function GET() {
  try {
    const res = await fetch(`${ADK_API_URL}/kb/documents`, {
      cache: "no-store",
    });
    
    if (!res.ok) {
      return NextResponse.json(
        { documents: [], error: `Backend returned ${res.status}` },
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
