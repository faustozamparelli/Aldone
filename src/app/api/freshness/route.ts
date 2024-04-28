import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const index = pc.index("diairy");

  const stats = await index.describeIndexStats();

  return NextResponse.json({ stats });
}
