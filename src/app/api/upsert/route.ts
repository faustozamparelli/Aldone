import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  //   const body = await req.json();

  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const index = pc.index("diairy");

  await index.namespace("ns1").upsert([
    {
      id: "vec1",
      values: Array.from({ length: 1536 }, () => Math.random()),
      metadata: { genre: "drama" },
    },
    {
      id: "vec2",
      values: Array.from({ length: 1536 }, () => Math.random()),
      metadata: { genre: "action" },
    },
    {
      id: "vec3",
      values: Array.from({ length: 1536 }, () => Math.random()),
      metadata: { genre: "drama" },
    },
    {
      id: "vec4",
      values: Array.from({ length: 1536 }, () => Math.random()),
      metadata: { genre: "action" },
    },
  ]);

  return NextResponse.json({ success: true });
}
