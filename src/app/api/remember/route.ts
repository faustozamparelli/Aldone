import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import { v4 as uuidv4 } from "uuid";
import OpenAI from "openai";

export const runtime = "edge";

export async function remember(text: string) {
  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
  const index = pc.index("diairy");

  const openai = new OpenAI();

  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const id = uuidv4();

  await index.upsert([
    {
      id: id,
      values: embedding.data[0].embedding,
    },
  ]);

  // TODO: push to DynamoBD
}

export async function POST(req: NextRequest) {
  const { text } = await req.json();
  await remember(text);
  return NextResponse.json({ success: true });
}
