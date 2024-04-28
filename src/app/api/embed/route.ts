import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

export const runtime = "edge";

const openai = new OpenAI();

export async function getEmbedding(text: string) {
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  return embedding;
}

export async function POST(req: NextRequest) {
  const embedding = await getEmbedding(
    "Hello, world! My name is Fausto and I absolutely love Virigina, Lemon Tea, and Chocolate Ice Cream."
  );

  return NextResponse.json({ embedding });
}
