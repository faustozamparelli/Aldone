import { NextRequest, NextResponse } from "next/server";

export interface QuestionClassifierRequest {
  input: string;
}

export interface QuestionClassifierResponse {
  isQuery: boolean;
  agentReply: string;
}

export async function POST(req: NextRequest) {
  const { input }: QuestionClassifierRequest = await req.json();

  const isQuery = (
    await (
      await fetch("http://localhost:3001/question_classifier", {
        method: "POST",
        body: JSON.stringify({ text: input }),
        headers: { "Content-Type": "application/json" },
      })
    ).json()
  )["is_it_a_query"];

  const result: QuestionClassifierResponse = {
    isQuery,
    agentReply: isQuery ? "Let me see." : "Ok. Done.",
  };

  console.log({
    questionClassifierRequest: { input },
    questionClassifierResponse: result,
  });

  return NextResponse.json(result);
}
