import { NextRequest, NextResponse } from "next/server";

export interface ProcessVoiceRequest {
  input: string[];
}

export interface VoiceAction {
  addingTodo?: string;
  completingTodoIndex?: number;
  retriveTodos?: boolean;
}

export interface ProcessVoiceResponse {
  action: VoiceAction;
  agentReply: string;
  payload?: any;
}

export async function POST(req: NextRequest) {
  const { input }: ProcessVoiceRequest = await req.json();

  const isItARequest = (
    await (
      await fetch("http://localhost:3001/question_classifier", {
        method: "POST",
        body: JSON.stringify({ text: input }),
        headers: { "Content-Type": "application/json" },
      })
    ).json()
  )["is_it_a_question"];

  const result: ProcessVoiceResponse = {
    action: isItARequest ? { addingTodo: `${input}` } : { retriveTodos: true },
    agentReply: isItARequest ? "Let me see." : "Ok, done.",
    payload: { isItARequest },
  };

  return NextResponse.json(result);
}
