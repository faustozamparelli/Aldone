import { NextRequest, NextResponse } from "next/server";

export interface TodoClassifierRequest {
  input: string;
}

export interface TodoClassifierResponse {
  category: string;
}

export async function POST(req: NextRequest) {
  const { input }: TodoClassifierRequest = await req.json();

  const category = (
    await (
      await fetch("http://localhost:3001/todo_classifier", {
        method: "POST",
        body: JSON.stringify({ text: input }),
        headers: { "Content-Type": "application/json" },
      })
    ).json()
  )["category"];

  const result: TodoClassifierResponse = {
    category,
  };

  console.log({
    todoClassifierRequest: { input },
    todoClassifierResponse: result,
  });
  return NextResponse.json(result);
}
