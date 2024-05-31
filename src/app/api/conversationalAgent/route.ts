import { TodoItem } from "@/app/[username]/todo/page";
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

interface ExplodingTodo {
  category: "todo" | "grocery";
  index: number;
  subtasks: TodoItem[];
}

interface ConversationalAgentResponse {
  narration: string;
  explodingTodo?: ExplodingTodo;
}

interface ConversationalAgentRequest {
  text: string;
  groceryList: TodoItem[];
  todoList: TodoItem[];
}

export async function askConversationalAgent(
  body: ConversationalAgentRequest
): Promise<ConversationalAgentResponse> {
  const { text, groceryList, todoList } = body;

  const result = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content: "The following is the user's todo list:",
      },
      { role: "user", content: todoList.join(", ") },

      {
        role: "system",
        content: "The following is the user's grocery list:",
      },
      { role: "user", content: groceryList.join(", ") },

      {
        role: "system",
        content: `Finally, the following is the user's request.

        If the user asks for more details about a certain task that's already in one of the two lists, return a json object with the task's category ("todo" or "grocery"), index, and subtasks. The result should be formatted exactly like {narration: string, explodingTodo: {category: "todo"|"grocery", index: number, subtasks: {text: string, completed: false}[]}}.  For example "{ narration: "Ok, I'll divide the task into smaller and easier pieces", explodingTodo: {index: 0, subtasks: [{text: 'subtask 1', completed: false}; {text: 'subtask 2'; completed: false}]}}". If it's about a grocery item, the subtasks should be the foods for the recipe.

        If the user asks something that's not about splitting or making an existing task simpler, return a json object with the narration and no explodingTodo. For example "{narration: "While California is the best place in the world where to launch a startup, you'll probably miss the beautiful italian girls."}
`,
      },
      {
        role: "user",
        content: text,
      },
    ],
    response_format: { type: "json_object" },
  });

  const responseObject: ConversationalAgentResponse = JSON.parse(
    result.choices[0].message.content || "{}"
  );

  return responseObject;
}

export async function POST(req: NextRequest) {
  const body: ConversationalAgentRequest = await req.json();

  const response: ConversationalAgentResponse = await askConversationalAgent(
    body
  );

  return NextResponse.json(response);
}
