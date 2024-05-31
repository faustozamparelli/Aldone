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

export interface ConversationalAgentResponse {
  narration: string;
  explodingTodo?: ExplodingTodo;
}

export interface ConversationalAgentRequest {
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
      { role: "user", content: JSON.stringify(todoList) },

      {
        role: "system",
        content: "The following is the user's grocery list:",
      },
      { role: "user", content: JSON.stringify(groceryList) },

      {
        role: "system",
        content: `Finally, the following is the user's request.

        If the user asks for more details about a certain task that's already in one of the two lists, return a json object with the task's category ("todo" or "grocery"), index, and subtasks. The result should be formatted exactly like {narration: string, explodingTodo: {category: "todo"|"grocery", index: number, subtasks: {text: string, completed: false}[]}}.  For example "{ narration: "Ok, I'll divide the task into smaller and easier pieces", explodingTodo: {index: 0, subtasks: [{text: 'subtask 1', completed: false}; {text: 'subtask 2'; completed: false}]}}". If it's about a grocery item, the subtasks should be the foods for the recipe.

        Please check very carefully whether there's something similar in one of the two lists. If there's a task that's very similar to the one the user is asking about, return the subtasks of the similar task. For example, if the user asks about "buying apples", and there's a task in the todo list that's "buying fruits", return the subtasks of "buying fruits". If there's a task in the grocery list that's "buying apples", return the subtasks of "buying apples".

        Indices start at 0, make sure to return the correct index. todolist: [{index 0 item}, {index 1 item}, {index 2 item}], grocerylist: [{index 0 item}, {index 1 item}, {index 2 item}].

        If the user asks something that's not about splitting or making an existing task simpler, return a json object with the narration and no explodingTodo. For example "{narration: "While California is the best place in the world where to launch a startup, you'll probably miss the beautiful italian girls."}

        If the user is asking for the items in the grocery list or the todo list, just read them normally without any explodingTodo. For example, "{narration: 'The items in the grocery list are: tiramisu, bananas, apple pie, 200g avocado'}" or "{narration: 'Today remember to: 1) Complete the AI Lab Project, 2) Collect Raspberri Pi at delivery center, 3) Pack for the trip to Rome'}"
        
        If there are subtasks, read them like a human would.
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

  console.log({
    conversationalAgentRequest: body,
    conversationalAgentResponse: responseObject,
  });

  return responseObject;
}

export async function POST(req: NextRequest) {
  const body: ConversationalAgentRequest = await req.json();

  const response: ConversationalAgentResponse = await askConversationalAgent(
    body
  );

  return NextResponse.json(response);
}
