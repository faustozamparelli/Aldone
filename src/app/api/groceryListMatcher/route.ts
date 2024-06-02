import { TodoItem } from "@/app/[username]/page";
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

async function updateGroceryListBasedOnProductsSeenOnTheTable(
  products: string[],
  groceryList: TodoItem[]
): Promise<TodoItem[]> {
  const result = await openai.chat.completions.create({
    model: "gpt-3.5-turbo",
    messages: [
      {
        role: "system",
        content: "The following are the products that were seen on the table:",
      },
      { role: "user", content: products.join(", ") },
      {
        role: "system",
        content: `The following is a json array of grocery list items. Return me a json array of grocery list items ids of items that were also seen on the table, if any. The result should be formatted like {ids: ["red_apples", "berries"]}. If none of the items were seen on the table, return an empty array "{ids: []}".`,
      },
      {
        role: "user",
        content: groceryList.map((item) => item.text).join(", "),
      },
    ],
    response_format: { type: "json_object" },
  });

  const idsObject: { ids: string[] } = JSON.parse(
    result.choices[0].message.content || "{}"
  );

  const ids = idsObject["ids"];

  console.log({ ids });

  const newGroceryList = [...groceryList];
  for (const id of ids) {
    const index = newGroceryList.findIndex((item) => item.id === id);
    newGroceryList[index] = { ...newGroceryList[index], completed: true };
  }

  return newGroceryList;
}

export interface GroceryListMatcherRequest {
  products: string[];
  groceryList: TodoItem[];
}

export interface GroceryListMatcherResponse {
  updatedGroceryList: TodoItem[];
}

export async function POST(req: NextRequest) {
  const body: GroceryListMatcherRequest = await req.json();

  const { products, groceryList } = body;

  const updatedGroceryList: TodoItem[] =
    await updateGroceryListBasedOnProductsSeenOnTheTable(products, groceryList);

  console.log({
    groceryListMatcherRequest: body,
    groceryListMatcherResponse: { updatedGroceryList },
  });

  return NextResponse.json({ updatedGroceryList });
}
