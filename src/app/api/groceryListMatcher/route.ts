import { TodoItem } from "@/app/[username]/page";
import { NextRequest, NextResponse } from "next/server";
import OpenAI from "openai";

const openai = new OpenAI();
export const runtime = "edge";

async function updateGroceryListBasedOnProductsSeenOnTheTable(
  products: string[],
  groceryList: TodoItem[]
): Promise<string[]> {
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
        content: `
The following is a json array of grocery list items structured like [{text: string, completed: bool, id: string}].
Return me a json array of grocery list items ids of items that were also seen on the table, if any.

Even if the ids don't match exactly, you can still match them if they are similar. For example, if the table has "apple" and the grocery list has "red_apple", you can still match them. Be smart and human about it. Make sure to return the id as seen in the "id" field of the grocery list item, not the products seen on the table.

The result should be formatted like {ids: ["red_apples", "berries"]}. If none of the items were seen on the table, return an empty array "{ids: []}".       
`,
      },
      { role: "user", content: JSON.stringify(groceryList) },
    ],
    response_format: { type: "json_object" },
  });

  const idsObject: { ids: string[] } = JSON.parse(
    result.choices[0].message.content || "{}"
  );

  const ids = idsObject["ids"];
  return ids;
}

export interface GroceryListMatcherRequest {
  products: string[];
  groceryList: TodoItem[];
}

export interface GroceryListMatcherResponse {
  completingIds: string[];
}

export async function POST(req: NextRequest) {
  const body: GroceryListMatcherRequest = await req.json();

  const { products, groceryList } = body;

  const completingIds: string[] =
    await updateGroceryListBasedOnProductsSeenOnTheTable(products, groceryList);

  console.log({
    groceryListMatcherRequest: body,
    groceryListMatcherResponse: { completingIds },
  });

  const response: GroceryListMatcherResponse = { completingIds };

  return NextResponse.json(response);
}
