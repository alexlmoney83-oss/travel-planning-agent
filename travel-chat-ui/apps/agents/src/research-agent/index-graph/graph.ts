/**
 * This "graph" simply exposes an endpoint for a user to upload docs to be indexed.
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph, END, START } from "@langchain/langgraph";
import fs from "fs/promises";

import { IndexStateAnnotation } from "./state.js";
import { makeRetriever } from "../shared/retrieval.js";
import {
  ensureIndexConfiguration,
  IndexConfigurationAnnotation,
} from "./configuration.js";
import { reduceDocs } from "../shared/state.js";

async function indexDocs(
  state: typeof IndexStateAnnotation.State,
  config?: RunnableConfig,
): Promise<typeof IndexStateAnnotation.Update> {
  if (!config) {
    throw new Error("Configuration required to run index_docs.");
  }

  const configuration = ensureIndexConfiguration(config);
  let docs = state.docs;

  if (!docs.length) {
    const fileContent = await fs.readFile(configuration.docsFile, "utf-8");
    const serializedDocs = JSON.parse(fileContent);
    docs = reduceDocs([], serializedDocs);
  }

  const retriever = await makeRetriever(config);
  await retriever.addDocuments(docs);

  return { docs: "delete" };
}

// Define the graph
const builder = new StateGraph(
  IndexStateAnnotation,
  IndexConfigurationAnnotation,
)
  .addNode("indexDocs", indexDocs)
  .addEdge(START, "indexDocs")
  .addEdge("indexDocs", END);

// Compile into a graph object that you can invoke and deploy.
export const graph = builder.compile().withConfig({ runName: "IndexGraph" });
