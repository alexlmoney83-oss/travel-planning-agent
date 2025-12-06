/**
 * This "graph" simply exposes an endpoint for a user to upload docs to be indexed.
 */

import { Document } from "@langchain/core/documents";
import { RunnableConfig } from "@langchain/core/runnables";
import { StateGraph } from "@langchain/langgraph";

import { IndexStateAnnotation } from "./state.js";
import { makeRetriever } from "./retrieval.js";
import {
  ensureIndexConfiguration,
  IndexConfigurationAnnotation,
} from "./configuration.js";

function ensureDocsHaveUserId(
  docs: Document[],
  config: RunnableConfig,
): Document[] {
  const configuration = ensureIndexConfiguration(config);
  const userId = configuration.userId;
  return docs.map((doc) => {
    return new Document({
      pageContent: doc.pageContent,
      metadata: { ...doc.metadata, user_id: userId },
    });
  });
}

async function indexDocs(
  state: typeof IndexStateAnnotation.State,
  config?: RunnableConfig,
): Promise<typeof IndexStateAnnotation.Update> {
  if (!config) {
    throw new Error("ConfigurationAnnotation required to run index_docs.");
  }
  const docs = state.docs;
  const retriever = await makeRetriever(config);
  const stampedDocs = ensureDocsHaveUserId(docs, config);

  await retriever.addDocuments(stampedDocs);
  return { docs: "delete" };
}

// Define a new graph

const builder = new StateGraph(
  IndexStateAnnotation,
  IndexConfigurationAnnotation,
)
  .addNode("indexDocs", indexDocs)
  .addEdge("__start__", "indexDocs");

// Finally, we compile it!
// This compiles it into a graph you can invoke and deploy.
export const graph = builder.compile();

graph.name = "Index Graph"; // Customizes the name displayed in LangSmith
