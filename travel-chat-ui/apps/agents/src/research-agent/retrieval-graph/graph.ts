import {
  StateGraph,
  END,
  START,
  LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { z } from "zod";
import { RunnableConfig } from "@langchain/core/runnables";

import {
  AgentConfigurationAnnotation,
  ensureAgentConfiguration,
} from "./configuration.js";
import { graph as researcherGraph } from "./researcher-graph/graph.js";
import { AgentStateAnnotation, InputStateAnnotation } from "./state.js";
import { formatDocs, loadChatModel } from "../shared/utils.js";

async function analyzeAndRouteQuery(
  state: typeof AgentStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof AgentStateAnnotation.Update> {
  const configuration = ensureAgentConfiguration(config);
  const model = await loadChatModel(configuration.queryModel);
  const messages = [
    { role: "system", content: configuration.routerSystemPrompt },
    ...state.messages,
  ];
  const Router = z
    .object({
      logic: z.string(),
      type: z.enum(["more-info", "langchain", "general"]),
    })
    .describe("Classify user query.");
  const response = await model.withStructuredOutput(Router).invoke(messages);
  return { router: response };
}

function routeQuery(
  state: typeof AgentStateAnnotation.State,
): "createResearchPlan" | "askForMoreInfo" | "respondToGeneralQuery" {
  const type = state.router.type;
  if (type === "langchain") {
    return "createResearchPlan";
  } else if (type === "more-info") {
    return "askForMoreInfo";
  } else if (type === "general") {
    return "respondToGeneralQuery";
  } else {
    throw new Error(`Unknown router type ${type}`);
  }
}

async function askForMoreInfo(
  state: typeof AgentStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof AgentStateAnnotation.Update> {
  const configuration = ensureAgentConfiguration(config);
  const model = await loadChatModel(configuration.queryModel);
  const systemPrompt = configuration.moreInfoSystemPrompt.replace(
    "{logic}",
    state.router.logic,
  );
  const messages = [
    { role: "system", content: systemPrompt },
    ...state.messages,
  ];
  const response = await model.invoke(messages);
  return { messages: [response] };
}

async function respondToGeneralQuery(
  state: typeof AgentStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof AgentStateAnnotation.Update> {
  const configuration = ensureAgentConfiguration(config);
  const model = await loadChatModel(configuration.queryModel);
  const systemPrompt = configuration.generalSystemPrompt.replace(
    "{logic}",
    state.router.logic,
  );
  const messages = [
    { role: "system", content: systemPrompt },
    ...state.messages,
  ];
  const response = await model.invoke(messages);
  return { messages: [response] };
}

async function createResearchPlan(
  state: typeof AgentStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof AgentStateAnnotation.Update> {
  const Plan = z
    .object({
      steps: z.array(z.string()),
    })
    .describe("Generate research plan.");

  const configuration = ensureAgentConfiguration(config);
  const model = (
    await loadChatModel(configuration.queryModel)
  ).withStructuredOutput(Plan);
  const messages = [
    { role: "system", content: configuration.researchPlanSystemPrompt },
    ...state.messages,
  ];
  const response = await model.invoke(messages);
  return { steps: response.steps, documents: "delete" };
}

async function conductResearch(
  state: typeof AgentStateAnnotation.State,
  config: LangGraphRunnableConfig,
): Promise<typeof AgentStateAnnotation.Update> {
  const result = await researcherGraph.invoke(
    { question: state.steps[0] },
    { ...config },
  );
  return { documents: result.documents, steps: state.steps.slice(1) };
}

function checkFinished(
  state: typeof AgentStateAnnotation.State,
): "conductResearch" | "respond" {
  return state.steps && state.steps.length > 0 ? "conductResearch" : "respond";
}

async function respond(
  state: typeof AgentStateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof AgentStateAnnotation.Update> {
  const configuration = ensureAgentConfiguration(config);
  const model = await loadChatModel(configuration.responseModel);
  const context = formatDocs(state.documents);
  const prompt = configuration.responseSystemPrompt.replace(
    "{context}",
    context,
  );
  const messages = [{ role: "system", content: prompt }, ...state.messages];
  const response = await model.invoke(messages);
  return { messages: [response] };
}

// Define the graph
const builder = new StateGraph(
  {
    stateSchema: AgentStateAnnotation,
    input: InputStateAnnotation,
  },
  AgentConfigurationAnnotation,
)
  .addNode("analyzeAndRouteQuery", analyzeAndRouteQuery)
  .addNode("askForMoreInfo", askForMoreInfo)
  .addNode("respondToGeneralQuery", respondToGeneralQuery)
  .addNode("createResearchPlan", createResearchPlan)
  .addNode("conductResearch", conductResearch, { subgraphs: [researcherGraph] })
  .addNode("respond", respond)
  .addEdge(START, "analyzeAndRouteQuery")
  .addConditionalEdges("analyzeAndRouteQuery", routeQuery, [
    "askForMoreInfo",
    "respondToGeneralQuery",
    "createResearchPlan",
  ])
  .addEdge("createResearchPlan", "conductResearch")
  .addConditionalEdges("conductResearch", checkFinished, [
    "conductResearch",
    "respond",
  ])
  .addEdge("askForMoreInfo", END)
  .addEdge("respondToGeneralQuery", END)
  .addEdge("respond", END);

// Compile into a graph object that you can invoke and deploy.
export const graph = builder
  .compile()
  .withConfig({ runName: "RetrievalGraph" });
