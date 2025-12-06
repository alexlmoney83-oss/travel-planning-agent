import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { Document } from "@langchain/core/documents";
import { reduceDocs } from "../shared/state.js";

/**
 * Represents the input state for the agent.
 * This is a restricted version of the State that defines a narrower interface
 * to the outside world compared to what is maintained internally.
 */
export const InputStateAnnotation = Annotation.Root({
  /**
   * Messages track the primary execution state of the agent.
   * @type {BaseMessage[]}
   * @description
   * Typically accumulates a pattern of Human/AI/Human/AI messages. If combined with a
   * tool-calling ReAct agent pattern, it may follow this sequence:
   * 1. HumanMessage - user input
   * 2. AIMessage with .tool_calls - agent picking tool(s) to use
   * 3. ToolMessage(s) - responses (or errors) from executed tools
   *    (... repeat steps 2 and 3 as needed ...)
   * 4. AIMessage without .tool_calls - agent's unstructured response to user
   * 5. HumanMessage - user's next conversational turn
   *    (... repeat steps 2-5 as needed ...)
   */
  ...MessagesAnnotation.spec,
});

/**
 * Classifies user query.
 * @typedef {Object} Router
 * @property {string} logic - The logic behind the classification.
 * @property {'more-info' | 'langchain' | 'general'} type - The type of the query.
 */

type Router = {
  logic: string;
  type: "more-info" | "langchain" | "general";
};

/**
 * Represents the state of the retrieval graph / agent.
 */
export const AgentStateAnnotation = Annotation.Root({
  ...InputStateAnnotation.spec,

  /**
   * The router's classification of the user's query.
   * @type {Router}
   */
  router: Annotation<Router>({
    default: () => ({ type: "general", logic: "" }),
    reducer: (existing: Router, newRouter: Router) => ({
      ...existing,
      ...newRouter,
    }),
  }),

  /**
   * A list of steps in the research plan.
   * @type {string[]}
   */
  steps: Annotation<string[]>,

  /**
   * Populated by the retriever. This is a list of documents that the agent can reference.
   * @type {Document[]}
   */
  documents: Annotation<
    Document[],
    Document[] | { [key: string]: any }[] | string[] | string | "delete"
  >({
    default: () => [],
    reducer: reduceDocs,
  }),

  // Additional attributes can be added here as needed
  // Examples might include retrieved documents, extracted entities, API connections, etc.
});
