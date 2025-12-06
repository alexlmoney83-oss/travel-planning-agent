import { Annotation } from "@langchain/langgraph";
import { RunnableConfig } from "@langchain/core/runnables";

import {
  ROUTER_SYSTEM_PROMPT,
  MORE_INFO_SYSTEM_PROMPT,
  GENERAL_SYSTEM_PROMPT,
  RESEARCH_PLAN_SYSTEM_PROMPT,
  GENERATE_QUERIES_SYSTEM_PROMPT,
  RESPONSE_SYSTEM_PROMPT,
} from "./prompts.js";
import {
  BaseConfigurationAnnotation,
  ensureBaseConfiguration,
} from "../shared/configuration.js";

/**
 * The configuration for the agent.
 */
export const AgentConfigurationAnnotation = Annotation.Root({
  ...BaseConfigurationAnnotation.spec,

  // models
  /**
   * The language model used for processing and refining queries.
   * Should be in the form: provider/model-name.
   */
  queryModel: Annotation<string>,

  /**
   * The language model used for generating responses.
   * Should be in the form: provider/model-name.
   */
  responseModel: Annotation<string>,

  // prompts
  /**
   * The system prompt used for classifying user questions to route them to the correct node.
   */
  routerSystemPrompt: Annotation<string>,

  /**
   * The system prompt used for asking for more information from the user.
   */
  moreInfoSystemPrompt: Annotation<string>,

  /**
   * The system prompt used for responding to general questions.
   */
  generalSystemPrompt: Annotation<string>,

  /**
   * The system prompt used for generating a research plan based on the user's question.
   */
  researchPlanSystemPrompt: Annotation<string>,

  /**
   * The system prompt used by the researcher to generate queries based on a step in the research plan.
   */
  generateQueriesSystemPrompt: Annotation<string>,

  /**
   * The system prompt used for generating responses.
   */
  responseSystemPrompt: Annotation<string>,
});

/**
 * Create a typeof ConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof ConfigurationAnnotation.State with the specified configuration.
 */
export function ensureAgentConfiguration(
  config: RunnableConfig,
): typeof AgentConfigurationAnnotation.State {
  const configurable = (config?.configurable || {}) as Partial<
    typeof AgentConfigurationAnnotation.State
  >;
  const baseConfig = ensureBaseConfiguration(config);
  return {
    ...baseConfig,
    queryModel: configurable.queryModel || "anthropic/claude-3-5-haiku-latest",
    responseModel:
      configurable.responseModel || "anthropic/claude-3-7-sonnet-latest",
    routerSystemPrompt: configurable.routerSystemPrompt || ROUTER_SYSTEM_PROMPT,
    moreInfoSystemPrompt:
      configurable.moreInfoSystemPrompt || MORE_INFO_SYSTEM_PROMPT,
    generalSystemPrompt:
      configurable.generalSystemPrompt || GENERAL_SYSTEM_PROMPT,
    researchPlanSystemPrompt:
      configurable.researchPlanSystemPrompt || RESEARCH_PLAN_SYSTEM_PROMPT,
    generateQueriesSystemPrompt:
      configurable.generateQueriesSystemPrompt ||
      GENERATE_QUERIES_SYSTEM_PROMPT,
    responseSystemPrompt:
      configurable.responseSystemPrompt || RESPONSE_SYSTEM_PROMPT,
  };
}
