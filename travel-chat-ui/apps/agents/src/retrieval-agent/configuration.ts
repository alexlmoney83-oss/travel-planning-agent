/**
 * Define the configurable parameters for the agent.
 */
import { RunnableConfig } from "@langchain/core/runnables";
import {
  RESPONSE_SYSTEM_PROMPT_TEMPLATE,
  QUERY_SYSTEM_PROMPT_TEMPLATE,
} from "./prompts.js";
import { Annotation } from "@langchain/langgraph";

/**
 * typeof ConfigurationAnnotation.State class for indexing and retrieval operations.
 *
 * This annotation defines the parameters needed for configuring the indexing and
 * retrieval processes, including user identification, embedding model selection,
 * retriever provider choice, and search parameters.
 */
export const IndexConfigurationAnnotation = Annotation.Root({
  /**
   * Unique identifier for the user.
   */
  userId: Annotation<string>,

  /**
   * Name of the embedding model to use. Must be a valid embedding model name.
   */
  embeddingModel: Annotation<string>,

  /**
   * The vector store provider to use for retrieval.
   * Options are 'elastic', 'elastic-local', 'pinecone', or 'mongodb'.
   */
  retrieverProvider: Annotation<
    "elastic" | "elastic-local" | "pinecone" | "mongodb"
  >,

  /**
   * Additional keyword arguments to pass to the search function of the retriever.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  searchKwargs: Annotation<Record<string, any>>,
});

/**
 * Create an typeof IndexConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof IndexConfigurationAnnotation.State with the specified configuration.
 */
export function ensureIndexConfiguration(
  config: RunnableConfig | undefined = undefined,
): typeof IndexConfigurationAnnotation.State {
  const configurable = (config?.configurable || {}) as Partial<
    typeof IndexConfigurationAnnotation.State
  >;
  return {
    userId: configurable.userId || "default", // Give a default user for shared docs
    embeddingModel:
      configurable.embeddingModel || "openai/text-embedding-3-small",
    retrieverProvider: configurable.retrieverProvider || "elastic",
    searchKwargs: configurable.searchKwargs || {},
  };
}

/**
 * The complete configuration for the agent.
 */
export const ConfigurationAnnotation = Annotation.Root({
  ...IndexConfigurationAnnotation.spec,
  /**
   * The system prompt used for generating responses.
   */
  responseSystemPromptTemplate: Annotation<string>,

  /**
   * The language model used for generating responses. Should be in the form: provider/model-name.
   */
  responseModel: Annotation<string>,

  /**
   * The system prompt used for processing and refining queries.
   */
  querySystemPromptTemplate: Annotation<string>,

  /**
   * The language model used for processing and refining queries. Should be in the form: provider/model-name.
   */
  queryModel: Annotation<string>,
});

/**
 * Create a typeof ConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof ConfigurationAnnotation.State with the specified configuration.
 */
export function ensureConfiguration(
  config: RunnableConfig | undefined = undefined,
): typeof ConfigurationAnnotation.State {
  const indexConfig = ensureIndexConfiguration(config);
  const configurable = (config?.configurable || {}) as Partial<
    typeof ConfigurationAnnotation.State
  >;

  return {
    ...indexConfig,
    responseSystemPromptTemplate:
      configurable.responseSystemPromptTemplate ||
      RESPONSE_SYSTEM_PROMPT_TEMPLATE,
    responseModel:
      configurable.responseModel || "anthropic/claude-3-7-sonnet-latest",
    querySystemPromptTemplate:
      configurable.querySystemPromptTemplate || QUERY_SYSTEM_PROMPT_TEMPLATE,
    queryModel: configurable.queryModel || "anthropic/claude-3-5-haiku-latest",
  };
}
