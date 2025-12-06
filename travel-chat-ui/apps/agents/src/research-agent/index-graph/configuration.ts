import { Annotation } from "@langchain/langgraph";
import { BaseConfigurationAnnotation } from "../shared/configuration.js";
import { RunnableConfig } from "@langchain/core/runnables";
import { ensureBaseConfiguration } from "../shared/configuration.js";

// This file contains sample documents to index, based on the following LangChain and LangGraph documentation pages:
// - https://python.langchain.com/docs/concepts/
// - https://langchain-ai.github.io/langgraph/concepts/low_level/
const DEFAULT_DOCS_FILE = "src/research-agent/sample_docs.json";

/**
 * The configuration for the indexing process.
 */
export const IndexConfigurationAnnotation = Annotation.Root({
  ...BaseConfigurationAnnotation.spec,

  /**
   * Path to a JSON file containing default documents to index.
   */
  docsFile: Annotation<string>,
});

/**
 * Create an typeof IndexConfigurationAnnotation.State instance from a RunnableConfig object.
 *
 * @param config - The configuration object to use.
 * @returns An instance of typeof IndexConfigurationAnnotation.State with the specified configuration.
 */
export function ensureIndexConfiguration(
  config: RunnableConfig,
): typeof IndexConfigurationAnnotation.State {
  const configurable = (config?.configurable || {}) as Partial<
    typeof IndexConfigurationAnnotation.State
  >;
  const baseConfig = ensureBaseConfiguration(config);
  return {
    ...baseConfig,
    docsFile: configurable.docsFile || DEFAULT_DOCS_FILE,
  };
}
