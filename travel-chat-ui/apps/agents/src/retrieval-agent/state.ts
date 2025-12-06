import { Document } from "@langchain/core/documents";
import { BaseMessage } from "@langchain/core/messages";
import { Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";

/**
 * Reduces the document array based on the provided new documents or actions.
 *
 * @param existing - The existing array of documents.
 * @param newDocs - The new documents or actions to apply.
 * @returns The updated array of documents.
 */
export function reduceDocs(
  existing?: Document[],
  newDocs?:
    | Document[]
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    | { [key: string]: any }[]
    | string[]
    | string
    | "delete",
) {
  // Supports deletion by returning an empty array when "delete" is specified
  if (newDocs === "delete") {
    return [];
  }
  // Supports adding a single string document
  if (typeof newDocs === "string") {
    const docId = uuidv4();
    return [{ pageContent: newDocs, metadata: { id: docId }, id: docId }];
  }
  // User can provide "docs" content in a few different ways
  if (Array.isArray(newDocs)) {
    const coerced: Document[] = [];
    for (const item of newDocs) {
      if (typeof item === "string") {
        coerced.push({ pageContent: item, metadata: { id: uuidv4() } });
      } else if (typeof item === "object") {
        const doc = item as Document;
        const docId = item?.id || uuidv4();
        item.id = docId;
        if (!doc.metadata || !doc.metadata.id) {
          doc.metadata = doc.metadata || {};
          doc.metadata.id = docId;
        }
        coerced.push(doc);
      }
    }
    return coerced;
  }
  // Returns existing documents if no valid update is provided
  return existing || [];
}

/**
 * Defines the structure and behavior of the index state.
 * This state is used to manage the documents in the index.
 */
export const IndexStateAnnotation = Annotation.Root({
  /**
   * Stores the documents in the index.
   *
   * @type {Document[]} - An array of Document objects.
   * @reducer reduceDocs - A function that handles updates to the documents array.
   *                       It can add new documents, replace existing ones, or delete all documents.
   * @default An empty array ([]).
   * @see reduceDocs for detailed behavior on how updates are processed.
   */
  docs: Annotation<
    Document[],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    Document[] | { [key: string]: any }[] | string[] | string | "delete"
  >({
    reducer: reduceDocs,
    default: () => [],
  }),
});

/**
 * This narrows the interface with the user.
 */
export const InputStateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>,
});

/**
 * The State defines three things:
 * 1. The structure of the graph's state (which "channels" are available to read/write)
 * 2. The default values for the state's channels
 * 3. The reducers for the state's channels. Reducers are functions that determine how to apply updates to the state.
 * See [Reducers](https://langchain-ai.github.io/langgraphjs/concepts/low_level/#reducers) for more information.
 */
export const StateAnnotation = Annotation.Root({
  /**
   * Stores the conversation messages.
   * @type {BaseMessage[]}
   * @reducer Default reducer that appends new messages to the existing ones.
   * @default An empty array.
   *
   * Nodes can return a list of "MessageLike" objects, which can be LangChain messages
   * or dictionaries following a common message format.
   *
   * To delete messages, use RemoveMessage.
   * @see https://langchain-ai.github.io/langgraphjs/how-tos/delete-messages/
   *
   * For more information, see:
   * @see https://langchain-ai.github.io/langgraphjs/reference/variables/langgraph.MessagesAnnotation.html
   */
  ...MessagesAnnotation.spec,

  /**
   * Stores the user queries.
   * @type {string[]}
   * @reducer A custom reducer function that appends new queries to the existing array.
   *          It handles both single string and string array inputs.
   * @default An empty array ([]).
   * @description This annotation manages the list of user queries in the state.
   *              It uses a reducer to add new queries while preserving existing ones.
   *              The reducer supports adding either a single query (string) or multiple queries (string[]).
   */
  queries: Annotation<string[], string | string[]>({
    reducer: (existing: string[], newQueries: string[] | string) => {
      /**
       * This reducer is currently "append only" - it only adds new queries to the existing list.
       *
       * To extend this reducer to support more complex operations, you could modify it in ways like this:
       *
       * reducer: (existing: string[], action: { type: string; payload: string | string[] }) => {
       *   switch (action.type) {
       *     case 'ADD':
       *       return [...existing, ...(Array.isArray(action.payload) ? action.payload : [action.payload])];
       *     case 'DELETE':
       *       return existing.filter(query => query !== action.payload);
       *     case 'REPLACE':
       *       return Array.isArray(action.payload) ? action.payload : [action.payload];
       *     default:
       *       return existing;
       *   }
       * }
       */
      return [
        ...existing,
        ...(Array.isArray(newQueries) ? newQueries : [newQueries]),
      ];
    },
    default: () => [],
  }),

  /**
   * Stores the retrieved documents.
   * @type {Document[]}
   */
  retrievedDocs: Annotation<Document[]>,
});
