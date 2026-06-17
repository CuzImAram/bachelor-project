import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI as OpenAIClient

# NEW: Import the OpenAILike class for custom OpenAI-compatible endpoints
from llama_index.llms.openai_like import OpenAILike

# Import the centralized config
from config import setup_environment


class HelmholtzEmbedding(BaseEmbedding):
    """
    /**
     * Custom embedding class to bypass LlamaIndex's strict OpenAI model validation.
     * Uses the official openai client to connect to the Helmholtz Blablador API.
     */
    """

    def __init__(self, api_key: str, api_base: str, model_name: str = "alias-embeddings", **kwargs):
        super().__init__(**kwargs)
        self._client = OpenAIClient(api_key=api_key, base_url=api_base)
        self._model_name = model_name

    def _get_query_embedding(self, query: str) -> list[float]:
        """
        /**
         * Generates an embedding for a single query string.
         *
         * @param query The search query text.
         * @return A list of floats representing the vector embedding.
         */
        """
        response = self._client.embeddings.create(input=query, model=self._model_name)
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> list[float]:
        """
        /**
         * Generates an embedding for a single document text string.
         *
         * @param text The document text to embed.
         * @return A list of floats representing the vector embedding.
         */
        """
        response = self._client.embeddings.create(input=text, model=self._model_name)
        return response.data[0].embedding

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        /**
         * Generates embeddings for a batch of text strings.
         *
         * @param texts A list of document text strings.
         * @return A list containing lists of floats representing the vector embeddings.
         */
        """
        response = self._client.embeddings.create(input=texts, model=self._model_name)
        return [data.embedding for data in response.data]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


def build_mes_doc_index(docs_dir: str = "mes_docs") -> VectorStoreIndex:
    """
    /**
     * Configures the global LlamaIndex settings using the centralized config,
     * loads unstructured MES documentation, and creates a searchable vector index.
     *
     * @param docs_dir The name of the folder inside the data directory containing the documentation.
     * @return A VectorStoreIndex object that can be used for semantic queries.
     */
    """
    # 1. Load credentials from centralized config
    env_vars = setup_environment()
    api_key = env_vars["api_key"]
    api_url = env_vars["api_url"]

    # 2. Configure LlamaIndex using OpenAILike to bypass strict model name validation.
    # The context_window is explicitly provided (8192 is standard for Ministral-8B) to prevent inference errors.
    Settings.llm = OpenAILike(
        model="alias-fast",
        api_key=api_key,
        api_base=api_url,
        is_chat_model=True,
        context_window=8192
    )

    # 3. Inject our custom embedding class
    Settings.embed_model = HelmholtzEmbedding(
        api_key=api_key,
        api_base=api_url
    )

    # 4. Resolve path to the document folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    full_docs_path = os.path.join(base_dir, "data", docs_dir)

    # 5. Load and index documents
    print(f"Reading documents from: {full_docs_path}")
    documents = SimpleDirectoryReader(full_docs_path).load_data()

    index = VectorStoreIndex.from_documents(documents)
    return index


if __name__ == "__main__":
    doc_index = build_mes_doc_index()

    query_engine = doc_index.as_query_engine()
    response = query_engine.query("Wie funktioniert das beschriebene MES-Modul?")

    print("\nSystem Response:")
    print(response)