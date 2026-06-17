import os
from pydantic import BaseModel, Field
from llama_index.core import Settings, PromptTemplate
# NEW: Import the Pydantic Output Parser
from llama_index.core.output_parsers import PydanticOutputParser
from document_indexer import build_mes_doc_index


class ProblemExtraction(BaseModel):
    """
    /**
     * Defines the exact JSON structure we expect the LLM to generate
     * when analyzing an unstructured meeting transcript.
     */
    """
    symptom: str = Field(description="The technical symptom or error mentioned by the team in the meeting.")
    search_query: str = Field(
        description="A precise, semantic search query formulated to find the solution for this symptom in the technical MES documentation.")


def analyze_meeting_and_query_docs(transcript_filename: str = "dummy_transcript.txt"):
    """
    /**
     * Reads a raw meeting transcript, extracts the core issue using structured LLM output via text completion,
     * and queries the vector index to find a technical solution.
     *
     * @param transcript_filename The name of the transcript file in the data/transcripts folder.
     */
    """
    # 1. Initialize the vector index
    print("1. Initializing vector index and loading Settings...")
    doc_index = build_mes_doc_index()
    query_engine = doc_index.as_query_engine()

    # 2. Read the unstructured transcript
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    transcript_path = os.path.join(base_dir, "data", "transcripts", transcript_filename)

    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript_text = file.read()

    # 3. Use PydanticOutputParser to extract the problem cleanly via standard text completion
    print(f"\n2. Analyzing transcript: {transcript_filename}...")

    parser = PydanticOutputParser(ProblemExtraction)

    prompt_template_str = (
        "Analyze the following meeting transcript and extract the technical issue:\n\n"
        "--- Transcript ---\n"
        "{transcript}\n\n"
        "--- Instructions ---\n"
        "{format_instructions}"
    )

    template = PromptTemplate(prompt_template_str)

    # Format the prompt with both the transcript and the auto-generated JSON instructions
    formatted_prompt = template.format(
        transcript=transcript_text,
        format_instructions=parser.get_format_string()
    )

    # Use .complete() instead of .structured_predict() to bypass the buggy tool-calling API
    response = Settings.llm.complete(formatted_prompt)

    # Parse the text response back into our Pydantic object
    extraction = parser.parse(response.text)

    print("\n--- LLM Extraction Results ---")
    print(f"Extracted Symptom: {extraction.symptom}")
    print(f"Generated Search Query: {extraction.search_query}")

    # 4. Query the documentation using the extracted query
    print("\n3. Querying MES documentation with the generated query...")
    doc_response = query_engine.query(extraction.search_query)

    print("\n--- Proposed Solution from Documentation ---")
    print(doc_response)


if __name__ == "__main__":
    analyze_meeting_and_query_docs()