from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def main():
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query involving clams.

        Args:
            query: The query to retrieve information for
        """
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        max_tokens=1000
    )
    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt="You are a helpful assistant",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "How does water temperature affect clam growth?"}]}
    )
    final_message = result["messages"][-1]
    print(final_message.content)

if __name__ == "__main__":
    main()
