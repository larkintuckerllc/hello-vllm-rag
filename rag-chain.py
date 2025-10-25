from langchain.agents.middleware import dynamic_prompt
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def main():
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )

    @dynamic_prompt
    def prompt_with_context(request):
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        system_message = (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )
        return system_message

    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        max_tokens=1000
    )
    agent = create_agent(
        model=llm,
        middleware=[prompt_with_context]
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "How does water temperature affect clam growth?"}]}
    )
    final_message = result["messages"][-1]
    print(final_message.content)

if __name__ == "__main__":
    main()
