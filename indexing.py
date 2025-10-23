from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    loader = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"Split blog post into {len(all_splits)} sub-documents.")

if __name__ == "__main__":
    main()
