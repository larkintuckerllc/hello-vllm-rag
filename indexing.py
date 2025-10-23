from langchain_community.document_loaders import DirectoryLoader, TextLoader


def main():
    loader = DirectoryLoader("./data/", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Total characters: {len(documents[0].page_content)}")
    print(documents[0].page_content[:500])

if __name__ == "__main__":
    main()
