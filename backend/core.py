from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from ingestion import embeddings

load_dotenv()

INDEX_NAME ="langchain-doc-index"

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    text_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=text_embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    # using memory with RAG, 1. The 'rephrase' prompt supplies the 'chat_history'
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, # 2. chaining in the history_aware_retriever
        combine_docs_chain=stuff_documents_chain
    )
    #3. pass in the chat history
    result = qa.invoke(input={"input":query, "chat_history":chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="What is LangChain Chain?")
    # print the result
    print(res["result"])
    # print the sources used for giving us the answer.
    [print(doc.metadata['source']) for doc in res['context']]
