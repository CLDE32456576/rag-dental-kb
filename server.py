import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dental-kb"

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

# Sample dental office documents
DENTAL_DOCS = """
FREQUENTLY ASKED QUESTIONS

Q: How much does a cleaning cost?
A: A standard cleaning costs $150. Deep cleaning (scaling and root planing) costs $300.

Q: What are your hours?
A: Monday-Friday 9 AM - 5 PM, Saturday 10 AM - 2 PM, Closed Sunday.

Q: Do you accept insurance?
A: Yes, we accept most major dental insurance. Please bring your card to your first appointment.

Q: What's the cost of a root canal?
A: Root canal treatment costs $1,200-$1,500 depending on tooth location and complexity.

Q: How long is an appointment?
A: Regular cleanings take 45-60 minutes. First-time exams take 60-90 minutes.

Q: Do you offer payment plans?
A: Yes, we offer financing through CareCredit with 0% interest for 12 months on approved credit.

Q: Can I get same-day appointments?
A: We try to accommodate emergency appointments. Call us immediately if you have severe pain.

Q: What's your cancellation policy?
A: Cancellations must be made 24 hours in advance. No-shows are charged a $50 fee.

PRICING

Cleaning: $150
Deep Cleaning: $300
Fillings: $200-$400
Root Canal: $1,200-$1,500
Crown: $1,000-$1,500
Extraction: $200-$500
Whitening: $400
"""

def ingest_documents():
    """Split documents and upload to Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(DENTAL_DOCS)

    vectorstore = PineconeVectorStore.from_texts(chunks, embeddings, index_name=index_name)
    return vectorstore

def query_rag(question: str) -> str:
    """Query the RAG system using hybrid retrieval (BM25 + embeddings)."""
    from rank_bm25 import BM25Okapi
    
    # Split docs for BM25
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(DENTAL_DOCS)
    
    # BM25 for keyword search
    bm25 = BM25Okapi([chunk.split() for chunk in chunks])
    
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # Get top 2 from BM25
    bm25_results = bm25.get_top_n(question.split(), chunks, n=2)
    
    # Get top 1 from semantic search
    semantic_docs = retriever.invoke(question)
    
    # Combine results
    combined = bm25_results + [doc.page_content for doc in semantic_docs]
    context = "\n\n".join(combined)
    
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question accurately.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    print("Ingesting dental documents...")
    ingest_documents()
    print("✓ Documents uploaded to Pinecone")

    # Test queries
    test_questions = [
        "How much is a cleaning?",
        "What are your hours?",
        "How much does a root canal cost?"
    ]

    print("\nTesting RAG retrieval:\n")
    for q in test_questions:
        answer = query_rag(q)
        print(f"Q: {q}\nA: {answer}\n")
