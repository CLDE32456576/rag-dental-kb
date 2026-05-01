import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone
import json
from rank_bm25 import BM25Okapi


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "dental-kb"

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

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

Q: What services do you offer?
A: We offer cleanings, fillings, root canals, crowns, extractions, whitening, and more.

Q: Are you open on weekends?
A: Yes, we're open Saturday 10 AM - 2 PM. Closed Sunday and Monday.

PRICING

Cleaning: $150
Deep Cleaning: $300
Fillings: $200-$400
Root Canal: $1,200-$1,500
Crown: $1,000-$1,500
Extraction: $200-$500
Whitening: $400
"""

# Test questions with expected keywords
TEST_QUESTIONS = [
    ("How much is a cleaning?", "150"),
    ("What are your hours?", "9 AM"),
    ("Do you take insurance?", "insurance"),
    ("Root canal cost?", "1,200"),
    ("How long is an appointment?", "45"),
    ("Payment plans?", "CareCredit"),
    ("Same-day appointments?", "emergency"),
    ("Cancellation policy?", "24 hours"),
    ("What services?", "cleaning"),
    ("Open on weekends?", "Saturday"),
]

def strategy_1_naive() -> dict:
    """Strategy 1: Embeddings only."""
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    correct = 0
    for question, expected_keyword in TEST_QUESTIONS:
        answer = chain.invoke(question)
        if expected_keyword.lower() in answer.lower():
            correct += 1
    
    return {
        "strategy": "Naive (Embeddings Only)",
        "correct": correct,
        "total": len(TEST_QUESTIONS),
        "accuracy": correct / len(TEST_QUESTIONS)
    }

def strategy_2_hybrid() -> dict:
    """Strategy 2: Hybrid (BM25 keyword + embeddings)."""
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
    
    # Split docs for BM25
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(DENTAL_DOCS)
    
    # BM25 for keyword search
    bm25 = BM25Okapi([c.split() for c in chunks])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    correct = 0
    for question, expected_keyword in TEST_QUESTIONS:
        # Get top 2 from BM25 keyword search
        bm25_results = bm25.get_top_n(question.split(), chunks, n=2)
        
        # Get top 1 from semantic search
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        semantic_docs = retriever.invoke(question)
        
        # Combine
        combined = bm25_results + [doc.page_content for doc in semantic_docs]
        context = "\n\n".join(combined)
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        if expected_keyword.lower() in answer.lower():
            correct += 1
    
    return {
        "strategy": "Hybrid (BM25 + Semantic)",
        "correct": correct,
        "total": len(TEST_QUESTIONS),
        "accuracy": correct / len(TEST_QUESTIONS)
    }

def strategy_3_contextual() -> dict:
    """Strategy 3: Contextual (Claude rewrites chunks for better retrieval)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(DENTAL_DOCS)
    
    # Claude rewrites each chunk to be more retrievable
    rewritten_chunks = []
    for chunk in chunks:
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite this dental FAQ to be clearer and more searchable:\n\n{chunk}"
        )
        rewritten = (rewrite_prompt | llm | StrOutputParser()).invoke({"chunk": chunk})
        rewritten_chunks.append(rewritten)
    
    # Re-embed the rewritten chunks
    vectorstore = PineconeVectorStore.from_texts(rewritten_chunks, embeddings, index_name=index_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_template(
        "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}"
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    correct = 0
    for question, expected_keyword in TEST_QUESTIONS:
        answer = chain.invoke(question)
        if expected_keyword.lower() in answer.lower():
            correct += 1
    
    return {
        "strategy": "Contextual (Rewritten Chunks)",
        "correct": correct,
        "total": len(TEST_QUESTIONS),
        "accuracy": correct / len(TEST_QUESTIONS)
    }

if __name__ == "__main__":
    print("=" * 60)
    print("RAG RETRIEVAL STRATEGY COMPARISON")
    print("=" * 60)
    print(f"Testing {len(TEST_QUESTIONS)} dental office Q&A pairs\n")
    
    results = []
    
    print("Running Strategy 1: Naive (Embeddings Only)...")
    results.append(strategy_1_naive())
    
    print("Running Strategy 2: Hybrid...")
    results.append(strategy_2_hybrid())
    
    print("Running Strategy 3: Contextual...")
    results.append(strategy_3_contextual())
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for result in results:
        print(f"\n{result['strategy']}")
        print(f"Accuracy: {result['correct']}/{result['total']} ({result['accuracy']:.1%})")
    
    # Determine winner
    winner = max(results, key=lambda x: x['accuracy'])
    print(f"\n🏆 WINNER: {winner['strategy']} ({winner['accuracy']:.1%})")
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to results.json")