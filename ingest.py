from server import ingest_documents, query_rag

print("Ingesting dental documents...")
ingest_documents()
print("✓ Documents uploaded to Pinecone")

test_questions = [
    "How much is a cleaning?",
    "What are your hours?",
    "How much does a root canal cost?"
]
print("\nTesting RAG retrieval:\n")
for q in test_questions:
    answer = query_rag(q)
    print(f"Q: {q}\nA: {answer}\n")
