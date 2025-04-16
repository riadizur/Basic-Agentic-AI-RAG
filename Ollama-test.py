import asyncio
import httpx
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

async def ask_ollama_with_pdf_context(file_path):
    pdf_context = extract_text_from_pdf(file_path)
    print("Extracted PDF Context:\n")
    print(pdf_context[:100000])  # Print the first 500 characters of the context for brevity
    prompt = "Summarize the latest document."

    # Trim context if too long (Ollama has ~4K-8K token limit)
    max_context_length = 1000000
    if len(pdf_context) > max_context_length:
        pdf_context = pdf_context[:max_context_length]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an enterprise-level AI assistant integrated with a RAG system. "
                "You have access to enterprise documents. Use the following extracted content from a PDF "
                "to answer the user query:\n\n"
                f"{pdf_context}\n\n"
                "When answering, cite the context when possible. If the answer cannot be found in the document, reply with "
                "'Information not available in current dataset.'"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": messages,
                "stream": False
            }
        )

        try:
            print("\n🧠 Response:\n")
            print(response.json()['message']['content'])
        except Exception:
            print("Error:", response.text)

# 👇 Replace this with your actual PDF file
pdf_file_path = "./files/file.pdf"
asyncio.run(ask_ollama_with_pdf_context(pdf_file_path))