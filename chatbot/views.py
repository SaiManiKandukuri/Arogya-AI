"""
Django Views for Medical Chatbot Application.

Handles template rendering for the primary chat interface and processes incoming
AJAX POST requests by executing similarity searches on Pinecone and generating
context-aware answers using Groq LLM (llama-3.3-70b-versatile) via LangChain.
"""

import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods
from dotenv import load_dotenv

from src.prompt import system_prompt

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Cache vector store retriever
docsearch_instance = None
embeddings_instance = None

def get_retriever():
    global docsearch_instance, embeddings_instance
    if docsearch_instance is None:
        print("[RAG] Initializing HuggingFace Embeddings & Pinecone Vector Store...")
        from langchain_pinecone import PineconeVectorStore
        from src.helper import download_hugging_face_embeddings
        embeddings_instance = download_hugging_face_embeddings()
        index_name = "medical-chatbot"
        docsearch_instance = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings_instance
        )
    return docsearch_instance.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )


def create_rag_chain():
    """
    Constructs a fresh RAG pipeline per request.
    Ensures HTTP client session (Groq/OpenAI) remains open and active.
    """
    from langchain_core.prompts import ChatPromptTemplate
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        try:
            from langchain_classic.chains.retrieval import create_retrieval_chain
            from langchain_classic.chains.combine_documents import create_stuff_documents_chain
        except ImportError:
            from langchain.chains.retrieval import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain

    retriever = get_retriever()
    
    # Instantiate fresh LLM per request to prevent closed client transport errors
    if GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            chat_model = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,
                groq_api_key=GROQ_API_KEY
            )
        except ImportError:
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        chat_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3
        )
    else:
        raise ValueError("No valid API key found! Please set GROQ_API_KEY or OPENAI_API_KEY in .env.")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)



def index(request):
    """
    Renders the main chat GUI page (templates/chat.html).
    """
    return render(request, 'chat.html')


@require_http_methods(["POST"])
def get_response(request):
    """
    AJAX Endpoint handling user chat queries via POST.
    
    1. Extracts 'msg' parameter from request.POST.
    2. Invokes RAG chain.
    3. Returns JSON response containing generated answer.
    """
    user_message = request.POST.get('msg', '').strip()
    
    if not user_message:
        return HttpResponseBadRequest("Empty message received.")
    
    try:
        # Construct fresh RAG pipeline per request
        chain = create_rag_chain()
        
        # Execute RAG query (Retrieve context + LLM response generation)
        chain_response = chain.invoke({"input": user_message})
        bot_answer = chain_response.get("answer", "I'm sorry, I couldn't process your query.")
        
        # Return JsonResponse for AJAX consumption
        return JsonResponse({"answer": bot_answer, "status": "success"})
        
    except Exception as e:
        print(f"[ERROR] Error during RAG execution: {str(e)}")
        return JsonResponse(
            {"answer": f"An error occurred while generating response: {str(e)}", "status": "error"},
            status=500
        )


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts plain text content from uploaded PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_file)
        text_content = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content.append(extracted)
        return "\n".join(text_content).strip()
    except Exception as e:
        print(f"[PDF EXTRACTION ERROR] {e}")
        return ""


def clean_for_tts(text: str) -> str:
    """Sanitizes text output to remove markdown headers, ASCII bars, and tables for smooth TTS reading."""
    import re
    cleaned = re.sub(r'\[[█░\-]+\]', '', text)       # Remove ASCII gauge bars
    cleaned = re.sub(r'[#*`|_~]', '', text)         # Remove markdown formatting characters
    cleaned = re.sub(r'🟢|🟡|🔴|⚠️|📋|🏫|📊', '', text) # Remove emojis
    cleaned = re.sub(r'\s+', ' ', cleaned)          # Collapse extra whitespaces
    return cleaned.strip()


@require_http_methods(["POST"])
def analyze_report(request):
    """
    AJAX Endpoint handling PDF Medical Report uploads.
    Extracts text from PDF, runs LLM chain with report_analyzer_prompt,
    returns structured answer and clean TTS audio text.
    """
    if 'report' not in request.FILES:
        return JsonResponse({"answer": "No medical report PDF file uploaded.", "status": "error"}, status=400)
    
    pdf_file = request.FILES['report']
    user_message = request.POST.get('msg', '').strip()
    language = request.POST.get('language', 'en').strip()

    report_text = extract_text_from_pdf(pdf_file)
    if not report_text:
        return JsonResponse({
            "answer": "Could not extract readable text from the uploaded PDF. Please upload a clear text-based PDF medical report.",
            "status": "error"
        }, status=400)

    try:
        # Instantiate LLM (Groq / OpenAI)
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                chat_model = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    groq_api_key=GROQ_API_KEY
                )
            except ImportError:
                from langchain_openai import ChatOpenAI
                chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        elif OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        else:
            raise ValueError("No valid API key configured! Please set GROQ_API_KEY or OPENAI_API_KEY in .env.")

        from langchain_core.prompts import PromptTemplate
        from src.prompt import report_analyzer_prompt

        prompt_template = PromptTemplate(
            template=report_analyzer_prompt,
            input_variables=["language", "report_text", "user_message"]
        )
        
        target_lang = "Telugu (తెలుగు script)" if language.lower() in ['te', 'telugu'] else "English"
        formatted_prompt = prompt_template.format(
            language=target_lang,
            report_text=report_text[:7000],
            user_message=user_message or "Please analyze this medical report and summarize key values."
        )

        response = chat_model.invoke(formatted_prompt)
        raw_answer = response.content if hasattr(response, 'content') else str(response)

        # Generate clean plain text for Text-To-Speech (TTS)
        tts_text = clean_for_tts(raw_answer)

        return JsonResponse({
            "answer": raw_answer,
            "tts_text": tts_text,
            "filename": pdf_file.name,
            "status": "success"
        })

    except Exception as e:
        print(f"[REPORT ANALYZER ERROR] {str(e)}")
        return JsonResponse({
            "answer": f"An error occurred while analyzing the medical report: {str(e)}",
            "status": "error"
        }, status=500)

