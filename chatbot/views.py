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

# Cache LLM instances and vector store retriever
docsearch_instance = None
embeddings_instance = None
cached_llm_fast = None
cached_llm_versatile = None

def get_llm(model_type="fast"):
    """Returns cached ChatGroq / ChatOpenAI model instance."""
    global cached_llm_fast, cached_llm_versatile
    if GROQ_API_KEY:
        from langchain_groq import ChatGroq
        model_name = "llama-3.1-8b-instant" if model_type == "fast" else "llama-3.3-70b-versatile"
        if model_type == "fast":
            if cached_llm_fast is None:
                cached_llm_fast = ChatGroq(model_name=model_name, temperature=0.3, groq_api_key=GROQ_API_KEY)
            return cached_llm_fast
        else:
            if cached_llm_versatile is None:
                cached_llm_versatile = ChatGroq(model_name=model_name, temperature=0.3, groq_api_key=GROQ_API_KEY)
            return cached_llm_versatile
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    else:
        raise ValueError("No valid API key found! Please set GROQ_API_KEY or OPENAI_API_KEY in .env.")


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
        search_kwargs={"k": 2}
    )


def create_rag_chain():
    """Constructs a RAG pipeline using fast Groq LLM."""
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
    chat_model = get_llm("fast")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def is_greeting_or_smalltalk(msg: str) -> bool:
    """Checks if message is short greeting/chitchat to bypass vector search."""
    clean = msg.lower().strip().strip("!.,?")
    greetings = {"hi", "hello", "hey", "namaste", "good morning", "good evening", "good afternoon", "who are you", "what can you do", "help", "thanks", "thank you", "bye"}
    return clean in greetings or len(clean.split()) <= 2 and clean in greetings


def index(request):
    """
    Renders the main chat GUI page (templates/chat.html).
    """
    return render(request, 'chat.html')


@require_http_methods(["POST"])
def get_response(request):
    """
    AJAX Endpoint handling user chat queries via POST.
    Optimized with short-circuiting for greetings & fast Groq model.
    """
    user_message = request.POST.get('msg', '').strip()
    language = request.POST.get('language', 'en').strip()
    
    if not user_message:
        return HttpResponseBadRequest("Empty message received.")
    
    try:
        # Instant response for simple greetings without vector lookup
        if is_greeting_or_smalltalk(user_message):
            if language.lower() in ['te', 'telugu']:
                answer = "నమస్కారం! నేను మీ ఆరోగ్య-AI వైద్య సహాయకుడిని. ఈరోజు మీకు ఎలా సహాయపడగలను?"
            else:
                answer = "Hello! I am your Arogya-AI Medical Assistant. How can I assist you with your health today?"
            return JsonResponse({"answer": answer, "tts_text": clean_for_tts(answer), "status": "success"})

        # RAG query execution for medical questions using fast Groq model
        chain = create_rag_chain()
        chain_response = chain.invoke({"input": user_message})
        bot_answer = chain_response.get("answer", "I'm sorry, I couldn't process your query.")
        
        return JsonResponse({"answer": bot_answer, "tts_text": clean_for_tts(bot_answer), "status": "success"})
        
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
        raw_text = "\n".join(text_content).strip()
        return compress_report_text(raw_text)
    except Exception as e:
        print(f"[PDF EXTRACTION ERROR] {e}")
        return ""


def compress_report_text(raw_text: str, max_chars: int = 3500) -> str:
    """Removes excessive blank lines, boilerplate footers, and caps report text size for LLM token limits."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    cleaned = "\n".join(lines)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + "\n...[Report text trimmed for token optimization]"
    return cleaned


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
    Handles Groq TPM rate limits gracefully with automatic fast-model fallback.
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

    from langchain_core.prompts import PromptTemplate
    from src.prompt import report_analyzer_prompt

    prompt_template = PromptTemplate(
        template=report_analyzer_prompt,
        input_variables=["language", "report_text", "user_message"]
    )
    
    target_lang = "Telugu (తెలుగు script)" if language.lower() in ['te', 'telugu'] else "English"
    formatted_prompt = prompt_template.format(
        language=target_lang,
        report_text=report_text,
        user_message=user_message or "Please analyze this medical report and summarize key values."
    )

    try:
        raw_answer = None

        # 1. Primary Attempt with Groq (llama-3.3-70b-versatile)
        if GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                chat_model = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3,
                    groq_api_key=GROQ_API_KEY
                )
                response = chat_model.invoke(formatted_prompt)
                raw_answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                err_str = str(e).lower()
                if "413" in err_str or "rate_limit" in err_str or "tpm" in err_str or "tokens" in err_str:
                    print(f"[GROQ RATE LIMIT] Falling back to llama-3.1-8b-instant for fast execution...")
                    try:
                        fallback_model = ChatGroq(
                            model_name="llama-3.1-8b-instant",
                            temperature=0.3,
                            groq_api_key=GROQ_API_KEY
                        )
                        response = fallback_model.invoke(formatted_prompt)
                        raw_answer = response.content if hasattr(response, 'content') else str(response)
                    except Exception as fallback_err:
                        print(f"[GROQ FALLBACK ERROR] {fallback_err}")
                if not raw_answer:
                    raise e

        # 2. OpenAI Fallback if Groq is unavailable
        if not raw_answer and OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            response = chat_model.invoke(formatted_prompt)
            raw_answer = response.content if hasattr(response, 'content') else str(response)

        if not raw_answer:
            raise ValueError("No LLM model returned a response.")

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


