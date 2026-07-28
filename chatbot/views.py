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
