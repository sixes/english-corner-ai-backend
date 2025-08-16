import os
import json
import glob
import numpy as np
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.genai as genai
from typing import List, Dict, Any, Optional
import requests  # For Groq and other free AI APIs

# LangChain imports
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables import ConfigurableField
from langchain_core.retrievers import BaseRetriever

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please add it to your .env file.")

# Optional: Groq API key for additional free models
groq_api_key = os.getenv("GROQ_API_KEY")  # Optional fallback

# Configure proxy settings only on macOS
import platform
import os

# Detect if we're running on macOS
is_mac = platform.system() == "Darwin"

if is_mac:
    print("Detected macOS - configuring SOCKS5 proxy for Google API access")
    # Set proxy environment variables for requests/httpx
    os.environ['HTTP_PROXY'] = 'socks5://127.0.0.1:51837'
    os.environ['HTTPS_PROXY'] = 'socks5://127.0.0.1:51837'
    os.environ['http_proxy'] = 'socks5://127.0.0.1:51837'
    os.environ['https_proxy'] = 'socks5://127.0.0.1:51837'
else:
    print("Detected non-macOS system - using direct connection")

# Configure the new google-genai client
client = genai.Client(api_key=api_key)

# Check if we're in development mode (no API access)
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_DATA_DIR = "/home/ef/fetch_ec_signup_data/sessions_data"  # Path where session JSON files are stored

# For local development, you can override this
if not os.path.exists(SESSION_DATA_DIR):
    # Try local alternatives
    local_alternatives = [
        "./sessions_data",
        "../sessions_data",
        "~/sessions_data"
    ]
    for alt_path in local_alternatives:
        expanded_path = os.path.expanduser(alt_path)
        if os.path.exists(expanded_path):
            SESSION_DATA_DIR = expanded_path
            break
    print(f"Using session data directory: {SESSION_DATA_DIR}")
else:
    print(f"Using VPS session data directory: {SESSION_DATA_DIR}")

# Combined documents = static known info + dynamic sessions
DOCUMENTS = [
    {"id": "info_1", "text": "Our English Corner runs weekly on Wednesdays and Fridays at 19:30, ending at 22:00, lasting about 2.5 hours."},
    {"id": "info_2", "text": "The venue is Starbucks (联通大厦店) near Futian Station (subway station)."},
    {"id": "info_3", "text": "Each session consists of 3 sections: 30 minutes of self-introduction (one by one), 1 hour playing a warmup game, and 1 hour dedicated to discussing a specific topic."},
    {"id": "info_4", "text": "To join: 1) Scan QR code from WeChat Official Account (公众号：深圳英语角) to add volunteer as contact, 2) Send 1-minute self-introduction voice message to ensure eligibility, 3) Read group notice after being added to WeChat group."},
    {"id": "info_5", "text": "Sometimes we have game sessions on Sundays at 9:30 located at Shenzhen North Station."},
    {"id": "info_6", "text": "The English Corner has been running for more than 7 years, with the earliest session dating back to 2017. It continued running even during the pandemic."},
    {"id": "info_7", "text": "We welcome diverse people from countries like Germany, Australia, South Korea, Hong Kong, Taiwan, UK, USA, Russia, and more."},
    {"id": "info_8", "text": "Age requirement: Attendees must be over 18 years old. We have had attendees over 80 years old as well."},
    {"id": "info_9", "text": "The sessions are currently free of charge."},
    {"id": "info_10", "text": "Uncle Eric founded this English corner around 2017 and is a strong supporter. Currently, Tony organizes English gatherings at Futian station, while Uncle Eric runs board game sessions at Shenzhen North Station."},
    {"id": "info_11", "text": "We meet people from all walks of life: foreign trade workers, engineers, teachers, tutors, freelancers, and more. Expats join us occasionally, but we can't guarantee foreigners at every session."},
    {"id": "info_12", "text": "The warmup game is called 'One Truth, One Lie' (adapted from '2 Truths, 1 Lie'). It helps practice speaking and listening skills. Participation is encouraged but not compulsory."},
    {"id": "info_13", "text": "We don't accept walk-ins. You need basic English skills to make a self-introduction. If you're not at entry level, consider joining other English corners."},
    {"id": "info_14", "text": "After sending self-introduction voice messages, eligible participants are invited to 'Language Exchange 2' WeChat group. We also have 'English Corner Forever' group which is at capacity."},
    {"id": "info_15", "text": "Check WeChat Official Account (公众号：深圳英语角) for latest posts to know upcoming session topics in advance."},
    {"id": "info_16", "text": "Since we have friends from other countries speaking varied languages, we want to be more inclusive, so our second group is named 'Language Exchange 2' rather than just English Corner."},
]

SESSIONS = []  # Full session dicts with structured fields, loaded from files

# LangChain components
vector_store = None
conversation_chain = None
memory = None

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None  # Optional session ID for conversation tracking

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"  # Session ID for conversation tracking

# Google AI model rotation configuration (expanded list)
GOOGLE_AI_MODELS = [
    "gemini-1.5-flash",           # Primary model (fast, efficient)
    "gemini-1.5-pro",             # Secondary model (more capable) 
    "gemini-1.0-pro",             # Stable fallback model
    "gemini-2.0-flash-exp",       # Latest experimental
    "gemini-1.5-flash-exp",       # Flash experimental
    "gemini-1.5-pro-exp",         # Pro experimental
    "gemini-exp-1114",            # Other experimental models
    "gemini-exp-1121",            # Additional experimental
]

# Track which models have hit quota limits (reset on startup)
quota_exceeded_models = set()  # For LLM models
quota_exceeded_embedding_models = set()  # For embedding models (separate tracking)

# Groq free models (as backup when Google AI exhausted)
GROQ_MODELS = [
    "llama-3.1-70b-versatile",    # Meta Llama 3.1 70B
    "llama-3.1-8b-instant",       # Meta Llama 3.1 8B (faster)
    "mixtral-8x7b-32768",         # Mixtral 8x7B
    "gemma2-9b-it",               # Google Gemma 2 9B
]

def reset_quota_tracking():
    """Reset quota tracking on startup or when needed"""
    global quota_exceeded_models, quota_exceeded_embedding_models
    quota_exceeded_models.clear()
    quota_exceeded_embedding_models.clear()
    print("Quota tracking reset - all models marked as available")

# Custom Google AI LLM for LangChain
class GoogleAILLM(LLM):
    """Custom Google AI LLM wrapper with model rotation for quota management"""

    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self._primary_model = model_name or "gemini-1.5-flash"

    def _get_available_models(self) -> List[str]:
        """Get list of available models that haven't exceeded quota"""
        if DEV_MODE:
            return GOOGLE_AI_MODELS
            
        print(f"[DEBUG] Checking available models. quota_exceeded_models: {quota_exceeded_models}")
        
        try:
            # Get actual available models from Google AI
            print("[DEBUG] Calling client.models.list()...")
            models = client.models.list()
            print(f"[DEBUG] Raw models response: {models}")
            
            available_model_names = []
            for model in models:
                print(f"[DEBUG] Processing model: {model}")
                if hasattr(model, 'name'):
                    available_model_names.append(model.name)
                    print(f"[DEBUG] Added model name: {model.name}")
                else:
                    print(f"[DEBUG] Model has no 'name' attribute: {model}")
            
            print(f"[DEBUG] Google AI reports {len(available_model_names)} total models available")
            print(f"[DEBUG] Available model names: {available_model_names}")
            
            # Filter out models that have exceeded quota and match our preferred list
            available_models = []
            for model in GOOGLE_AI_MODELS:
                if model in available_model_names and model not in quota_exceeded_models:
                    available_models.append(model)
                    print(f"[DEBUG] Model {model}: Available")
                elif model not in available_model_names:
                    print(f"[DEBUG] Model {model}: Not found in Google AI list")
                elif model in quota_exceeded_models:
                    print(f"[DEBUG] Model {model}: Quota exceeded")
            
            # If all preferred models are quota-exceeded, try other available models
            if not available_models:
                print("[DEBUG] No preferred models available, checking other Gemini models...")
                for model_name in available_model_names:
                    if model_name not in quota_exceeded_models and 'gemini' in model_name.lower():
                        available_models.append(model_name)
                        print(f"[DEBUG] Found alternative model: {model_name}")
            
            print(f"[DEBUG] Final available models: {available_models}")
            return available_models
            
        except Exception as e:
            print(f"[ERROR] Exception getting available models: {e}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            # Fallback to our predefined list when API listing fails
            fallback_models = [model for model in GOOGLE_AI_MODELS if model not in quota_exceeded_models]
            print(f"[DEBUG] API listing failed, using fallback models: {fallback_models}")
            # If no fallback models available, at least try the most common one
            if not fallback_models:
                print("[DEBUG] No fallback models, trying gemini-1.5-flash as last resort")
                fallback_models = ["gemini-1.5-flash"]
            return fallback_models

    @property
    def _llm_type(self) -> str:
        return "google_genai_rotating"

    def _call_with_model(self, prompt: str, model_name: str) -> str:
        """Call Google AI with a specific model"""
        try:
            print(f"Calling Google AI with model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            print(f"Success with model {model_name}")
            return response.text
            
        except Exception as e:
            error_str = str(e)
            print(f"Error with model {model_name}: {error_str}")
            
            # Check if it's a quota error
            if any(keyword in error_str.lower() for keyword in ["429", "quota", "resource_exhausted", "rate limit"]):
                print(f"Quota exceeded for model {model_name}, marking as unavailable")
                quota_exceeded_models.add(model_name)
                raise Exception(f"QUOTA_EXCEEDED_{model_name}")
            else:
                # Other error, re-raise
                raise e

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if DEV_MODE:
            return f"Mock response for development mode. Query: {prompt[:100]}..."

        print(f"[DEBUG] Starting LLM call. Current quota_exceeded_models: {quota_exceeded_models}")

        # Get available models
        available_models = self._get_available_models()
        print(f"[DEBUG] Available models after filtering: {available_models}")
        
        if not available_models:
            # Reset quota tracking if all models are exhausted (maybe quotas have reset)
            print("All models quota-exceeded, resetting quota tracking...")
            quota_exceeded_models.clear()
            available_models = self._get_available_models()
            print(f"[DEBUG] Available models after reset: {available_models}")
        
        if not available_models:
            # Still no models available, return helpful error
            return (
                "I'm sorry, but all Google AI models have reached their daily quota limits. "
                "This typically resets at midnight Pacific Time. Please try again later, "
                "or consider upgrading to a paid Google AI plan for higher limits."
            )

        # Try each available model in order
        last_error = None
        for model_name in available_models:
            print(f"[DEBUG] Trying model: {model_name}")
            try:
                result = self._call_with_model(prompt, model_name)
                print(f"[DEBUG] Model {model_name} succeeded!")
                return result
                
            except Exception as e:
                last_error = str(e)
                print(f"[DEBUG] Model {model_name} failed: {last_error}")
                if "QUOTA_EXCEEDED" in last_error:
                    print(f"Model {model_name} quota exceeded, trying next model...")
                    continue
                else:
                    print(f"Model {model_name} failed with non-quota error: {last_error}")
                    # For non-quota errors, still try next model but with a delay
                    import time
                    time.sleep(1)
                    continue

        # If we get here, all models failed
        print(f"[DEBUG] All models failed. Last error: {last_error}")
        return (
            f"I'm sorry, but I'm currently experiencing technical difficulties with Google AI services. "
            f"Please try again in a moment. Technical details: {last_error}"
        )

# Groq LLM class for free fallback when Google AI is exhausted
class GroqLLM(LLM):
    """Custom Groq LLM wrapper for free API access as fallback"""
    
    def __init__(self, api_key: str = None, model_name: str = "llama-3.1-70b-versatile"):
        super().__init__()
        self.api_key = api_key or groq_api_key
        self.model_name = model_name
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call Groq API with the given prompt"""
        if not self.api_key:
            return "Groq API key not configured. Please add GROQ_API_KEY to your .env file."
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"Groq API error: {str(e)}"

# Multi-provider LLM that tries Google AI first, then Groq as fallback
class MultiProviderLLM(LLM):
    """LLM that tries Google AI models first, then falls back to Groq if all Google models are exhausted"""
    
    def __init__(self):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, '_google_llm', GoogleAILLM())
        object.__setattr__(self, '_groq_llm', GroqLLM() if groq_api_key else None)
        
    @property
    def _llm_type(self) -> str:
        return "multi_provider"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Try Google AI first, then Groq if all Google models exhausted"""
        
        # First try Google AI with all its models
        try:
            result = self._google_llm._call(prompt, stop, run_manager)
            
            # Check if Google AI returned a quota/limit error message
            quota_indicators = [
                "daily quota limits",
                "technical difficulties with Google AI services",
                "all Google AI models have reached their daily quota"
            ]
            
            google_exhausted = any(indicator in result for indicator in quota_indicators)
            
            if not google_exhausted:
                return result
                
        except Exception as e:
            print(f"Google AI failed with exception: {e}")
            google_exhausted = True
        
        # If Google AI is exhausted and we have Groq configured, try Groq
        if google_exhausted and self._groq_llm:
            print("Google AI exhausted, trying Groq as fallback...")
            try:
                result = self._groq_llm._call(prompt, stop, run_manager)
                # Add a note that we're using fallback
                return f"{result}\n\n*(Response generated using fallback AI service due to quota limits)*"
            except Exception as e:
                print(f"Groq also failed: {e}")
        
        # If both failed or Groq not configured, return Google AI result
        return result if 'result' in locals() else (
            "I'm sorry, but I'm currently experiencing technical difficulties with AI services. "
            "Please try again later."
        )

# Custom Google AI Embeddings for LangChain
class GoogleAIEmbeddings(Embeddings):
    def __init__(self, model_name: str = "text-embedding-004"):
        self.model_name = model_name
        self.embedding_models = ["text-embedding-004", "text-embedding-003"]  # Alternative embedding models
        # Use global embedding quota tracking (separate from LLM quota tracking)

    def _get_embedding_with_rotation(self, text: str) -> Optional[List[float]]:
        """Get embedding with model rotation"""
        global quota_exceeded_embedding_models
        
        if DEV_MODE:
            print(f"DEV_MODE: Creating mock embedding for text: {text[:50]}...")
            import hashlib
            import numpy as np
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            mock_embedding = np.random.normal(0, 1, 768).tolist()
            return mock_embedding

        # Try each embedding model
        available_models = [m for m in self.embedding_models if m not in quota_exceeded_embedding_models]
        
        if not available_models:
            # Reset if all models are quota-exceeded
            quota_exceeded_embedding_models.clear()
            available_models = self.embedding_models

        for model_name in available_models:
            try:
                print(f"Getting embedding with model: {model_name}")
                
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Embedding API call timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)

                try:
                    response = client.models.embed_content(
                        model=model_name,
                        contents=text
                    )
                    signal.alarm(0)

                    # Extract embedding data (same logic as before)
                    if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                        embedding_obj = response.embeddings[0]
                        if hasattr(embedding_obj, 'values'):
                            embedding_data = embedding_obj.values
                        else:
                            embedding_data = embedding_obj
                    elif hasattr(response, 'embedding'):
                        embedding_obj = response.embedding
                        if hasattr(embedding_obj, 'values'):
                            embedding_data = embedding_obj.values
                        else:
                            embedding_data = embedding_obj
                    elif hasattr(response, 'values'):
                        embedding_data = response.values
                    else:
                        print(f"Unknown response format for {model_name}")
                        continue

                    if hasattr(embedding_data, 'values'):
                        embedding_data = embedding_data.values

                    if isinstance(embedding_data, (list, tuple)):
                        embedding_list = [float(x) for x in embedding_data]
                        if embedding_list and len(embedding_list) > 0:
                            print(f"Embedding successful with {model_name}, got {len(embedding_list)} dimensions")
                            return embedding_list

                finally:
                    signal.alarm(0)

            except Exception as e:
                error_str = str(e)
                if any(keyword in error_str.lower() for keyword in ["429", "quota", "resource_exhausted"]):
                    print(f"Embedding quota exceeded for {model_name}")
                    quota_exceeded_embedding_models.add(model_name)
                    continue
                else:
                    print(f"Embedding error with {model_name}: {e}")
                    continue

        print("All embedding models failed or quota exceeded")
        return None

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Use the new rotation method"""
        return self._get_embedding_with_rotation(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                # Fallback to zero vector if embedding fails
                embeddings.append([0.0] * 768)  # Assuming 768-dim embeddings
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self._get_embedding(text)
        return embedding if embedding else [0.0] * 768

# Global memory storage for different sessions
session_memories: Dict[str, ConversationBufferWindowMemory] = {}

# Simple in-memory vector store to avoid SQLite dependency issues
class SimpleVectorStore(VectorStore):
    """Simple in-memory vector store using numpy for similarity search"""

    def __init__(self, embedding_function: Embeddings):
        self.embedding_function = embedding_function
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        self._embeddings: List[List[float]] = []
        self._documents: List[Document] = []

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None, **kwargs) -> List[str]:
        """Add texts to the vector store"""
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Generate embeddings for new texts
        new_embeddings = self.embedding_function.embed_documents(texts)

        # Store everything
        start_idx = len(self._texts)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)
        self._embeddings.extend(new_embeddings)

        # Create document objects
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            doc = Document(page_content=text, metadata=metadata)
            self._documents.append(doc)

        return [str(start_idx + i) for i in range(len(texts))]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Return docs most similar to query"""
        if not self._texts:
            return []

        # Get query embedding
        query_embedding = self.embedding_function.embed_query(query)

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self._embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, i))

        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in similarities[:k]]

        return [self._documents[i] for i in top_indices]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        **kwargs
    ) -> "SimpleVectorStore":
        """Create a vector store from texts (required abstract method)"""
        instance = cls(embedding)
        instance.add_texts(texts, metadatas)
        return instance

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs
    ) -> "SimpleVectorStore":
        """Create a vector store from documents"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(texts, embedding, metadatas)

    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        """Return a retriever interface"""
        if search_kwargs is None:
            search_kwargs = {}

        def retrieve(query: str) -> List[Document]:
            k = search_kwargs.get("k", 4)
            return self.similarity_search(query, k=k)

        return SimpleRetriever(retrieve)

class SimpleRetriever(BaseRetriever):
    """Simple retriever wrapper that properly inherits from BaseRetriever"""

    def __init__(self, retrieve_func, **kwargs):
        super().__init__(**kwargs)
        # Store the function in a way that doesn't conflict with Pydantic
        object.__setattr__(self, '_retrieve_func', retrieve_func)

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Required method for BaseRetriever"""
        return self._retrieve_func(query)

# --- Utility functions ---

def get_embedding(text, model="text-embedding-004"):
    """Legacy function kept for backward compatibility"""
    try:
        response = client.models.embed_content(
            model=model,
            contents=text
        )
        return response.values
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0
    return np.dot(vec1, vec2) / denom

def get_memory_for_session(session_id: str) -> ConversationBufferWindowMemory:
    """Get or create memory for a specific session"""
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            k=20,  # Remember last 10 exchanges
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Add output_key to avoid deprecation warning
        )
    return session_memories[session_id]

def create_documents_from_data() -> List[Document]:
    """Convert static info and sessions into LangChain Documents"""
    documents = []

    # Add static documents
    for doc in DOCUMENTS:
        documents.append(Document(
            page_content=doc["text"],
            metadata={"id": doc["id"], "type": "static_info"}
        ))

    # Add session documents
    for session in SESSIONS:
        documents.append(Document(
            page_content=session["text"],
            metadata={
                "id": session["id"] + 366,
                "type": "session",
                "date": session["date_str"],
                "topic": session["topic"],
                "location": session["location"],
                "participant_count": session["participant_count"]
            }
        ))

    return documents

def setup_vector_store_and_chain():
    """Initialize simple in-memory vector store"""
    global vector_store, conversation_chain

    print("Creating embeddings instance...")
    # Create embeddings
    embeddings = GoogleAIEmbeddings()

    print("Creating documents from data...")
    # Create documents
    documents = create_documents_from_data()

    if not documents:
        raise ValueError("No documents available to create vector store")

    print(f"Created {len(documents)} documents, building vector store...")
    # Create vector store using simple in-memory implementation
    vector_store = SimpleVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

    print("Vector store setup completed!")
    # Note: We're not using the complex ConversationalRetrievalChain anymore
    # to avoid recursion issues. We'll handle conversation logic directly.

def parse_session_item(session_raw):
    # Extract structured metadata from raw session dict loaded from file
    # Supports your DynamoDB session JSON format converted to standard dict with keys:
    # id (str), date (str, yyyy-mm-dd), time (str), topic (str), location (str), participants (list of dict with 'name' key)
    id_ = int(session_raw.get("id"))
    date_str = session_raw.get("date")
    time = session_raw.get("time", "Unknown time")
    topic = session_raw.get("topic", "No topic")
    location = session_raw.get("location", "No location")
    participants = session_raw.get("participants", [])

    # Parse date to datetime date_obj
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
    except Exception:
        date_obj = None

    # Participant names extraction handling different structures
    participant_names = []
    if isinstance(participants, list):
        for p in participants:
            # Some JSON may have nested dict structure (DynamoDB style), flatten it
            # e.g. {"M": {"name": {"S": "Tony"}}}
            if isinstance(p, dict):
                if "M" in p and "name" in p["M"]:
                    # DynamoDB style
                    name = p["M"]["name"].get("S") if isinstance(p["M"]["name"], dict) else p["M"]["name"]
                    if name:
                        participant_names.append(name)
                elif isinstance(p.get("name"), dict):
                    # Fallback in nested dict
                    name = p["name"].get("S")
                    if name:
                        participant_names.append(name)
                elif isinstance(p.get("name"), str):
                    participant_names.append(p["name"])

            elif isinstance(p, str):
                participant_names.append(p)

    # Compose human-readable text description for embedding
    text = (
        f"Session ID: {id_}\n"
        f"Date: {date_str}\n"
        f"Time: {time}\n"
        f"Location: {location}\n"
        f"Topic: {topic}\n"
        f"Participants ({len(participant_names)}): {', '.join(participant_names)}"
    )

    return {
        "id": id_,
        "date_str": date_str,
        "date_obj": date_obj,
        "time": time,
        "topic": topic,
        "location": location,
        "participant_names": participant_names,
        "participant_count": len(participant_names),
        "text": text,
    }

def load_sessions():
    sessions = []
    files = glob.glob(os.path.join(SESSION_DATA_DIR, "session_*.json"))
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                session = parse_session_item(raw_data)
                if session["id"] is not None:
                    sessions.append(session)
        except Exception as e:
            print(f"Failed to load session from {fpath}: {e}")
    return sessions

# Legacy functions removed - now using LangChain vector store

# --- Intent detection & rule-based direct answers ---

def detect_intent(question: str):
    q_lower = question.lower()

    if any(kw in q_lower for kw in ["last session", "most recent session", "when was the last"]):
        return "last_session_date"
    elif any(kw in q_lower for kw in ["who attended most", "most frequent attendee", "most attendees", "who has attended the most"]):
        return "top_participant"
    return None

def get_last_session_date():
    sessions_with_date = [s for s in SESSIONS if s["date_obj"] is not None]
    if not sessions_with_date:
        return "No sessions with valid dates were found."

    last_sess = max(sessions_with_date, key=lambda s: s["date_obj"])
    return (
        f"The last English Corner session was on {last_sess['date_str']} "
        f"with topic '{last_sess['topic']}' held at {last_sess['location']}. "
        f"It had {last_sess['participant_count']} participants."
    )

def get_top_attendee():
    all_names = []
    for s in SESSIONS:
        if s.get("participant_names"):
            all_names.extend(s["participant_names"])

    if not all_names:
        return "There are no participants recorded in any session."

    counts = Counter(all_names)
    top_attendee, count = counts.most_common(1)[0]
    return f"The participant who has attended the most sessions is {top_attendee}, attending {count} session(s)."

# build_prompt function removed - LangChain handles prompt formatting internally

# --- FastAPI Lifecycle and Routes ---

@app.on_event("startup")
async def startup_event():
    global SESSIONS
    print("Loading sessions from files...")
    SESSIONS = load_sessions()
    print(f"Loaded {len(SESSIONS)} sessions.")

    print("Resetting quota tracking...")
    reset_quota_tracking()

    print("Setting up vector store...")
    setup_vector_store_and_chain()
    print("Setup complete!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "English Corner RAG Backend is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "sessions_loaded": len(SESSIONS),
        "vector_store_ready": vector_store is not None,
        "active_chat_sessions": len(session_memories)
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    """Legacy endpoint for backward compatibility"""
    print(f"Received request at /ask: {request.question}")  # Debug log
    return await chat(ChatRequest(question=request.question, session_id=request.session_id or "default"))

@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"Received chat request: {request.question} (session: {request.session_id})")  # Debug log
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = request.session_id or "default"

    # Get session-specific memory
    session_memory = get_memory_for_session(session_id)

    # Intent detection: direct logic for some queries (enhanced with context awareness)
    intent = detect_intent(question)
    if intent == "last_session_date":
        answer = get_last_session_date()
        # Add to conversation memory
        session_memory.chat_memory.add_user_message(HumanMessage(content=question))
        session_memory.chat_memory.add_ai_message(AIMessage(content=answer))
        return {"answer": answer, "session_id": session_id}
    elif intent == "top_participant":
        answer = get_top_attendee()
        # Add to conversation memory
        session_memory.chat_memory.add_user_message(HumanMessage(content=question))
        session_memory.chat_memory.add_ai_message(AIMessage(content=answer))
        return {"answer": answer, "session_id": session_id}

    # Use a simpler approach without potential recursion issues
    try:
        # Get relevant documents from vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Get chat history for context
        chat_history = session_memory.chat_memory.messages
        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in chat_history[-6:]  # Last 3 exchanges
            ])

        # Create prompt with context and history
        prompt = f"""You are a helpful assistant for the Forever English Corner. Use the provided context and chat history to answer questions accurately.

Context from documents:
{context}

Chat History:
{history_text}

Current Question: {question}

Please provide a helpful and accurate answer based on the context and conversation history.
If you don't find accurate answers in the context, try your best to answer the question using your full capacity, but should let the user know.
Always answer the questions in the same language with the user quetsion. However, if you identify the user's English is poor, answer in both English and Chinese.
Since you'are an assistant, always to encourage users to keep up and learn English. It would be better in a light and humurous way."""

        # Get response from LLM - using Google AI only
        llm = GoogleAILLM()
        answer = llm._call(prompt)

        # Add to conversation memory
        session_memory.chat_memory.add_user_message(HumanMessage(content=question))
        session_memory.chat_memory.add_ai_message(AIMessage(content=answer))

        # Extract source information for debugging
        sources = [{"type": doc.metadata.get("type"), "id": doc.metadata.get("id")}
                  for doc in relevant_docs]

        return {
            "answer": answer,
            "session_id": session_id,
            "sources_used": sources
        }

    except Exception as e:
        print(f"Chat error: {e}")  # Add debug logging
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {e}")

@app.post("/reset_session")
async def reset_session(session_id: str = "default"):
    """Reset conversation memory for a specific session"""
    if session_id in session_memories:
        del session_memories[session_id]
    return {"message": f"Session {session_id} reset successfully"}

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to check loaded sessions data"""
    return {
        "session_count": len(SESSIONS),
        "first_5_sessions": SESSIONS[:5] if SESSIONS else [],
        "vector_store_initialized": vector_store is not None,
        "active_chat_sessions": list(session_memories.keys())
    }

@app.get("/debug/test-google-ai")
async def test_google_ai():
    """Test Google AI API directly"""
    try:
        # Test 1: Direct model call without listing
        print("[TEST] Testing direct model call...")
        test_prompt = "Hello, please respond with 'API test successful'"
        
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=test_prompt
        )
        
        return {
            "test": "direct_model_call",
            "status": "success",
            "response": response.text,
            "model_used": "gemini-1.5-flash"
        }
        
    except Exception as e:
        print(f"[TEST ERROR] Direct model call failed: {e}")
        import traceback
        return {
            "test": "direct_model_call", 
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/ai-status")
async def debug_ai_status():
    """Debug endpoint to check AI provider status"""
    google_status = {
        "available_models": [m for m in GOOGLE_AI_MODELS if m not in quota_exceeded_models],
        "quota_exceeded_llm_models": list(quota_exceeded_models),
        "quota_exceeded_embedding_models": list(quota_exceeded_embedding_models),
        "total_google_models": len(GOOGLE_AI_MODELS)
    }
    
    groq_status = {
        "api_key_configured": bool(groq_api_key),
        "available_models": GROQ_MODELS if groq_api_key else []
    }
    
    return {
        "google_ai": google_status,
        "groq": groq_status,
        "overall_status": "operational" if (
            google_status["available_models"] or groq_status["api_key_configured"]
        ) else "degraded"
    }
