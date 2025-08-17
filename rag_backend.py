import os
import json
import glob
import numpy as np
import logging
import time
import uuid
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, HTTPException, Request, Response
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a separate logger for request/response logging
request_logger = logging.getLogger('request_response')
request_logger.setLevel(logging.INFO)
request_handler = logging.FileHandler('requests.log')
request_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
request_logger.addHandler(request_handler)

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
    logger.info("Detected macOS - configuring SOCKS5 proxy for Google API access")
    # Set proxy environment variables for requests/httpx
    os.environ['HTTP_PROXY'] = 'socks5://127.0.0.1:51837'
    os.environ['HTTPS_PROXY'] = 'socks5://127.0.0.1:51837'
    os.environ['http_proxy'] = 'socks5://127.0.0.1:51837'
    os.environ['https_proxy'] = 'socks5://127.0.0.1:51837'
else:
    logger.info("Detected non-macOS system - using direct connection")

# Configure the new google-genai client
client = genai.Client(api_key=api_key)

# Check if we're in development mode (no API access)
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

app = FastAPI()

# Middleware for logging all requests and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generate unique request ID
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Log request details
    request_body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:
                request_body = body.decode('utf-8')
        except Exception as e:
            request_body = f"Error reading body: {str(e)}"
    
    request_info = {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "client_ip": request.client.host if request.client else "unknown",
        "body": request_body,
        "timestamp": datetime.now().isoformat()
    }
    
    request_logger.info(f"REQUEST [{request_id}]: {json.dumps(request_info, ensure_ascii=False)}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response details
    response_info = {
        "request_id": request_id,
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "process_time_seconds": round(process_time, 4),
        "timestamp": datetime.now().isoformat()
    }
    
    request_logger.info(f"RESPONSE [{request_id}]: {json.dumps(response_info, ensure_ascii=False)}")
    
    # Add request ID to response headers for tracking
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://english-corner-ai-frontend-[a-zA-Z0-9\-]+-sixes2010-3296s-projects\.vercel\.app/?$",
    allow_origins=[
        "https://www.englishcorner.cyou",
        "https://englishcorner.cyou", 
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ],  # Specific domains for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
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
    logger.info(f"Using session data directory: {SESSION_DATA_DIR}")
else:
    logger.info(f"Using VPS session data directory: {SESSION_DATA_DIR}")

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

# Google AI model rotation configuration (updated with stable models first)
GOOGLE_AI_MODELS = [
    "gemini-1.5-flash",           # Primary model (fast, efficient)
    "gemini-1.5-pro",             # Secondary model (more capable) 
    "gemini-2.0-flash-exp",       # Latest experimental (if available)
    # Removed models that are consistently returning 404 errors:
    # "gemini-1.0-pro" - deprecated
    # "gemini-1.5-flash-exp" - not found
    # "gemini-1.5-pro-exp" - not found
]

# This will be populated with actual available models at startup
AVAILABLE_GOOGLE_AI_MODELS = []

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
    logger.info("Quota tracking reset - all models marked as available")

def discover_available_models():
    """Discover available Google AI models at startup"""
    global AVAILABLE_GOOGLE_AI_MODELS
    
    try:
        logger.info("Discovering available Google AI models...")
        logger.info(f"Using API key: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else '***'}")
        
        # Try to list models with timeout and better error handling
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("Model discovery API call timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        try:
            logger.info("Calling client.models.list()...")
            models_pager = client.models.list()
            logger.info(f"Got models_pager: {type(models_pager)}")
            signal.alarm(0)  # Cancel timeout
            
            discovered_models = []
            full_model_names = []
            model_count = 0
            
            logger.info("Iterating through models...")
            for model in models_pager:
                model_count += 1
                logger.info(f"Processing model #{model_count}: {model}")
                
                if hasattr(model, 'name'):
                    full_model_name = model.name
                    full_model_names.append(full_model_name)
                    logger.info(f"Model name: {full_model_name}")
                    
                    # Extract short model name from full path (e.g., "models/gemini-1.5-flash" -> "gemini-1.5-flash")
                    if full_model_name.startswith('models/'):
                        model_name = full_model_name.replace('models/', '')
                    else:
                        model_name = full_model_name
                    
                    # Filter for text generation models (not embedding or vision-only models)
                    if any(keyword in model_name.lower() for keyword in ['gemini']):  # Focus on gemini models
                        discovered_models.append(model_name)
                        logger.info(f"Discovered model: {model_name} (full name: {full_model_name})")
                else:
                    logger.warning(f"Model has no 'name' attribute: {model}")
            
            logger.info(f"Iteration completed. Total models processed: {model_count}")
            logger.info(f"Total models from API: {len(full_model_names)}")
            logger.info(f"All model names: {full_model_names}")
            logger.info(f"Filtered Gemini models: {discovered_models}")
            
            # If we got no models at all, there's an API issue
            if model_count == 0:
                logger.error("No models returned from Google AI API - this indicates an API access problem")
                logger.info("Possible causes: 1) Invalid API key, 2) Network/proxy issues, 3) API service down, 4) Rate limiting")
                
                # Try a simple test API call to verify connectivity
                try:
                    logger.info("Testing API connectivity with a simple generate_content call...")
                    test_response = client.models.generate_content(
                        model="gemini-1.5-flash",  # Try the most common model
                        contents="Hello"
                    )
                    logger.info("API connectivity test successful - API key and network are working")
                    logger.info("The models.list() endpoint might be restricted or temporarily unavailable")
                except Exception as e:
                    logger.error(f"API connectivity test failed: {e}")
                    logger.error("This confirms there's an issue with API access or authentication")
            
            # Prioritize our preferred models, then add any others found
            prioritized_models = []
            
            # First, add our preferred models that are actually available
            for preferred_model in GOOGLE_AI_MODELS:
                if preferred_model in discovered_models:
                    prioritized_models.append(preferred_model)
                    logger.info(f"Preferred model {preferred_model} is available")
                else:
                    logger.warning(f"Preferred model {preferred_model} not found in discovery")
            
            # Then add any other discovered models we haven't listed
            for discovered_model in discovered_models:
                if discovered_model not in prioritized_models:
                    prioritized_models.append(discovered_model)
                    logger.info(f"Adding additional discovered model: {discovered_model}")
            
            AVAILABLE_GOOGLE_AI_MODELS = prioritized_models
            logger.info(f"Available Google AI models: {AVAILABLE_GOOGLE_AI_MODELS}")
            
        except TimeoutError:
            logger.error("Model discovery timed out after 30 seconds")
            raise
        finally:
            signal.alarm(0)  # Ensure timeout is cancelled
        
        # If discovery failed but we want to test our fallback models
        if not AVAILABLE_GOOGLE_AI_MODELS and len(full_model_names) == 0:
            logger.info("Since model discovery failed, testing our fallback models directly...")
            # Skip testing if we already know there's a quota issue
            if model_count == 0:
                logger.warning("Skipping individual model tests since models.list() returned empty")
                logger.warning("This typically indicates quota exhaustion or API access restrictions")
            else:
                for test_model in GOOGLE_AI_MODELS[:2]:  # Test first 2 models
                    try:
                        logger.info(f"Testing fallback model: {test_model}")
                        test_response = client.models.generate_content(
                            model=test_model,
                            contents="Hello"
                        )
                        logger.info(f"Fallback model {test_model} works! Adding to available models.")
                        AVAILABLE_GOOGLE_AI_MODELS.append(test_model)
                        break  # If one works, we're good
                    except Exception as e:
                        logger.warning(f"Fallback model {test_model} failed: {e}")
                        continue
        
        # Final fallback
        if not AVAILABLE_GOOGLE_AI_MODELS:
            logger.warning("No Google AI models discovered or tested successfully - using fallback list")
            logger.info("Note: Models in fallback list may still hit quota limits at runtime")
            AVAILABLE_GOOGLE_AI_MODELS = GOOGLE_AI_MODELS
            
    except Exception as e:
        logger.error(f"Failed to discover Google AI models: {e}")
        import traceback
        logger.error(f"Discovery error traceback: {traceback.format_exc()}")
        logger.info("Using fallback model list")
        AVAILABLE_GOOGLE_AI_MODELS = GOOGLE_AI_MODELS

# Custom Google AI LLM for LangChain
class GoogleAILLM(LLM):
    """Custom Google AI LLM wrapper with model rotation for quota management"""

    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self._primary_model = model_name or "gemini-1.5-flash"

    def _get_available_models(self) -> List[str]:
        """Get list of available models that haven't exceeded quota"""
        if DEV_MODE:
            return AVAILABLE_GOOGLE_AI_MODELS or GOOGLE_AI_MODELS
            
        logger.debug(f"Checking available models. quota_exceeded_models: {quota_exceeded_models}")
        logger.debug(f"AVAILABLE_GOOGLE_AI_MODELS: {AVAILABLE_GOOGLE_AI_MODELS}")
        
        # Use the models discovered at startup, or fallback if empty
        model_source = AVAILABLE_GOOGLE_AI_MODELS if AVAILABLE_GOOGLE_AI_MODELS else GOOGLE_AI_MODELS
        available_models = [model for model in model_source if model not in quota_exceeded_models]
        
        logger.debug(f"Available models after filtering: {available_models}")
        
        # If all models are quota-exceeded, reset quota tracking (quotas may have reset)
        if not available_models:
            logger.debug("All models quota-exceeded, resetting quota tracking...")
            quota_exceeded_models.clear()
            # Try again with reset quota tracking
            available_models = [model for model in model_source if model not in quota_exceeded_models]
            logger.debug(f"Available models after quota reset: {available_models}")
        
        # Final fallback - ensure we always have models to try
        if not available_models:
            logger.warning("No models available even after quota reset, using full fallback list")
            available_models = GOOGLE_AI_MODELS[:3]  # Use first 3 models as last resort
        
        return available_models

    @property
    def _llm_type(self) -> str:
        return "google_genai_rotating"

    def _call_with_model(self, prompt: str, model_name: str) -> str:
        """Call Google AI with a specific model"""
        try:
            logger.debug(f"Calling Google AI with model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            logger.debug(f"Success with model {model_name}")
            return response.text
            
        except Exception as e:
            error_str = str(e)
            logger.debug(f"Error with model {model_name}: {error_str}")
            
            # Log the full exception details for debugging
            import traceback
            logger.debug(f"Full traceback for {model_name}: {traceback.format_exc()}")
            
            # Check if it's a 404 or model not found error
            if any(keyword in error_str.lower() for keyword in ["404", "not_found", "not found", "is not found", "model_not_found"]):
                logger.warning(f"Model {model_name} not found (404), marking as unavailable and auto-switching to next model")
                quota_exceeded_models.add(model_name)
                # Remove from available models list to prevent future attempts
                if model_name in AVAILABLE_GOOGLE_AI_MODELS:
                    AVAILABLE_GOOGLE_AI_MODELS.remove(model_name)
                    logger.warning(f"Removed {model_name} from available models due to 404 error")
                raise Exception(f"MODEL_NOT_FOUND_{model_name}")
            
            # Check if it's a quota error
            elif any(keyword in error_str.lower() for keyword in ["429", "quota", "resource_exhausted", "rate limit"]):
                logger.warning(f"Quota exceeded for model {model_name}, marking as unavailable and auto-switching to next model")
                quota_exceeded_models.add(model_name)
                raise Exception(f"QUOTA_EXCEEDED_{model_name}")
            
            # Check for network/proxy errors
            elif any(keyword in error_str.lower() for keyword in ["connection", "timeout", "proxy", "network"]):
                logger.error(f"Network/connection error with model {model_name}: {error_str}")
                raise Exception(f"NETWORK_ERROR_{model_name}")
            
            else:
                # Other error, re-raise with more context
                logger.error(f"Unexpected error with model {model_name}: {error_str}")
                raise Exception(f"UNKNOWN_ERROR_{model_name}_{error_str}")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if DEV_MODE:
            return f"Mock response for development mode. Query: {prompt[:100]}..."

        logger.debug(f"Starting LLM call. Current quota_exceeded_models: {quota_exceeded_models}")

        # Get available models
        available_models = self._get_available_models()
        logger.debug(f"Available models after filtering: {available_models}")
        
        if not available_models:
            # Reset quota tracking if all models are exhausted (maybe quotas have reset)
            logger.info("All models quota-exceeded, resetting quota tracking...")
            quota_exceeded_models.clear()
            available_models = self._get_available_models()
            logger.debug(f"Available models after reset: {available_models}")
        
        if not available_models:
            # Still no models available, return helpful error
            return (
                "I'm sorry, but all Google AI models have reached their daily quota limits. "
                "This typically resets at midnight Pacific Time. Please try again later, "
                "or consider upgrading to a paid Google AI plan for higher limits."
            )

        # Try each available model in order
        last_error = None
        models_tried = []
        
        for model_name in available_models:
            logger.debug(f"Trying model: {model_name}")
            models_tried.append(model_name)
            
            try:
                result = self._call_with_model(prompt, model_name)
                logger.debug(f"Model {model_name} succeeded! Auto-switching worked.")
                return result
                
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Model {model_name} failed: {last_error}")
                
                if "QUOTA_EXCEEDED" in last_error:
                    logger.info(f"Model {model_name} quota exceeded, auto-switching to next model...")
                    continue
                elif "MODEL_NOT_FOUND" in last_error:
                    logger.info(f"Model {model_name} not found, auto-switching to next model...")
                    continue
                elif "NETWORK_ERROR" in last_error:
                    logger.warning(f"Model {model_name} network error, auto-switching to next model...")
                    continue
                else:
                    logger.warning(f"Model {model_name} failed with error: {last_error}, trying next model...")
                    # For other errors, still try next model but with a delay
                    import time
                    time.sleep(2)  # Increased delay to avoid rate limiting
                    continue

        # If we get here, all models failed
        logger.error(f"All models failed after trying: {models_tried}. Last error: {last_error}")
        
        # Categorize the failures for better error messages
        network_errors = sum(1 for e in [last_error] if "NETWORK_ERROR" in str(e))
        quota_errors = len([m for m in models_tried if m in quota_exceeded_models])
        not_found_errors = sum(1 for e in [last_error] if "MODEL_NOT_FOUND" in str(e))
        
        # Provide more helpful error message based on the type of failures
        if models_tried:
            if quota_errors >= len(models_tried):
                # Calculate hours until midnight PT (UTC-8)
                import datetime
                now_utc = datetime.datetime.utcnow()
                # Convert to PT (UTC-8)
                pt_offset = datetime.timedelta(hours=-8)
                now_pt = now_utc + pt_offset
                # Calculate hours until next midnight PT
                tomorrow_pt = now_pt.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
                hours_until_reset = int((tomorrow_pt - now_pt).total_seconds() / 3600)
                
                return (
                    f"I'm sorry, but Google AI has reached its daily quota limits. "
                    f"The free tier allows 50 requests per day per model, and this limit has been exceeded. "
                    f"The quota resets at midnight Pacific Time (approximately {hours_until_reset} hours from now). "
                    f"You can try again later, or upgrade to a paid Google AI plan for higher limits. "
                    f"Models attempted: {', '.join(models_tried[:3])}{'...' if len(models_tried) > 3 else ''}"
                )
            elif network_errors > 0:
                return (
                    f"I'm sorry, but I'm experiencing network connectivity issues with Google AI services. "
                    f"This might be due to proxy settings or internet connection problems. "
                    f"Please check your network connection or try again later. "
                    f"Models attempted: {', '.join(models_tried[:3])}{'...' if len(models_tried) > 3 else ''}"
                )
            elif not_found_errors > 0:
                return (
                    f"I'm sorry, but some AI models are not accessible. This might be due to "
                    f"API configuration issues or model availability changes. "
                    f"The service is being updated to use only available models. "
                    f"Models attempted: {', '.join(models_tried[:3])}{'...' if len(models_tried) > 3 else ''}"
                )
            else:
                return (
                    f"I'm sorry, but I'm currently experiencing technical difficulties. "
                    f"I tried {len(models_tried)} different AI models but all failed. "
                    f"Last error: {last_error}. "
                    f"Please try again in a few minutes. "
                    f"Models attempted: {', '.join(models_tried[:3])}{'...' if len(models_tried) > 3 else ''}"
                )
        else:
            return (
                f"I'm sorry, but no AI models are currently available. "
                f"This might be due to service maintenance. Please try again later."
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
                "all Google AI models have reached their daily quota",
                "quota limits that reset at midnight Pacific Time"
            ]
            
            google_exhausted = any(indicator in result for indicator in quota_indicators)
            
            if not google_exhausted:
                return result
                
        except Exception as e:
            error_str = str(e)
            # Check if the exception itself indicates quota exhaustion
            if any(keyword in error_str.lower() for keyword in ["429", "quota", "resource_exhausted", "exceeded your current quota"]):
                logger.warning(f"Google AI quota exhausted at MultiProvider level: {e}")
                google_exhausted = True
            else:
                logger.warning(f"Google AI failed with exception: {e}")
                google_exhausted = True
        
        # If Google AI is exhausted and we have Groq configured, try Groq
        if google_exhausted and self._groq_llm:
            logger.info("Google AI exhausted, automatically switching to Groq fallback...")
            try:
                result = self._groq_llm._call(prompt, stop, run_manager)
                # Add a note that we're using fallback
                return f"{result}\n\n*(Response generated using Groq AI due to Google AI quota limits)*"
            except Exception as e:
                logger.error(f"Groq also failed: {e}")
        
        # If both failed or Groq not configured, return helpful message
        if google_exhausted:
            if self._groq_llm:
                return (
                    "I'm sorry, but both Google AI and Groq services are currently unavailable. "
                    "Google AI has reached its daily quota limits, and the Groq fallback also failed. "
                    "Please try again later or contact the administrator."
                )
            else:
                return (
                    "I'm sorry, but Google AI has reached its daily quota limits (50 requests per day on free tier). "
                    "The quota resets at midnight Pacific Time. You can:\n"
                    "1. Try again after midnight PT\n"
                    "2. Upgrade to a paid Google AI plan for higher limits\n"
                    "3. Ask the administrator to configure Groq as a fallback service"
                )
        else:
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
            logger.debug(f"DEV_MODE: Creating mock embedding for text: {text[:50]}...")
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
                logger.debug(f"Getting embedding with model: {model_name}")
                
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
                        logger.warning(f"Unknown response format for {model_name}")
                        continue

                    if hasattr(embedding_data, 'values'):
                        embedding_data = embedding_data.values

                    if isinstance(embedding_data, (list, tuple)):
                        embedding_list = [float(x) for x in embedding_data]
                        if embedding_list and len(embedding_list) > 0:
                            logger.debug(f"Embedding successful with {model_name}, got {len(embedding_list)} dimensions")
                            return embedding_list

                finally:
                    signal.alarm(0)

            except Exception as e:
                error_str = str(e)
                if any(keyword in error_str.lower() for keyword in ["429", "quota", "resource_exhausted"]):
                    logger.warning(f"Embedding quota exceeded for {model_name}")
                    quota_exceeded_embedding_models.add(model_name)
                    continue
                else:
                    logger.warning(f"Embedding error with {model_name}: {e}")
                    continue

        logger.warning("All embedding models failed or quota exceeded")
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
        logger.error(f"Embedding error: {e}")
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

    logger.info("Creating embeddings instance...")
    # Create embeddings
    embeddings = GoogleAIEmbeddings()

    logger.info("Creating documents from data...")
    # Create documents
    documents = create_documents_from_data()

    if not documents:
        raise ValueError("No documents available to create vector store")

    logger.info(f"Created {len(documents)} documents, building vector store...")
    # Create vector store using simple in-memory implementation
    vector_store = SimpleVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

    logger.info("Vector store setup completed!")
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
            logger.error(f"Failed to load session from {fpath}: {e}")
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
    logger.info("Starting up RAG Backend...")
    
    logger.info("Resetting quota tracking...")
    reset_quota_tracking()
    
    logger.info("Discovering available Google AI models...")
    discover_available_models()
    logger.info(f"Discovery completed. AVAILABLE_GOOGLE_AI_MODELS = {AVAILABLE_GOOGLE_AI_MODELS}")
    
    # Check Groq fallback configuration
    if groq_api_key:
        logger.info("Groq API key configured - fallback service available")
    else:
        logger.warning("Groq API key not configured - no fallback service available")
    
    logger.info("Loading sessions from files...")
    SESSIONS = load_sessions()
    logger.info(f"Loaded {len(SESSIONS)} sessions.")

    logger.info("Setting up vector store...")
    setup_vector_store_and_chain()
    logger.info("Setup complete! Backend is ready to serve requests.")

@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"message": "English Corner RAG Backend is running", "status": "healthy"}

@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle preflight OPTIONS requests"""
    logger.debug(f"OPTIONS request for path: {path}")
    return {"message": "OK"}

@app.get("/health")
async def health():
    """Detailed health check"""
    logger.info("Detailed health check requested")
    health_data = {
        "status": "healthy",
        "sessions_loaded": len(SESSIONS),
        "vector_store_ready": vector_store is not None,
        "active_chat_sessions": len(session_memories)
    }
    logger.info(f"Health check result: {health_data}")
    return health_data

@app.post("/ask")
async def ask(request: QuestionRequest):
    """Legacy endpoint for backward compatibility"""
    logger.info(f"Received request at /ask: {request.question}")
    return await chat(ChatRequest(question=request.question, session_id=request.session_id or "default"))

@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received chat request: {request.question} (session: {request.session_id})")
    question = request.question.strip()
    if not question:
        logger.warning("Empty question received")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    session_id = request.session_id or "default"
    logger.info(f"Processing question for session {session_id}: {question[:100]}...")

    # Get session-specific memory
    session_memory = get_memory_for_session(session_id)

    # Intent detection: direct logic for some queries (enhanced with context awareness)
    intent = detect_intent(question)
    if intent == "last_session_date":
        logger.info(f"Detected intent: {intent}")
        answer = get_last_session_date()
        # Add to conversation memory
        session_memory.chat_memory.add_user_message(HumanMessage(content=question))
        session_memory.chat_memory.add_ai_message(AIMessage(content=answer))
        
        # Log the response
        response_data = {"answer": answer, "session_id": session_id}
        logger.info(f"Intent-based response: {answer[:100]}...")
        request_logger.info(f"CHAT_RESPONSE [session: {session_id}]: {json.dumps(response_data, ensure_ascii=False)}")
        return response_data
        
    elif intent == "top_participant":
        logger.info(f"Detected intent: {intent}")
        answer = get_top_attendee()
        # Add to conversation memory
        session_memory.chat_memory.add_user_message(HumanMessage(content=question))
        session_memory.chat_memory.add_ai_message(AIMessage(content=answer))
        
        # Log the response
        response_data = {"answer": answer, "session_id": session_id}
        logger.info(f"Intent-based response: {answer[:100]}...")
        request_logger.info(f"CHAT_RESPONSE [session: {session_id}]: {json.dumps(response_data, ensure_ascii=False)}")
        return response_data

    # Use a simpler approach without potential recursion issues
    try:
        logger.info("Retrieving relevant documents from vector store...")
        # Get relevant documents from vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

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

        logger.info("Calling LLM for response generation...")
        # Get response from LLM - using multi-provider with Groq fallback
        llm = MultiProviderLLM()
        answer = llm._call(prompt)
        logger.info(f"LLM response generated: {answer[:100]}...")

        # Add to conversation memory
        session_memory.chat_memory.add_user_message(HumanMessage(content=question))
        session_memory.chat_memory.add_ai_message(AIMessage(content=answer))

        # Extract source information for debugging
        sources = [{"type": doc.metadata.get("type"), "id": doc.metadata.get("id")}
                  for doc in relevant_docs]

        response_data = {
            "answer": answer,
            "session_id": session_id,
            "sources_used": sources
        }
        
        # Log the complete response
        request_logger.info(f"CHAT_RESPONSE [session: {session_id}]: {json.dumps(response_data, ensure_ascii=False)}")
        logger.info(f"Chat response completed for session {session_id}")
        
        return response_data

    except Exception as e:
        error_msg = f"Chat error: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Log the error response
        error_response = {"error": str(e), "session_id": session_id}
        request_logger.error(f"CHAT_ERROR [session: {session_id}]: {json.dumps(error_response, ensure_ascii=False)}")
        
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {e}")

@app.post("/reset_session")
async def reset_session(session_id: str = "default"):
    """Reset conversation memory for a specific session"""
    logger.info(f"Resetting session: {session_id}")
    if session_id in session_memories:
        del session_memories[session_id]
        logger.info(f"Session {session_id} reset successfully")
    else:
        logger.info(f"Session {session_id} not found in memory")
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
    # Get available models
    available_models = [m for m in GOOGLE_AI_MODELS if m not in quota_exceeded_models]
    
    if not available_models:
        return {
            "test": "direct_model_call",
            "status": "error", 
            "error": "No available models - all quota exceeded or unavailable"
        }
    
    # Try each available model
    for model_name in available_models:
        try:
            logger.debug(f"Testing model: {model_name}")
            test_prompt = "Hello, please respond with 'API test successful'"
            
            response = client.models.generate_content(
                model=model_name,
                contents=test_prompt
            )
            
            return {
                "test": "direct_model_call",
                "status": "success",
                "response": response.text,
                "model_used": model_name
            }
            
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Model {model_name} failed: {error_str}")
            
            # Check if it's a 404 or model not found error
            if any(keyword in error_str.lower() for keyword in ["404", "not_found", "not found", "is not found"]):
                logger.warning(f"Model {model_name} not found (404), marking as unavailable")
                quota_exceeded_models.add(model_name)
                continue  # Try next model
            
            # Check if it's a quota error
            elif any(keyword in error_str.lower() for keyword in ["429", "quota", "resource_exhausted", "rate limit"]):
                logger.warning(f"Quota exceeded for model {model_name}, marking as unavailable")
                quota_exceeded_models.add(model_name)
                continue  # Try next model
            else:
                # Other error, return it
                import traceback
                return {
                    "test": "direct_model_call", 
                    "status": "error",
                    "error": str(e),
                    "model_tested": model_name,
                    "traceback": traceback.format_exc()
                }
    
    # If we get here, all models failed
    return {
        "test": "direct_model_call",
        "status": "error",
        "error": "All available models failed",
        "quota_exceeded_models": list(quota_exceeded_models)
    }

@app.get("/debug/ai-status")
async def debug_ai_status():
    """Debug endpoint to check AI provider status"""
    google_status = {
        "discovered_models": AVAILABLE_GOOGLE_AI_MODELS,
        "fallback_models": GOOGLE_AI_MODELS,
        "available_models": [m for m in AVAILABLE_GOOGLE_AI_MODELS if m not in quota_exceeded_models],
        "quota_exceeded_llm_models": list(quota_exceeded_models),
        "quota_exceeded_embedding_models": list(quota_exceeded_embedding_models),
        "total_discovered_models": len(AVAILABLE_GOOGLE_AI_MODELS)
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

@app.get("/debug/model-test")
async def debug_model_test():
    """Test the first available model directly"""
    try:
        # Get available models using the same logic as the LLM
        google_llm = GoogleAILLM()
        available_models = google_llm._get_available_models()
        
        if not available_models:
            return {
                "status": "error",
                "error": "No available models",
                "discovered_models": AVAILABLE_GOOGLE_AI_MODELS,
                "fallback_models": GOOGLE_AI_MODELS,
                "quota_exceeded": list(quota_exceeded_models)
            }
        
        # Test first available model with MultiProvider (includes Groq fallback)
        multi_llm = MultiProviderLLM()
        try:
            result = multi_llm._call("Say hello")
            return {
                "status": "success",
                "response": result,
                "available_google_models": available_models,
                "groq_configured": bool(groq_api_key)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "available_google_models": available_models,
                "groq_configured": bool(groq_api_key)
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to test models: {str(e)}"
        }

@app.get("/debug/discover-models")
async def debug_discover_models():
    """Manually trigger model discovery for debugging"""
    try:
        logger.info("Manual model discovery triggered...")
        
        # Test basic API connectivity first
        try:
            logger.info("Testing basic API connectivity...")
            test_response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Test"
            )
            api_working = True
            api_test_result = "Success"
        except Exception as e:
            api_working = False
            api_test_result = str(e)
        
        # Try model listing
        try:
            logger.info("Testing models.list() API...")
            models_pager = client.models.list()
            models_list = []
            
            for i, model in enumerate(models_pager):
                if i >= 20:  # Limit to prevent too much output
                    break
                models_list.append({
                    "index": i,
                    "model": str(model),
                    "name": getattr(model, 'name', 'No name'),
                    "attributes": [attr for attr in dir(model) if not attr.startswith('_')]
                })
            
            models_list_result = "Success"
            models_count = len(models_list)
            
        except Exception as e:
            models_list = []
            models_list_result = str(e)
            models_count = 0
        
        # Trigger discovery function
        old_available = AVAILABLE_GOOGLE_AI_MODELS.copy()
        discover_available_models()
        new_available = AVAILABLE_GOOGLE_AI_MODELS.copy()
        
        return {
            "api_connectivity_test": {
                "working": api_working,
                "result": api_test_result
            },
            "models_list_test": {
                "result": models_list_result,
                "models_count": models_count,
                "models": models_list
            },
            "discovery_results": {
                "old_available_models": old_available,
                "new_available_models": new_available,
                "quota_exceeded": list(quota_exceeded_models)
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
