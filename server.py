from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CodeGeeX4 Inference Service", version="1.0.0")

# Global model cache
MODEL_NAME = "THUDM/codegeex4-all-9b"
MODEL_CACHE = {"model": None, "tokenizer": None}

class CodePrompt(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

class CodeResponse(BaseModel):
    prompt: str
    completion: str
    tokens_generated: int
    model: str

def load_model():
    """Load model into cache."""
    if MODEL_CACHE["model"] is None:
        logger.info(f"Loading {MODEL_NAME}...")
        try:
            MODEL_CACHE["tokenizer"] = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                cache_dir=os.environ.get("HF_HOME", "/app/models")
            )
            MODEL_CACHE["model"] = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=os.environ.get("HF_HOME", "/app/models")
            )
            logger.info(f"Model loaded successfully on device: {MODEL_CACHE['model'].device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "online",
        "model": MODEL_NAME,
        "model_loaded": MODEL_CACHE["model"] is not None
    }

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodePrompt):
    """Generate code completion."""
    try:
        if MODEL_CACHE["model"] is None:
            load_model()

        model = MODEL_CACHE["model"]
        tokenizer = MODEL_CACHE["tokenizer"]

        # Tokenize input
        inputs = tokenizer.encode(request.prompt, return_tensors="pt").to(model.device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode output
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = completion[len(request.prompt):]  # Remove prompt from output

        return CodeResponse(
            prompt=request.prompt,
            completion=completion.strip(),
            tokens_generated=outputs.shape[1] - inputs.shape[1],
            model=MODEL_NAME
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [MODEL_NAME],
        "active": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
