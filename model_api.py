from fastapi import FastAPI, Request
from agents import LlamaCppInterface
import os

app = FastAPI()

# Load the model at startup
llama = LlamaCppInterface(
    model_path=os.environ.get("LLAMA_MODEL_PATH", "/app/unsloth.Q4_K_M.gguf"),
    n_gpu_layers=-1,
    n_ctx=2048
)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return {"error": "No prompt provided"}
    result = llama.generate(prompt)
    return {"result": result} 