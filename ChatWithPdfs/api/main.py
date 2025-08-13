import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from orchestrator.workflow import orchestrator
from api.dependencies import VECTOR_INDEX_PATH, get_vector_store

app = FastAPI(title="Chat With PDF Backend")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={
        "error": exc.__class__.__name__,
        "message": str(exc)
    })

class AskRequest(BaseModel):
    question: str = Field(..., examples=[
        "String"
    ])

@app.get("/")
async def root():
    return {
        "name": "Chat With PDF Backend",
        "status": "ok",
        "docs": "/docs",
        "health": "/healthz"
    }


@app.get("/healthz")
async def healthz():
    if not os.path.isdir(VECTOR_INDEX_PATH):
        return {"ready": False, "reason": f"Missing vector index dir: {VECTOR_INDEX_PATH}"}
    try:
        _ = get_vector_store()
    except Exception as e:
        return {"ready": False, "reason": f"Vector store not ready: {e.__class__.__name__}: {e}"}
    return {"ready": True}


@app.post('/ask')
async def ask(req: AskRequest):
    q = (req.question or '').strip()
    if not q:
        raise HTTPException(status_code=400, detail='Question cannot be empty')
    response = orchestrator.handle(q)
    return {'answer': response}


@app.post('/clear')
async def clear():
    orchestrator.history.clear()
    return {'status': 'memory cleared'}


@app.get('/ask')
async def ask_get(q: str):
    q = (q or '').strip()
    if not q:
        raise HTTPException(status_code=400, detail='q cannot be empty')
    response = orchestrator.handle(q)
    return {'answer': response}
