from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
from app.rag_pipeline import answer_question

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
UPLOAD_DIR = Path("data/clinical_data")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    answer = answer_question(question)
    return JSONResponse({"reply": answer})

@app.post("/upload")
async def upload_file(file: UploadFile):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"message": f" {file.filename} uploaded. It will be auto-ingested shortly."})
