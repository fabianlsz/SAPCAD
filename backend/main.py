# ----------------------------------------
# SAPCAD Backend Main API
# ----------------------------------------

# Import required libraries and modules
import uuid
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx
from ifc_parser import IFCParser
from ifc_modifier import IFCModifier
from window_modification import WindowModificationData
import tempfile
import os
from typing import Optional, Dict, Any
from chatbot_workflow import chatbot_handler
import re
import json
import shutil
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import PyPDF2
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
# ----------------------------------------
# FastAPI App Initialization and CORS Setup
# ----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Data Models for API Endpoints
# ----------------------------------------
class Prompt(BaseModel):
    prompt: str
    ifc_file_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None  # Additional context for LLM

class WindowModification(BaseModel):
    window_id: int
    new_width: float
    new_height: float

class ChatMessage(BaseModel):
    user_id: str
    message: str
# ----------------------------------------
# Directory and Server Configurations
# ----------------------------------------
UPLOADS_DIR = "./uploaded_ifcs"
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

SERVER_HOST = "localhost"
SERVER_PORT = "8000"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# ----------------------------------------
# Llama Model Response Function
# ----------------------------------------
async def get_llama_response(prompt: str, context: Optional[Dict[str, Any]] = None,  json_only: bool = False) -> str:
    """Get response from Llama 3.2 model"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare the system message with context
            system_message = "You are a helpful assistant for IFC file modifications. "
            if json_only:
                system_message = (
                    "You are a BIM assistant. Your task is to ONLY return a valid JSON object based on the user's instruction and the building standard. "
                    "Do NOT include explanations, formatting, or any non-JSON content. Output must be pure JSON."
                )
            else:
                system_message = (
                    "You are a helpful assistant for BIM and IFC-related questions. "
                    "You may explain your answers naturally unless the user asks otherwise."
                )
            if context:
                system_message += f"Context: {json.dumps(context)}"
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:latest",
                    "prompt": prompt,
                    "system": system_message,
                    "stream": False,
                    "options": {
                        "temperature": 0.3 if json_only else 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
            )
            data = response.json()
            return data.get("response", "No response from model")
    except Exception as e:
        print(f"Error getting Llama response: {str(e)}")
        return "Error getting response from model"

# ----------------------------------------
# Main Chat and Window Modification Endpoint
# ----------------------------------------
@app.get("/llm")
async def llm_get():
    """GET endpoint for /llm - instructs to use POST method"""
    return {"message": "Please use POST method to send your request"}

@app.post("/llm")
async def chat_with_llm(request: Request):
    """
    Main endpoint for chat and window modification logic.
    Accepts both JSON and plain text requests.
    Handles window size change commands and general chat prompts.
    """
    try:
        # Try to parse JSON body
        try:
            data = await request.json()
            prompt = data.get("prompt")
            context = data.get("context")
        except Exception:
            # If not JSON, treat as plain text
            prompt = (await request.body()).decode("utf-8")
            context = None
        # For non-window modification requests, just use LLM
        llm_response = await get_llama_response(prompt, context)
        return {"response": llm_response}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[LLM ERROR] {str(e)}\n{tb}")
        return {"response": "An error occurred while processing your request."}

# ----------------------------------------
# Endpoint to Upload and Parse IFC File
# ----------------------------------------
@app.post("/api/parse-ifc")
async def parse_ifc(file: UploadFile = File(...)):
    """
    Endpoint to upload and parse IFC file.
    Returns parsed information about the IFC file.
    """
    parser = IFCParser()
    result = await parser.parse_ifc_file(file)
    return result

# ----------------------------------------
# Endpoint to Modify Window Dimensions in Uploaded IFC File
# ----------------------------------------
@app.post("/api/modify-window")
async def modify_window(modification: WindowModification, file: UploadFile = File(...)):
    """
    Endpoint to modify window dimensions in an uploaded IFC file.
    Returns the modified file content.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ifc') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Create modifier instance
        modifier = IFCModifier(temp_file_path)
        # Apply modification
        result = modifier.modify_window_size(
            modification.window_id,
            modification.new_width,
            modification.new_height
        )
        if result['status'] == 'success':
            # Save modified file
            save_result = modifier.save_modified_file()
            if save_result['status'] == 'success':
                # Read the modified file
                with open(save_result['file_path'], 'rb') as f:
                    modified_content = f.read()
                # Clean up temporary file
                os.unlink(save_result['file_path'])
                return {
                    'status': 'success',
                    'message': result['message'],
                    'modified_file': modified_content
                }
            else:
                raise HTTPException(status_code=500, detail=save_result['message'])
        else:
            raise HTTPException(status_code=400, detail=result['message'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# ----------------------------------------
# Root Endpoint for Health Check
# ----------------------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "IFC Parser API is running"}

# ----------------------------------------
# Endpoint for Chatbot Workflow
# ----------------------------------------
@app.post("/chatbot")
async def chatbot_endpoint(msg: ChatMessage):
    """
    Endpoint for chatbot workflow logic.
    Handles chat messages and returns AI responses.
    """
    response = chatbot_handler(msg.user_id, msg.message)
    return {"response": response}

# ----------------------------------------
# Endpoint to Download Modified IFC Files
# ----------------------------------------
@app.get("/download/{file_name}")
async def download_file(file_name: str):
    """
    Endpoint to download modified IFC files by file name.
    Returns the file as an attachment if found.
    """
    try:
        # Check if file exists in the uploads directory
        file_path = os.path.join(UPLOADS_DIR, file_name)
        if not os.path.exists(file_path):
            # Check if file exists in temp directory
            temp_file_path = os.path.join(tempfile.gettempdir(), file_name)
            if os.path.exists(temp_file_path):
                # Move file from temp to uploads directory
                shutil.copy2(temp_file_path, file_path)
            else:
                raise HTTPException(status_code=404, detail=f"File {file_name} not found")
        # Verify file exists and is readable
        if not os.access(file_path, os.R_OK):
            raise HTTPException(status_code=403, detail="File is not readable")
        return FileResponse(
            path=file_path,
            filename=file_name,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={file_name}"
            }
        )
    except Exception as e:
        print(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")

# ----------------------------------------
# Main Entrypoint for Running the App
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ----------------------------------------
# For extracting standards
# ----------------------------------------
# This function extracts text from a PDF file
def extractTextFromPdf(pdf_path):
    """Extract all text content from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
    
    return text

def generatePrompt(building_standard: str, user_instruction: str) -> str:
    return f"""
            You are an expert assistant for analyzing building standard documents (e.g. GEG) and producing structured outputs for updating IFC building models.

            Your job is to:
            - Read the provided {building_standard} (in German or English)
            - Understand the {user_instruction}
            - Extract relevant BIM element information (e.g. windows, doors) and organize it in a structured format
            - Return the result in the specific JSON structure described below.

            Please follow this **strict JSON schema** in your output:

            ```json
          
            {{
                "element_updates": [
                    {{
                    "type": "<IfcElement>",
                    "action": "modify" | "remove" | "add",
                    "target_properties": {{
                        "<property_name>": <value or "auto" or null>
                    }},
                    "unit": "<unit_from_spec>"
                    }}
                ],
                "update_scope": "<describe_scope>"
            }}
            """
# Main function that runs the complete workflow
async def processBimPdf(pdf_path, user_instruction):
    try:
        # Extract text from PDF
        text = extractTextFromPdf(pdf_path)
        
       # 2. Split long text into chunks (e.g., every 3000 characters)
        chunk_size = 3000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_results = []
        
        # 3. For each chunk: build prompt and get LLM response
        for chunk in chunks:
            prompt = generatePrompt(chunk, user_instruction)
            result = await get_llama_response(prompt,None,True)
            all_results.append(result)

        return all_results

    except Exception as e:
        return f"Error processing BIM PDF: {str(e)}"


@app.get("/test-pdf")
async def test_pdf():
    pdf_path = "./uploaded_pdfs/mock_bim_standard.pdf" 
    # user_instruction= """Please review the provided building standard and apply all relevant updates to the IFC model.
    # Use the technical values in the document (e.g., dimensions, thermal transmittance, energy performance) to improve the model wherever applicable.
    # Focus on aligning the IFC elements (like windows, doors, or roofs) with the specified norms."""
    user_instruction="Remove the roof from the house"
    response_list =await processBimPdf(pdf_path, user_instruction)
    full_response = response_list[0]

    # Remove Markdown code block wrapper (```json ... ```)
    cleaned = re.sub(r"^```(?:json)?\n?|```$", "", full_response.strip(), flags=re.MULTILINE)

    try:
        json_data = json.loads(cleaned)
        return json_data
    except Exception as e:
        return {"error": f"Failed to parse JSON: {str(e)}", "raw": cleaned}

# Flag to switch between development mode and frontend-integrated mode
dev_mode = False

@app.post("/modify-ifc")
async def modify_ifc_endpoint(
    ifc_file: UploadFile = File(None),
    norm_pdf_file: UploadFile = File(None),
    user_instruction: str = Form("Please update the model based on the provided building standards.")
):
    try:
        if dev_mode:
            # Development mode: use local static files
            temp_ifc_path = "./uploaded_ifc/ISO.ifc"
            temp_pdf_path = "./uploaded_pdfs/Nomr_Mock.pdf"
        else:
            # Production mode: get uploaded files from user request
            temp_ifc_path = f"./temp/{uuid.uuid4()}_{ifc_file.filename}"
            temp_pdf_path = f"./temp/{uuid.uuid4()}_{norm_pdf_file.filename}"
            os.makedirs(os.path.dirname(temp_ifc_path), exist_ok=True)
            with open(temp_ifc_path, "wb") as f:
                shutil.copyfileobj(ifc_file.file, f)
            with open(temp_pdf_path, "wb") as f:
                shutil.copyfileobj(norm_pdf_file.file, f)
        
        # Run: extract structured information from PDF using Olama
        response_list = await processBimPdf(temp_pdf_path, user_instruction)
        full_response = response_list[0]

        # Clean: remove markdown formatting (e.g., ```json blocks)
        cleaned = re.sub(r"^```(?:json)?\n?|```$", "", full_response.strip(), flags=re.MULTILINE)
        try:
            llm_data = json.loads(cleaned)
           

        except Exception as e:
            return {"error": f"Failed to parse JSON from model output: {str(e)}", "raw": cleaned}

   
        # Apply modifications using IFCModifier
        modifier = IFCModifier(temp_ifc_path)
        results = []
        print(f"llm_data: {json.dumps(llm_data, indent=2)}")
        
        for update in llm_data.get("element_updates", []):
            element_type = update.get("type")
            target_properties = update.get("target_properties", {})
            action = update.get("action", "modify")  # Default to "modify" if not provided

            if not element_type:
                results.append({"error": "Missing element type", "update": update})
                continue

            if action == "modify":
                if target_properties:
                    result = modifier.modify_element(element_type, target_properties)
                    results.append({"type": element_type, "action": "modify", "result": result})
                else:
                    results.append({"error": "Missing properties for modify", "type": element_type})
            
            elif action == "remove":
                result = modifier.remove_elements(element_type=element_type)
                # result = modifier.remove_all_elements_of_type(element_type)
                results.append({"type": element_type, "action": "remove", "result": result})
            
            elif action == "add":
                if target_properties:
                    result = modifier.add_element(element_type, target_properties)
                    results.append({"type": element_type, "action": "add", "result": result})
                else:
                    results.append({"error": "Missing properties for add", "type": element_type})
            
            else:
                results.append({"error": f"Unknown action '{action}'", "type": element_type})

        # Save the modified IFC file
        output_path = temp_ifc_path.replace(".ifc", "_modified.ifc")
        save_result = modifier.save_modified_file(output_path)
        if save_result["status"] != "success":
            return {"error": "Failed to save modified IFC", "details": save_result}

        final_download_path = os.path.join(UPLOADS_DIR, os.path.basename(output_path))
        shutil.copy2(output_path, final_download_path)
      
        return {
            "change": save_result["message"],  # Example: "File saved successfully to ..."
            "modification_data": results,      # Detailed list of modified elements
            "llm_response": full_response,     # Original LLM output string (raw)
            "download_url": f"/download/{os.path.basename(final_download_path)}"
            }
    except Exception as e:
        return {"error": str(e)}

# @app.post("/api/natural-to-ifc-json")
# async def natural_to_ifc_json(request: NaturalLanguagePrompt):
#     """
#     Endpoint to convert natural language instructions to IFC modification JSON.
#     Takes natural language input and returns structured JSON for IFC modifications.
#     """
#     try:
#         # Get response from Llama model with JSON-only flag
#         llm_response = await get_llama_response(
#             request.prompt,
#             request.context,
#             json_only=True
#         )
        
#         # Try to parse the response as JSON
#         try:
#             json_response = json.loads(llm_response)
#             return {
#                 "status": "success",
#                 "modifications": json_response
#             }
#         except json.JSONDecodeError:
#             return {
#                 "status": "error",
#                 "message": "Failed to generate valid JSON from natural language input",
#                 "raw_response": llm_response
#             }
            
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error processing natural language input: {str(e)}"
#         )
        

# Load the model only once (global cache)
_model = None
_tokenizer = None

def load_phi2_lora_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    base_model_id = "microsoft/phi-2"
    lora_path = "./phi2_lora_model"  

    print(" Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,  
            device_map=None             
        )
    print(" Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print(" Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    _model = model
    _tokenizer = tokenizer
    return model, tokenizer


def phi2_lora_generate(prompt: str, max_new_tokens=256) -> str:
    model, tokenizer = load_phi2_lora_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only JSON output (if prompt contains prompt+output)
    start_index = decoded.find("{")
    if start_index != -1:
        decoded = decoded[start_index:]
    return decoded
