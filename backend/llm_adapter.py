import json
from main import get_llama_response, phi2_lora_generate

REQUIRED_FIELDS = {"action", "element_type", "parameters"}

def parse_model_output(output_str: str) -> dict:
    try:
        data = json.loads(output_str)
        if not REQUIRED_FIELDS.issubset(data.keys()):
            raise ValueError("Missing required fields in model output.")
        return data
    except Exception as e:
        raise ValueError(f"Output parsing error: {e}")


def query_llm(prompt: str, model: str = "llama3") -> dict:
    if model == "phi2":
        raw = phi2_lora_generate(prompt)
    elif model == "llama3":
        raw = get_llama_response(prompt)
    else:
        raise ValueError("Invalid model name")

    return parse_model_output(raw)
