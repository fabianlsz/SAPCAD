# Utilities for sending prompts to the LLM and simulating responses

def send_prompt_to_llm(prompt):
    """
    Simulate sending a prompt to the LLM (replace with real API call if needed).
    For now, just echo the prompt and return a fake modifications dict.
    """
    return {
        "modifications": {},
        "llm_response": f"LLM received prompt:\n{prompt}"
    } 