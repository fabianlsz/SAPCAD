# Utility for building prompts for the LLM based on IFC element modifications

def build_prompt(element_type, element_id, new_values):
    """
    Build a prompt string for the LLM to modify a specific IFC element.
    """
    prompt = f"Please modify {element_type}#{element_id} in the IFC file:"
    for key, value in new_values.items():
        prompt += f"\n- {key}: {value}"
    return prompt 