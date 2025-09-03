import ifcopenshell
from typing import Dict, Any

# In-memory user state storage (for demo; use persistent storage in production)
user_states: Dict[str, Dict[str, Any]] = {}

EDITABLE_TYPES = ["IfcWindow", "IfcDoor", "IfcWall", "IfcSlab", "IfcRoof"]

# Step 1: Ask user to select an element type
def step_select_type(user_id, ifc_file_path):
    ifc_file = ifcopenshell.open(ifc_file_path)
    available_types = [t for t in EDITABLE_TYPES if ifc_file.by_type(t)]
    user_states[user_id] = {
        "step": 1,
        "ifc_file_path": ifc_file_path,
        "available_types": available_types
    }
    return f"Which building element do you want to edit? Available types: {', '.join(available_types)}"

# Step 2: Ask user to select an element of the chosen type
def step_select_element(user_id, element_type):
    state = user_states[user_id]
    ifc_file = ifcopenshell.open(state["ifc_file_path"])
    elements = ifc_file.by_type(element_type)
    element_list = [
        {"id": el.id(), "name": getattr(el, "Name", ""), "description": getattr(el, "Description", "")}
        for el in elements
    ]
    state["step"] = 2
    state["element_type"] = element_type
    state["element_list"] = element_list
    user_states[user_id] = state
    msg = "Select the element you want to edit:\n"
    for el in element_list:
        msg += f"ID: {el['id']}, Name: {el['name']}, Desc: {el['description']}\n"
    return msg

# Step 3: Show properties and ask for new values
import json

def load_reference_fields():
    with open("reference_fields.json") as f:
        return json.load(f)

def step_show_properties(user_id, element_id):
    state = user_states[user_id]
    ifc_file = ifcopenshell.open(state["ifc_file_path"])
    element = ifc_file.by_id(int(element_id))
    reference = load_reference_fields()[state["element_type"]]
    props = {}
    for field in reference["required"] + reference.get("optional", []):
        value = getattr(element, field, None)
        props[field] = value
    state["step"] = 3
    state["element_id"] = int(element_id)
    state["properties"] = props
    user_states[user_id] = state
    msg = "Current properties:\n"
    for k, v in props.items():
        msg += f"{k}: {v}\n"
    msg += "\nPlease enter new values as a JSON object (e.g. {\"OverallWidth\": 2.0, \"FrameMaterial\": \"Aluminum\"})"
    return msg

# Step 4: Build prompt for LLM

def build_prompt(element_type, element_id, new_values):
    prompt = f"Please modify {element_type}#{element_id} in the IFC file:"
    for key, value in new_values.items():
        prompt += f"\n- {key}: {value}"
    return prompt

def step_build_prompt(user_id, new_values: Dict[str, Any]):
    state = user_states[user_id]
    prompt = build_prompt(state["element_type"], state["element_id"], new_values)
    state["step"] = 4
    state["prompt"] = prompt
    user_states[user_id] = state
    return f"Generated prompt for LLM:\n{prompt}"

# Workflow logic for handling chatbot messages and responses

def chatbot_handler(user_id: str, message: str) -> str:
    """
    Handle a chatbot message and return a response.
    """
    # Example: Echo the message with user ID
    return f"User {user_id} said: {message}"

# Example chat handler

def chatbot_handler(user_id, message):
    state = user_states.get(user_id)
    if not state or state.get("step") == 1:
        # Expecting IFC file path in message
        return step_select_type(user_id, message)
    elif state["step"] == 1:
        # User should select element type
        return step_select_element(user_id, message)
    elif state["step"] == 2:
        # User should select element id
        return step_show_properties(user_id, message)
    elif state["step"] == 3:
        # User should send new values as JSON
        try:
            new_values = json.loads(message)
        except Exception:
            return "Invalid JSON. Please enter new values as a JSON object."
        return step_build_prompt(user_id, new_values)
    else:
        return "Workflow complete. Restart to edit another element." 