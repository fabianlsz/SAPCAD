from main import phi2_lora_generate

if __name__ == "__main__":
    instruction = "Add a new IfcDoor next to the main entrance with width 900 mm."
    prompt = f"Instruction: {instruction}\nOutput:"
    output = phi2_lora_generate(prompt)
    print("📤 Model Output:")
    print(output)