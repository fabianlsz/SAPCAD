import ifcopenshell
from ifc_extra_modifier import modify_element_material, modify_element_relationship

def auto_modify_ifc(input_file_path, output_file_path, modifications):
    """
    Automatically apply multiple modifications to an IFC file.
    
    Args:
        input_file_path (str): Path to the input IFC file
        output_file_path (str): Path where the modified IFC file will be saved
        modifications (list): List of modifications to apply. Each modification is a dict with:
            - type: 'material' or 'relationship'
            - element_id: ID of the element to modify
            - new_value: new material name or new container ID
    """
    try:
        # Open the IFC file
        ifc_file = ifcopenshell.open(input_file_path)
        
        # Apply each modification
        results = []
        for mod in modifications:
            if mod['type'] == 'material':
                result = modify_element_material(ifc_file, mod['element_id'], mod['new_value'])
            elif mod['type'] == 'relationship':
                result = modify_element_relationship(ifc_file, mod['element_id'], mod['new_value'])
            else:
                result = {"status": "error", "message": f"Unknown modification type: {mod['type']}"}
            results.append(result)
        
        # Save the modified file
        ifc_file.write(output_file_path)
        
        return {
            "status": "success",
            "message": f"File saved to {output_file_path}",
            "modifications": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during modification: {str(e)}"
        }

# Example usage
if __name__ == "__main__":
    # Example modifications
    modifications = [
        {
            "type": "material",
            "element_id": 123,  # Replace with actual element ID
            "new_value": "Concrete"
        },
        {
            "type": "relationship",
            "element_id": 456,  # Replace with actual element ID
            "new_value": 789    # Replace with actual container ID
        }
    ]
    
    # Run the modifications
    result = auto_modify_ifc(
        input_file_path="path/to/input.ifc",
        output_file_path="path/to/output.ifc",
        modifications=modifications
    )
    
    print(result) 