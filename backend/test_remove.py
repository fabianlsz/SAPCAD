from ifc_apply import apply_modifications

if __name__ == '__main__':
    ifc_path = "uploaded_ifcs/ISO.ifc"
    
    # Test removing all windows (batch removal is not supported in apply_modifications, so remove by id example)
    print("\nTest 1: Removing a specific window (example)")
    # Replace 23024 with a real window id from your file
    result = apply_modifications(ifc_path, element_id=23024, modifications={"action": "remove"})
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    # Test removing a specific element (if you know an element ID)
    print("\nTest 2: Removing a specific element")
    # Replace 123 with an actual element ID from your IFC file
    result = apply_modifications(ifc_path, element_id=123, modifications={"action": "remove"})
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result['status'] == 'success':
        print(f"Element removed successfully") 