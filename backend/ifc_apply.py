from ifc_modifier import IFCModifier

def apply_modifications(ifc_file_path, element_id=None, modifications=None):
    """
    Apply modifications (property/geometry changes, remove, or add) to an IFC file.
    Args:
        ifc_file_path (str): Path to the IFC file.
        element_id (int, optional): ID of the element (required for modify/remove).
        modifications (dict): Dictionary containing changes and action type.
    Returns:
        dict: Result of the operation (status, message, etc.).
    """
    try:
        modifier = IFCModifier(ifc_file_path)
        action = modifications.get("action", "modify") if modifications else "modify"
        # REMOVE
        if action == "remove":
            if element_id is None:
                return {"status": "error", "message": "element_id required for remove action"}
            return modifier.remove_elements(element_id=element_id)
        # ADD
        elif action == "add":
            element_type = modifications.get("element_type")
            attributes = modifications.get("attributes", {})
            geometry = modifications.get("geometry", None)
            if not element_type:
                return {"status": "error", "message": "element_type required for add action"}
            return modifier.add_element(element_type, attributes, geometry)
        # MODIFY (default)
        else:
            if element_id is None:
                return {"status": "error", "message": "element_id required for modify action"}
            result = {}
            non_geom_mods = {k: v for k, v in modifications.items() if k != "geometry" and k != "action"}
            if non_geom_mods:
                result['properties'] = modifier.modify_element(element_id, non_geom_mods)
            if modifications and "geometry" in modifications:
                result['geometry'] = modifier.modify_element_geometry(element_id, modifications["geometry"])
            save_result = modifier.save_modified_file()
            if save_result["status"] == "success":
                result['file_path'] = save_result["message"]
                return result
            else:
                raise Exception(save_result["message"])
    except Exception as e:
        return {"status": "error", "message": str(e)} 