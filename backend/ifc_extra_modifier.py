import ifcopenshell
from ifcopenshell.util.element import get_material, assign_material
from ifcopenshell.util.element import get_container, assign_container

# Function to change or assign material to an IFC element
def modify_element_material(ifc_file, element_id, new_material_name):
    """
    Assign or change the material of an IFC element.
    Args:
        ifc_file: Opened ifcopenshell file object
        element_id: ID of the element to modify
        new_material_name: Name of the new material to assign
    Returns:
        dict: Status and message
    """
    element = ifc_file.by_id(element_id)
    if not element:
        return {"status": "error", "message": f"Element {element_id} not found"}
    # Try to find existing material
    material = None
    for mat in ifc_file.by_type('IfcMaterial'):
        if mat.Name == new_material_name:
            material = mat
            break
    # If material does not exist, create it
    if not material:
        material = ifc_file.create_entity('IfcMaterial', Name=new_material_name)
    # Assign material to element
    assign_material(ifc_file, element, material)
    return {"status": "success", "message": f"Material '{new_material_name}' assigned to element {element_id}"}

# Function to change the container (spatial/parent) relationship of an IFC element
def modify_element_relationship(ifc_file, element_id, new_container_id):
    """
    Change the container (parent) of an IFC element (e.g., move a wall to another storey).
    Args:
        ifc_file: Opened ifcopenshell file object
        element_id: ID of the element to move
        new_container_id: ID of the new container (e.g., IfcBuildingStorey)
    Returns:
        dict: Status and message
    """
    element = ifc_file.by_id(element_id)
    new_container = ifc_file.by_id(new_container_id)
    if not element or not new_container:
        return {"status": "error", "message": "Element or new container not found"}
    # Assign new container (parent) to element
    assign_container(ifc_file, element, new_container)
    return {"status": "success", "message": f"Element {element_id} moved to container {new_container_id}"} 