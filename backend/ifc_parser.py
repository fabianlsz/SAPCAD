# IFC file parsing utilities and API
import ifcopenshell
import ifcopenshell.util.pset as pset_util
import json
from typing import Dict, List, Any, Optional, Union
from fastapi import UploadFile
import tempfile
import os
from constants import EDITABLE_TYPES
from ifc_prompt import build_prompt
from ifc_llm import send_prompt_to_llm
from ifc_apply import apply_modifications
import re

NUMERIC_FIELDS = {"OverallWidth", "OverallHeight", "Length", "Height", "Thickness"}

def load_reference_fields():
    with open("reference_fields.json") as f:
        return json.load(f)

def get_element_properties(ifc_file_path, element_type, element_id):
    """
    Extract properties of a specific element from the IFC file, including required and optional fields.
    Returns a tuple: (properties_dict, reference_fields_dict)
    """
    ifc_file = ifcopenshell.open(ifc_file_path)
    element = ifc_file.by_id(int(element_id))
    reference = load_reference_fields()[element_type]
    props = {}
    for field in reference["required"] + reference.get("optional", []):
        value = getattr(element, field, None)
        props[field] = value
    return props, reference

def validate_new_values(new_values, reference):
    errors = []
    for field in reference["required"]:
        value = new_values.get(field)
        if value is None or value == "":
            errors.append(f"Field '{field}' is required.")
        elif field in NUMERIC_FIELDS:
            try:
                float(value)
            except Exception:
                errors.append(f"Field '{field}' must be a number.")
        else:
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Field '{field}' must be a non-empty string.")
    for field in reference.get("optional", []):
        if field in new_values and field in NUMERIC_FIELDS:
            try:
                float(new_values[field])
            except Exception:
                errors.append(f"Field '{field}' must be a number.")
    return errors

def extract_element_properties(ifc_file_path: str, element_id: int) -> Dict[str, Any]:
    """
    Extract properties of a specific element from the IFC file.
    """
    ifc_file = ifcopenshell.open(ifc_file_path)
    element = ifc_file.by_id(element_id)
    props = {}
    # Get basic properties
    props['Name'] = element.Name if hasattr(element, 'Name') else None
    props['Description'] = element.Description if hasattr(element, 'Description') else None
    # Get Pset properties
    for definition in element.IsDefinedBy:
        if definition.is_a('IfcRelDefinesByProperties'):
            property_set = definition.RelatingPropertyDefinition
            if property_set.is_a('IfcPropertySet'):
                for property in property_set.HasProperties:
                    props[property.Name] = property.NominalValue.wrappedValue if property.NominalValue else None
    return props

def extract_psets(element):
    psets = {}
    if hasattr(element, 'IsDefinedBy'):
        for definition in element.IsDefinedBy:
            if definition.is_a('IfcRelDefinesByProperties'):
                property_set = definition.RelatingPropertyDefinition
                if property_set.is_a('IfcPropertySet'):
                    for prop in property_set.HasProperties:
                        if hasattr(prop, "Name") and hasattr(prop, "NominalValue"):
                            psets[prop.Name] = prop.NominalValue.wrappedValue if prop.NominalValue else None
    return psets

class IFCParser:
    def __init__(self):
        # Initialize the IFCParser with empty file and elements dictionary
        self.ifc_file = None
        self.elements = {}

    def get_element_properties(self, ifc_file_path: str, element_id: int) -> Dict[str, Any]:
        """
        Get properties of a specific element from the IFC file
        Args:
            ifc_file_path: Path to the IFC file
            element_id: ID of the element to get properties for
        Returns:
            Dictionary containing element properties or error message
        """
        ifc_file = ifcopenshell.open(ifc_file_path)
        element = ifc_file.by_id(element_id)
        if not element:
            return {'status': 'error', 'message': f'Element with ID {element_id} not found'}
        return extract_psets(element)

    def get_elements_by_type(self, ifc_file_path: str, element_type: str) -> List[Dict[str, Any]]:
        """
        Get elements of a specific type from the IFC file
        Args:
            ifc_file_path: Path to the IFC file
            element_type: Type of elements to retrieve (e.g., 'IfcWall', 'IfcWindow')
        Returns:
            List of dictionaries containing element information
        """
        ifc_file = ifcopenshell.open(ifc_file_path)
        elements = ifc_file.by_type(element_type)
        return [
            {
                "id": el.id(),
                "type": el.is_a(),
                "name": getattr(el, "Name", ""),
                "description": getattr(el, "Description", ""),
                "properties": extract_psets(el),
                "geometry": self._get_element_geometry(el)
            }
            for el in elements
        ]

    def get_elements_by_name(self, ifc_file_path: str, name: str) -> List[Dict[str, Any]]:
        """
        Get elements by their name
        Args:
            ifc_file_path: Path to the IFC file
            name: Name of elements to find
        Returns:
            List of dictionaries containing element information
        """
        ifc_file = ifcopenshell.open(ifc_file_path)
        elements = []
        for element in ifc_file.by_type("IfcRoot"):
            if getattr(element, "Name", None) == name:
                elements.append({
                    "id": element.id(),
                    "type": element.is_a(),
                    "name": element.Name,
                    "description": getattr(element, "Description", ""),
                    "properties": extract_psets(element),
                    "geometry": self._get_element_geometry(element)
                })
        return elements

    def get_elements_by_property(self, ifc_file_path: str, property_name: str, property_value: Any) -> List[Dict[str, Any]]:
        """
        Get elements by a specific property value
        Args:
            ifc_file_path: Path to the IFC file
            property_name: Name of the property to search for
            property_value: Value of the property to match
        Returns:
            List of dictionaries containing element information
        """
        ifc_file = ifcopenshell.open(ifc_file_path)
        elements = []
        for element in ifc_file.by_type("IfcRoot"):
            psets = extract_psets(element)
            if property_name in psets and psets[property_name] == property_value:
                elements.append({
                    "id": element.id(),
                    "type": element.is_a(),
                    "name": getattr(element, "Name", ""),
                    "description": getattr(element, "Description", ""),
                    "properties": psets,
                    "geometry": self._get_element_geometry(element)
                })
        return elements

    async def parse_ifc_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Parse the uploaded IFC file and extract all elements and their information
        Args:
            file: Uploaded IFC file
        Returns:
            Dictionary containing parsed IFC data or error message
        """
        try:
            # Read the file content and parse with ifcopenshell
            ifc = ifcopenshell.file.from_string(file.file.read().decode('utf-8'))
            
            # Get all element types in the file
            element_types = set()
            for element in ifc.by_type("IfcRoot"):
                element_types.add(element.is_a())

            # Extract elements of each type
            elements_by_type = {}
            for element_type in element_types:
                elements = ifc.by_type(element_type)
                elements_by_type[element_type] = [
                    {
                        'id': el.id(),
                        'type': el.is_a(),
                        'name': getattr(el, "Name", ""),
                        'description': getattr(el, "Description", ""),
                        'properties': extract_psets(el),
                        'geometry': self._get_element_geometry(el)
                    }
                    for el in elements
                ]

            return {
                'status': 'success',
                'element_types': list(element_types),
                'elements_by_type': elements_by_type,
                'total_elements': sum(len(elements) for elements in elements_by_type.values())
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _get_element_geometry(self, element) -> Dict[str, Any]:
        """
        Extract geometry information of an IFC element
        Args:
            element: IFC element to extract geometry from
        Returns:
            Dictionary containing element geometry information
        """
        geometry = {}
        
        # Get placement information
        if hasattr(element, 'ObjectPlacement'):
            placement = element.ObjectPlacement
            if placement.is_a('IfcLocalPlacement'):
                location = placement.RelativePlacement.Location
                geometry['location'] = {
                    'x': location.Coordinates[0],
                    'y': location.Coordinates[1],
                    'z': location.Coordinates[2]
                }
                
                # Get rotation information if available
                if hasattr(placement.RelativePlacement, 'Axis'):
                    axis = placement.RelativePlacement.Axis
                    if axis:
                        geometry['rotation'] = {
                            'x': axis.DirectionRatios[0],
                            'y': axis.DirectionRatios[1],
                            'z': axis.DirectionRatios[2]
                        }

        # Get shape representation information
        if hasattr(element, 'Representation'):
            representation = element.Representation
            if representation and representation.Representations:
                geometry['representations'] = []
                for rep in representation.Representations:
                    if rep.is_a('IfcShapeRepresentation'):
                        rep_info = {
                            'type': rep.RepresentationType,
                            'identifier': rep.RepresentationIdentifier,
                            'items': []
                        }
                        
                        # Get representation items
                        for item in rep.Items:
                            if item.is_a('IfcExtrudedAreaSolid'):
                                rep_info['items'].append({
                                    'type': 'extruded_area',
                                    'depth': item.Depth,
                                    'position': {
                                        'x': item.Position.Location.Coordinates[0],
                                        'y': item.Position.Location.Coordinates[1],
                                        'z': item.Position.Location.Coordinates[2]
                                    }
                                })
                            elif item.is_a('IfcMappedItem'):
                                rep_info['items'].append({
                                    'type': 'mapped_item',
                                    'mapping_source': str(item.MappingSource)
                                })
                        
                        geometry['representations'].append(rep_info)

        return geometry

    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the IFC model
        Returns:
            Dictionary containing model summary information
        """
        return {
            'total_elements': sum(len(elements) for elements in self.elements.values()),
            'element_counts': {
                element_type: len(elements)
                for element_type, elements in self.elements.items()
            }
        } 