import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element as element_util
import ifcopenshell.util.pset as pset_util
import ifcopenshell.api
from typing import Dict, Any, Optional, List, Union
import tempfile
import os
from fastapi import FastAPI, Request
import json
from constants import EDITABLE_TYPES
from ifc_prompt import build_prompt
from ifc_llm import send_prompt_to_llm
# from ifc_apply import apply_modifications  # circular import
import re

# Initialize FastAPI app (for direct API endpoints if needed)
app = FastAPI()

# ----------------------------------------
# IFCModifier Class
# ----------------------------------------
# This class loads an IFC file and provides methods to modify its elements,
# such as changing window dimensions or updating properties.
class IFCModifier:
    def __init__(self, ifc_file_path: str):
        """
        Initialize the IFCModifier with an IFC file
        Args:
            ifc_file_path: Path to the IFC file to modify
        """
        self.ifc_file = ifcopenshell.open(ifc_file_path)

    def find_elements(self, name: str = None, element_type: str = None, property_filter: dict = None) -> List[Any]:
        """
        Generic search for elements by name, type, or property.
        """
        elements = self.ifc_file.by_type("IfcRoot")
        if name:
            elements = [el for el in elements if getattr(el, "Name", None) == name]
        if element_type:
            elements = [el for el in elements if el.is_a() == element_type]
        if property_filter:
            elements = [el for el in elements if all(pset_util.get_pset(el, k) == v for k, v in property_filter.items())]
        return elements

    def get_element_info(self, element_id: int) -> Dict[str, Any]:
        el = self.ifc_file.by_id(element_id)
        if not el:
            return {'status': 'error', 'message': f'Element {element_id} not found'}
        return {
            'status': 'success',
            'element': {
                'id': el.id(),
                'type': el.is_a(),
                'name': getattr(el, 'Name', ''),
                'description': getattr(el, 'Description', ''),
                'properties': pset_util.get_psets(el),
                'geometry': self._extract_geometry(el)
            }
        }

    def modify_element(self, element_id: int, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an element in the IFC file
        Args:
            element_id: ID of the element to modify
            modifications: Dictionary containing modifications
                - attributes: dict with Name, Description, etc.
                - geometry: dict with dimensions and position
                - properties: dict with property sets and values
        Returns:
            Dict containing status and message
        """
        print(f"[DEBUG] Attempting to modify element {element_id}")
        print(f"[DEBUG] Modifications: {modifications}")
        
        try:
            el = self.ifc_file.by_id(element_id)
            if not el:
                return {'status': 'error', 'message': f'Element {element_id} not found'}
            
            changed = {}
            
            # Modify attributes
            if 'attributes' in modifications:
                for attr_name, attr_value in modifications['attributes'].items():
                    if hasattr(el, attr_name):
                        setattr(el, attr_name, attr_value)
                        changed[f'attribute_{attr_name}'] = attr_value
            
            # Modify geometry
            if 'geometry' in modifications:
                changed.update(self._modify_geometry(el, modifications['geometry']))
            
            # Modify properties
            if 'properties' in modifications:
                changed.update(self._modify_properties(el, modifications['properties']))
            
            # Save the modified file
            save_result = self.save_modified_file()
            if save_result['status'] == 'error':
                return save_result
            
            return {
                'status': 'success' if changed else 'error',
                'message': f'Element modified: {changed}' if changed else 'No changes made',
                'modified_properties': changed
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to modify element: {str(e)}")
            return {'status': 'error', 'message': f'Failed to modify element: {str(e)}'}

    def _modify_geometry(self, element, geometry_mods: Dict[str, Any]) -> Dict[str, Any]:
        changed = {}
        if not hasattr(element, 'Representation') or not element.Representation:
            return changed

        for rep in element.Representation.Representations:
            if rep.is_a('IfcShapeRepresentation'):
                for item in rep.Items:
                    if item.is_a('IfcExtrudedAreaSolid'):
                        if hasattr(item, 'SweptArea') and item.SweptArea.is_a('IfcRectangleProfileDef'):
                            if 'dimensions' in geometry_mods:
                                dims = geometry_mods['dimensions']
                                if 'width' in dims:
                                    item.SweptArea.XDim = dims['width']
                                    changed['width'] = dims['width']
                                if 'height' in dims:
                                    item.SweptArea.YDim = dims['height']
                                    changed['height'] = dims['height']
                                if 'depth' in dims:
                                    item.Depth = dims['depth']
                                    changed['depth'] = dims['depth']
        return changed

    def _modify_properties(self, element, properties: Dict[str, Any]) -> Dict[str, Any]:
        changed = {}
        for pset_name, props in properties.items():
            if pset_name not in ifcopenshell.util.element.get_psets(element):
                ifcopenshell.api.run('pset.add_pset', self.ifc_file, product=element, name=pset_name)
            
            for prop_name, prop_value in props.items():
                ifcopenshell.api.run('pset.edit_pset', self.ifc_file,
                    pset=ifcopenshell.util.element.get_psets(element)[pset_name],
                    properties={prop_name: prop_value}
                )
                changed[f'property_{pset_name}_{prop_name}'] = prop_value
        return changed

    def verify_changes(self, expected_dimensions: Dict[str, float], element_type: str = None) -> Dict[str, Any]:
        """
        Verify that the changes were applied correctly for all elements or specific type
        Args:
            expected_dimensions: Dictionary containing expected dimensions (width, height, depth)
            element_type: Optional element type to verify (e.g., 'IfcWindow', 'IfcDoor')
        Returns:
            Dict containing verification results for all matching elements
        """
        print(f"[VERIFY] Verifying changes for all elements")
        print(f"[VERIFY] Expected dimensions: {expected_dimensions}")
        if element_type:
            print(f"[VERIFY] Filtering by element type: {element_type}")
        
        verification = {
            'elements': {},
            'status': 'success',
            'total_elements': 0,
            'matching_elements': 0
        }
        
        # Get all elements or filter by type
        elements = self.ifc_file.by_type(element_type) if element_type else self.ifc_file.by_type("IfcRoot")
        
        for el in elements:
            # Skip elements without representation
            if not hasattr(el, 'Representation') or not el.Representation:
                continue
                
            element_verification = {
                'id': el.id(),
                'type': el.is_a(),
                'geometry': {},
                'properties': {},
                'status': 'success'
            }
            
            # Verify geometry changes
            for rep in el.Representation.Representations:
                if rep.is_a('IfcShapeRepresentation'):
                    for item in rep.Items:
                        if item.is_a('IfcExtrudedAreaSolid'):
                            if hasattr(item, 'SweptArea') and item.SweptArea.is_a('IfcRectangleProfileDef'):
                                # Check width
                                if 'width' in expected_dimensions:
                                    actual_width = item.SweptArea.XDim
                                    element_verification['geometry']['width'] = {
                                        'expected': expected_dimensions['width'],
                                        'actual': actual_width,
                                        'matches': abs(actual_width - expected_dimensions['width']) < 0.001
                                    }
                                
                                # Check height
                                if 'height' in expected_dimensions:
                                    actual_height = item.SweptArea.YDim
                                    element_verification['geometry']['height'] = {
                                        'expected': expected_dimensions['height'],
                                        'actual': actual_height,
                                        'matches': abs(actual_height - expected_dimensions['height']) < 0.001
                                    }
                                
                                # Check depth
                                if 'depth' in expected_dimensions:
                                    actual_depth = item.Depth
                                    element_verification['geometry']['depth'] = {
                                        'expected': expected_dimensions['depth'],
                                        'actual': actual_depth,
                                        'matches': abs(actual_depth - expected_dimensions['depth']) < 0.001
                                    }
            
            # Verify property changes
            pset = ifcopenshell.util.element.get_psets(el)
            if 'Pset_ElementCommon' in pset:
                props = pset['Pset_ElementCommon']
                if 'width' in expected_dimensions:
                    actual_width = props.get('Width', 0)
                    element_verification['properties']['width'] = {
                        'expected': expected_dimensions['width'],
                        'actual': actual_width,
                        'matches': abs(actual_width - expected_dimensions['width']) < 0.001
                    }
                if 'height' in expected_dimensions:
                    actual_height = props.get('Height', 0)
                    element_verification['properties']['height'] = {
                        'expected': expected_dimensions['height'],
                        'actual': actual_height,
                        'matches': abs(actual_height - expected_dimensions['height']) < 0.001
                    }
                if 'depth' in expected_dimensions:
                    actual_depth = props.get('Depth', 0)
                    element_verification['properties']['depth'] = {
                        'expected': expected_dimensions['depth'],
                        'actual': actual_depth,
                        'matches': abs(actual_depth - expected_dimensions['depth']) < 0.001
                    }
            
            # Check if all changes match for this element
            all_match = True
            for dim_type in ['geometry', 'properties']:
                for dim in element_verification[dim_type].values():
                    if not dim['matches']:
                        all_match = False
                        element_verification['status'] = 'error'
                        break
            
            # Only add elements that have geometry or property changes
            if element_verification['geometry'] or element_verification['properties']:
                verification['elements'][el.id()] = element_verification
                verification['total_elements'] += 1
                if all_match:
                    verification['matching_elements'] += 1
        
        # Update overall status
        if verification['total_elements'] > 0:
            verification['status'] = 'success' if verification['matching_elements'] == verification['total_elements'] else 'error'
            verification['message'] = f'Verified {verification["matching_elements"]} out of {verification["total_elements"]} elements'
        else:
            verification['status'] = 'error'
            verification['message'] = 'No elements found to verify'
        
        print(f"[VERIFY] Verification results: {verification['message']}")
        return verification

    def modify_element_geometry(self, element_id: int, geometry_mods: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify the geometry of an element in the IFC file
        Args:
            element_id: ID of the element to modify
            geometry_mods: Dictionary containing geometry modifications
                - dimensions: dict with width, height, depth
                - position: dict with x, y, z coordinates
        Returns:
            Dict containing status and message
        """
        print(f"[DEBUG] Starting geometry modification for element {element_id}")
        print(f"[DEBUG] Geometry modifications requested: {geometry_mods}")
        
        el = self.ifc_file.by_id(element_id)
        if not el:
            print(f"[ERROR] Element {element_id} not found")
            return {'status': 'error', 'message': f'Element {element_id} not found'}
        
        print(f"[DEBUG] Found element: {el.is_a()}")
        changed = {}
        
        # Modify dimensions if specified
        if 'dimensions' in geometry_mods:
            dims = geometry_mods['dimensions']
            print(f"[DEBUG] Modifying dimensions: {dims}")
            
            # Get the window's representation
            if not hasattr(el, 'Representation'):
                print("[ERROR] Element has no Representation attribute")
                return {'status': 'error', 'message': 'Element has no representation'}
                
            if not el.Representation:
                print("[ERROR] Element's Representation is None")
                return {'status': 'error', 'message': 'Element has no representation'}
                
            print(f"[DEBUG] Element has {len(el.Representation.Representations)} representations")
            
            # Try to modify using IfcOpenShell API first
            try:
                print("[DEBUG] Attempting to modify using IfcOpenShell API")
                if 'width' in dims and 'height' in dims:
                    # Create a new representation
                    settings = {
                        'context': self.ifc_file.by_type('IfcGeometricRepresentationContext')[0],
                        'representation_type': 'SweptSolid',
                        'representation_identifier': 'Body',
                        'items': []
                    }
                    
                    # Create a new rectangle profile
                    profile = self.ifc_file.create_entity('IfcRectangleProfileDef', 
                        ProfileType='AREA',
                        XDim=dims['width'],
                        YDim=dims['height']
                    )
                    
                    # Create a new extruded solid
                    placement = self.ifc_file.create_entity('IfcAxis2Placement3D',
                        Location=self.ifc_file.create_entity('IfcCartesianPoint', Coordinates=(0., 0., 0.))
                    )
                    
                    extruded_solid = self.ifc_file.create_entity('IfcExtrudedAreaSolid',
                        SweptArea=profile,
                        Position=placement,
                        ExtrudedDirection=self.ifc_file.create_entity('IfcDirection', DirectionRatios=(0., 0., 1.)),
                        Depth=dims.get('depth', 0.1)
                    )
                    
                    settings['items'].append(extruded_solid)
                    
                    # Create new representation
                    new_rep = self.ifc_file.create_entity('IfcShapeRepresentation', **settings)
                    
                    # Replace old representation
                    if el.Representation:
                        el.Representation.Representations = [new_rep]
                    
                    changed['width'] = dims['width']
                    changed['height'] = dims['height']
                    if 'depth' in dims:
                        changed['depth'] = dims['depth']
                    
                    print("[DEBUG] Successfully modified geometry using API")
            except Exception as e:
                print(f"[ERROR] Failed to modify using API: {str(e)}")
                # Fallback to direct modification
                try:
                    print("[DEBUG] Attempting direct modification")
                    for rep in el.Representation.Representations:
                        if rep.is_a('IfcShapeRepresentation'):
                            for item in rep.Items:
                                if item.is_a('IfcExtrudedAreaSolid'):
                                    if hasattr(item, 'SweptArea') and item.SweptArea.is_a('IfcRectangleProfileDef'):
                                        if 'width' in dims:
                                            item.SweptArea.XDim = dims['width']
                                            changed['width'] = dims['width']
                                        if 'height' in dims:
                                            item.SweptArea.YDim = dims['height']
                                            changed['height'] = dims['height']
                                        if 'depth' in dims:
                                            item.Depth = dims['depth']
                                            changed['depth'] = dims['depth']
                    print("[DEBUG] Successfully modified geometry directly")
                except Exception as e:
                    print(f"[ERROR] Failed to modify directly: {str(e)}")
                    return {'status': 'error', 'message': f'Failed to modify geometry: {str(e)}'}

        # Update element properties to match geometry changes
        if changed:
            print("[DEBUG] Updating element properties")
            try:
                pset = ifcopenshell.util.element.get_psets(el)
                if 'dimensions' in geometry_mods:
                    dims = geometry_mods['dimensions']
                    if 'width' in dims:
                        pset.get('Pset_ElementCommon', {})['Width'] = dims['width']
                    if 'height' in dims:
                        pset.get('Pset_ElementCommon', {})['Height'] = dims['height']
                    if 'depth' in dims:
                        pset.get('Pset_ElementCommon', {})['Depth'] = dims['depth']
                print("[DEBUG] Successfully updated properties")
            except Exception as e:
                print(f"[ERROR] Failed to update properties: {str(e)}")

        print(f"[DEBUG] Changes made: {changed}")
        save_result = self.save_modified_file()
        if save_result['status'] == 'error':
            print(f"[ERROR] Failed to save file: {save_result['message']}")
            return save_result
            
        # Verify changes after modification
        if changed:
            verification = self.verify_changes(geometry_mods.get('dimensions', {}))
            if verification['status'] == 'error':
                print(f"[ERROR] Changes verification failed: {verification['message']}")
                return {
                    'status': 'error',
                    'message': f'Changes made but verification failed: {verification["message"]}',
                    'verification': verification
                }
            
        return {
            'status': 'success' if changed else 'error',
            'message': f'Geometry modified: {changed}' if changed else 'No changes made',
            'modified_geometry': changed,
            'verification': verification if changed else None
        }

    def add_element(self, element_type: str, attributes: dict, geometry: dict = None) -> Dict[str, Any]:
        """
        Add a new element to the IFC file
        Args:
            element_type: Type of element to add (e.g., 'IfcWindow', 'IfcDoor', 'IfcWall')
            attributes: Dictionary containing element attributes (Name, Description, etc.)
            geometry: Dictionary containing geometry information
                - dimensions: dict with width, height, depth
                - position: dict with x, y, z coordinates
        Returns:
            Dict containing status and message
        """
        print(f"[DEBUG] Attempting to add new {element_type}")
        print(f"[DEBUG] Attributes: {attributes}")
        print(f"[DEBUG] Geometry: {geometry}")
        
        try:
            settings = {'type': element_type, 'attributes': attributes}
            
            if geometry:
                settings['representation'] = self._create_representation(geometry)
            
            new_el = ifcopenshell.api.run('root.create_entity', self.ifc_file, **settings)
            
            if 'properties' in attributes:
                self._add_properties(new_el, attributes['properties'])
            
            save_result = self.save_modified_file()
            if save_result['status'] == 'error':
                return save_result
            
            return {
                "status": "success",
                "message": f"Successfully added new {element_type}",
                "element_id": new_el.id(),
                "element_type": element_type
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to add element: {str(e)}")
            return {"status": "error", "message": f"Failed to add element: {str(e)}"}

    def _create_representation(self, geometry: dict) -> dict:
        if 'dimensions' not in geometry:
            return {}
            
        dims = geometry['dimensions']
        return {
            'context': self.ifc_file.by_type('IfcGeometricRepresentationContext')[0],
            'representation_type': 'SweptSolid',
            'representation_identifier': 'Body',
            'items': [self._create_extruded_solid(dims, geometry.get('position', {}))]
        }

    def _create_extruded_solid(self, dims: dict, position: dict) -> Any:
        profile = self.ifc_file.create_entity('IfcRectangleProfileDef', 
            ProfileType='AREA',
            XDim=dims.get('width', 1.0),
            YDim=dims.get('height', 1.0)
        )
        
        placement = self.ifc_file.create_entity('IfcAxis2Placement3D',
            Location=self.ifc_file.create_entity('IfcCartesianPoint', 
                Coordinates=(
                    position.get('x', 0.0),
                    position.get('y', 0.0),
                    position.get('z', 0.0)
                )
            )
        )
        
        return self.ifc_file.create_entity('IfcExtrudedAreaSolid',
            SweptArea=profile,
            Position=placement,
            ExtrudedDirection=self.ifc_file.create_entity('IfcDirection', DirectionRatios=(0., 0., 1.)),
            Depth=dims.get('depth', 0.1)
        )

    def _add_properties(self, element, properties: Dict[str, Any]):
        for pset_name, props in properties.items():
            if pset_name not in ifcopenshell.util.element.get_psets(element):
                ifcopenshell.api.run('pset.add_pset', self.ifc_file, product=element, name=pset_name)
            
            for prop_name, prop_value in props.items():
                ifcopenshell.api.run('pset.edit_pset', self.ifc_file,
                    pset=ifcopenshell.util.element.get_psets(element)[pset_name],
                    properties={prop_name: prop_value}
                )

    def remove_elements(self, element_type: str = None, element_id: int = None) -> Dict[str, Any]:
        """
        Remove elements from the IFC file. Can remove either a specific element or all elements of a type.
        Args:
            element_type (str, optional): Type of elements to remove (e.g., 'IfcWindow', 'IfcDoor')
            element_id (int, optional): ID of specific element to remove
        Returns:
            Dict containing status, message and count of removed elements
        """
        print(f"[DEBUG] Attempting to remove elements")
        if element_type:
            print(f"[DEBUG] Type: {element_type}")
        if element_id:
            print(f"[DEBUG] Element ID: {element_id}")
        
        try:
            count = 0
            
            if element_id is not None:
                # Remove specific element
                el = self.ifc_file.by_id(element_id)
                if not el:
                    return {"status": "error", "message": f"Element {element_id} not found"}
                print(f"[DEBUG] Removing element: {el.id()} ({getattr(el, 'Name', 'Unnamed')})")
                ifcopenshell.api.run("root.remove_product", self.ifc_file, product=el)
                count = 1
            elif element_type is not None:
                # Remove all elements of type
                elements = self.ifc_file.by_type(element_type)
                for el in elements:
                    print(f"[DEBUG] Removing element: {el.id()} ({getattr(el, 'Name', 'Unnamed')})")
                    ifcopenshell.api.run("root.remove_product", self.ifc_file, product=el)
                    count += 1
            else:
                return {"status": "error", "message": "Either element_type or element_id must be provided"}
            
            # Save the modified file
            save_result = self.save_modified_file()
            if save_result['status'] == 'error':
                return save_result
            
            return {
                "status": "success",
                "message": f"Successfully removed {count} element(s)",
                "removed_count": count
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to remove elements: {str(e)}")
            return {"status": "error", "message": f"Failed to remove elements: {str(e)}"}

    def save_modified_file(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not output_path:
                # For testing: always save to uploaded_ifcs/ISO_modified.ifc
                output_path = os.path.join("uploaded_ifcs", "ISO_modified.ifc")
            print(f"[DEBUG] Saving file to: {output_path}")
            self.ifc_file.write(output_path)
            return {'status': 'success', 'message': f'File saved to {output_path}'}
        except Exception as e:
            print(f"[ERROR] Failed to save file: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _extract_geometry(self, el) -> Dict[str, Any]:
        geometry = {}
        if hasattr(el, 'ObjectPlacement'):
            plc = el.ObjectPlacement
            if plc.is_a('IfcLocalPlacement'):
                loc = plc.RelativePlacement.Location
                geometry['location'] = {'x': loc.Coordinates[0], 'y': loc.Coordinates[1], 'z': loc.Coordinates[2]}
        if hasattr(el, 'Representation') and el.Representation and el.Representation.Representations:
            geometry['representations'] = [
                {
                    'type': rep.RepresentationType,
                    'identifier': rep.RepresentationIdentifier,
                    'items': [
                        {'type': 'extruded_area', 'depth': item.Depth} if item.is_a('IfcExtrudedAreaSolid')
                        else {'type': 'mapped_item'} if item.is_a('IfcMappedItem')
                        else {'type': 'unknown'}
                        for item in rep.Items
                    ]
                }
                for rep in el.Representation.Representations
                if rep.is_a('IfcShapeRepresentation')
            ]
        return geometry

    def find_element_by_name(self, name: str) -> List[Any]:
        """
        Find elements by their name.
        Returns a list of elements that match the given name.
        """
        elements = []
        for element in self.ifc_file.by_type("IfcRoot"):
            if hasattr(element, "Name") and element.Name == name:
                elements.append(element)
        return elements

    def find_elements_by_type(self, element_type: str) -> List[Any]:
        """
        Find elements by their IFC type.
        Returns a list of elements of the specified type.
        """
        return self.ifc_file.by_type(element_type)

    def find_elements_by_property(self, property_name: str, property_value: Any) -> List[Any]:
        """
        Find elements by a specific property value.
        Returns a list of elements that have the specified property value.
        """
        elements = []
        for element in self.ifc_file.by_type("IfcRoot"):
            if hasattr(element, "IsDefinedBy"):
                for definition in element.IsDefinedBy:
                    if definition.is_a("IfcRelDefinesByProperties"):
                        property_set = definition.RelatingPropertyDefinition
                        if property_set.is_a("IfcPropertySet"):
                            for prop in property_set.HasProperties:
                                if prop.Name == property_name and prop.NominalValue == property_value:
                                    elements.append(element)
        return elements

    def get_element_type_info(self, element) -> Dict[str, Any]:
        """
        Get detailed information about an element's type and properties.
        """
        info = {
            'type': element.is_a(),
            'properties': {},
            'quantities': {},
            'attributes': {}
        }
        
        # Get direct attributes
        for attr in element.get_info():
            if not attr.startswith('__'):
                info['attributes'][attr] = getattr(element, attr)

        # Get properties
        if hasattr(element, 'IsDefinedBy'):
            for definition in element.IsDefinedBy:
                if definition.is_a('IfcRelDefinesByProperties'):
                    property_set = definition.RelatingPropertyDefinition
                    if property_set.is_a('IfcPropertySet'):
                        for prop in property_set.HasProperties:
                            if prop.is_a('IfcPropertySingleValue'):
                                info['properties'][prop.Name] = prop.NominalValue.wrappedValue
                    elif property_set.is_a('IfcElementQuantity'):
                        for quantity in property_set.Quantities:
                            if quantity.is_a('IfcQuantityLength'):
                                info['quantities'][quantity.Name] = quantity.LengthValue
                            elif quantity.is_a('IfcQuantityArea'):
                                info['quantities'][quantity.Name] = quantity.AreaValue
                            elif quantity.is_a('IfcQuantityVolume'):
                                info['quantities'][quantity.Name] = quantity.VolumeValue

        return info

    def _get_element_properties(self, element) -> Dict[str, Any]:
        """
        Extract properties of an IFC element
        Args:
            element: IFC element to extract properties from
        Returns:
            Dictionary containing element properties
        """
        properties = {}
        
        # Get basic properties
        properties['Name'] = element.Name if hasattr(element, 'Name') else None
        properties['Description'] = element.Description if hasattr(element, 'Description') else None
        properties['ObjectType'] = element.ObjectType if hasattr(element, 'ObjectType') else None
        
        # Get Pset properties
        if hasattr(element, 'IsDefinedBy'):
            for definition in element.IsDefinedBy:
                if definition.is_a('IfcRelDefinesByProperties'):
                    property_set = definition.RelatingPropertyDefinition
                    if property_set.is_a('IfcPropertySet'):
                        for property in property_set.HasProperties:
                            if property.is_a('IfcPropertySingleValue'):
                                properties[property.Name] = property.NominalValue.wrappedValue if property.NominalValue else None
                            elif property.is_a('IfcPropertyEnumeratedValue'):
                                properties[property.Name] = [v.wrappedValue for v in property.EnumerationValues]
                            elif property.is_a('IfcPropertyListValue'):
                                properties[property.Name] = [v.wrappedValue for v in property.ListValues]

        return properties

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

# Example FastAPI endpoints for direct testing (not used in main workflow)
@app.post("/llm")
async def llm_post(request: Request):
    # Process POST request
    return {"result": "ok"}

@app.get("/llm")
async def llm_get():
    # Inform that GET is not supported for this endpoint
    return {"message": "GET not supported, use POST"} 