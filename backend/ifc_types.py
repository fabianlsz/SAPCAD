# Utility for listing editable IFC element types present in a given IFC file

import ifcopenshell
from constants import EDITABLE_TYPES

def list_editable_types(ifc_file_path):
    """
    Return a list of editable IFC element types that exist in the given IFC file.
    """
    ifc_file = ifcopenshell.open(ifc_file_path)
    return [t for t in EDITABLE_TYPES if ifc_file.by_type(t)] 