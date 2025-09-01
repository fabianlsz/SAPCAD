import re
import sys
import argparse
import json
from typing import List, Dict, Any
import ifcopenshell
import PyPDF2
import os
from ifc_modifier import IFCModifier

class IFCCommandGenerator:
    def __init__(self):
        """Initialize the command generator"""
        self.commands = []
        self.modifier = None
        
    def set_modifier(self, ifc_file_path: str):
        """Set the IFC modifier instance"""
        self.modifier = IFCModifier(ifc_file_path)
        
    def send_commands_to_modifier(self) -> Dict[str, Any]:
        """Send the generated commands to the IFC modifier"""
        if not self.modifier:
            return {'status': 'error', 'message': 'IFC modifier not initialized'}
            
        results = []
        for command in self.commands:
            try:
                # Parse the command
                parts = command.split()
                if len(parts) < 3:
                    continue
                    
                action = parts[0]
                element_type = parts[1]
                element_id = parts[2]
                
                if action == 'rename':
                    # Handle rename command
                    new_name = ' '.join(parts[3:])
                    result = self.modifier.modify_element(element_id, {'Name': new_name})
                elif action == 'move':
                    # Handle move command
                    x, y, z = map(float, parts[3:6])
                    result = self.modifier.modify_element(element_id, {
                        'ObjectPlacement': {'Location': {'Coordinates': (x, y, z)}}
                    })
                elif action == 'set':
                    # Handle set property command
                    if len(parts) >= 6 and parts[1] == 'property':
                        prop_name = parts[4]
                        prop_value = ' '.join(parts[5:])
                        result = self.modifier.modify_element(element_id, {prop_name: prop_value})
                elif action == 'delete':
                    # Handle delete command
                    # Note: Delete functionality needs to be implemented in IFCModifier
                    result = {'status': 'error', 'message': 'Delete operation not implemented'}
                    
                results.append(result)
                
            except Exception as e:
                results.append({'status': 'error', 'message': str(e)})
                
        return {
            'status': 'success' if all(r['status'] == 'success' for r in results) else 'error',
            'results': results
        }
    
    def parse_input_file(self, input_file: str) -> bool:
        """Parse an input file (TXT, PDF, IFC, or JSON) and generate IFC commands"""
        file_ext = os.path.splitext(input_file)[1].lower()
        
        if file_ext == '.pdf':
            return self._parse_pdf(input_file)
        elif file_ext == '.ifc':
            return self._parse_ifc(input_file)
        elif file_ext == '.txt':
            return self.parse_text_file(input_file)
        elif file_ext == '.json':
            return self._parse_json(input_file)
        else:
            print(f"Unsupported file format: {file_ext}")
            return False
    
    def _parse_pdf(self, pdf_file: str) -> bool:
        """Parse a PDF file and extract commands"""
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    # Split text into lines and process each line
                    for line in text.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            command = self._parse_line(line, page_num + 1)
                            if command:
                                self.commands.append(command)
            return True
        except Exception as e:
            print(f"Error parsing PDF file: {e}")
            return False
    
    def _parse_ifc(self, ifc_file: str) -> bool:
        """Parse an IFC file and generate commands based on its contents"""
        try:
            ifc = ifcopenshell.open(ifc_file)
            
            # Process all products in the IFC file
            for product in ifc.by_type('IfcProduct'):
                element_type = product.is_a()
                element_id = product.GlobalId
                name = product.Name if hasattr(product, "Name") and product.Name else "Unnamed"
                
                # Generate rename command
                self.commands.append(f"rename {element_type} {element_id} {name}")
                
                # Generate position command if available
                if hasattr(product, "ObjectPlacement"):
                    placement = product.ObjectPlacement
                    if placement.is_a("IfcLocalPlacement"):
                        location = placement.RelativePlacement.Location
                        if hasattr(location, "Coordinates"):
                            x, y, z = location.Coordinates
                            self.commands.append(f"move {element_type} {element_id} {x} {y} {z}")
                
                # Generate property commands
                for definition in getattr(product, "IsDefinedBy", []):
                    if hasattr(definition, "RelatingPropertyDefinition"):
                        prop_def = definition.RelatingPropertyDefinition
                        if prop_def.is_a('IfcPropertySet'):
                            for prop in prop_def.HasProperties:
                                if hasattr(prop, "NominalValue"):
                                    value = prop.NominalValue.wrappedValue
                                    self.commands.append(f"set property {element_type} {element_id} {prop.Name} {value}")
            
            return True
        except Exception as e:
            print(f"Error parsing IFC file: {e}")
            return False
    
    def parse_text_file(self, input_file: str) -> bool:
        """Parse a text file and generate IFC commands"""
        try:
            with open(input_file, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        command = self._parse_line(line, line_num)
                        if command:
                            self.commands.append(command)
            return True
        except Exception as e:
            print(f"Error parsing file: {e}")
            return False
    
    def _parse_line(self, line: str, line_num: int) -> str:
        """Parse a single line and convert it to an IFC command"""
        try:
            # Remove any extra whitespace
            line = ' '.join(line.split())
            
            # Try to match different patterns
            # Pattern 1: Element definition with properties
            # Format: ElementType:ID:Name:Property1=Value1:Property2=Value2
            element_pattern = r'^([^:]+):([^:]+):([^:]+)(?::([^:]+))?$'
            match = re.match(element_pattern, line)
            if match:
                element_type, element_id, name, properties = match.groups()
                commands = []
                
                # Add rename command
                commands.append(f"rename {element_type} {element_id} {name}")
                
                # Add property commands if any
                if properties:
                    for prop in properties.split(':'):
                        if '=' in prop:
                            prop_name, prop_value = prop.split('=')
                            commands.append(f"set property {element_type} {element_id} {prop_name} {prop_value}")
                
                return '\n'.join(commands)
            
            # Pattern 2: Position definition
            # Format: ElementType:ID:Position(x,y,z)
            position_pattern = r'^([^:]+):([^:]+):Position\(([^)]+)\)$'
            match = re.match(position_pattern, line)
            if match:
                element_type, element_id, coords = match.groups()
                x, y, z = map(float, coords.split(','))
                return f"move {element_type} {element_id} {x} {y} {z}"
            
            # Pattern 3: Delete command
            # Format: Delete:ElementType:ID
            delete_pattern = r'^Delete:([^:]+):([^:]+)$'
            match = re.match(delete_pattern, line)
            if match:
                element_type, element_id = match.groups()
                return f"delete {element_type} {element_id}"
            
            print(f"Warning: Line {line_num} does not match any known pattern: {line}")
            return None
            
        except Exception as e:
            print(f"Error parsing line {line_num}: {e}")
            return None
    
    def save_commands(self, output_file: str) -> bool:
        """Save the generated commands to a file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                for command in self.commands:
                    file.write(command + '\n')
            print(f"Commands saved to: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving commands: {e}")
            return False

    def _parse_json(self, json_file: str) -> bool:
        """Parse a JSON file and extract commands"""
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Handle array of commands
                if isinstance(data, list):
                    for item in data:
                        self._process_json_command(item)
                # Handle single command object
                elif isinstance(data, dict):
                    self._process_json_command(data)
                else:
                    print("Invalid JSON format: expected array or object")
                    return False
                    
            return True
        except Exception as e:
            print(f"Error parsing JSON file: {e}")
            return False
            
    def _process_json_command(self, command_data: Dict[str, Any]) -> None:
        """Process a single command from JSON data"""
        try:
            action = command_data.get('action')
            element_type = command_data.get('element_type')
            element_id = command_data.get('element_id')
            
            if not all([action, element_type, element_id]):
                print("Missing required fields in JSON command")
                return
                
            if action == 'rename':
                new_name = command_data.get('new_name')
                if new_name:
                    self.commands.append(f"rename {element_type} {element_id} {new_name}")
                    
            elif action == 'move':
                location = command_data.get('location', {})
                x = location.get('x')
                y = location.get('y')
                z = location.get('z')
                if all(v is not None for v in [x, y, z]):
                    self.commands.append(f"move {element_type} {element_id} {x} {y} {z}")
                    
            elif action == 'set':
                property_name = command_data.get('property_name')
                property_value = command_data.get('property_value')
                if property_name and property_value is not None:
                    self.commands.append(f"set property {element_type} {element_id} {property_name} {property_value}")
                    
            elif action == 'delete':
                self.commands.append(f"delete {element_type} {element_id}")
                
        except Exception as e:
            print(f"Error processing JSON command: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate IFC commands from various input files")
    parser.add_argument("input_file", help="Path to the input file (TXT, PDF, or IFC)")
    parser.add_argument("-o", "--output", help="Output file for commands (default: commands.txt)")
    parser.add_argument("-m", "--modifier", help="Path to the IFC file to modify")
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        args.output = "commands.txt"
    
    generator = IFCCommandGenerator()
    
    if generator.parse_input_file(args.input_file):
        # Save commands to file
        generator.save_commands(args.output)
        
        # If modifier file is specified, send commands to modifier
        if args.modifier:
            generator.set_modifier(args.modifier)
            result = generator.send_commands_to_modifier()
            print(f"Modifier results: {result}")
            
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main()) 