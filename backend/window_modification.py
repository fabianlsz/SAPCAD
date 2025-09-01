import json
from datetime import datetime
from typing import Dict, Any

# Data structure for window modification information
class WindowModificationData:
    def __init__(self, window_id: int, new_width: float, new_height: float):
        # Initialize window modification data
        self.data = {
            "window_id": window_id,
            "new_width": new_width,
            "new_height": new_height,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
    
    def to_json(self) -> str:
        # Convert the data to a JSON string
        return json.dumps(self.data, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        # Return the data as a dictionary
        return self.data
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WindowModificationData':
        # Create an instance from a JSON string
        data = json.loads(json_str)
        return cls(
            window_id=data["window_id"],
            new_width=data["new_width"],
            new_height=data["new_height"]
        )
    
    def update_status(self, status: str):
        # Update the status of the modification
        self.data["status"] = status
    
    def add_metadata(self, key: str, value: Any):
        # Add additional metadata to the modification data
        self.data[key] = value 