from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

@dataclass
class Filter:
    """Represents a filter applied to an image"""
    id: str  # Unique identifier
    name: str  # Display name
    type: str  # Filter type
    preview: bool = False  # Flag for preview mode
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for the filter
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


class FilterStack:
    """Manages a stack of filters applied to an image"""
    
    def __init__(self):
        self.filters: List[Filter] = []
    
    def add_filter(self, name: str, filter_type: str, params: Dict[str, Any] = None, preview=False) -> Filter:
        """Add a filter to the stack and return it"""
        if params is None:
            params = {}
        
        filter_id = str(uuid.uuid4())
        new_filter = Filter(id=filter_id, name=name, type=filter_type, params=params, preview=preview)
        if len(self.filters) > 0 and self.filters[-1].preview:
            self.filters[-1] = new_filter
        else:
            self.filters.append(new_filter)
        return new_filter
    
    def remove_filter(self, filter_id: str) -> bool:
        """Remove a filter by ID"""
        for i, filter_obj in enumerate(self.filters):
            if filter_obj.id == filter_id:
                self.filters.pop(i)
                return True
        return False
    
    def clear(self) -> None:
        """Clear all filters"""
        self.filters.clear()
    
    def get_filter(self, filter_id: str) -> Optional[Filter]:
        """Get a filter by ID"""
        for filter_obj in self.filters:
            if filter_obj.id == filter_id:
                return filter_obj
        return None
    
    def get_filters(self) -> List[Filter]:
        """Get all filters"""
        return self.filters
    
    def move_filter(self, filter_id: str, new_position: int) -> bool:
        """Move a filter to a new position"""
        if new_position < 0 or new_position >= len(self.filters):
            return False
        
        for i, filter_obj in enumerate(self.filters):
            if filter_obj.id == filter_id:
                filter_obj = self.filters.pop(i)
                self.filters.insert(new_position, filter_obj)
                return True
        
        return False