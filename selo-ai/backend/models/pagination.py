"""
Pagination models and utilities for SELO AI Backend.

Provides standardized pagination support across all API endpoints
to improve performance and prevent large payload issues.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Generic, TypeVar, List, Optional, Any
from math import ceil

T = TypeVar('T')

class PaginationParams(BaseModel):
    """Standard pagination parameters for API requests."""
    
    page: int = Field(default=1, ge=1, description="Page number (1-based)")
    size: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")
    
    @field_validator('page')
    @classmethod
    def validate_page(cls, v):
        if v < 1:
            raise ValueError('Page must be >= 1')
        return v
    
    @field_validator('size')
    @classmethod
    def validate_size(cls, v):
        if v < 1:
            raise ValueError('Size must be >= 1')
        if v > 100:
            raise ValueError('Size must be <= 100')
        return v
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size
    
    @property
    def limit(self) -> int:
        """Get limit for database queries."""
        return self.size

class PaginationMeta(BaseModel):
    """Pagination metadata for API responses."""
    
    page: int = Field(description="Current page number")
    size: int = Field(description="Items per page")
    total_items: int = Field(description="Total number of items")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_prev: bool = Field(description="Whether there are previous pages")
    
    @classmethod
    def from_params(cls, params: PaginationParams, total_items: int) -> 'PaginationMeta':
        """Create pagination metadata from parameters and total count."""
        total_pages = ceil(total_items / params.size) if total_items > 0 else 0
        
        return cls(
            page=params.page,
            size=params.size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_prev=params.page > 1
        )

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    
    items: List[T] = Field(description="List of items for current page")
    pagination: PaginationMeta = Field(description="Pagination metadata")
    
    @classmethod
    def create(cls, items: List[T], params: PaginationParams, total_items: int) -> 'PaginatedResponse[T]':
        """Create a paginated response."""
        pagination = PaginationMeta.from_params(params, total_items)
        return cls(items=items, pagination=pagination)

class SortParams(BaseModel):
    """Standard sorting parameters for API requests."""
    
    sort_by: Optional[str] = Field(default=None, description="Field to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v):
        if v not in ('asc', 'desc'):
            raise ValueError('Sort order must be "asc" or "desc"')
        return v

class FilterParams(BaseModel):
    """Base class for filter parameters."""
    pass

def paginate_query_result(items: List[T], params: PaginationParams, total_count: int) -> PaginatedResponse[T]:
    """
    Helper function to create paginated response from query results.
    
    Args:
        items: List of items for current page
        params: Pagination parameters
        total_count: Total number of items available
        
    Returns:
        PaginatedResponse with items and metadata
    """
    return PaginatedResponse.create(items, params, total_count)
