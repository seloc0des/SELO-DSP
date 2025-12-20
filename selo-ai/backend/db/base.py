"""
Shared SQLAlchemy Base for SELO AI database models.

All ORM models should import `Base` from this module to ensure a single
metadata registry across the application.
"""

from sqlalchemy.orm import declarative_base

# Single shared declarative base for all models
Base = declarative_base()
