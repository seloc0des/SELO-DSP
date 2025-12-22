"""
Socket.IO Registry

Global registry to avoid circular imports when accessing Socket.IO server instance.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("selo.socketio.registry")

# Global Socket.IO server instance
_socketio_server = None

def register_socketio_server(server: Any) -> None:
    """
    Register the Socket.IO server instance globally.
    
    Args:
        server: Socket.IO AsyncServer instance
    """
    global _socketio_server
    _socketio_server = server
    logger.info("Socket.IO server registered in global registry")

def get_socketio_server() -> Optional[Any]:
    """
    Get the registered Socket.IO server instance.
    
    Returns:
        The registered Socket.IO server, or None if not registered
    """
    return _socketio_server

def is_socketio_available() -> bool:
    """
    Check if Socket.IO server is available.
    
    Returns:
        bool: True if server is registered, False otherwise
    """
    return _socketio_server is not None
