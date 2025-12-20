"""
Socket.IO Registry

Global registry to avoid circular imports when accessing Socket.IO server instance.
"""

import logging

logger = logging.getLogger("selo.socketio.registry")

# Global Socket.IO server instance
_socketio_server = None

def register_socketio_server(server):
    """
    Register the Socket.IO server instance globally.
    
    Args:
        server: Socket.IO server instance
    """
    global _socketio_server
    _socketio_server = server
    logger.info("Socket.IO server registered in global registry")

def get_socketio_server():
    """
    Get the registered Socket.IO server instance.
    
    Returns:
        Socket.IO server instance or None if not registered
    """
    return _socketio_server

def is_socketio_available():
    """
    Check if Socket.IO server is available.
    
    Returns:
        bool: True if Socket.IO server is registered
    """
    return _socketio_server is not None
