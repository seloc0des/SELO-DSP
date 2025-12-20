"""
Reflection SocketIO Namespace

This module implements the Socket.IO namespace for reflection-related events.
"""

import logging
import json
from typing import Dict, List, Optional, Any

logger = logging.getLogger("selo.socketio.reflection")

class ReflectionNamespace:
    """
    Socket.IO namespace for reflection events.
    
    This class handles real-time communication related to reflections,
    including generation events, updates, and queries.
    """
    
    def __init__(self, reflection_processor, reflection_scheduler=None):
        """
        Initialize the reflection namespace with dependency-injected processor.
        Args:
            reflection_processor: Reflection processor instance (should use analytical LLM)
            reflection_scheduler: Reflection scheduler instance
        """
        self.reflection_processor = reflection_processor
        self.reflection_scheduler = reflection_scheduler
        self.connected_clients = {}
        self.sio_server = None  # Will be set during registration
        # This namespace always uses the analytical LLM for backend reflection tasks.
        
    def register(self, sio):
        """
        Register event handlers with the Socket.IO server.
        
        Args:
            sio: Socket.IO server instance
        """
        # Store Socket.IO server reference for later use
        self.sio_server = sio
        try:
            setattr(sio, "reflection_namespace", self)
        except Exception:
            logger.debug("Unable to attach reflection namespace to Socket.IO server instance", exc_info=True)
        @sio.on('connect', namespace='/reflection')
        async def connect(sid, environ):
            self.connected_clients[sid] = {
                'connected_at': environ.get('time', 0),
                'user_id': None  # Will be set on authentication
            }
            logger.info(f"Client connected to reflection namespace: {sid}")
            
        @sio.on('disconnect', namespace='/reflection')
        async def disconnect(sid):
            if sid in self.connected_clients:
                del self.connected_clients[sid]
            logger.info(f"Client disconnected from reflection namespace: {sid}")
            
        @sio.on('authenticate', namespace='/reflection')
        async def authenticate(sid, data):
            try:
                user_id = data.get('user_id')
                if not user_id:
                    await sio.emit('error', {'message': 'User ID required'}, room=sid, namespace='/reflection')
                    return
                    
                # Store user ID with connection
                if sid in self.connected_clients:
                    self.connected_clients[sid]['user_id'] = user_id
                    
                await sio.emit('authenticated', {'status': 'success'}, room=sid, namespace='/reflection')
                logger.info(f"Client {sid} authenticated as user {user_id}")
                
            except Exception as e:
                logger.error(f"Authentication error: {str(e)}", exc_info=True)
                await sio.emit('error', {'message': str(e)}, room=sid, namespace='/reflection')
                
        @sio.on('generate_reflection', namespace='/reflection')
        async def generate_reflection(sid, data):
            try:
                # Validate request
                if sid not in self.connected_clients:
                    await sio.emit('error', {'message': 'Not connected'}, room=sid, namespace='/reflection')
                    return
                    
                user_id = self.connected_clients[sid].get('user_id')
                if not user_id:
                    await sio.emit('error', {'message': 'Not authenticated'}, room=sid, namespace='/reflection')
                    return
                    
                # Extract parameters
                reflection_type = data.get('reflection_type')
                memory_ids = data.get('memory_ids')
                
                if not reflection_type:
                    await sio.emit('error', {'message': 'Reflection type required'}, room=sid, namespace='/reflection')
                    return
                    
                # Emit 'generating' event to indicate start of processing
                await sio.emit('reflection_generating', {
                    'reflection_type': reflection_type,
                    'user_id': user_id
                }, room=sid, namespace='/reflection')
                
                # Generate reflection
                if self.reflection_scheduler:
                    result = await self.reflection_scheduler.trigger_reflection(
                        reflection_type=reflection_type,
                        user_profile_id=user_id,
                        memory_ids=memory_ids
                    )
                elif self.reflection_processor:
                    result = await self.reflection_processor.generate_reflection(
                        reflection_type=reflection_type,
                        user_profile_id=user_id,
                        memory_ids=memory_ids,
                        trigger_source='socketio'
                    )
                else:
                    await sio.emit('error', {'message': 'Reflection service not available'}, room=sid, namespace='/reflection')
                    return
                
                # Check for errors
                if 'error' in result:
                    await sio.emit('error', {'message': result['error']}, room=sid, namespace='/reflection')
                    return
                    
                # Emit result
                await sio.emit('reflection_generated', {
                    'reflection_id': result.get('reflection_id'),
                    'reflection_type': reflection_type,
                    'result': result.get('result')
                }, room=sid, namespace='/reflection')
                
                logger.info(f"Generated reflection for user {user_id}, type: {reflection_type}")
                
            except Exception as e:
                logger.error(f"Error generating reflection: {str(e)}", exc_info=True)
                await sio.emit('error', {'message': str(e)}, room=sid, namespace='/reflection')
                
        @sio.on('list_reflections', namespace='/reflection')
        async def list_reflections(sid, data):
            try:
                # Validate request
                if sid not in self.connected_clients:
                    await sio.emit('error', {'message': 'Not connected'}, room=sid, namespace='/reflection')
                    return
                    
                user_id = self.connected_clients[sid].get('user_id')
                if not user_id:
                    await sio.emit('error', {'message': 'Not authenticated'}, room=sid, namespace='/reflection')
                    return
                    
                # Extract parameters
                reflection_type = data.get('reflection_type')
                limit = data.get('limit', 10)
                offset = data.get('offset', 0)
                
                # Get reflections from repository
                if self.reflection_processor and self.reflection_processor.reflection_repo:
                    repo = self.reflection_processor.reflection_repo
                    reflections = await repo.list_reflections(
                        user_profile_id=user_id,
                        reflection_type=reflection_type,
                        limit=limit,
                        offset=offset
                    )
                    
                    await sio.emit('reflections_list', {
                        'reflections': reflections,
                        'count': len(reflections),
                        'limit': limit,
                        'offset': offset
                    }, room=sid, namespace='/reflection')
                    
                else:
                    await sio.emit('error', {'message': 'Reflection repository not available'}, room=sid, namespace='/reflection')
                    
            except Exception as e:
                logger.error(f"Error listing reflections: {str(e)}", exc_info=True)
                await sio.emit('error', {'message': str(e)}, room=sid, namespace='/reflection')
                
    async def emit_reflection_event(self, event_name, data, user_id=None):
        """
        Emit a reflection-related event to connected clients.
        
        Args:
            event_name: Name of the event to emit
            data: Event data
            user_id: Optional user ID to target specific user, or None for broadcast
        """
        if not self.sio_server:
            logger.warning("Socket.IO server not available, cannot emit reflection event")
            return
            
        try:
            if user_id:
                # Target specific user - find their session ID(s)
                target_sids = [
                    sid for sid, client_info in self.connected_clients.items() 
                    if client_info.get('user_id') == user_id
                ]
                
                if target_sids:
                    for sid in target_sids:
                        await self.sio_server.emit(event_name, data, room=sid, namespace='/reflection')
                    logger.debug(f"Emitted {event_name} to {len(target_sids)} sessions for user {user_id}")
                else:
                    logger.debug(f"No active sessions found for user {user_id}")
            else:
                # Broadcast to all connected clients in reflection namespace
                await self.sio_server.emit(event_name, data, namespace='/reflection')
                logger.debug(f"Broadcasted {event_name} to all clients in reflection namespace")
                
        except Exception as e:
            logger.error(f"Error emitting reflection event {event_name}: {str(e)}", exc_info=True)
