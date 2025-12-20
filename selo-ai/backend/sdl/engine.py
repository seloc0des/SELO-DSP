"""
SDL Engine for the Self-Development Learning module.

This is the core component that processes reflections and interactions 
to generate learnings and update the AI's knowledge base.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Set, Tuple

from ..llm.router import LLMRouter
from ..memory.vector_store import VectorStore
from .repository import LearningRepository
from ..db.repositories.reflection import ReflectionRepository
from .concept_mapper import ConceptMapper
from .learning_models import Learning, Concept, Connection

logger = logging.getLogger("selo.sdl.engine")


class SDLEngine:
    """
    Self-Development Learning Engine.
    """
    def __init__(
        self, 
        llm_router: LLMRouter,
        vector_store: VectorStore,
        learning_repo: Optional[LearningRepository] = None,
        reflection_repo: Optional[ReflectionRepository] = None,
        concept_mapper: Optional[ConceptMapper] = None
    ):
        self.llm_router = llm_router
        self.vector_store = vector_store
        self.learning_repo = learning_repo or LearningRepository()
        self.reflection_repo = reflection_repo or ReflectionRepository()
        self.concept_mapper = concept_mapper or ConceptMapper(self.llm_router, self.learning_repo)
        
        # Define valid learning domains
        self.domains = {
            "personality", "preferences", "knowledge", "beliefs", 
            "relationships", "goals", "experiences", "skills", "meta_learning"
        }
        
        # Set minimum confidence threshold for storing learnings
        self.min_learning_confidence = 0.6
    
    async def close(self):
        """Close all resources used by the engine."""
        if self.learning_repo:
            await self.learning_repo.close()
        if self.reflection_repo:
            await self.reflection_repo.close()
    
    async def process_reflection(self, reflection_id: str) -> List[Dict[str, Any]]:
        """
        Process a reflection to extract learnings.
        
        Args:
            reflection_id: ID of the reflection to process
            
        Returns:
            List of learning objects that were extracted and stored
        """
        # Get the reflection
        reflection = await self.reflection_repo.get_reflection(reflection_id)
        if not reflection:
            logger.warning(f"Reflection {reflection_id} not found")
            return []
        
        user_id = reflection.user_id
        reflection_dict = reflection.to_dict()
        
        # Extract learnings using LLM
        learnings_data = await self._extract_learnings_from_reflection(reflection_dict)
        
        # Store learnings
        stored_learnings = []
        for learning_data in learnings_data:
            # Add user ID and source info
            learning_data["user_id"] = user_id
            learning_data["source_type"] = "reflection"
            learning_data["source_id"] = reflection_id
            
            # Skip low-confidence learnings
            if learning_data.get("confidence", 0) < self.min_learning_confidence:
                # Log full content without truncation (may be verbose by design)
                logger.debug(f"Skipping low-confidence learning: {learning_data.get('content', '')}")
                continue
                
            # Store the learning
            try:
                learning = await self.learning_repo.create_learning(learning_data)
                
                # Update vector store
                vector_id = await self._update_vector_store(learning)
                if vector_id:
                    await self.learning_repo.update_learning(
                        learning.id, {"vector_id": vector_id}
                    )
                
                stored_learnings.append(learning.to_dict())
                
                # Process concepts and connections
                await self._process_learning_concepts(learning.id)
                
            except Exception as e:
                logger.error(f"Error storing learning: {str(e)}", exc_info=True)
        
        logger.info(f"Processed reflection {reflection_id}, extracted {len(stored_learnings)} learnings")
        return stored_learnings
    
    async def process_conversation(
        self, 
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a conversation to extract learnings.
        
        Args:
            conversation_id: ID of the conversation
            messages: List of message objects with 'role' and 'content'
            
        Returns:
            List of learning objects that were extracted and stored
        """
        # Extract user ID from the first user message
        user_id = None
        for msg in messages:
            if msg.get("role") == "user" and "user_id" in msg:
                user_id = msg["user_id"]
                break
                
        if not user_id:
            logger.warning("No user ID found in conversation messages")
            return []
        
        # Extract learnings using LLM
        learnings_data = await self._extract_learnings_from_conversation(messages)
        
        # Store learnings
        stored_learnings = []
        for learning_data in learnings_data:
            # Add user ID and source info
            learning_data["user_id"] = user_id
            learning_data["source_type"] = "conversation"
            learning_data["source_id"] = conversation_id
            
            # Skip low-confidence learnings
            if learning_data.get("confidence", 0) < self.min_learning_confidence:
                logger.debug(f"Skipping low-confidence learning: {learning_data.get('content', '')}")
                continue
                
            # Store the learning
            try:
                learning = await self.learning_repo.create_learning(learning_data)
                
                # Update vector store
                vector_id = await self._update_vector_store(learning)
                if vector_id:
                    await self.learning_repo.update_learning(
                        learning.id, {"vector_id": vector_id}
                    )
                
                stored_learnings.append(learning.to_dict())
                
                # Process concepts and connections
                await self._process_learning_concepts(learning.id)
                
            except Exception as e:
                logger.error(f"Error storing learning: {str(e)}", exc_info=True)
        
        logger.info(f"Processed conversation {conversation_id}, extracted {len(stored_learnings)} learnings")
        return stored_learnings
    
    async def consolidate_learnings(self, user_id: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Consolidate learnings into a higher-level understanding.
        
        Args:
            user_id: User ID to consolidate learnings for
            domain: Optional domain to focus on
            
        Returns:
            Summary of consolidation results
        """
        # Get recent learnings to consolidate
        learnings = await self.learning_repo.get_learnings_for_user(
            user_id=user_id,
            limit=50,
            domain=domain
        )
        
        if not learnings:
            logger.info(f"No learnings found to consolidate for user {user_id}")
            return {"success": False, "reason": "No learnings found"}
        
        # Extract learning contents for consolidation
        learning_contents = [
            {
                "id": learning.id,
                "content": learning.content,
                "domain": learning.domain,
                "confidence": learning.confidence,
                "importance": learning.importance,
                "concepts": [c.name for c in learning.concepts] if learning.concepts else []
            }
            for learning in learnings
        ]
        
        # Use LLM to consolidate learnings
        consolidated = await self._consolidate_learnings_with_llm(
            user_id, learning_contents, domain
        )
        
        results = {
            "success": True,
            "learnings_count": len(learnings),
            "domain": domain,
            "consolidation": consolidated,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Consolidated {len(learnings)} learnings for user {user_id}")
        return results
    
    async def generate_meta_learning(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a meta-learning about the learning process itself.
        
        This helps the AI understand its own learning patterns and improve.
        
        Args:
            user_id: User ID to generate meta-learning for
            
        Returns:
            Generated meta-learning
        """
        # Get recent learnings stats
        learnings = await self.learning_repo.get_learnings_for_user(
            user_id=user_id,
            limit=100
        )
        
        # Create stats about the learnings
        domains_count = {}
        sources_count = {}
        concepts_set = set()
        
        for learning in learnings:
            # Count domains
            domain = learning.domain
            domains_count[domain] = domains_count.get(domain, 0) + 1
            
            # Count sources
            source = learning.source_type
            sources_count[source] = sources_count.get(source, 0) + 1
            
            # Collect concepts
            if learning.concepts:
                for concept in learning.concepts:
                    concepts_set.add(concept.name)
        
        # Prepare stats for LLM
        stats = {
            "total_learnings": len(learnings),
            "domains_distribution": domains_count,
            "sources_distribution": sources_count,
            "unique_concepts_count": len(concepts_set),
            "time_range": {
                "oldest": min([l.created_at for l in learnings]).isoformat() if learnings else None,
                "newest": max([l.created_at for l in learnings]).isoformat() if learnings else None
            }
        }
        
        # Generate meta-learning using LLM
        meta_learning = await self._generate_meta_learning_with_llm(user_id, stats)
        
        # Store the meta-learning
        learning_data = {
            "user_id": user_id,
            "content": meta_learning["content"],
            "source_type": "meta_analysis",
            "source_id": "system",
            "confidence": meta_learning.get("confidence", 0.85),
            "importance": meta_learning.get("importance", 0.7),
            "domain": "meta_learning",
            "concepts": meta_learning.get("concepts", [])
        }
        
        try:
            learning = await self.learning_repo.create_learning(learning_data)
            logger.info(f"Generated and stored meta-learning for user {user_id}")
            return learning.to_dict()
        except Exception as e:
            logger.error(f"Error storing meta-learning: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "meta_learning": meta_learning
            }
    
    async def search_knowledge(
        self, 
        user_id: str, 
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the AI's knowledge base for relevant learnings.
        
        Args:
            user_id: User ID to search learnings for
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching learning objects
        """
        # Convert query to vector
        vector = await self.llm_router.route(
            task_type="embedding",
            prompt=query,
            model="analytical"
        )
        
        # Search vector store
        results = await self.vector_store.search_by_embedding(
            embedding=vector,
            top_k=limit,
            threshold=0.5
        )
        
        # Get full learning objects, filtering by user_id
        learnings = []
        for result in results:
            # Check if this result belongs to the user
            if (result.get("metadata") or {}).get("user_id") == user_id:
                learning_id = result.get("id") or (result.get("metadata") or {}).get("id")
                if learning_id:
                    learning = await self.learning_repo.get_learning(learning_id)
                    if learning:
                        learnings.append(learning.to_dict())
        
        logger.info(f"Search for '{query}' returned {len(learnings)} results")
        return learnings
    
    async def get_related_concepts(
        self, 
        user_id: str,
        concept_name: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get concepts related to the given concept.
        
        Args:
            user_id: User ID
            concept_name: Name of the concept to find relations for
            max_depth: Maximum depth of relation traversal
            
        Returns:
            Dictionary with concept graph data
        """
        # Find the concept
        concept = await self.learning_repo.get_concept_by_name(user_id, concept_name)
        if not concept:
            return {
                "success": False,
                "reason": f"Concept '{concept_name}' not found"
            }
            
        # Build the concept graph
        nodes = {}
        edges = []
        visited = set()
        
        async def traverse_concept(concept_id: str, depth: int):
            if depth > max_depth or concept_id in visited:
                return
                
            visited.add(concept_id)
            
            # Get the concept
            concept = await self.learning_repo.get_concept(concept_id)
            if not concept:
                return
                
            # Add the node
            nodes[concept.id] = {
                "id": concept.id,
                "name": concept.name,
                "category": concept.category,
                "importance": concept.importance
            }
            
            # Get connections
            connections = await self.learning_repo.get_connections_for_concept(concept.id)
            
            for conn in connections:
                # Add edge
                edge = {
                    "source": conn.source_id,
                    "target": conn.target_id,
                    "type": conn.relation_type,
                    "strength": conn.strength
                }
                edges.append(edge)
                
                # Traverse connected concept
                next_id = conn.target_id if conn.source_id == concept.id else conn.source_id
                await traverse_concept(next_id, depth + 1)
        
        # Start traversal from the requested concept
        await traverse_concept(concept.id, 0)
        
        result = {
            "success": True,
            "central_concept": concept.name,
            "nodes": list(nodes.values()),
            "edges": edges
        }
        
        logger.info(f"Found {len(nodes)} related concepts for '{concept_name}'")
        return result
    
    # Internal methods
    
    async def _extract_learnings_from_reflection(
        self, 
        reflection: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract learnings from a reflection using LLM."""
        # Construct prompt for learning extraction
        system_prompt = """
        You are an expert at extracting meaningful learnings from reflections. 
        Analyze the following reflection from a persona AI and extract key learnings.
        
        For each learning, provide:
        - content: The clear, specific learning statement
        - confidence: A score from 0.0 to 1.0 indicating confidence in this learning
        - importance: A score from 0.0 to 1.0 indicating importance of this learning
        - domain: The knowledge domain (personality, preferences, knowledge, beliefs, relationships, goals, experiences, skills)
        - concepts: A list of 1-5 key concepts related to this learning
        
        Format your response as a JSON array of learning objects.
        """
        
        user_prompt = f"""
        Reflection type: {reflection.get('reflection_type')}
        Reflection content:
        {reflection.get('content')}
        
        Additional context:
        {json.dumps(reflection.get('metadata', {}))}
        
        Extract the key learnings from this reflection.
        """
        
        # Call LLM to extract learnings
        response = await self.llm_router.route(
            task_type="sdl",
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="analytical",
            json_output=True,
            max_tokens=1500
        )
        
        try:
            # Parse and validate learnings
            learnings = json.loads(response) if isinstance(response, str) else response
            
            # Ensure it's a list
            if not isinstance(learnings, list):
                logger.warning("LLM response is not a list, attempting to extract array")
                # Try to extract array if we got an object with an array property
                if isinstance(learnings, dict) and "learnings" in learnings:
                    learnings = learnings["learnings"]
                else:
                    logger.error("Failed to parse learnings from LLM response")
                    return []
                    
            # Validate each learning
            valid_learnings = []
            for learning in learnings:
                if not isinstance(learning, dict):
                    continue
                    
                # Ensure required fields
                if "content" not in learning:
                    continue
                    
                # Set default values for missing fields
                learning["confidence"] = learning.get("confidence", 0.7)
                learning["importance"] = learning.get("importance", 0.5)
                learning["domain"] = learning.get("domain", "knowledge")
                learning["concepts"] = learning.get("concepts", [])
                
                # Validate domain
                if learning["domain"] not in self.domains:
                    learning["domain"] = "knowledge"
                    
                valid_learnings.append(learning)
                
            return valid_learnings
            
        except Exception as e:
            logger.error(f"Error parsing learnings from LLM: {str(e)}", exc_info=True)
            return []
    
    async def _extract_learnings_from_conversation(
        self, 
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract learnings from a conversation using LLM."""
        # Construct prompt for learning extraction
        system_prompt = """
        You are an expert at extracting meaningful learnings from conversations. 
        Analyze the following conversation with a persona AI and extract key learnings 
        about the user's preferences, personality, and interests.
        
        For each learning, provide:
        - content: The clear, specific learning statement
        - confidence: A score from 0.0 to 1.0 indicating confidence in this learning
        - importance: A score from 0.0 to 1.0 indicating importance of this learning
        - domain: The knowledge domain (personality, preferences, knowledge, beliefs, relationships, goals, experiences, skills)
        - concepts: A list of 1-5 key concepts related to this learning
        
        Format your response as a JSON array of learning objects.
        """
        
        # Prepare conversation text
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
            
        user_prompt = f"""
        Conversation:
        {conversation_text}
        
        Extract the key learnings about the user from this conversation.
        """
        
        # Call LLM to extract learnings
        response = await self.llm_router.route(
            task_type="sdl",
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="analytical",
            json_output=True,
            max_tokens=1500
        )
        
        try:
            # Parse and validate learnings
            learnings = json.loads(response) if isinstance(response, str) else response
            
            # Ensure it's a list
            if not isinstance(learnings, list):
                logger.warning("LLM response is not a list, attempting to extract array")
                # Try to extract array if we got an object with an array property
                if isinstance(learnings, dict) and "learnings" in learnings:
                    learnings = learnings["learnings"]
                else:
                    logger.error("Failed to parse learnings from LLM response")
                    return []
                    
            # Validate each learning
            valid_learnings = []
            for learning in learnings:
                if not isinstance(learning, dict):
                    continue
                    
                # Ensure required fields
                if "content" not in learning:
                    continue
                    
                # Set default values for missing fields
                learning["confidence"] = learning.get("confidence", 0.7)
                learning["importance"] = learning.get("importance", 0.5)
                learning["domain"] = learning.get("domain", "preferences")
                learning["concepts"] = learning.get("concepts", [])
                
                # Validate domain
                if learning["domain"] not in self.domains:
                    learning["domain"] = "preferences"
                    
                valid_learnings.append(learning)
                
            return valid_learnings
            
        except Exception as e:
            logger.error(f"Error parsing learnings from LLM: {str(e)}", exc_info=True)
            return []
    
    async def _process_learning_concepts(self, learning_id: str) -> None:
        """Process concepts in a learning and build connections."""
        # Get the learning with concepts
        learning = await self.learning_repo.get_learning(learning_id)
        if not learning or not learning.concepts or len(learning.concepts) < 2:
            return
            
        # Use the concept mapper to analyze relationships
        await self.concept_mapper.process_learning_concepts(learning)
    
    async def _update_vector_store(self, learning: Learning) -> Optional[str]:
        """Update vector store with learning content."""
        try:
            # Get vector embedding for the learning content
            vector = await self.llm_router.route(
                task_type="embedding",
                prompt=learning.content,
                model="analytical"
            )
            
            # Store in vector database
            vector_id = await self.vector_store.store_embedding(
                text=learning.content,
                embedding_id=learning.id,
                metadata={
                    "user_id": learning.user_id,
                    "domain": learning.domain,
                    "importance": learning.importance,
                    "type": "learning"
                }
            )
            
            return vector_id
            
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}", exc_info=True)
            return None
    
    async def _consolidate_learnings_with_llm(
        self, 
        user_id: str,
        learning_contents: List[Dict[str, Any]],
        domain: Optional[str]
    ) -> Dict[str, Any]:
        """Consolidate learnings using LLM."""
        system_prompt = """
        You are an expert at analyzing and consolidating learnings to form a coherent 
        understanding of a person. Review the following learnings and provide a 
        consolidated understanding and key insights.
        
        Format your response as a JSON object with:
        - summary: A concise summary of the key insights
        - themes: The main themes or patterns identified
        - contradictions: Any contradictions or inconsistencies in the learnings
        - confidence: Overall confidence in these consolidated insights (0.0-1.0)
        """
        
        # Prepare learnings text
        learnings_text = json.dumps(learning_contents, indent=2)
        domain_text = f" for the '{domain}' domain" if domain else ""
        
        user_prompt = f"""
        Here are the learnings{domain_text} for user {user_id}:
        
        {learnings_text}
        
        Please analyze these learnings and provide a consolidated understanding.
        """
        
        # Call LLM to consolidate
        response = await self.llm_router.route(
            task_type="sdl",
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="analytical",
            json_output=True,
            max_tokens=1000
        )
        
        try:
            # Parse response
            consolidated = json.loads(response) if isinstance(response, str) else response
            return consolidated
        except Exception as e:
            logger.error(f"Error parsing consolidation from LLM: {str(e)}", exc_info=True)
            return {
                "summary": "Failed to consolidate learnings due to processing error.",
                "error": str(e)
            }
    
    async def _generate_meta_learning_with_llm(
        self, 
        user_id: str,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate meta-learning using LLM."""
        system_prompt = """
        You are an expert at analyzing learning patterns and generating meta-insights.
        Review the following statistics about a persona AI's learning process and 
        generate a meta-learning insight about the learning process itself.
        
        Format your response as a JSON object with:
        - content: The meta-learning insight
        - confidence: Confidence in this insight (0.0-1.0)
        - importance: Importance of this insight (0.0-1.0)
        - concepts: List of relevant concepts for this meta-learning
        """
        
        # Prepare stats text
        stats_text = json.dumps(stats, indent=2)
        
        user_prompt = f"""
        Here are the learning statistics for user {user_id}:
        
        {stats_text}
        
        Please analyze these statistics and generate a meta-learning insight 
        about the learning patterns and process.
        """
        
        # Call LLM to generate meta-learning
        response = await self.llm_router.route(
            task_type="sdl",
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="analytical",
            json_output=True,
            max_tokens=800
        )
        
        try:
            # Parse response
            meta_learning = json.loads(response) if isinstance(response, str) else response
            return meta_learning
        except Exception as e:
            logger.error(f"Error parsing meta-learning from LLM: {str(e)}", exc_info=True)
            return {
                "content": "Failed to generate meta-learning due to processing error.",
                "confidence": 0.5,
                "importance": 0.5,
                "concepts": ["error", "learning_process"]
            }
