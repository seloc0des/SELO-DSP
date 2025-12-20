"""
Concept Mapper for the SDL (Self-Development Learning) module.

This module is responsible for mapping and organizing concepts 
into a knowledge graph, identifying relationships between concepts,
and maintaining the conceptual structure of the AI's knowledge.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple

from ..llm.router import LLMRouter
from .learning_models import Learning, Concept, Connection

logger = logging.getLogger("selo.sdl.concept_mapper")


class ConceptMapper:
    """
    Maps and organizes concepts into a knowledge graph.
    """
    def __init__(self, llm_router: LLMRouter, learning_repo):
        self.llm_router = llm_router
        self.learning_repo = learning_repo
        
        # Define valid relationship types
        self.relation_types = {
            "is-a", "has-a", "part-of", "related-to", "causes", 
            "opposite-of", "similar-to", "depends-on", "enables", "contains"
        }
    
    async def process_learning_concepts(self, learning: Learning) -> List[Dict[str, Any]]:
        """
        Process concepts in a learning and identify relationships.
        
        Args:
            learning: The learning object with concepts to process
            
        Returns:
            List of created connections
        """
        if not learning or not learning.concepts or len(learning.concepts) < 2:
            logger.debug(f"Learning {getattr(learning, 'id', 'unknown')} has fewer than 2 concepts, skipping")
            return []
            
        user_id = learning.user_id
        concepts = learning.concepts
        
        # Analyze concept relationships
        relationships = await self._analyze_concept_relationships(
            user_id,
            [(c.name, c.id) for c in concepts], 
            learning.content
        )
        
        # Create connections
        created_connections = []
        for rel in relationships:
            # Get source and target concept IDs
            source_name, target_name = rel["source"], rel["target"]
            
            source_id = None
            target_id = None
            
            # Find concept IDs from the concepts list
            for concept in concepts:
                if concept.name == source_name:
                    source_id = concept.id
                if concept.name == target_name:
                    target_id = concept.id
            
            # Skip if we couldn't find both concepts
            if not source_id or not target_id:
                logger.warning(f"Could not find concepts {source_name} or {target_name}")
                continue
                
            # Create the connection
            try:
                connection_data = {
                    "user_id": user_id,
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": rel["relation_type"],
                    "strength": rel["strength"],
                    "bidirectional": rel["bidirectional"]
                }
                
                connection = await self.learning_repo.create_connection(connection_data)
                created_connections.append(connection.to_dict())
                
                logger.debug(f"Created connection: {source_name} -{rel['relation_type']}-> {target_name}")
                
            except Exception as e:
                logger.error(f"Error creating connection: {str(e)}", exc_info=True)
        
        logger.info(f"Created {len(created_connections)} connections for learning {learning.id}")
        return created_connections
    
    async def reorganize_concepts(self, user_id: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Reorganize concepts to improve the knowledge graph structure.
        
        This involves analyzing the current concept relationships, identifying
        missing connections, redundant connections, and suggesting improvements.
        
        Args:
            user_id: The user ID
            category: Optional category to focus on
            
        Returns:
            Summary of reorganization changes
        """
        # Get concepts to analyze
        concepts = await self.learning_repo.get_concepts_for_user(
            user_id=user_id,
            limit=100,  # Analyze a reasonable number of concepts
            category=category
        )
        
        if not concepts or len(concepts) < 5:
            logger.info(f"Not enough concepts to reorganize for user {user_id}")
            return {
                "success": False,
                "reason": "Not enough concepts available for reorganization"
            }
        
        # Extract concept data
        concept_data = []
        for concept in concepts:
            # Get connections
            connections = await self.learning_repo.get_connections_for_concept(concept.id)
            
            # Build connection data
            connection_data = []
            for conn in connections:
                source = await self.learning_repo.get_concept(conn.source_id)
                target = await self.learning_repo.get_concept(conn.target_id)
                
                if source and target:
                    connection_data.append({
                        "source": source.name,
                        "target": target.name,
                        "relation_type": conn.relation_type,
                        "strength": conn.strength,
                        "bidirectional": conn.bidirectional
                    })
            
            # Add concept with connections
            concept_data.append({
                "id": concept.id,
                "name": concept.name,
                "category": concept.category,
                "importance": concept.importance,
                "connections": connection_data
            })
        
        # Analyze for reorganization
        reorganization = await self._analyze_concept_organization(user_id, concept_data)
        
        # Apply suggested changes
        changes_made = {
            "connections_added": 0,
            "connections_removed": 0,
            "connections_modified": 0,
            "concepts_categorized": 0
        }
        
        # Add new connections
        for conn in reorganization.get("suggested_connections", []):
            source_concept = await self.learning_repo.get_concept_by_name(user_id, conn["source"])
            target_concept = await self.learning_repo.get_concept_by_name(user_id, conn["target"])
            
            if source_concept and target_concept:
                try:
                    connection_data = {
                        "user_id": user_id,
                        "source_id": source_concept.id,
                        "target_id": target_concept.id,
                        "relation_type": conn["relation_type"],
                        "strength": conn["strength"],
                        "bidirectional": conn.get("bidirectional", False)
                    }
                    
                    await self.learning_repo.create_connection(connection_data)
                    changes_made["connections_added"] += 1
                    
                except Exception as e:
                    logger.error(f"Error creating connection: {str(e)}", exc_info=True)
        
        # Remove redundant connections
        for conn in reorganization.get("redundant_connections", []):
            source_concept = await self.learning_repo.get_concept_by_name(user_id, conn["source"])
            target_concept = await self.learning_repo.get_concept_by_name(user_id, conn["target"])
            
            if source_concept and target_concept:
                # Find the connection
                connections = await self.learning_repo.get_connections_for_concept(source_concept.id)
                
                for connection in connections:
                    if connection.target_id == target_concept.id:
                        try:
                            await self.learning_repo.delete_connection(connection.id)
                            changes_made["connections_removed"] += 1
                        except Exception as e:
                            logger.error(f"Error deleting connection: {str(e)}", exc_info=True)
        
        # Update concept categories
        for cat in reorganization.get("suggested_categories", []):
            concept = await self.learning_repo.get_concept_by_name(user_id, cat["concept"])
            
            if concept:
                try:
                    await self.learning_repo.update_concept(
                        concept.id,
                        {"category": cat["category"]}
                    )
                    changes_made["concepts_categorized"] += 1
                except Exception as e:
                    logger.error(f"Error updating concept: {str(e)}", exc_info=True)
        
        result = {
            "success": True,
            "concepts_analyzed": len(concepts),
            "changes_made": changes_made,
            "analysis": reorganization.get("analysis", {}),
            "categories": reorganization.get("categories", [])
        }
        
        logger.info(f"Reorganized concepts for user {user_id}: {changes_made}")
        return result
    
    async def generate_concept_summary(self, user_id: str, concept_name: str) -> Dict[str, Any]:
        """
        Generate a summary for a concept based on related learnings and connections.
        
        Args:
            user_id: The user ID
            concept_name: The name of the concept to summarize
            
        Returns:
            Concept summary
        """
        # Find the concept
        concept = await self.learning_repo.get_concept_by_name(user_id, concept_name)
        if not concept:
            logger.warning(f"Concept {concept_name} not found for user {user_id}")
            return {
                "success": False,
                "reason": f"Concept '{concept_name}' not found"
            }
        
        # Get related learnings
        related_learnings = []
        if concept.learnings:
            for learning in concept.learnings[:10]:  # Limit to 10 most relevant
                related_learnings.append({
                    "content": learning.content,
                    "domain": learning.domain,
                    "importance": learning.importance,
                    "confidence": learning.confidence
                })
        
        # Get connections
        connections = await self.learning_repo.get_connections_for_concept(concept.id)
        
        related_concepts = []
        for conn in connections:
            other_id = conn.target_id if conn.source_id == concept.id else conn.source_id
            other_concept = await self.learning_repo.get_concept(other_id)
            
            if other_concept:
                direction = "outgoing" if conn.source_id == concept.id else "incoming"
                related_concepts.append({
                    "name": other_concept.name,
                    "relation_type": conn.relation_type,
                    "direction": direction,
                    "strength": conn.strength
                })
        
        # Generate summary using LLM
        concept_data = {
            "name": concept.name,
            "description": concept.description,
            "category": concept.category,
            "importance": concept.importance,
            "familiarity": concept.familiarity,
            "related_learnings": related_learnings,
            "related_concepts": related_concepts
        }
        
        summary = await self._generate_concept_summary_with_llm(user_id, concept_data)
        
        # Update concept description if it improved
        if (not concept.description or 
            len(summary.get("description", "")) > len(concept.description)):
            try:
                await self.learning_repo.update_concept(
                    concept.id,
                    {"description": summary.get("description", "")}
                )
            except Exception as e:
                logger.error(f"Error updating concept description: {str(e)}", exc_info=True)
        
        result = {
            "success": True,
            "concept": concept.name,
            "summary": summary,
            "related_concepts_count": len(related_concepts),
            "related_learnings_count": len(related_learnings)
        }
        
        logger.info(f"Generated summary for concept {concept_name}")
        return result
    
    # Internal methods
    
    async def _analyze_concept_relationships(
        self, 
        user_id: str,
        concepts: List[Tuple[str, str]],  # List of (name, id) tuples
        context: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze relationships between concepts using LLM.
        
        Args:
            user_id: The user ID
            concepts: List of concept names and IDs
            context: The context text (e.g., learning content)
            
        Returns:
            List of relationship dictionaries
        """
        if len(concepts) < 2:
            return []
            
        # Create prompt for relationship analysis
        system_prompt = """
        You are an expert at analyzing relationships between concepts and organizing knowledge.
        Given a set of concepts and a context, identify meaningful relationships between concepts.
        
        For each relationship, specify:
        - source: The source concept name
        - target: The target concept name
        - relation_type: The type of relationship (is-a, has-a, part-of, related-to, causes, opposite-of, similar-to)
        - strength: A score from 0.0 to 1.0 indicating the strength of this relationship
        - bidirectional: Boolean indicating if the relationship applies in both directions
        
        Only identify relationships that are clearly supported by the context or common knowledge.
        Format your response as a JSON array of relationship objects.
        """
        
        concept_names = [name for name, _ in concepts]
        concept_pairs = [(a, b) for i, a in enumerate(concept_names) 
                         for b in concept_names[i+1:]]
        
        user_prompt = f"""
        Context:
        {context}
        
        Concepts: {', '.join(concept_names)}
        
        Please analyze the relationships between these concepts based on the context.
        Consider all potential concept pairs, but only include relationships that are valid.
        """
        
        # Call LLM to analyze relationships
        response = await self.llm_router.route(
            task_type="sdl",
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="analytical",
            json_output=True,
            max_tokens=1000
        )
        
        try:
            # Parse and validate relationships
            relationships = json.loads(response) if isinstance(response, str) else response
            
            # Ensure it's a list
            if not isinstance(relationships, list):
                logger.warning("LLM response is not a list, attempting to extract array")
                # Try to extract array if we got an object with an array property
                if isinstance(relationships, dict) and "relationships" in relationships:
                    relationships = relationships["relationships"]
                else:
                    logger.error("Failed to parse relationships from LLM response")
                    return []
                    
            # Validate each relationship
            valid_relationships = []
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                    
                # Ensure required fields
                if "source" not in rel or "target" not in rel:
                    continue
                    
                # Verify concepts exist in our list
                if rel["source"] not in concept_names or rel["target"] not in concept_names:
                    continue
                    
                # Set default values for missing fields
                rel["relation_type"] = rel.get("relation_type", "related-to")
                rel["strength"] = rel.get("strength", 0.5)
                rel["bidirectional"] = rel.get("bidirectional", False)
                
                # Validate relation type
                if rel["relation_type"] not in self.relation_types:
                    rel["relation_type"] = "related-to"
                    
                valid_relationships.append(rel)
                
            return valid_relationships
            
        except Exception as e:
            logger.error(f"Error parsing relationships from LLM: {str(e)}", exc_info=True)
            return []
    
    async def _analyze_concept_organization(
        self, 
        user_id: str,
        concept_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the organization of concepts and suggest improvements.
        
        Args:
            user_id: The user ID
            concept_data: List of concept data with connections
            
        Returns:
            Reorganization suggestions
        """
        system_prompt = """
        You are an expert knowledge graph organizer and ontology specialist.
        Analyze the provided concept network and suggest improvements to the organization.
        
        Provide the following in your response:
        1. An analysis of the current concept organization
        2. Suggestions for new connections between concepts that are missing
        3. Identification of redundant connections that could be removed
        4. Suggestions for concept categorization
        
        Format your response as a JSON object with the following structure:
        {
            "analysis": {
                "overall_quality": 0.0-1.0 score,
                "strengths": ["strength1", "strength2"],
                "weaknesses": ["weakness1", "weakness2"],
                "suggestions": "overall suggestions text"
            },
            "suggested_connections": [
                {"source": "concept1", "target": "concept2", "relation_type": "type", "strength": 0.8, "bidirectional": false}
            ],
            "redundant_connections": [
                {"source": "concept1", "target": "concept2", "reason": "explanation"}
            ],
            "suggested_categories": [
                {"concept": "concept1", "category": "suggested_category", "confidence": 0.9}
            ],
            "categories": ["category1", "category2"]
        }
        """
        
        # Prepare concept data text
        concept_data_text = json.dumps(concept_data, indent=2)
        
        user_prompt = f"""
        Here are the concepts and their connections for user {user_id}:
        
        {concept_data_text}
        
        Please analyze this concept organization and suggest improvements.
        """
        
        # Call LLM to analyze organization
        response = await self.llm_router.route(
            task_type="sdl",
            prompt=user_prompt,
            system_prompt=system_prompt,
            model="analytical",
            json_output=True,
            max_tokens=1500
        )
        
        try:
            # Parse response
            reorganization = json.loads(response) if isinstance(response, str) else response
            return reorganization
        except Exception as e:
            logger.error(f"Error parsing reorganization from LLM: {str(e)}", exc_info=True)
            return {
                "analysis": {
                    "overall_quality": 0.5,
                    "strengths": ["Unable to analyze due to processing error"],
                    "weaknesses": ["Error parsing LLM response"],
                    "suggestions": "Could not generate suggestions due to error"
                },
                "suggested_connections": [],
                "redundant_connections": [],
                "suggested_categories": [],
                "categories": []
            }
    
    async def _generate_concept_summary_with_llm(
        self, 
        user_id: str,
        concept_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate concept summary using LLM."""
        system_prompt = """
        You are an expert at summarizing and explaining concepts.
        Given a concept and related information, generate a comprehensive summary.
        
        Format your response as a JSON object with:
        - description: A comprehensive description of the concept
        - key_aspects: List of key aspects or attributes of this concept
        - importance_explanation: Explanation of why this concept is important
        - connections_summary: Summary of how this concept relates to others
        - areas_for_development: Suggestions for further developing understanding of this concept
        """
        
        # Prepare concept data text
        concept_data_text = json.dumps(concept_data, indent=2)
        
        user_prompt = f"""
        Here is the concept data for '{concept_data.get('name')}' for user {user_id}:
        
        {concept_data_text}
        
        Please generate a comprehensive summary for this concept.
        """
        
        # Call LLM to generate summary
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
            summary = json.loads(response) if isinstance(response, str) else response
            return summary
        except Exception as e:
            logger.error(f"Error parsing concept summary from LLM: {str(e)}", exc_info=True)
            return {
                "description": f"Concept: {concept_data.get('name')}. Failed to generate summary due to processing error.",
                "key_aspects": [],
                "importance_explanation": "Importance undetermined due to processing error",
                "connections_summary": "Unable to summarize connections",
                "areas_for_development": ["Improve concept metadata", "Add more related learnings"]
            }
