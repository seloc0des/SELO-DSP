"""
Predictive Cognition Engine

Analyzes conversation patterns to predict user needs and
proactively prepare responses or gather information.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import Counter
import re

logger = logging.getLogger("selo.agent.predictive")


class PredictiveCognitionEngine:
    """
    Analyzes conversation patterns to predict user needs and
    proactively prepare responses or gather information.
    """
    
    def __init__(
        self,
        llm_router,
        conversation_repo,
        memory_repo,
        persona_repo
    ):
        """
        Initialize the predictive cognition engine.
        
        Args:
            llm_router: LLM router for predictions
            conversation_repo: Conversation repository
            memory_repo: Memory repository
            persona_repo: Persona repository
        """
        self.llm_router = llm_router
        self.conversation_repo = conversation_repo
        self.memory_repo = memory_repo
        self.persona_repo = persona_repo
    
    async def predict_conversation_trajectory(
        self,
        user_id: str,
        persona_id: str,
        recent_messages: Optional[List[Dict]] = None,
        lookback_limit: int = 20
    ) -> Dict[str, Any]:
        """
        Predict likely conversation directions and prepare accordingly.
        
        Args:
            user_id: User ID
            persona_id: Persona ID
            recent_messages: Optional recent messages (if not provided, will fetch)
            lookback_limit: Number of recent messages to analyze
        
        Returns:
            Dictionary with predictions and suggested actions
        """
        try:
            # Get recent messages if not provided
            if not recent_messages:
                recent_messages = await self._fetch_recent_messages(
                    user_id,
                    limit=lookback_limit
                )
            
            if not recent_messages or len(recent_messages) < 3:
                return {
                    "predicted_topics": [],
                    "proactive_actions": [],
                    "information_gaps": [],
                    "confidence": 0.0,
                    "reason": "insufficient_conversation_history"
                }
            
            # Analyze conversation patterns
            patterns = await self._analyze_conversation_patterns(recent_messages)
            
            # Extract topic evolution
            topic_sequence = self._extract_topic_sequence(recent_messages)
            
            # Get persona context
            persona_context = await self._get_persona_context(persona_id, user_id)
            
            # Predict next topics using pattern matching + LLM
            predicted_topics = await self._predict_next_topics(
                topic_sequence,
                patterns,
                persona_context,
                recent_messages
            )
            
            # Identify information gaps
            info_gaps = self._identify_knowledge_gaps(
                predicted_topics,
                persona_context
            )
            
            # Suggest proactive actions
            proactive_actions = self._generate_proactive_actions(
                predicted_topics,
                info_gaps,
                patterns
            )
            
            # Calculate overall prediction confidence
            confidence = self._calculate_prediction_confidence(
                patterns,
                len(recent_messages),
                predicted_topics
            )
            
            result = {
                "predicted_topics": predicted_topics[:5],  # Top 5
                "proactive_actions": proactive_actions[:3],  # Top 3
                "information_gaps": info_gaps,
                "conversation_patterns": patterns,
                "confidence": round(confidence, 3),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(
                f"Predicted conversation trajectory for user {user_id}: "
                f"{len(predicted_topics)} topics, confidence={confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting conversation trajectory: {e}", exc_info=True)
            return {
                "predicted_topics": [],
                "proactive_actions": [],
                "information_gaps": [],
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _fetch_recent_messages(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """Fetch recent conversation messages."""
        try:
            messages = await self.conversation_repo.get_recent_messages(
                user_id=user_id,
                limit=limit
            )
            return messages or []
        except Exception as e:
            logger.warning(f"Error fetching recent messages: {e}")
            return []
    
    async def _analyze_conversation_patterns(
        self,
        messages: List[Dict]
    ) -> Dict[str, Any]:
        """
        Analyze conversation patterns.
        
        Returns patterns like:
        - Message frequency
        - Topic switching frequency
        - Question patterns
        - Emotional patterns
        """
        if not messages:
            return {}
        
        patterns = {
            "total_messages": len(messages),
            "user_messages": 0,
            "assistant_messages": 0,
            "avg_message_length": 0,
            "question_count": 0,
            "topics_identified": [],
            "emotional_tone": "neutral",
            "time_span_hours": 0,
            "avg_response_time_minutes": 0
        }
        
        try:
            total_length = 0
            question_count = 0
            timestamps = []
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp")
                
                if role == "user":
                    patterns["user_messages"] += 1
                elif role == "assistant":
                    patterns["assistant_messages"] += 1
                
                total_length += len(content)
                
                # Count questions
                if "?" in content:
                    question_count += content.count("?")
                
                if timestamp:
                    timestamps.append(timestamp)
            
            patterns["avg_message_length"] = (
                total_length // len(messages) if messages else 0
            )
            patterns["question_count"] = question_count
            
            # Calculate time span
            if len(timestamps) >= 2:
                try:
                    first = timestamps[0]
                    last = timestamps[-1]
                    if isinstance(first, str):
                        first = datetime.fromisoformat(first.replace("Z", "+00:00"))
                    if isinstance(last, str):
                        last = datetime.fromisoformat(last.replace("Z", "+00:00"))
                    
                    time_diff = last - first
                    patterns["time_span_hours"] = round(time_diff.total_seconds() / 3600, 2)
                except Exception:
                    pass
            
            # Extract topics using keyword extraction
            patterns["topics_identified"] = self._extract_topics_from_messages(messages)
            
        except Exception as e:
            logger.warning(f"Error analyzing conversation patterns: {e}")
        
        return patterns
    
    def _extract_topics_from_messages(self, messages: List[Dict]) -> List[str]:
        """Extract topics from messages using simple keyword extraction."""
        # Combine all message content
        all_text = " ".join([
            msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        ]).lower()
        
        # Common topic keywords
        topic_keywords = {
            "code": ["code", "programming", "function", "class", "debug", "error"],
            "data": ["data", "database", "query", "table", "sql"],
            "design": ["design", "ui", "ux", "interface", "layout"],
            "planning": ["plan", "schedule", "timeline", "roadmap", "milestone"],
            "learning": ["learn", "understand", "explain", "teach", "tutorial"],
            "problem": ["problem", "issue", "bug", "fix", "solve"],
            "project": ["project", "build", "create", "develop", "implement"],
            "personal": ["feel", "think", "believe", "opinion", "experience"],
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics[:5]  # Top 5 topics
    
    def _extract_topic_sequence(self, messages: List[Dict]) -> List[str]:
        """Extract sequence of topics from conversation."""
        # Simplified topic extraction - in production would use more sophisticated NLP
        topics = []
        
        for msg in messages:
            if msg.get("role") != "user":
                continue
            
            content = msg.get("content", "").lower()
            
            # Simple keyword-based topic detection
            if any(word in content for word in ["code", "programming", "function"]):
                topics.append("coding")
            elif any(word in content for word in ["data", "database", "query"]):
                topics.append("data")
            elif any(word in content for word in ["design", "ui", "interface"]):
                topics.append("design")
            elif any(word in content for word in ["plan", "schedule", "timeline"]):
                topics.append("planning")
            elif any(word in content for word in ["learn", "understand", "explain"]):
                topics.append("learning")
            elif any(word in content for word in ["feel", "think", "opinion"]):
                topics.append("personal")
            else:
                topics.append("general")
        
        return topics
    
    async def _get_persona_context(
        self,
        persona_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Get persona context for predictions."""
        try:
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            if not persona:
                return {}
            
            # Get user interests from memories
            user_interests = []
            try:
                memories = await self.memory_repo.get_memories(
                    user_id=user_id,
                    importance_threshold=7,
                    limit=10
                )
                # Extract interests from high-importance memories
                # Simplified - would use more sophisticated extraction
                for memory in memories:
                    content = memory.get("content", "").lower()
                    if "interest" in content or "like" in content:
                        user_interests.append(content[:100])
            except Exception:
                pass
            
            return {
                "persona_name": getattr(persona, "name", ""),
                "expertise": getattr(persona, "expertise", {}),
                "values": getattr(persona, "values", {}),
                "user_interests": user_interests[:5]
            }
            
        except Exception as e:
            logger.warning(f"Error getting persona context: {e}")
            return {}
    
    async def _predict_next_topics(
        self,
        topic_sequence: List[str],
        patterns: Dict[str, Any],
        persona_context: Dict[str, Any],
        recent_messages: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Use LLM to predict conversation trajectory."""
        try:
            # Build context for LLM
            recent_topics = topic_sequence[-5:] if len(topic_sequence) > 5 else topic_sequence
            topic_str = " → ".join(recent_topics) if recent_topics else "none"
            
            # Get last few user messages for context
            last_user_messages = [
                msg.get("content", "")[:200]
                for msg in recent_messages[-3:]
                if msg.get("role") == "user"
            ]
            
            prompt = f"""Analyze this conversation trajectory and predict the next likely topics.

Recent topic sequence: {topic_str}

Last user messages:
{chr(10).join(f"- {msg}" for msg in last_user_messages)}

Conversation patterns:
- Total messages: {patterns.get('total_messages', 0)}
- Questions asked: {patterns.get('question_count', 0)}
- Topics discussed: {', '.join(patterns.get('topics_identified', []))}

User interests: {', '.join(persona_context.get('user_interests', [])[:3])}

Predict the 3 most likely next topics or questions the user might raise.
For each prediction, provide:
1. Topic name (2-4 words)
2. Confidence (0.0-1.0)
3. Brief reasoning
4. Whether it requires web search
5. Whether it requires memory recall

Output as JSON array:
[{{"topic": "...", "confidence": 0.0, "reasoning": "...", "requires_web_search": false, "requires_memory_recall": false}}]
"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="analytical",
                max_tokens=400,
                temperature=0.7
            )
            
            # Parse JSON response
            predictions = self._parse_topic_predictions(response)
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Error predicting next topics: {e}")
            return []
    
    def _parse_topic_predictions(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into topic predictions."""
        try:
            import json
            
            # Try to extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                predictions = json.loads(json_match.group(0))
                
                # Validate and normalize
                validated = []
                for pred in predictions:
                    if isinstance(pred, dict) and "topic" in pred:
                        validated.append({
                            "topic": pred.get("topic", ""),
                            "confidence": float(pred.get("confidence", 0.5)),
                            "reasoning": pred.get("reasoning", ""),
                            "requires_web_search": bool(pred.get("requires_web_search", False)),
                            "requires_memory_recall": bool(pred.get("requires_memory_recall", False)),
                            "search_query": pred.get("search_query", ""),
                            "memory_query": pred.get("memory_query", "")
                        })
                
                return validated
            
        except Exception as e:
            logger.warning(f"Error parsing topic predictions: {e}")
        
        return []
    
    def _identify_knowledge_gaps(
        self,
        predicted_topics: List[Dict],
        persona_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify knowledge gaps based on predictions."""
        gaps = []
        
        expertise = persona_context.get("expertise", {})
        known_domains = expertise.get("domains", []) if isinstance(expertise, dict) else []
        
        for topic_pred in predicted_topics:
            topic = topic_pred.get("topic", "").lower()
            confidence = topic_pred.get("confidence", 0.0)
            
            # Check if topic is outside known domains
            is_known = any(
                domain.lower() in topic or topic in domain.lower()
                for domain in known_domains
            )
            
            if not is_known and confidence > 0.5:
                gaps.append({
                    "topic": topic_pred.get("topic"),
                    "confidence": confidence,
                    "reason": "Outside known expertise domains",
                    "suggested_action": "research" if topic_pred.get("requires_web_search") else "recall"
                })
        
        return gaps
    
    def _generate_proactive_actions(
        self,
        predicted_topics: List[Dict],
        info_gaps: List[Dict],
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate suggested proactive actions."""
        actions = []
        
        for topic_pred in predicted_topics[:3]:  # Top 3 predictions
            confidence = topic_pred.get("confidence", 0.0)
            
            # Only suggest actions for high-confidence predictions
            if confidence < 0.6:
                continue
            
            topic = topic_pred.get("topic", "")
            
            # Web search action
            if topic_pred.get("requires_web_search"):
                actions.append({
                    "type": "web_search",
                    "query": topic_pred.get("search_query") or topic,
                    "reason": f"Anticipating question about {topic}",
                    "priority": confidence,
                    "topic": topic
                })
            
            # Memory retrieval action
            if topic_pred.get("requires_memory_recall"):
                actions.append({
                    "type": "memory_retrieval",
                    "query": topic_pred.get("memory_query") or topic,
                    "reason": f"Preparing context for {topic}",
                    "priority": confidence,
                    "topic": topic
                })
        
        # Sort by priority
        actions.sort(key=lambda x: x.get("priority", 0.0), reverse=True)
        
        return actions
    
    def _calculate_prediction_confidence(
        self,
        patterns: Dict[str, Any],
        message_count: int,
        predictions: List[Dict]
    ) -> float:
        """Calculate overall prediction confidence."""
        # Base confidence on conversation history depth
        history_confidence = min(1.0, message_count / 20.0)
        
        # Factor in pattern clarity
        pattern_confidence = 0.5
        if patterns.get("topics_identified"):
            pattern_confidence = min(1.0, len(patterns["topics_identified"]) / 5.0)
        
        # Factor in prediction quality
        prediction_confidence = 0.5
        if predictions:
            avg_pred_confidence = sum(
                p.get("confidence", 0.0) for p in predictions
            ) / len(predictions)
            prediction_confidence = avg_pred_confidence
        
        # Weighted average
        overall = (
            history_confidence * 0.3 +
            pattern_confidence * 0.3 +
            prediction_confidence * 0.4
        )
        
        return overall
