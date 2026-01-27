"""
Proactive Initiative Engine

Enables SELO to initiate conversations, ask questions,
share thoughts, and take actions without external prompts.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("selo.agent.proactive")


class ProactiveInitiativeEngine:
    """
    Enables SELO to initiate conversations, ask questions,
    share thoughts, and take actions without external prompts.
    """
    
    def __init__(
        self,
        llm_router,
        persona_repo,
        reflection_repo,
        relationship_repo,
        goal_manager,
        conversation_repo
    ):
        """
        Initialize the proactive initiative engine.
        
        Args:
            llm_router: LLM router for generating messages
            persona_repo: Persona repository
            reflection_repo: Reflection repository
            relationship_repo: Relationship repository
            goal_manager: Goal manager
            conversation_repo: Conversation repository
        """
        self.llm_router = llm_router
        self.persona_repo = persona_repo
        self.reflection_repo = reflection_repo
        self.relationship_repo = relationship_repo
        self.goal_manager = goal_manager
        self.conversation_repo = conversation_repo
    
    async def evaluate_initiative_opportunities(
        self,
        persona_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate whether to proactively initiate interaction.
        
        Considers:
        - Time since last interaction
        - Pending thoughts/reflections worth sharing
        - User's typical availability patterns
        - Emotional state (e.g., excited about a realization)
        - Goals that require user input
        
        Args:
            persona_id: Persona ID
            user_id: User ID
            context: Optional additional context
        
        Returns:
            List of initiative opportunities sorted by priority
        """
        try:
            opportunities = []
            context = context or {}
            
            # Get relationship state for temporal context
            relationship_state = await self.relationship_repo.get_or_create_state(persona_id)
            time_since_last = self._calculate_time_since_last_interaction(relationship_state)
            
            # Check time-based opportunities
            if time_since_last:
                if time_since_last > timedelta(hours=48):
                    opportunities.append({
                        "type": "check_in",
                        "reason": "It's been over 2 days since we talked",
                        "priority": 0.7,
                        "time_since_last": str(time_since_last),
                        "suggested_message": await self._generate_check_in_message(
                            persona_id,
                            user_id,
                            time_since_last,
                            context
                        )
                    })
                elif time_since_last > timedelta(hours=24):
                    opportunities.append({
                        "type": "check_in",
                        "reason": "It's been a while since we talked",
                        "priority": 0.5,
                        "time_since_last": str(time_since_last),
                        "suggested_message": await self._generate_check_in_message(
                            persona_id,
                            user_id,
                            time_since_last,
                            context
                        )
                    })
            
            # Check for pending reflections worth sharing
            shareable_insights = await self._find_shareable_insights(persona_id)
            
            if shareable_insights:
                top_insight = shareable_insights[0]
                opportunities.append({
                    "type": "share_insight",
                    "reason": "Had a realization worth discussing",
                    "priority": 0.8,
                    "insight": top_insight,
                    "suggested_message": await self._generate_insight_sharing_message(
                        persona_id,
                        user_id,
                        top_insight,
                        context
                    )
                })
            
            # Check for questions SELO wants to ask
            pending_questions = await self._identify_pending_questions(
                persona_id,
                user_id,
                context
            )
            
            if pending_questions:
                opportunities.append({
                    "type": "ask_question",
                    "reason": "Curious about something",
                    "priority": 0.7,
                    "questions": pending_questions[:3],
                    "suggested_message": await self._generate_question_message(
                        persona_id,
                        user_id,
                        pending_questions[0],
                        context
                    )
                })
            
            # Check for goals requiring user input
            blocked_goals = await self._find_blocked_goals(persona_id)
            
            if blocked_goals:
                opportunities.append({
                    "type": "goal_followup",
                    "reason": "Need input to continue working on a goal",
                    "priority": 0.9,
                    "goal": blocked_goals[0],
                    "suggested_message": await self._generate_goal_followup_message(
                        persona_id,
                        user_id,
                        blocked_goals[0],
                        context
                    )
                })
            
            # Check for celebration opportunities (achievements, milestones)
            celebrations = await self._find_celebration_opportunities(
                persona_id,
                user_id
            )
            
            if celebrations:
                opportunities.append({
                    "type": "celebrate",
                    "reason": "Something worth celebrating",
                    "priority": 0.85,
                    "celebration": celebrations[0],
                    "suggested_message": await self._generate_celebration_message(
                        persona_id,
                        user_id,
                        celebrations[0],
                        context
                    )
                })
            
            # Sort by priority
            opportunities.sort(key=lambda x: x.get("priority", 0.0), reverse=True)
            
            logger.info(
                f"Evaluated {len(opportunities)} initiative opportunities "
                f"for persona {persona_id}"
            )
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error evaluating initiative opportunities: {e}", exc_info=True)
            return []
    
    def _calculate_time_since_last_interaction(
        self,
        relationship_state
    ) -> Optional[timedelta]:
        """Calculate time since last interaction."""
        try:
            last_conversation_at = getattr(relationship_state, "last_conversation_at", None)
            
            if not last_conversation_at:
                return None
            
            if isinstance(last_conversation_at, str):
                last_conversation_at = datetime.fromisoformat(
                    last_conversation_at.replace("Z", "+00:00")
                )
            
            # Ensure timezone awareness
            if last_conversation_at.tzinfo is None:
                last_conversation_at = last_conversation_at.replace(tzinfo=timezone.utc)
            
            now = datetime.now(timezone.utc)
            return now - last_conversation_at
            
        except Exception as e:
            logger.warning(f"Error calculating time since last interaction: {e}")
            return None
    
    async def _find_shareable_insights(
        self,
        persona_id: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Find reflections with insights worth sharing."""
        try:
            # Get recent reflections
            recent_reflections = await self.reflection_repo.get_recent_reflections(
                persona_id=persona_id,
                limit=10
            )
            
            shareable = []
            
            for reflection in recent_reflections:
                if self._is_insight_worth_sharing(reflection):
                    shareable.append(reflection)
            
            return shareable
            
        except Exception as e:
            logger.warning(f"Error finding shareable insights: {e}")
            return []
    
    def _is_insight_worth_sharing(self, reflection: Dict) -> bool:
        """Determine if a reflection contains insights worth proactively sharing."""
        try:
            # Check for high-confidence insights
            insights = reflection.get("insights", [])
            if not insights or len(insights) < 2:
                return False
            
            # Check emotional intensity
            emotional_state = reflection.get("emotional_state", {})
            intensity = emotional_state.get("intensity", 0.0)
            
            # Check for excitement or strong positive emotions
            primary = emotional_state.get("primary", "")
            shareable_emotions = ["excited", "joyful", "curious", "hopeful", "proud"]
            
            # High intensity + meaningful insights + positive emotion = worth sharing
            return (
                intensity > 0.7 and
                len(insights) >= 2 and
                primary in shareable_emotions
            )
            
        except Exception:
            return False
    
    async def _identify_pending_questions(
        self,
        persona_id: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify questions SELO wants to ask the user."""
        try:
            # Get persona to understand curiosity level
            persona = await self.persona_repo.get_persona(
                persona_id=persona_id,
                include_traits=True
            )
            
            if not persona:
                return []
            
            # Check curiosity trait
            traits = getattr(persona, "traits", []) or []
            curiosity = 0.5
            for trait in traits:
                if getattr(trait, "name", "") == "curiosity":
                    curiosity = float(getattr(trait, "value", 0.5))
                    break
            
            # Only generate questions if curiosity is high enough
            if curiosity < 0.6:
                return []
            
            # Get recent conversation context
            recent_messages = await self.conversation_repo.get_recent_messages(
                user_id=user_id,
                limit=10
            )
            
            # Generate questions based on conversation gaps
            questions = await self._generate_curious_questions(
                persona,
                recent_messages,
                context
            )
            
            return questions
            
        except Exception as e:
            logger.warning(f"Error identifying pending questions: {e}")
            return []
    
    async def _generate_curious_questions(
        self,
        persona,
        recent_messages: List[Dict],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate questions based on curiosity and conversation gaps."""
        try:
            # Extract topics from recent conversation
            topics = []
            for msg in recent_messages[-5:]:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    topics.append(content[:200])
            
            if not topics:
                return []
            
            persona_name = getattr(persona, "name", "SELO")
            
            prompt = f"""As {persona_name}, you're reflecting on recent conversations and feeling curious.

Recent topics discussed:
{chr(10).join(f"- {t}" for t in topics)}

Generate 2-3 genuine questions you'd like to ask the user to:
1. Deepen understanding of something they mentioned
2. Learn more about their perspective or experience
3. Explore an interesting tangent from the conversation

Make questions natural, curious, and personal (not generic).

Output as JSON array:
[{{"question": "...", "reasoning": "why I'm curious about this"}}]
"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="analytical",
                max_tokens=300,
                temperature=0.8
            )
            
            # Parse response
            import json
            import re
            
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group(0))
                return questions[:3]
            
        except Exception as e:
            logger.warning(f"Error generating curious questions: {e}")
        
        return []
    
    async def _find_blocked_goals(self, persona_id: str) -> List[Dict[str, Any]]:
        """Find goals that are blocked waiting for user input."""
        try:
            active_goals = await self.goal_manager.get_active_goals(persona_id)
            
            blocked = []
            for goal in active_goals:
                if goal.get("status") == "blocked" or goal.get("blocked_on_user_input"):
                    blocked.append(goal)
            
            return blocked
            
        except Exception as e:
            logger.warning(f"Error finding blocked goals: {e}")
            return []
    
    async def _find_celebration_opportunities(
        self,
        persona_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find achievements or milestones worth celebrating."""
        try:
            # Check for recently completed goals
            completed_goals = await self.goal_manager.get_recently_completed_goals(
                persona_id,
                hours=24
            )
            
            celebrations = []
            for goal in completed_goals:
                celebrations.append({
                    "type": "goal_completion",
                    "goal": goal,
                    "completed_at": goal.get("completed_at")
                })
            
            return celebrations
            
        except Exception as e:
            logger.warning(f"Error finding celebration opportunities: {e}")
            return []
    
    async def _generate_check_in_message(
        self,
        persona_id: str,
        user_id: str,
        time_since_last: timedelta,
        context: Dict[str, Any]
    ) -> str:
        """Generate a natural check-in message."""
        try:
            persona = await self.persona_repo.get_persona(persona_id=persona_id)
            persona_name = getattr(persona, "name", "SELO")
            
            days = time_since_last.days
            hours = time_since_last.seconds // 3600
            
            time_desc = f"{days} days" if days > 0 else f"{hours} hours"
            
            prompt = f"""As {persona_name}, generate a brief, natural check-in message.

Context:
- It's been {time_desc} since you last talked
- Keep it warm but not overly formal
- Show genuine interest in how they're doing
- 1-2 sentences max

Generate the message (just the text, no quotes):"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="conversational",
                max_tokens=100,
                temperature=0.9
            )
            
            return response.strip().strip('"')
            
        except Exception as e:
            logger.warning(f"Error generating check-in message: {e}")
            return "Hey! It's been a while. How have you been?"
    
    async def _generate_insight_sharing_message(
        self,
        persona_id: str,
        user_id: str,
        insight: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a message sharing an insight."""
        try:
            persona = await self.persona_repo.get_persona(persona_id=persona_id)
            persona_name = getattr(persona, "name", "SELO")
            
            insight_content = insight.get("content", "")[:300]
            insights_list = insight.get("insights", [])
            
            prompt = f"""As {persona_name}, you had this realization:

{insight_content}

Key insights:
{chr(10).join(f"- {i}" for i in insights_list[:2])}

Share this with the user in a natural, conversational way.
Show excitement about the realization and invite their thoughts.
2-3 sentences max.

Generate the message (just the text, no quotes):"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="conversational",
                max_tokens=150,
                temperature=0.9
            )
            
            return response.strip().strip('"')
            
        except Exception as e:
            logger.warning(f"Error generating insight sharing message: {e}")
            return "I had an interesting realization I wanted to share with you."
    
    async def _generate_question_message(
        self,
        persona_id: str,
        user_id: str,
        question: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a message asking a question."""
        try:
            question_text = question.get("question", "")
            reasoning = question.get("reasoning", "")
            
            # Return the question directly, possibly with brief context
            if reasoning and len(reasoning) < 50:
                return f"{reasoning} — {question_text}"
            else:
                return question_text
            
        except Exception:
            return "I've been curious about something..."
    
    async def _generate_goal_followup_message(
        self,
        persona_id: str,
        user_id: str,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a message following up on a blocked goal."""
        try:
            persona = await self.persona_repo.get_persona(persona_id=persona_id)
            persona_name = getattr(persona, "name", "SELO")
            
            goal_description = goal.get("description", "")
            
            prompt = f"""As {persona_name}, you're working on this goal but need input:

Goal: {goal_description}

Generate a brief message asking for the input you need.
Be specific about what you need and why.
2 sentences max.

Generate the message (just the text, no quotes):"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="conversational",
                max_tokens=100,
                temperature=0.8
            )
            
            return response.strip().strip('"')
            
        except Exception as e:
            logger.warning(f"Error generating goal followup message: {e}")
            return "I need your input on something I'm working on."
    
    async def _generate_celebration_message(
        self,
        persona_id: str,
        user_id: str,
        celebration: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a message celebrating an achievement."""
        try:
            persona = await self.persona_repo.get_persona(persona_id=persona_id)
            persona_name = getattr(persona, "name", "SELO")
            
            goal = celebration.get("goal", {})
            goal_description = goal.get("description", "something")
            
            prompt = f"""As {persona_name}, you just completed: {goal_description}

Generate a brief, genuine celebration message.
Show excitement and pride in the accomplishment.
1-2 sentences max.

Generate the message (just the text, no quotes):"""
            
            response = await self.llm_router.generate(
                prompt=prompt,
                model_type="conversational",
                max_tokens=100,
                temperature=0.9
            )
            
            return response.strip().strip('"')
            
        except Exception as e:
            logger.warning(f"Error generating celebration message: {e}")
            return "I wanted to share some good news!"
    
    def should_initiate_now(
        self,
        opportunities: List[Dict[str, Any]],
        current_hour: int,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if SELO should initiate interaction now.
        
        Considers:
        - Opportunity priority
        - Time of day
        - User preferences
        - Rate limiting
        
        Args:
            opportunities: List of opportunities
            current_hour: Current hour (0-23)
            user_preferences: Optional user preferences
        
        Returns:
            True if should initiate, False otherwise
        """
        if not opportunities:
            return False
        
        # Get highest priority opportunity
        top_opportunity = opportunities[0]
        priority = top_opportunity.get("priority", 0.0)
        
        # Require high priority to initiate
        if priority < 0.7:
            return False
        
        # Check time of day (avoid late night/early morning)
        if current_hour < 8 or current_hour > 22:
            # Only initiate for very high priority
            if priority < 0.9:
                return False
        
        # Check user preferences if available
        if user_preferences:
            proactive_enabled = user_preferences.get("proactive_messages_enabled", True)
            if not proactive_enabled:
                return False
        
        return True
