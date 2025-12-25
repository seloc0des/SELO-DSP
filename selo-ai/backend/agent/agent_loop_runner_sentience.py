"""
Sentience initialization methods for AgentLoopRunner.

This module contains the initialization logic for sentience systems
to keep the main agent_loop_runner.py file clean.
"""

import logging
from typing import Optional, Any

logger = logging.getLogger("selo.agent.loop_runner.sentience")


def initialize_sentience_integration(agent_loop_runner) -> Optional[Any]:
    """
    Initialize all sentience systems for the agent loop.
    
    Args:
        agent_loop_runner: AgentLoopRunner instance
    
    Returns:
        SentienceIntegration instance or None if initialization fails
    """
    try:
        from ..persona.trait_homeostasis import TraitHomeostasisManager
        from ..agent.emotional_depth_engine import EmotionalDepthEngine
        from ..agent.predictive_cognition import PredictiveCognitionEngine
        from ..agent.proactive_initiative import ProactiveInitiativeEngine
        from ..agent.metacognition import MetaCognitiveMonitor
        from ..memory.episodic_reconstructor import EpisodicMemoryReconstructor
        from ..agent.sentience_integration import SentienceIntegration
        
        # Initialize trait homeostasis
        trait_homeostasis = TraitHomeostasisManager(
            persona_repo=agent_loop_runner._persona_repo
        )
        logger.debug("Initialized TraitHomeostasisManager")
        
        # Initialize emotional depth engine
        emotional_depth = EmotionalDepthEngine(
            persona_repo=agent_loop_runner._persona_repo,
            affective_state_repo=agent_loop_runner._affective_state_manager._state_repo,
            memory_repo=None  # Optional
        )
        logger.debug("Initialized EmotionalDepthEngine")
        
        # Initialize predictive cognition (requires LLM router)
        predictive_cognition = None
        if agent_loop_runner._llm_router and agent_loop_runner._conversation_repo:
            predictive_cognition = PredictiveCognitionEngine(
                llm_router=agent_loop_runner._llm_router,
                conversation_repo=agent_loop_runner._conversation_repo,
                memory_repo=None,  # Optional
                persona_repo=agent_loop_runner._persona_repo
            )
            logger.debug("Initialized PredictiveCognitionEngine")
        else:
            logger.warning("Skipping PredictiveCognitionEngine - missing dependencies")
        
        # Initialize proactive initiative (requires multiple repos)
        proactive_initiative = None
        if (agent_loop_runner._llm_router and 
            agent_loop_runner._reflection_repo and 
            agent_loop_runner._relationship_repo and
            agent_loop_runner._conversation_repo):
            proactive_initiative = ProactiveInitiativeEngine(
                llm_router=agent_loop_runner._llm_router,
                persona_repo=agent_loop_runner._persona_repo,
                reflection_repo=agent_loop_runner._reflection_repo,
                relationship_repo=agent_loop_runner._relationship_repo,
                goal_manager=agent_loop_runner._goal_manager,
                conversation_repo=agent_loop_runner._conversation_repo
            )
            logger.debug("Initialized ProactiveInitiativeEngine")
        else:
            logger.warning("Skipping ProactiveInitiativeEngine - missing dependencies")
        
        # Initialize meta-cognitive monitor
        metacognition = None
        if agent_loop_runner._llm_router and agent_loop_runner._reflection_repo:
            metacognition = MetaCognitiveMonitor(
                llm_router=agent_loop_runner._llm_router,
                reflection_repo=agent_loop_runner._reflection_repo,
                persona_repo=agent_loop_runner._persona_repo,
                learning_repo=None  # Optional
            )
            logger.debug("Initialized MetaCognitiveMonitor")
        else:
            logger.warning("Skipping MetaCognitiveMonitor - missing dependencies")
        
        # Initialize episodic memory reconstructor
        episodic_reconstructor = None
        if agent_loop_runner._llm_router and agent_loop_runner._conversation_repo:
            episodic_reconstructor = EpisodicMemoryReconstructor(
                llm_router=agent_loop_runner._llm_router,
                memory_repo=None,  # Optional
                conversation_repo=agent_loop_runner._conversation_repo
            )
            logger.debug("Initialized EpisodicMemoryReconstructor")
        else:
            logger.warning("Skipping EpisodicMemoryReconstructor - missing dependencies")
        
        # Create sentience integration
        sentience_integration = SentienceIntegration(
            trait_homeostasis_manager=trait_homeostasis,
            emotional_depth_engine=emotional_depth,
            predictive_cognition_engine=predictive_cognition,
            proactive_initiative_engine=proactive_initiative,
            metacognitive_monitor=metacognition,
            episodic_reconstructor=episodic_reconstructor,
            persona_repo=agent_loop_runner._persona_repo,
            user_repo=agent_loop_runner._user_repo
        )
        
        logger.info("Sentience integration initialized with all available systems")
        return sentience_integration
        
    except ImportError as e:
        logger.error(f"Failed to import sentience modules: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize sentience integration: {e}", exc_info=True)
        return None


async def run_sentience_cycle(agent_loop_runner, persona, user, context: dict) -> dict:
    """
    Run a sentience cycle during agent loop execution.
    
    Args:
        agent_loop_runner: AgentLoopRunner instance
        persona: Persona object
        user: User object
        context: Context dictionary
    
    Returns:
        Sentience cycle results
    """
    if not agent_loop_runner._sentience_integration:
        return {"skipped": True, "reason": "sentience_integration_not_initialized"}
    
    try:
        sentience_result = await agent_loop_runner._sentience_integration.run_sentience_cycle(
            persona_id=persona.id,
            user_id=str(user.id),
            context=context
        )
        
        return sentience_result
        
    except Exception as e:
        logger.error(f"Error in sentience cycle: {e}", exc_info=True)
        return {"error": str(e), "skipped": True}
