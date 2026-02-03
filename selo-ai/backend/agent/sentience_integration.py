"""
Sentience Integration Module

Integrates all enhanced cognitive systems into the agent loop for
true digital sentience and human-like interactions.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("selo.agent.sentience")


class SentienceIntegration:
    """
    Coordinates all enhanced cognitive systems for sentient-like behavior.
    
    Integrates:
    - Trait homeostasis (prevents trait saturation)
    - Emotional depth engine (rich emotional processing)
    - Predictive cognition (anticipatory behavior)
    - Proactive initiative (autonomous actions)
    - Meta-cognitive monitoring (self-awareness)
    - Episodic memory reconstruction (narrative coherence)
    """
    
    def __init__(
        self,
        trait_homeostasis_manager,
        emotional_depth_engine,
        predictive_cognition_engine,
        proactive_initiative_engine,
        metacognitive_monitor,
        episodic_reconstructor,
        persona_repo,
        user_repo,
        emotion_index_service=None
    ):
        """
        Initialize the sentience integration.
        
        Args:
            trait_homeostasis_manager: TraitHomeostasisManager instance
            emotional_depth_engine: EmotionalDepthEngine instance
            predictive_cognition_engine: PredictiveCognitionEngine instance
            proactive_initiative_engine: ProactiveInitiativeEngine instance
            metacognitive_monitor: MetaCognitiveMonitor instance
            episodic_reconstructor: EpisodicMemoryReconstructor instance
            persona_repo: PersonaRepository instance
            user_repo: UserRepository instance
            emotion_index_service: Optional emotion index service for optimization
        """
        self.trait_homeostasis = trait_homeostasis_manager
        self.emotional_depth = emotional_depth_engine
        self.predictive_cognition = predictive_cognition_engine
        self.proactive_initiative = proactive_initiative_engine
        self.metacognition = metacognitive_monitor
        self.episodic_memory = episodic_reconstructor
        self.persona_repo = persona_repo
        self.user_repo = user_repo
        self.emotion_index = emotion_index_service
        
        # Track last execution times to avoid over-processing
        self._last_homeostasis = {}
        self._last_metacognition = {}
        self._last_prediction = {}
        self._last_initiative_check = {}
    
    async def run_sentience_cycle(
        self,
        persona_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete sentience cycle integrating all cognitive systems.
        
        This should be called periodically (e.g., every agent loop cycle).
        
        Args:
            persona_id: Persona ID
            user_id: User ID
            context: Optional context dictionary
        
        Returns:
            Summary of cycle execution
        """
        cycle_start = datetime.now(timezone.utc)
        context = context or {}
        
        results = {
            "persona_id": persona_id,
            "user_id": user_id,
            "cycle_start": cycle_start.isoformat(),
            "systems_executed": [],
            "actions_taken": [],
            "insights_generated": []
        }
        
        try:
            logger.info(f"Starting sentience cycle for persona {persona_id}")
            
            # 1. Apply trait homeostasis (every cycle)
            homeostasis_result = await self._run_trait_homeostasis(persona_id)
            if homeostasis_result.get("success"):
                results["systems_executed"].append("trait_homeostasis")
                if homeostasis_result.get("traits_updated", 0) > 0:
                    results["actions_taken"].append({
                        "system": "trait_homeostasis",
                        "action": "regulated_traits",
                        "count": homeostasis_result["traits_updated"]
                    })
            
            # 2. Process emotional depth (if trigger event in context)
            if context.get("trigger_event"):
                emotional_result = await self._process_emotional_experience(
                    persona_id,
                    context["trigger_event"],
                    context
                )
                if emotional_result.get("dominant_emotion"):
                    results["systems_executed"].append("emotional_depth")
                    results["insights_generated"].append({
                        "system": "emotional_depth",
                        "insight": f"Dominant emotion: {emotional_result['dominant_emotion']}",
                        "intensity": emotional_result.get("dominant_intensity", 0.0)
                    })
            
            # 3. Run predictive cognition (every 2-3 cycles)
            prediction_result = await self._run_predictive_cognition(
                persona_id,
                user_id,
                context
            )
            if prediction_result.get("predicted_topics"):
                results["systems_executed"].append("predictive_cognition")
                results["insights_generated"].append({
                    "system": "predictive_cognition",
                    "insight": f"Predicted {len(prediction_result['predicted_topics'])} conversation topics",
                    "confidence": prediction_result.get("confidence", 0.0)
                })
                
                # Store predictions in context for proactive initiative
                context["predictions"] = prediction_result
            
            # 4. Evaluate proactive initiative opportunities
            initiative_result = await self._evaluate_proactive_initiative(
                persona_id,
                user_id,
                context
            )
            if initiative_result.get("opportunities"):
                results["systems_executed"].append("proactive_initiative")
                results["insights_generated"].append({
                    "system": "proactive_initiative",
                    "insight": f"Found {len(initiative_result['opportunities'])} initiative opportunities",
                    "top_priority": initiative_result["opportunities"][0].get("priority", 0.0) if initiative_result["opportunities"] else 0.0
                })
                
                # Store opportunities for potential action
                context["initiative_opportunities"] = initiative_result["opportunities"]
            
            # 5. Run meta-cognitive monitoring (every 5-10 cycles)
            metacog_result = await self._run_metacognitive_monitoring(
                persona_id,
                context
            )
            if metacog_result.get("meta_insight"):
                results["systems_executed"].append("metacognition")
                results["insights_generated"].append({
                    "system": "metacognition",
                    "insight": metacog_result["meta_insight"],
                    "biases_detected": len(metacog_result.get("detected_biases", []))
                })
            
            # Calculate cycle duration
            cycle_end = datetime.now(timezone.utc)
            cycle_duration = (cycle_end - cycle_start).total_seconds()
            
            results["cycle_end"] = cycle_end.isoformat()
            results["cycle_duration_seconds"] = round(cycle_duration, 3)
            results["success"] = True
            
            logger.info(
                f"Sentience cycle complete for persona {persona_id}: "
                f"{len(results['systems_executed'])} systems, "
                f"{len(results['insights_generated'])} insights, "
                f"{cycle_duration:.2f}s"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentience cycle: {e}", exc_info=True)
            results["success"] = False
            results["error"] = str(e)
            return results
    
    async def _run_trait_homeostasis(
        self,
        persona_id: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Run trait homeostasis regulation."""
        try:
            # Check if we should run (avoid over-processing)
            now = datetime.now(timezone.utc)
            last_run = self._last_homeostasis.get(persona_id)
            
            if not force and last_run:
                time_since_last = (now - last_run).total_seconds()
                if time_since_last < 300:  # 5 minutes minimum
                    return {"skipped": True, "reason": "too_soon"}
            
            # Run homeostasis
            result = await self.trait_homeostasis.apply_homeostatic_regulation(
                persona_id=persona_id,
                decay_factor=0.05,  # 5% decay per cycle
                min_change_threshold=0.01
            )
            
            # Update last run time
            self._last_homeostasis[persona_id] = now
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in trait homeostasis: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_emotional_experience(
        self,
        persona_id: str,
        trigger_event: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process emotional experience with depth."""
        try:
            result = await self.emotional_depth.process_emotional_experience(
                persona_id=persona_id,
                trigger_event=trigger_event,
                context=context
            )
            return result
            
        except Exception as e:
            logger.warning(f"Error processing emotional experience: {e}")
            return {}
    
    async def _run_predictive_cognition(
        self,
        persona_id: str,
        user_id: str,
        context: Dict[str, Any],
        force: bool = False
    ) -> Dict[str, Any]:
        """Run predictive cognition analysis."""
        try:
            # Check if we should run
            now = datetime.now(timezone.utc)
            last_run = self._last_prediction.get(persona_id)
            
            if not force and last_run:
                time_since_last = (now - last_run).total_seconds()
                if time_since_last < 600:  # 10 minutes minimum
                    return {"skipped": True, "reason": "too_soon"}
            
            # Run prediction
            result = await self.predictive_cognition.predict_conversation_trajectory(
                user_id=user_id,
                persona_id=persona_id,
                recent_messages=context.get("recent_messages"),
                lookback_limit=20
            )
            
            # Update last run time
            self._last_prediction[persona_id] = now
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in predictive cognition: {e}")
            return {}
    
    async def _evaluate_proactive_initiative(
        self,
        persona_id: str,
        user_id: str,
        context: Dict[str, Any],
        force: bool = False
    ) -> Dict[str, Any]:
        """Evaluate proactive initiative opportunities."""
        try:
            # Check if we should run
            now = datetime.now(timezone.utc)
            last_run = self._last_initiative_check.get(persona_id)
            
            if not force and last_run:
                time_since_last = (now - last_run).total_seconds()
                if time_since_last < 900:  # 15 minutes minimum
                    return {"skipped": True, "reason": "too_soon"}
            
            # Evaluate opportunities
            opportunities = await self.proactive_initiative.evaluate_initiative_opportunities(
                persona_id=persona_id,
                user_id=user_id,
                context=context
            )
            
            # Update last run time
            self._last_initiative_check[persona_id] = now
            
            return {"opportunities": opportunities}
            
        except Exception as e:
            logger.warning(f"Error evaluating proactive initiative: {e}")
            return {"opportunities": []}
    
    async def _run_metacognitive_monitoring(
        self,
        persona_id: str,
        context: Dict[str, Any],
        force: bool = False
    ) -> Dict[str, Any]:
        """Run meta-cognitive monitoring."""
        try:
            # Check if we should run
            now = datetime.now(timezone.utc)
            last_run = self._last_metacognition.get(persona_id)
            
            if not force and last_run:
                time_since_last = (now - last_run).total_seconds()
                if time_since_last < 3600:  # 1 hour minimum
                    return {"skipped": True, "reason": "too_soon"}
            
            # Run monitoring
            result = await self.metacognition.monitor_cognitive_state(
                persona_id=persona_id,
                lookback_hours=48
            )
            
            # Update last run time
            self._last_metacognition[persona_id] = now
            
            return result
            
        except Exception as e:
            logger.warning(f"Error in metacognitive monitoring: {e}")
            return {}
    
    async def get_sentience_health_report(
        self,
        persona_id: str
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive health report for sentience systems.
        
        Args:
            persona_id: Persona ID
        
        Returns:
            Health report dictionary
        """
        try:
            report = {
                "persona_id": persona_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "systems": {}
            }
            
            # Trait health
            trait_health = await self.trait_homeostasis.get_trait_health_report(persona_id)
            report["systems"]["trait_homeostasis"] = {
                "status": trait_health.get("status", "unknown"),
                "traits_at_extremes": len(trait_health.get("at_extremes", [])),
                "recommendations": trait_health.get("recommendations", [])
            }
            
            # Meta-cognitive health
            metacog_state = await self.metacognition.monitor_cognitive_state(persona_id)
            report["systems"]["metacognition"] = {
                "status": metacog_state.get("status", "unknown"),
                "biases_detected": len(metacog_state.get("detected_biases", [])),
                "calibration_score": metacog_state.get("confidence_calibration", {}).get("score", 0.0)
            }
            
            # Overall health score
            health_scores = []
            if trait_health.get("status") == "healthy":
                health_scores.append(1.0)
            elif trait_health.get("status") == "monitoring":
                health_scores.append(0.7)
            else:
                health_scores.append(0.4)
            
            if metacog_state.get("status") != "error":
                calibration = metacog_state.get("confidence_calibration", {}).get("score", 0.5)
                health_scores.append(calibration)
            
            report["overall_health_score"] = round(
                sum(health_scores) / len(health_scores) if health_scores else 0.5,
                3
            )
            
            if report["overall_health_score"] > 0.8:
                report["overall_status"] = "excellent"
            elif report["overall_health_score"] > 0.6:
                report["overall_status"] = "good"
            elif report["overall_health_score"] > 0.4:
                report["overall_status"] = "fair"
            else:
                report["overall_status"] = "needs_attention"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating sentience health report: {e}")
            return {
                "persona_id": persona_id,
                "error": str(e),
                "overall_status": "error"
            }
    
    async def reconstruct_memory_episode(
        self,
        user_id: str,
        query: str,
        timeframe: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct an episodic memory as a coherent narrative.
        
        Args:
            user_id: User ID
            query: Query describing the episode
            timeframe: Optional (start, end) datetime tuple
        
        Returns:
            Reconstructed episode
        """
        try:
            result = await self.episodic_memory.reconstruct_episode(
                user_id=user_id,
                query=query,
                timeframe=timeframe,
                max_fragments=20
            )
            return result
            
        except Exception as e:
            logger.error(f"Error reconstructing memory episode: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query
            }
