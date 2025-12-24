"""
Database initialization script for SELO AI Digital Sentience Platform.
Creates necessary tables for reflection system.
"""

import asyncio
import logging
import argparse
from sqlalchemy.sql import text

# Centralize DB config by reusing the application's engine and session factory
from .session import engine, AsyncSessionLocal  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def create_tables():
    """
    Create all the necessary tables for the reflection system if they don't exist.
    """
    logger.info("Creating database tables for SELO AI reflection system...")
    
    async with engine.begin() as conn:
        # Create minimal users table required by repositories and services
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL DEFAULT 'user',
                display_name VARCHAR(200),
                installation_id TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                last_active TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
                api_key VARCHAR(255),
                preferences TEXT,
                is_active BOOLEAN DEFAULT TRUE NOT NULL
            );
        """))
        # Helpful indices
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_users_installation_id ON users (installation_id);
        """))

        # Conversations core tables
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(500),
                started_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                last_message_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                is_active BOOLEAN DEFAULT TRUE NOT NULL,
                message_count INTEGER DEFAULT 0 NOT NULL,
                summary TEXT,
                topics JSONB,
                sentiment JSONB,
                meta_json JSONB
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations (session_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_started_at ON conversations (started_at);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_last_message_at ON conversations (last_message_at);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id UUID PRIMARY KEY,
                conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                message_index INTEGER NOT NULL,
                role VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                model_used VARCHAR(100),
                token_count INTEGER,
                processing_time INTEGER,
                reflection_triggered BOOLEAN DEFAULT FALSE NOT NULL,
                sentiment_score JSONB,
                topics_extracted JSONB,
                entities_mentioned JSONB,
                meta_json JSONB
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conv_msgs_conversation_id ON conversation_messages (conversation_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conv_msgs_role ON conversation_messages (role);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conv_msgs_timestamp ON conversation_messages (timestamp);
        """))

        # Memories table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                memory_type VARCHAR(100) NOT NULL,
                content TEXT NOT NULL,
                summary VARCHAR(500),
                created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                last_accessed TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                access_count INTEGER DEFAULT 0 NOT NULL,
                importance_score INTEGER DEFAULT 5 NOT NULL,
                confidence_score INTEGER DEFAULT 5 NOT NULL,
                source_conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
                source_message_id UUID REFERENCES conversation_messages(id) ON DELETE SET NULL,
                tags JSONB,
                topics JSONB,
                is_active BOOLEAN DEFAULT TRUE NOT NULL,
                is_validated BOOLEAN DEFAULT FALSE NOT NULL,
                meta_json JSONB
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories (created_at);
        """))

        # Persona-related tables
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS personas (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                mantra TEXT,
                first_thoughts TEXT,
                boot_directive TEXT,
                personality JSONB NOT NULL DEFAULT '{}'::jsonb,
                communication_style JSONB NOT NULL DEFAULT '{}'::jsonb,
                expertise JSONB NOT NULL DEFAULT '{"domains": [], "skills": [], "knowledge_depth": 0.5}'::jsonb,
                preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
                goals JSONB NOT NULL DEFAULT '{}'::jsonb,
                values JSONB NOT NULL DEFAULT '{}'::jsonb,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                evolution_locked BOOLEAN DEFAULT FALSE,
                creation_date TIMESTAMPTZ DEFAULT NOW(),
                last_modified TIMESTAMPTZ DEFAULT NOW(),
                last_evolution TIMESTAMPTZ,
                evolution_count INTEGER DEFAULT 0,
                stability_score FLOAT DEFAULT 1.0
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_personas_user_id ON personas (user_id);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS persona_evolutions (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                changes JSONB NOT NULL,
                reasoning TEXT NOT NULL,
                evidence JSONB NOT NULL,
                confidence FLOAT NOT NULL,
                impact_score FLOAT DEFAULT 0.0,
                source_type TEXT NOT NULL,
                source_id TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                reviewed BOOLEAN DEFAULT FALSE,
                approved BOOLEAN DEFAULT TRUE,
                reviewer_id TEXT,
                review_notes TEXT
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_persona_evolutions_persona_id ON persona_evolutions (persona_id);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS persona_traits (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                value FLOAT NOT NULL,
                description TEXT,
                confidence FLOAT NOT NULL DEFAULT 1.0,
                stability FLOAT NOT NULL DEFAULT 1.0,
                last_updated TIMESTAMPTZ DEFAULT NOW(),
                evidence_count INTEGER DEFAULT 0
            );
        """))
        
        # Ensure persona columns exist for legacy databases
        await conn.execute(text("""
            ALTER TABLE personas ADD COLUMN IF NOT EXISTS mantra TEXT;
        """))
        await conn.execute(text("""
            ALTER TABLE personas ADD COLUMN IF NOT EXISTS first_thoughts TEXT;
        """))
        await conn.execute(text("""
            ALTER TABLE personas ADD COLUMN IF NOT EXISTS boot_directive TEXT;
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_persona_traits_persona_id ON persona_traits (persona_id);
        """))

        # Create relationship_state table (Week 1: single-user optimization)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS relationship_state (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL UNIQUE REFERENCES personas(id) ON DELETE CASCADE,
                intimacy_level FLOAT NOT NULL DEFAULT 0.0,
                trust_level FLOAT NOT NULL DEFAULT 0.5,
                comfort_level FLOAT NOT NULL DEFAULT 0.3,
                stage TEXT NOT NULL DEFAULT 'early',
                days_known INTEGER NOT NULL DEFAULT 0,
                conversations_count INTEGER NOT NULL DEFAULT 0,
                communication_style TEXT,
                shared_interests JSONB NOT NULL DEFAULT '[]'::jsonb,
                inside_jokes JSONB NOT NULL DEFAULT '[]'::jsonb,
                first_conversation TIMESTAMPTZ,
                first_deep_conversation TIMESTAMPTZ,
                first_vulnerability_moment TIMESTAMPTZ,
                first_disagreement TIMESTAMPTZ,
                first_inside_joke TIMESTAMPTZ,
                user_name TEXT,
                user_preferences JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_conversation_at TIMESTAMPTZ
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_relationship_state_persona_id ON relationship_state (persona_id);
        """))

        # Create relationship_memories table (Week 2: relationship memory system)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS relationship_memories (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                memory_type TEXT NOT NULL,
                emotional_significance FLOAT NOT NULL DEFAULT 0.5,
                emotional_tone TEXT,
                intimacy_delta FLOAT NOT NULL DEFAULT 0.0,
                trust_delta FLOAT NOT NULL DEFAULT 0.0,
                narrative TEXT NOT NULL,
                user_perspective TEXT,
                context TEXT,
                conversation_id TEXT,
                tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                recall_count FLOAT NOT NULL DEFAULT 0,
                last_recalled TIMESTAMPTZ
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_relationship_memories_persona_id ON relationship_memories (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_relationship_memories_memory_type ON relationship_memories (memory_type);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_relationship_memories_conversation_id ON relationship_memories (conversation_id);
        """))

        # Create anticipated_events table (Week 2: temporal awareness)
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS anticipated_events (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                event_description TEXT NOT NULL,
                event_type TEXT,
                anticipated_date TIMESTAMPTZ,
                mentioned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                conversation_id TEXT,
                followed_up FLOAT NOT NULL DEFAULT 0.0,
                follow_up_at TIMESTAMPTZ,
                outcome TEXT,
                importance FLOAT NOT NULL DEFAULT 0.5
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_anticipated_events_persona_id ON anticipated_events (persona_id);
        """))

        # Create reflections table
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reflections (
                id UUID PRIMARY KEY,
                user_profile_id TEXT NOT NULL,
                reflection_type TEXT NOT NULL,
                result JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                embedding BYTEA,
                reflection_metadata JSONB
            );
        """))

        # Backward compatibility: if an older schema created a column named
        # 'metadata', rename it to 'reflection_metadata'. Also ensure the new
        # column exists if neither is present.
        await conn.execute(text("""
            DO $$
            BEGIN
                -- If legacy column exists and new one does not, rename it
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reflections' AND column_name = 'metadata'
                ) AND NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reflections' AND column_name = 'reflection_metadata'
                ) THEN
                    ALTER TABLE reflections RENAME COLUMN metadata TO reflection_metadata;
                END IF;

                -- If neither legacy nor new column exists (edge case), add new column
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'reflections' AND column_name = 'reflection_metadata'
                ) THEN
                    ALTER TABLE reflections ADD COLUMN reflection_metadata JSONB;
                END IF;
            END
            $$;
        """))
        
        # Create indices for faster queries
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reflections_user_profile_id ON reflections (user_profile_id);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reflections_type ON reflections (reflection_type);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reflections_created_at ON reflections (created_at);
        """))
        
        # Create reflection_memories table to track which memories were used in a reflection
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reflection_memories (
                reflection_id UUID NOT NULL REFERENCES reflections(id) ON DELETE CASCADE,
                memory_id UUID NOT NULL,
                relevance_score FLOAT,
                PRIMARY KEY (reflection_id, memory_id)
            );
        """))
        
        # Create reflection_schedule table for managing scheduled reflections
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reflection_schedule (
                id UUID PRIMARY KEY,
                user_profile_id TEXT NOT NULL,
                reflection_type TEXT NOT NULL,
                scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                completed_at TIMESTAMP WITH TIME ZONE,
                reflection_id UUID REFERENCES reflections(id)
            );
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reflection_schedule_status ON reflection_schedule (status);
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_reflection_schedule_user ON reflection_schedule (user_profile_id);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS relationship_question_queue (
                id UUID PRIMARY KEY,
                user_profile_id TEXT NOT NULL,
                reflection_id UUID NOT NULL REFERENCES reflections(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                topic TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 3,
                suggested_delay_days INTEGER NOT NULL DEFAULT 7,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                insight_value TEXT,
                existing_conflicts JSONB DEFAULT '[]'::jsonb,
                queued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                available_at TIMESTAMP WITH TIME ZONE NOT NULL,
                raw_payload JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                delivered_at TIMESTAMP WITH TIME ZONE,
                answered_at TIMESTAMP WITH TIME ZONE,
                answer_memory_id UUID,
                answer_tags JSONB DEFAULT '[]'::jsonb,
                answer_importance_score INTEGER,
                answer_confidence_score INTEGER
            );
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_relationship_queue_user_status ON relationship_question_queue (user_profile_id, status);
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_relationship_queue_available_at ON relationship_question_queue (available_at);
        """))

        # Ensure new answer-tracking columns exist for earlier installs
        await conn.execute(text("""
            ALTER TABLE relationship_question_queue
                ADD COLUMN IF NOT EXISTS delivered_at TIMESTAMP WITH TIME ZONE;
        """))
        await conn.execute(text("""
            ALTER TABLE relationship_question_queue
                ADD COLUMN IF NOT EXISTS answered_at TIMESTAMP WITH TIME ZONE;
        """))
        await conn.execute(text("""
            ALTER TABLE relationship_question_queue
                ADD COLUMN IF NOT EXISTS answer_memory_id UUID;
        """))
        await conn.execute(text("""
            ALTER TABLE relationship_question_queue
                ADD COLUMN IF NOT EXISTS answer_tags JSONB DEFAULT '[]'::jsonb;
        """))
        await conn.execute(text("""
            ALTER TABLE relationship_question_queue
                ADD COLUMN IF NOT EXISTS answer_importance_score INTEGER;
        """))
        await conn.execute(text("""
            ALTER TABLE relationship_question_queue
                ADD COLUMN IF NOT EXISTS answer_confidence_score INTEGER;
        """))

        # Reflection Examples table for dynamic few-shot learning
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reflection_examples (
                id UUID PRIMARY KEY,
                category VARCHAR(50) NOT NULL,
                scenario VARCHAR(100) NOT NULL,
                requires_history BOOLEAN DEFAULT FALSE,
                is_emotional BOOLEAN DEFAULT FALSE,
                is_technical BOOLEAN DEFAULT FALSE,
                user_message TEXT NOT NULL,
                context_description TEXT NOT NULL,
                reflection_content TEXT NOT NULL,
                full_json JSONB NOT NULL,
                explanation TEXT,
                success_rate FLOAT DEFAULT 0.0,
                times_shown INTEGER DEFAULT 0,
                times_succeeded INTEGER DEFAULT 0,
                version INTEGER DEFAULT 1,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                tags JSONB DEFAULT '[]'::jsonb
            );
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_examples_category ON reflection_examples (category);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_examples_scenario ON reflection_examples (scenario);
        """))
        
        # Agent state tables for emergent agent roadmap foundations
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS affective_states (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                mood_vector JSONB NOT NULL DEFAULT '{"valence": 0.0, "arousal": 0.0}'::jsonb,
                energy FLOAT NOT NULL DEFAULT 0.5,
                stress FLOAT NOT NULL DEFAULT 0.5,
                confidence FLOAT NOT NULL DEFAULT 0.5,
                last_update TIMESTAMPTZ DEFAULT NOW(),
                state_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                homeostasis_active BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_affective_states_persona ON affective_states (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_affective_states_user ON affective_states (user_id);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_goals (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                origin TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                priority FLOAT NOT NULL DEFAULT 0.5,
                deadline TIMESTAMPTZ,
                progress FLOAT NOT NULL DEFAULT 0.0,
                evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                extra_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_goals_persona ON agent_goals (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_goals_user ON agent_goals (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_goals_status ON agent_goals (status);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS plan_steps (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL REFERENCES agent_goals(id) ON DELETE CASCADE,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                description TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                priority FLOAT NOT NULL DEFAULT 0.5,
                target_time TIMESTAMPTZ,
                evidence_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                extra_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMPTZ
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_plan_steps_goal ON plan_steps (goal_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_plan_steps_persona ON plan_steps (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_plan_steps_status ON plan_steps (status);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS autobiographical_episodes (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                narrative_text TEXT NOT NULL,
                summary TEXT,
                importance FLOAT NOT NULL DEFAULT 0.5,
                emotion_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
                participants JSONB NOT NULL DEFAULT '[]'::jsonb,
                linked_memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                start_time TIMESTAMPTZ,
                end_time TIMESTAMPTZ,
                extra_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_auto_episodes_persona ON autobiographical_episodes (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_auto_episodes_user ON autobiographical_episodes (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_auto_episodes_importance ON autobiographical_episodes (importance);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meta_reflections (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                directive_text TEXT NOT NULL,
                priority FLOAT NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'pending',
                due_time TIMESTAMPTZ,
                related_goal_id TEXT REFERENCES agent_goals(id) ON DELETE SET NULL,
                source_reflection_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                review_notes TEXT,
                extra_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_meta_reflections_persona ON meta_reflections (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_meta_reflections_user ON meta_reflections (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_meta_reflections_status ON meta_reflections (status);
        """))

        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_examples_requires_history ON reflection_examples (requires_history);
        """))
        
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_examples_is_active ON reflection_examples (is_active);
        """))

        # SDL (Self-Directed Learning) tables
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS learnings (
                id VARCHAR(36) PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                source_type VARCHAR(50) NOT NULL,
                source_id VARCHAR(36) NOT NULL,
                confidence FLOAT NOT NULL DEFAULT 0.7,
                importance FLOAT NOT NULL DEFAULT 0.5,
                domain VARCHAR(100) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                active BOOLEAN DEFAULT TRUE NOT NULL,
                vector_id VARCHAR(36),
                attributes JSONB
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learnings_user_id ON learnings (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learnings_domain ON learnings (domain);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learnings_source_type ON learnings (source_type);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learnings_source_id ON learnings (source_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learnings_importance ON learnings (importance);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learnings_created_at ON learnings (created_at);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS concepts (
                id VARCHAR(36) PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                category VARCHAR(100),
                importance FLOAT NOT NULL DEFAULT 0.5,
                familiarity FLOAT NOT NULL DEFAULT 0.3,
                created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                active BOOLEAN DEFAULT TRUE NOT NULL,
                vector_id VARCHAR(36),
                attributes JSONB
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concepts_user_id ON concepts (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts (name);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concepts_category ON concepts (category);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concepts_importance ON concepts (importance);
        """))
        await conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_concepts_user_name ON concepts (user_id, name);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS persona_concept_association (
                persona_id TEXT NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
                concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                relevance_score FLOAT DEFAULT 0.5,
                last_updated TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (persona_id, concept_id)
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_persona_concept_persona ON persona_concept_association (persona_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_persona_concept_concept ON persona_concept_association (concept_id);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS learning_concept (
                learning_id VARCHAR(36) NOT NULL REFERENCES learnings(id) ON DELETE CASCADE,
                concept_id VARCHAR(36) NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                strength FLOAT DEFAULT 0.5,
                created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                PRIMARY KEY (learning_id, concept_id)
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learning_concept_learning ON learning_concept (learning_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_learning_concept_concept ON learning_concept (concept_id);
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS concept_connections (
                id VARCHAR(36) PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                source_id VARCHAR(36) NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                target_id VARCHAR(36) NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                relation_type VARCHAR(100) NOT NULL,
                strength FLOAT NOT NULL DEFAULT 0.5,
                bidirectional BOOLEAN DEFAULT FALSE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
                active BOOLEAN DEFAULT TRUE NOT NULL
            );
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concept_connections_user_id ON concept_connections (user_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concept_connections_source ON concept_connections (source_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concept_connections_target ON concept_connections (target_id);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concept_connections_relation ON concept_connections (relation_type);
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_concept_connections_strength ON concept_connections (strength);
        """))
    
    logger.info("Database tables created successfully!")

async def drop_tables():
    """
    Drop all reflection system tables (for testing/reset purposes).
    WARNING: This will delete all reflection data.
    """
    logger.warning("Dropping all reflection system tables!")
    
    async with engine.begin() as conn:
        # Drop tables in reverse order of dependencies
        await conn.execute(text("DROP TABLE IF EXISTS reflection_schedule;"))
        await conn.execute(text("DROP TABLE IF EXISTS reflection_memories;"))
        await conn.execute(text("DROP TABLE IF EXISTS reflections;"))
        
        # Drop SDL tables
        await conn.execute(text("DROP TABLE IF EXISTS concept_connections;"))
        await conn.execute(text("DROP TABLE IF EXISTS learning_concept;"))
        await conn.execute(text("DROP TABLE IF EXISTS concepts;"))
        await conn.execute(text("DROP TABLE IF EXISTS learnings;"))
    
    logger.info("All reflection system tables dropped.")

if __name__ == "__main__":
    """
    Script can be run directly to initialize the database.
    """
    parser = argparse.ArgumentParser(description="SELO AI database initialization")
    parser.add_argument("--drop", action="store_true", help="Drop existing tables before creating new ones")
    args = parser.parse_args()

    async def main():
        # Ensure we can acquire a session from the centralized factory (sanity check)
        async with AsyncSessionLocal() as _test_session:  # type: ignore
            pass
        if args.drop:
            await drop_tables()
        await create_tables()

    asyncio.run(main())
