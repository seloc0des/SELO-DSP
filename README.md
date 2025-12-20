# SELO DSP - Self-Evolving Learning Organism Digital Sentient Platform

**Version:** 1.0.0  
**Created by:** SELOdev  
**License:** CC BY-NC-SA 4.0 (Non-Commercial)

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub stars](https://img.shields.io/github/stars/selodesigns/SELODSP-Linux.svg)](https://github.com/selodesigns/SELODSP-Linux/stargazers)

SELO AI is an autonomous artificial intelligence system that learns and evolves through self-reflection. Built on local LLMs (Ollama), it features dynamic personality development, semantic memory, and emergent agent behaviors. The system generates inner reflections after each interaction, extracts learnings, and adapts its persona over time.

**⚠️ Non-Commercial License:** This software is free for personal, educational, and research use. Commercial use requires a separate license from SELOdev. See [License](#license) section below.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)
- [License](#license)

## Features

### Core Capabilities

**Reflection System**
- Generates inner reflections after each conversation
- Analyzes emotional state, themes, and insights
- Tracks trait changes over time
- Selective reflection (decides when deep introspection is needed)

**Dynamic Persona**
- Bootstraps unique personality on first run
- Evolves traits based on interactions
- Maintains values, goals, and communication style
- Scheduled reassessments for holistic evolution

**Semantic Memory**
- FAISS vector store for semantic search
- Embeds reflections, themes, and learnings
- GPU-accelerated similarity search
- Persistent storage across restarts

**Emergent Agent State**
- Affective state tracking (energy, stress, confidence, mood)
- Self-directed goals and planning
- Meta-directives from reflection analysis
- Autobiographical episode generation

**Self-Directed Learning (SDL)**
- Extracts learnings from reflections
- Organizes by domain (personality, preferences, knowledge, etc.)
- Confidence-based filtering
- Informs persona evolution

### Technical Features

**Backend (FastAPI + Python)**
- Multi-model LLM routing (conversational, analytical, reflection)
- Real-time streaming responses via Socket.IO
- PostgreSQL database for persistence
- Async/await throughout
- Circuit breaker pattern for fault tolerance
- Structured logging with correlation IDs

**Frontend (React)**
- Real-time chat interface
- Reflection dashboard
- Persona management UI
- Agent state visualization
- System diagnostics

**LLM Integration (Ollama)**
- Local model execution (no API keys needed)
- Supports Llama 3, Qwen 2.5, Phi-3, and more
- GPU acceleration with CUDA
- Automatic model warmup and keepalive

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SELO AI System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  User Message → Chat → Reflection → SDL → Persona Evolution │
│                  ↓         ↓          ↓            ↓         │
│              Response  Analysis  Learnings   Trait Changes   │
│                                                               │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Frontend   │◄──►│   Backend    │◄──►│  PostgreSQL  │
│              │    │              │    │              │
│ • Chat UI    │    │ • LLM Router │    │ • Personas   │
│ • Reflection │    │ • Reflection │    │ • Reflections│
│ • Persona    │    │ • SDL Engine │    │ • Learnings  │
│ • Agent State│    │ • Vector Store│   │ • Agent State│
└──────────────┘    └──────────────┘    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    Ollama    │
                    │              │
                    │ • llama3:8b  │
                    │ • qwen2.5:3b │
                    │ • nomic-embed│
                    └──────────────┘
```

### Key Components

**Reflection Processor** (`backend/reflection/processor.py`)
- Generates reflections using LLM
- Validates output (word count, schema, constraints)
- Tracks example performance for few-shot learning
- Emits events for emergent behaviors

**Persona Engine** (`backend/persona/engine.py`)
- Bootstraps initial persona
- Generates system prompts
- Applies trait changes (bounded ±0.2 per reflection)
- Scheduled reassessments

**SDL Engine** (`backend/sdl/engine.py`)
- Extracts learnings from reflections
- Categorizes by domain
- Stores with confidence scores
- Provides semantic retrieval

**Agent State Managers** (`backend/agent/`)
- Affective state with homeostasis
- Goal creation and tracking
- Planning service
- Meta-reflection processor
- Episode builder

**Vector Store** (`backend/memory/vector_store.py`)
- FAISS index for embeddings
- GPU acceleration when available
- Persistent storage
- Semantic search

## System Requirements

### Minimum (8GB GPU)
- **OS:** Linux (Ubuntu 20.04+) or Windows 10/11
- **RAM:** 16GB
- **GPU:** 8GB VRAM (NVIDIA GTX 1070, RTX 3060, etc.)
- **Storage:** 20GB free space
- **CPU:** 4+ cores

### Recommended (16GB GPU)
- **OS:** Linux with systemd or Windows 10/11
- **RAM:** 32GB
- **GPU:** 16GB VRAM (NVIDIA RTX 4080, RTX 3090, etc.)
- **Storage:** 50GB SSD
- **CPU:** 8+ cores (Ryzen 7/Intel i7 or better)

### Software Dependencies
- **Python:** 3.10+
- **Node.js:** 18+
- **PostgreSQL:** 13+ (Linux) or SQLite (Windows - auto-configured)
- **Ollama:** Latest version
- **CUDA:** 11.8+ (for GPU acceleration, optional)

## Installation

### Linux Quick Start

```bash
# Clone repository
git clone https://github.com/selodesigns/SELODSP-Linux.git
cd SELODSP-Linux/selo-ai

# Run installer (detects GPU and configures automatically)
chmod +x install-complete.sh
sudo ./install-complete.sh

# Start the system
sudo systemctl start selodsp
# OR
./start-full-server.sh
```

### Windows Quick Start

```powershell
# Clone repository (or download and extract ZIP)
git clone https://github.com/selodesigns/SELODSP-Linux.git
cd SELODSP-Linux\selo-ai

# Run installer (PowerShell as Administrator recommended)
Set-ExecutionPolicy Bypass -Scope Process -Force
.\windows-install.ps1

# Start SELO AI
.\start-selo.bat
# OR
.\start-selo.ps1
```

> **Windows Note:** See [README-Windows.md](selo-ai/README-Windows.md) for detailed Windows installation instructions and troubleshooting.

The installer will:
1. Detect your GPU and RAM
2. Install Python and Node dependencies
3. Set up PostgreSQL database
4. Pull Ollama models (llama3:8b, qwen2.5:3b, nomic-embed-text)
5. Build the frontend
6. Configure systemd service

**Installation time:** 20-40 minutes (mostly model downloads)

### Manual Installation

See `selo-ai/install-complete.sh` for the full installation script with detailed comments.

## Configuration

### Default Models

The system uses three specialized models:

- **Conversational:** `llama3:8b` - Chat responses
- **Analytical:** `qwen2.5:3b` - Structured analysis (traits, seed generation)
- **Reflection:** `qwen2.5:3b` - Inner reflections
- **Embedding:** `nomic-embed-text` - Vector embeddings (2048-dim)

### Environment Variables

Key settings in `backend/.env`:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://seloai:password@localhost/seloai

# Models
CONVERSATIONAL_MODEL=llama3:8b
ANALYTICAL_MODEL=qwen2.5:3b
REFLECTION_LLM=qwen2.5:3b
EMBEDDING_MODEL=nomic-embed-text

# Token Budgets (auto-configured by tier)
REFLECTION_NUM_PREDICT=650  # High-tier: 650, Standard: 640
CHAT_NUM_PREDICT=1024
CHAT_NUM_CTX=8192

# Reflection Settings
REFLECTION_TEMPERATURE=0.35
REFLECTION_WORD_MIN=80
REFLECTION_WORD_MAX=250

# Agent Loop
AGENT_LOOP_ENABLED=true
AGENT_LOOP_INTERVAL_SECONDS=900  # 15 minutes
```

### Tier System

The installer auto-detects your hardware:

**High-Performance Tier (≥12GB GPU):**
- Reflection: 650 tokens, 80-250 words
- Analytical: 1536 tokens
- Richer philosophical depth

**Standard Tier (<12GB GPU):**
- Reflection: 640 tokens, 80-250 words
- Analytical: 640 tokens
- Optimized for 8GB GPU

## Usage

### Starting the System

**As Service:**
```bash
sudo systemctl start selodsp
sudo systemctl status selodsp
sudo journalctl -u selodsp -f  # View logs
```

**Development Mode:**
```bash
# Terminal 1: Backend
cd selo-ai/backend
source venv/bin/activate
python -m backend.main

# Terminal 2: Frontend
cd selo-ai/frontend
npm start
```

### Using the Chat Interface

1. Open `http://localhost:3000`
2. Send a message
3. Watch the reflection generate (10-30 seconds)
4. View reflection in the Reflection tab
5. Check persona evolution in Persona tab
6. Monitor agent state in Agent State tab

### API Endpoints

```bash
# Chat
POST /chat
GET /conversations/history?session_id={id}

# Reflections
GET /api/reflections?user_id={id}&page=1&per_page=20
POST /api/reflection/generate
GET /api/reflection/{reflection_id}

# Persona
GET /api/persona/status
GET /api/persona/default
GET /api/persona/traits/{persona_id}

# Agent State
GET /api/agent-state/affective?persona_id={id}
GET /api/agent-state/goals?persona_id={id}
GET /api/agent-state/episodes?persona_id={id}
GET /api/agent-state/meta-directives?persona_id={id}

# System
GET /health
GET /health/details
GET /diagnostics/env
GET /diagnostics/gpu
```

## How It Works

### Interaction Flow

1. **User sends message** → Stored in conversation history
2. **Reflection decision** → Heuristics + LLM classifier decide if reflection needed
3. **Reflection generation** → LLM generates inner monologue (80-250 words)
4. **Validation** → Checks word count, schema, constraints, identity violations
5. **Emergent behaviors** → Extracts meta-directives, updates affective state, creates goals
6. **SDL extraction** → Identifies learnings from reflection
7. **Persona evolution** → Applies trait changes (bounded ±0.2)
8. **Chat response** → LLM generates reply using persona system prompt + reflection context

### Reflection System

**Types:**
- **Message:** After each conversation turn (selective)
- **Daily:** Scheduled at midnight
- **Weekly:** Scheduled weekly
- **Relationship:** Asks questions to learn about user

**Structure:**
```json
{
  "content": "Inner monologue (80-250 words)",
  "themes": ["theme1", "theme2"],
  "insights": ["insight1", "insight2"],
  "actions": ["action1", "action2"],
  "emotional_state": {
    "primary": "curious",
    "intensity": 0.7,
    "secondary": ["engaged"]
  },
  "metadata": {"coherence_rationale": "..."},
  "trait_changes": [
    {"name": "curiosity", "delta": 0.05, "reason": "..."}
  ]
}
```

### Persona Evolution

**Trait Changes:**
- Bounded: ±0.01 to ±0.20 per reflection
- Clamped: 0.0 to 1.0 range
- Locked traits: authenticity, grounding, curiosity (protected)
- Evidence-based: Every change includes reason

**Reassessment:**
- Triggered every 50 reflections
- Reviews recent learnings and reflections
- Proposes holistic updates
- Applies approved changes

### Emergent Behaviors

**Affective State:**
- Energy, stress, confidence, mood vector
- Updated from reflection emotions
- Homeostasis decay over time
- Displayed in chat system prompt

**Goals:**
- Auto-created from reflection actions
- Similarity detection prevents duplicates
- Priority scoring
- Progress tracking

**Meta-Directives:**
- Extracted from high-intensity emotions
- Recurring themes
- Actionable insights
- Due dates and priorities

## Development

### Project Structure

```
SELODSP-Linux/
├── selo-ai/
│   ├── backend/
│   │   ├── agent/          # Emergent agent state
│   │   ├── api/            # REST API routes
│   │   ├── config/         # Configuration
│   │   ├── constraints/    # Identity constraints
│   │   ├── core/           # Core systems
│   │   ├── db/             # Database models & repos
│   │   ├── llm/            # LLM integration
│   │   ├── memory/         # Vector store
│   │   ├── persona/        # Persona engine
│   │   ├── prompt/         # Prompt templates
│   │   ├── reflection/     # Reflection system
│   │   ├── scheduler/      # Task scheduling
│   │   ├── sdl/            # Self-directed learning
│   │   └── utils/          # Utilities
│   ├── frontend/
│   │   └── src/
│   │       ├── components/ # React components
│   │       └── services/   # API services
│   ├── configs/            # Configuration templates
│   ├── scripts/            # Utility scripts
│   ├── install-complete.sh # Main installer
│   └── start-service.sh    # Service management
├── Reports/                # Documentation & audits
└── README.md
```

### Running Tests

```bash
cd selo-ai/backend
source venv/bin/activate
pytest tests/
```

### Code Style

- **Python:** PEP 8, type hints, async/await
- **JavaScript:** ESLint, React hooks
- **Commits:** Conventional commits

## Troubleshooting

### Common Issues

**Models not loading:**
```bash
ollama list  # Check installed models
ollama pull llama3:8b
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

**Database connection errors:**
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

**GPU not detected:**
```bash
nvidia-smi  # Check GPU
echo $CUDA_VISIBLE_DEVICES  # Should be 0
```

**Reflection taking too long:**
- Check GPU utilization: `nvidia-smi`
- Verify model is loaded: `ollama ps`
- Check token budgets in `.env`

**High memory usage:**
- Standard tier uses ~4.5GB VRAM
- High tier uses ~6-8GB VRAM
- Adjust `CHAT_NUM_CTX` if needed

### Logs

```bash
# Service logs
sudo journalctl -u selodsp -f

# Backend logs
tail -f selo-ai/backend/logs/selo.log

# Check for errors
grep ERROR selo-ai/backend/logs/selo.log
```

## Credits

**Creator & Lead Developer:** [SELOdev](https://github.com/selodesigns)

SELO AI was conceived, designed, and built by SELOdev. This project represents a comprehensive exploration of self-evolving artificial intelligence systems with autonomous learning capabilities.

### Project Vision

SELO AI embodies the vision of creating truly adaptive AI systems that can:
- Learn and evolve through self-reflection
- Develop unique personalities through interaction
- Build semantic understanding through experience
- Maintain coherent long-term memory and context

### Acknowledgments

Special thanks to:
- All contributors who help improve this project
- The open-source community for the amazing tools and libraries
- Everyone who provides feedback and suggestions

### Contributing

Contributions are welcome! If you find this project valuable, please consider:
- ⭐ Starring the repository
- 🐛 Reporting bugs and issues
- 💡 Suggesting new features
- 🔧 Contributing code improvements
- 📖 Improving documentation

---

## License

**Copyright (c) 2025 SELOdev**

This project is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International) - see the [LICENSE](LICENSE) file for details.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

### License Summary

**You are free to:**
- ✅ **Share** — copy and redistribute the material in any medium or format
- ✅ **Adapt** — remix, transform, and build upon the material

**Under the following terms:**
- 📝 **Attribution** — You must give appropriate credit to SELOdev, provide a link to the license, and indicate if changes were made
- 🚫 **NonCommercial** — You may NOT use the material for commercial purposes without obtaining a separate commercial license from SELOdev
- 🔄 **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

### What This Means

**FREE for:**
- ✅ Personal projects and learning
- ✅ Educational and academic use
- ✅ Research and experimentation
- ✅ Non-profit organizations
- ✅ Open-source projects

**Requires Commercial License for:**
- ❌ Commercial products or services
- ❌ SaaS (Software as a Service) deployments
- ❌ Selling software based on SELO AI
- ❌ Internal business use for profit
- ❌ Consulting services using SELO AI

### Commercial Licensing

Interested in using SELO AI for commercial purposes? Contact SELOdev for commercial licensing options.

Commercial licenses include:
- Full commercial usage rights
- Priority support
- Custom feature development (optional)
- White-label options (optional)

---

## Contact & Support

- **GitHub:** [github.com/selodesigns/SELODSP-Linux](https://github.com/selodesigns/SELODSP-Linux)
- **Issues:** [Report bugs or request features](https://github.com/selodesigns/SELODSP-Linux/issues)
- **Discussions:** [Join the conversation](https://github.com/selodesigns/SELODSP-Linux/discussions)

---

**SELO AI** - Advancing the frontier of autonomous artificial intelligence through continuous learning and adaptation.

*Created with ❤️ by SELOdev*
