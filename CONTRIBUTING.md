# Contributing to SELO DSP

Thank you for your interest in contributing to SELO DSP! This document provides guidelines and instructions for contributing to the project.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Coding Guidelines](#coding-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Review](#code-review)

---

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL 13+
- Ollama (latest version)
- CUDA 11.8+ (for GPU acceleration)
- Git

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/selodesigns/SELODSP-Linux.git
   cd SELODSP-Linux
   ```

2. **Install development dependencies:**
   ```bash
   # Backend
   cd selo-ai/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

3. **Install pre-commit hooks:**
   ```bash
   cd /path/to/SELODSP-Linux
   pip install pre-commit
   pre-commit install
   ```

4. **Configure environment:**
   ```bash
   cd selo-ai/backend
   cp .env.example .env
   # Edit .env and change default credentials!
   ```

5. **Set up database:**
   ```bash
   sudo systemctl start postgresql
   sudo -u postgres createdb seloai
   sudo -u postgres psql -c "CREATE USER seloai WITH PASSWORD 'your_strong_password';"
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE seloai TO seloai;"
   ```

---

## Code Quality Standards

### Pre-commit Hooks

All code must pass pre-commit checks before being committed. The hooks will automatically:

- ✅ Format code with **Black** (line length: 120)
- ✅ Sort imports with **isort**
- ✅ Lint code with **flake8**
- ✅ Scan for security issues with **bandit**
- ✅ Remove trailing whitespace
- ✅ Fix end-of-file issues
- ✅ Detect private keys
- ✅ Validate YAML/JSON/TOML files

**Run manually:**
```bash
pre-commit run --all-files
```

### Code Formatting

We use **Black** for Python code formatting:

```bash
black --line-length 120 selo-ai/backend/
```

**Configuration:** See `pyproject.toml`

### Import Sorting

We use **isort** with Black profile:

```bash
isort --profile black selo-ai/backend/
```

### Linting

We use **flake8** for linting:

```bash
flake8 --max-line-length=120 selo-ai/backend/
```

---

## Coding Guidelines

### Python Style Guide

1. **Follow PEP 8** with 120 character line length
2. **Use type hints** for all public functions:
   ```python
   def get_user(user_id: str) -> Optional[Dict[str, Any]]:
       """Get user by ID."""
       pass
   ```

3. **Use f-strings** for string formatting:
   ```python
   # Good
   message = f"User {user_id} logged in"
   
   # Avoid
   message = "User {} logged in".format(user_id)
   message = "User %s logged in" % user_id
   ```

4. **Specific exception handling** - Never use bare `except Exception:`:
   ```python
   # Good
   try:
       value = int(data)
   except (ValueError, TypeError) as e:
       logger.error(f"Failed to parse value: {e}")
       return default
   
   # Bad
   try:
       value = int(data)
   except Exception:
       return default
   ```

5. **Use named constants** instead of magic numbers:
   ```python
   # Good
   MAX_RETRY_ATTEMPTS = 3
   for attempt in range(MAX_RETRY_ATTEMPTS):
       pass
   
   # Bad
   for attempt in range(3):
       pass
   ```

6. **Proper logging levels:**
   - `logger.debug()` - Development details, expected fallbacks
   - `logger.info()` - Normal operations, milestones
   - `logger.warning()` - Unexpected but recoverable issues
   - `logger.error()` - Failures requiring attention

7. **Async/await patterns:**
   - Use `async def` for all I/O operations
   - Use `await` for async calls
   - Never block the event loop

### JavaScript/React Style Guide

1. **Use functional components** with hooks
2. **Use arrow functions** for consistency
3. **PropTypes or TypeScript** for type safety
4. **ESLint** configuration in package.json

---

## Testing

### Running Tests

**Backend (Python):**
```bash
cd selo-ai/backend
source venv/bin/activate
pytest tests/
```

**With coverage:**
```bash
pytest --cov=backend --cov-report=html tests/
```

**Frontend (React):**
```bash
cd selo-ai/frontend
npm test
```

### Writing Tests

1. **Unit tests** for individual functions
2. **Integration tests** for API endpoints
3. **E2E tests** for critical user flows

**Test file naming:**
- `test_*.py` or `*_test.py` for Python
- `*.test.js` or `*.test.jsx` for JavaScript

**Example test:**
```python
import pytest
from backend.persona.bootstrapper import PersonaBootstrapper

@pytest.mark.asyncio
async def test_persona_bootstrap():
    """Test persona bootstrap creates valid persona."""
    bootstrapper = PersonaBootstrapper(...)
    persona = await bootstrapper.ensure_persona()
    assert persona is not None
    assert "description" in persona
```

---

## Pull Request Process

### Before Submitting

1. ✅ **Run pre-commit hooks:**
   ```bash
   pre-commit run --all-files
   ```

2. ✅ **Run tests:**
   ```bash
   pytest tests/
   ```

3. ✅ **Update documentation** if needed

4. ✅ **Add tests** for new features

5. ✅ **Check for security issues:**
   ```bash
   bandit -r selo-ai/backend/
   ```

### Commit Message Format

Use **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

**Examples:**
```bash
feat(reflection): add daily reflection scheduling
fix(llm): handle timeout errors gracefully
docs(readme): update installation instructions
refactor(main): extract service initialization to startup.py
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

---

## Code Review

### Review Checklist

**Functionality:**
- [ ] Code does what it's supposed to do
- [ ] Edge cases handled
- [ ] Error handling appropriate

**Code Quality:**
- [ ] Follows style guidelines
- [ ] No code duplication
- [ ] Functions are focused and small
- [ ] Variable names are descriptive

**Security:**
- [ ] No hardcoded credentials
- [ ] Input validation present
- [ ] No SQL injection vulnerabilities
- [ ] Sensitive data not logged

**Testing:**
- [ ] Tests included
- [ ] Tests are meaningful
- [ ] Coverage maintained or improved

**Documentation:**
- [ ] Docstrings added
- [ ] README updated if needed
- [ ] Comments explain "why" not "what"

---

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feat/your-feature-name
```

### 2. Make Changes

- Write code following guidelines
- Add tests
- Update documentation

### 3. Commit Changes

```bash
git add .
git commit -m "feat(scope): description"
# Pre-commit hooks run automatically
```

### 4. Push and Create PR

```bash
git push origin feat/your-feature-name
# Create pull request on GitHub
```

### 5. Address Review Feedback

- Make requested changes
- Push updates to same branch
- PR updates automatically

---

## Common Tasks

### Adding a New API Endpoint

1. **Create route in appropriate router:**
   ```python
   # In api/your_router.py
   @router.get("/endpoint")
   async def your_endpoint(
       param: str,
       db: AsyncSession = Depends(get_db)
   ) -> Dict[str, Any]:
       """
       Endpoint description.
       
       Args:
           param: Parameter description
           
       Returns:
           Response description
       """
       # Implementation
   ```

2. **Add tests:**
   ```python
   # In tests/test_your_router.py
   async def test_your_endpoint():
       # Test implementation
   ```

3. **Update API documentation**

### Adding a New Database Model

1. **Create model in `db/models/`**
2. **Create repository in `db/repositories/`**
3. **Create Alembic migration:**
   ```bash
   cd selo-ai/backend
   alembic revision --autogenerate -m "Add new model"
   alembic upgrade head
   ```

### Adding a New Configuration Option

1. **Add to `.env.example` with documentation**
2. **Add validation in startup code**
3. **Document in README.md**

---

## Troubleshooting

### Pre-commit Hooks Failing

```bash
# Skip hooks temporarily (not recommended)
git commit --no-verify

# Fix issues and retry
pre-commit run --all-files
```

### Import Errors

```bash
# Ensure you're in the backend directory
cd selo-ai/backend
source venv/bin/activate

# Reinstall in development mode
pip install -e .
```

### Tests Failing

```bash
# Run with verbose output
pytest -v tests/

# Run specific test
pytest tests/test_specific.py::test_function_name

# Run with debugging
pytest --pdb tests/
```

---

## Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [React Docs](https://react.dev/)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Tools
- [Black](https://black.readthedocs.io/)
- [isort](https://pycqa.github.io/isort/)
- [flake8](https://flake8.pycqa.org/)
- [mypy](https://mypy.readthedocs.io/)
- [pytest](https://docs.pytest.org/)

---

## Getting Help

- **Issues:** [GitHub Issues](https://github.com/selodesigns/SELODSP-Linux/issues)
- **Discussions:** [GitHub Discussions](https://github.com/selodesigns/SELODSP-Linux/discussions)
- **Documentation:** See README.md and Reports/ directory

---

## License

By contributing, you agree that your contributions will be licensed under the same CC BY-NC-SA 4.0 license that covers the project.

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the project
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Publishing others' private information
- Other unprofessional conduct

---

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Ask in GitHub Discussions
4. Open a new issue

---

**Thank you for contributing to SELO DSP!** 🚀

*Created with ❤️ by the SELO community*
