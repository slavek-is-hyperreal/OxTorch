---
description: How to run tests in this repository
---
# Testing Policy for AI Agents

**CRITICAL RULE FOR ALL FUTURE AGENTS:**
Due to the massive scale of VNN's SSD streaming tests (e.g., streaming 24GB of data over simulated memory paths), running full test suites or large integration tests directly via the agent's `run_command` tool **WILL CAUSE THE ENVIRONMENT TO HANG AND CRASH**. 

Therefore, whenever you need to run tests (especially `pytest tests/` or any `tests/integration/` tests):
1. **DO NOT** run them yourself using the `run_command` tool.
2. **DO** provide the exact `pytest` command to the USER.
3. **ASK** the USER to run the command in a separate, external terminal window and paste the results back to you.

Example command to provide to the user:
```bash
cd /my_data/gaussian_room && source venv/bin/activate && PYTHONPATH=. pytest tests/ -v -s
```

You may run very fast, localized unit tests (e.g. `tests/core/test_basic_ops.py`) if you are 100% sure they finish in under 2 seconds and do not allocate large SSD tensors, but when in doubt or when verifying a full feature/phase, ALWAYS defer to the USER.
