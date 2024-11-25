# Guardrails-Genie

Guardrails-Genie is a tool that helps you implement guardrails in your LLM applications.

## Installation

```bash
git clone https://github.com/soumik12345/guardrails-genie
cd guardrails-genie
pip install -u pip uv
uv venv
# If you want to install for torch CPU, uncomment the following line
# export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
uv pip install -e .
source .venv/bin/activate
```

## Run Chat App

```bash
OPENAI_API_KEY="YOUR_OPENAI_API_KEY" streamlit run app.py
```
