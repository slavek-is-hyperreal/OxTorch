# Gemini to Markdown Converter

A self-contained Python script to convert shared Google Gemini conversations to Markdown.

## Features
- Opens a headed browser window so you can handle cookie consent and sign-ins manually.
- Extracts user prompts and model responses.
- Converts HTML to clean Markdown.
- Supports CLI and Library usage.

## Installation
Ensure you have the dependencies installed:
```bash
pip install playwright beautifulsoup4 markdownify
playwright install chromium
```

## Usage

### CLI
```bash
python side_tools/gemini_to_markdown/gemini_to_md.py "https://gemini.google.com/share/..."
```
Press **ENTER** in the terminal once the page is fully loaded in the browser.

### Library
```python
from side_tools.gemini_to_markdown import GeminiConverter

converter = GeminiConverter(headless=False)
converter.convert_to_md("https://gemini.google.com/share/...")
```
