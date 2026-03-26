import os
import sys
import time
import argparse
from playwright.sync_api import sync_playwright
import markdownify
from bs4 import BeautifulSoup

class GeminiConverter:
    def __init__(self, headless=False):
        self.headless = headless

    def convert_to_md(self, url, output_path=None):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            
            print(f"Opening {url}...")
            page.goto(url)
            
            print("\n" + "="*50)
            print("MANUAL ACTION REQUIRED:")
            print("1. Handle any cookie consent popups in the browser window.")
            print("2. Ensure the conversation is fully loaded and visible.")
            print("3. When ready, press ENTER here in the terminal to start conversion.")
            print("="*50 + "\n")
            
            input("Press ENTER to convert...")
            
            # Extract content
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get Title
            title_tag = soup.find('h1') or soup.find('h2') or soup.select_one('.title')
            title = title_tag.get_text(strip=True) if title_tag else "Gemini_Conversation"
            title_clean = "".join(x for x in title if x.isalnum() or x in " -_").strip()
            
            if not output_path:
                output_path = f"{title_clean}.md"
            
            md_output = f"# {title}\n\n"
            md_output += f"*Source: {url}*\n\n---\n\n"
            
            # Extract turns
            # We look for .share-turn-viewer which contains both prompt and response
            # Or fallback to individual containers if needed
            turns_handled = set()
            
            # Strategy: Find all turn containers
            # In some views it's .share-turn-viewer, in others it might be different.
            turn_containers = soup.select('.share-turn-viewer')
            
            if not turn_containers:
                # Fallback to loose selection if no shared container found
                # (e.g. if the DOM structure is slightly different)
                prompt_containers = soup.select('.user-query-container, .query-content')
                for pc in prompt_containers:
                    # Find following response
                    # This is trickier, so we prefer the shared container
                    pass

            for turn in turn_containers:
                # Deduplication logic
                # Gemini often renders the same turn multiple times for different breakpoints
                turn_text_raw = turn.get_text(strip=True)
                turn_hash = hash(turn_text_raw)
                if turn_hash in turns_handled:
                    continue
                turns_handled.add(turn_hash)
                
                # 1. Extract User Prompt
                prompt_el = turn.select_one('.user-query-container, .query-content, [aria-level="2"]')
                if prompt_el:
                    # Clean up "Treść Twojej wiadomości" if present
                    for hidden in prompt_el.select('.visually-hidden, [aria-hidden="true"]'):
                        hidden.decompose()
                    
                    prompt_text = prompt_el.get_text(separator='\n', strip=True)
                    # Remove the recurring Gemini UI labels
                    prompt_text = prompt_text.replace("Treść Twojej wiadomości", "").strip()
                    
                    if prompt_text:
                        md_output += f"### 👤 User\n\n{prompt_text}\n\n"
                
                # 2. Extract Gemini Response
                # Multiple response versions might exist, but usually one is 'presented'
                response_el = turn.select_one('.markdown, .model-response-text, .message-content')
                if response_el:
                    # Clean up before markdown conversion
                    for btn in response_el.select('button, .action-buttons'):
                        btn.decompose()
                    
                    response_md = markdownify.markdownify(str(response_el), heading_style="ATX")
                    # Post-process response_md to remove potential ghost labels
                    response_md = response_md.replace("Odpowiedź Gemini", "").strip()
                    
                    if response_md:
                        md_output += f"### 🤖 Gemini\n\n{response_md}\n\n"
                
                md_output += "---\n\n"
            
            if not turn_containers:
                 md_output += "> No conversation turns found. Please check if the selectors are still valid for this page structure.\n"

            # Final cleanup of MD
            md_output = md_output.replace("\n---\n\n---", "\n---")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_output)
            
            print(f"\nSuccess! Conversation saved to: {os.path.abspath(output_path)}")
            browser.close()
            return output_path

def main():
    parser = argparse.ArgumentParser(description="Convert Gemini Share Link to Markdown")
    parser.add_argument("url", nargs="?", help="The Gemini share URL")
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args()

    url = args.url
    if not url:
        url = input("Enter Gemini Share URL: ").strip()
    
    if not url.startswith("http"):
        print("Invalid URL format.")
        sys.exit(1)
        
    converter = GeminiConverter(headless=False)
    
    # Check if URL is actually a local file
    if getattr(args, 'local', False) or os.path.exists(url):
        # We need to hack the convert_to_md slightly or just provide a different path
        # Let's just add a simple check in convert_to_md or here
        if os.path.exists(url):
            with open(url, "r", encoding="utf-8") as f:
                html = f.read()
            # Simple mock-up of the conversion logic for local files
            # (or we could refactor convert_to_md to accept html string)
            print(f"Reading local file: {url}")
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            # Reuse the extraction logic (calling a private helper would be better but let's keep it simple)
            # Refactor: extract_from_soup(soup, url, output_path)
            # For now, let's just make convert_to_md handle file:// URLs
            if not url.startswith("file://"):
                url = "file://" + os.path.abspath(url)
    
    converter.convert_to_md(url, args.output)

if __name__ == "__main__":
    main()
