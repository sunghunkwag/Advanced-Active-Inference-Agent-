import arxiv
import re
# --- SECURITY WARNING ---
# This script clones and analyzes code from arbitrary, untrusted remote Git
# repositories. This is a significant security risk. In a production
# environment, this code should be executed in a sandboxed, isolated
# environment with no access to sensitive data or systems. The current
# implementation is for research and prototyping purposes only.
# --- END SECURITY WARNING ---

import os
import json
import subprocess
import logging
import ast
from datetime import datetime, timedelta

# --- Configuration ---
SEARCH_KEYWORDS = ["meta reinforcement learning", "world model", "continual learning", "model-based rl"]
MAX_RESULTS_PER_KEYWORD = 10
KNOWLEDGE_DB_PATH = "knowledge.json"
TEMP_CLONE_DIR = "temp_repos"
LOG_LEVEL = logging.INFO

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeCurator:
    """
    Searches for new knowledge from ArXiv and associated GitHub repositories,
    then structures and saves it for the Architect module.
    """

    def __init__(self):
        if not os.path.exists(TEMP_CLONE_DIR):
            os.makedirs(TEMP_CLONE_DIR)

    def run_curation_cycle(self):
        """Main method to run the full knowledge acquisition pipeline."""
        logging.info("Starting new knowledge curation cycle...")

        # 1. Search for recent, relevant papers on ArXiv
        papers = self._search_arxiv()
        if not papers:
            logging.info("No new relevant papers found.")
            return

        # 2. Extract GitHub links and analyze code
        knowledge_candidates = []
        for paper in papers:
            logging.info(f"Processing paper: {paper['title']}")
            github_url = self._find_github_url(paper['summary'])
            if not github_url:
                logging.warning(f"  -> No GitHub URL found in abstract.")
                continue

            logging.info(f"  -> Found GitHub URL: {github_url}")
            repo_path = self._clone_repo(github_url)
            if not repo_path:
                continue

            core_code = self._analyze_repo(repo_path)
            if not core_code:
                logging.warning(f"  -> Could not identify core code snippets.")
                self._cleanup_repo(repo_path)
                continue

            candidate = {
                "paper_title": paper['title'],
                "paper_summary": paper['summary'],
                "paper_authors": paper['authors'],
                "github_url": github_url,
                "core_code_snippets": core_code,
                "timestamp": datetime.utcnow().isoformat()
            }
            knowledge_candidates.append(candidate)
            self._cleanup_repo(repo_path)

        # 3. Save new knowledge to the database
        if knowledge_candidates:
            self._save_to_db(knowledge_candidates)
            logging.info(f"Successfully curated and saved {len(knowledge_candidates)} new knowledge candidates.")
        else:
            logging.info("Curation cycle finished. No new actionable knowledge was found.")

    def _search_arxiv(self):
        """
        Searches ArXiv for recent papers using a list of keywords.
        To avoid server errors, it searches for each keyword individually.
        """
        yesterday = datetime.utcnow() - timedelta(days=7)
        all_results = {} # Use a dict to avoid duplicate papers
        api_success = False

        for keyword in SEARCH_KEYWORDS:
            try:
                logging.info(f"Searching ArXiv for keyword: '{keyword}'")
                search = arxiv.Search(
                    query=f'"{keyword}" AND submittedDate: [{yesterday.strftime("%Y%m%d")} TO {datetime.utcnow().strftime("%Y%m%d")}]',
                    max_results=MAX_RESULTS_PER_KEYWORD,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )

                results_found = False
                for result in search.results():
                    all_results[result.entry_id] = {
                        "title": result.title,
                        "summary": result.summary,
                        "authors": [author.name for author in result.authors],
                    }
                    results_found = True
                if results_found:
                    api_success = True
            except Exception as e:
                logging.error(f"An error occurred while searching for keyword '{keyword}': {e}")

        if not api_success and os.path.exists("knowledge_fallback.json"):
            logging.warning("ArXiv API search failed for all keywords. Using fallback knowledge base.")
            with open("knowledge_fallback.json", 'r') as f:
                return json.load(f)

        return list(all_results.values())

    def _find_github_url(self, text):
        """Extracts the first GitHub URL from a block of text."""
        match = re.search(r'https?://github\.com/[\w\-\./]+', text)
        return match.group(0) if match else None

    def _clone_repo(self, url):
        """Clones a GitHub repository into the temporary directory."""
        try:
            repo_name = url.split('/')[-1]
            repo_path = os.path.join(TEMP_CLONE_DIR, repo_name)
            if os.path.exists(repo_path):
                self._cleanup_repo(repo_path) # Clean up previous clone if it exists

            subprocess.run(["git", "clone", "--depth", "1", url, repo_path], check=True, capture_output=True)
            logging.info(f"  -> Successfully cloned repo to {repo_path}")
            return repo_path
        except subprocess.CalledProcessError as e:
            logging.error(f"  -> Failed to clone repo {url}. Error: {e.stderr.decode()}")
            return None

    def _analyze_repo(self, repo_path):
        """
        Analyzes Python files using AST parsing to find nn.Module subclasses
        and potential loss functions.
        """
        snippets = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', errors='ignore') as f:
                            content = f.read()
                            tree = ast.parse(content)

                            models = []
                            losses = []

                            for node in ast.walk(tree):
                                # Find classes that inherit from nn.Module
                                if isinstance(node, ast.ClassDef):
                                    for base in node.bases:
                                        if isinstance(base, ast.Attribute) and \
                                           isinstance(base.value, ast.Name) and \
                                           base.value.id == 'nn' and base.attr == 'Module':
                                            models.append(node.name)
                                        elif isinstance(base, ast.Name) and base.id == 'Module': # Handles `from torch.nn import Module`
                                            models.append(node.name)

                                # Find functions with 'loss' in the name
                                if isinstance(node, ast.FunctionDef) and 'loss' in node.name.lower():
                                    losses.append(node.name)

                            if models or losses:
                                snippets.append({
                                    "file_path": os.path.relpath(file_path, repo_path),
                                    "models": models,
                                    "losses": losses,
                                    "code": content
                                })
                    except (SyntaxError, UnicodeDecodeError) as e:
                        logging.warning(f"  -> Could not parse AST for {file_path}. Skipping. Reason: {e}")
                        continue
        return snippets

    def _cleanup_repo(self, repo_path):
        """Removes a cloned repository directory."""
        if os.path.exists(repo_path):
            # A more robust way to remove directories on different OS
            subprocess.run(['rm', '-rf', repo_path], check=True)
            logging.info(f"  -> Cleaned up repo {repo_path}")

    def _save_to_db(self, new_knowledge):
        """Saves the curated knowledge to a JSON file."""
        db = []
        if os.path.exists(KNOWLEDGE_DB_PATH):
            with open(KNOWLEDGE_DB_PATH, 'r') as f:
                db = json.load(f)

        db.extend(new_knowledge)

        with open(KNOWLEDGE_DB_PATH, 'w') as f:
            json.dump(db, f, indent=2)

if __name__ == '__main__':
    curator = KnowledgeCurator()
    curator.run_curation_cycle()
