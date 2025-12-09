import os
import json
import ast
import logging
import anthropic
import subprocess

# --- Configuration ---
KNOWLEDGE_DB_PATH = "knowledge.json"
OUR_CODEBASE_ROOT = "."
CHALLENGER_DIR = "generated_challengers"
CHALLENGER_MANIFEST_PATH = os.path.join(CHALLENGER_DIR, "challengers.json")
LOG_LEVEL = logging.INFO

# --- Setup Logging ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

class FunctionReplacer(ast.NodeTransformer):
    """
    An AST transformer that replaces a target function definition
    with a new function definition node.
    """
    def __init__(self, target_function_name, new_function_node):
        self.target_function_name = target_function_name
        self.new_function_node = new_function_node
        self.replaced = False

    def visit_FunctionDef(self, node):
        if node.name == self.target_function_name:
            logging.info(f"Found target function '{node.name}'. Replacing it.")
            self.replaced = True
            # Replace the entire function definition node
            return self.new_function_node
        return node

class ImportCollector(ast.NodeVisitor):
    """
    An AST visitor that collects all import and from-import statements.
    """
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        self.imports.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.imports.append(node)
        self.generic_visit(node)

    def get_import_statements_as_strings(self):
        """Returns a set of unique import statements as strings."""
        return {ast.unparse(imp) for imp in self.imports}

class Architect:
    """
    Dynamically generates new 'challenger' versions of the agent's code
    by integrating ideas from the curated knowledge base.
    """
    def __init__(self):
        if not os.path.exists(CHALLENGER_DIR):
            os.makedirs(CHALLENGER_DIR)

        self.llm_client = None
        if "ANTHROPIC_API_KEY" in os.environ:
             self.llm_client = anthropic.Anthropic()
             logging.info("Anthropic client initialized for Architect.")
        else:
            logging.warning("ANTHROPIC_API_KEY env var not found. LLM-based code generation will be disabled.")

    def generate_challengers(self):
        """Main method to generate new experimental code versions using LLM."""
        logging.info("Architect starting generation of challenger architectures...")

        if not self.llm_client:
            logging.error("LLM client not initialized. Cannot generate challengers.")
            return

        if not os.path.exists(KNOWLEDGE_DB_PATH):
            logging.warning("Knowledge database not found.")
            return

        with open(KNOWLEDGE_DB_PATH, 'r') as f:
            knowledge_db = json.load(f)

        challenger_manifest = []
        challenger_count = 0

        for entry in knowledge_db:
            if not entry.get("implementation_plan"):
                continue

            logging.info(f"Considering knowledge from paper: {entry['paper_title']}")

            # For this PoC, we will try to generate a new VAE model
            target_file = "world_model.py"
            with open(target_file, 'r') as f:
                our_original_code = f.read()

            challenger_metadata = self._try_create_challenger_with_llm(
                original_code=our_original_code,
                implementation_plan=entry["implementation_plan"],
                origin_paper=entry["paper_title"],
                challenger_id=challenger_count
            )

            if challenger_metadata:
                challenger_manifest.append(challenger_metadata)
                challenger_count += 1

        self._save_manifest(challenger_manifest)
        logging.info(f"Architect finished. Generated {len(challenger_manifest)} new challengers.")

    def _try_create_challenger_with_llm(self, original_code, implementation_plan, origin_paper, challenger_id, max_retries=3):
        """
        Attempts to create and verify a challenger file using an LLM with a self-correction loop.
        """
        prompt = f"""
Based on the following implementation plan and our existing code, generate a new, complete Python script for a 'challenger' version.
The new script should integrate the core ideas from the plan into our existing architecture.

**Implementation Plan:**
- **Core Idea:** {implementation_plan['core_idea']}
- **Key Components:** {', '.join(implementation_plan['key_components'])}
- **Proposed Architecture:** {implementation_plan['proposed_architecture']}

**Our Existing Code (`world_model.py`):**
```python
{original_code}
```

Your task is to generate a new, complete Python script that:
1.  Keeps the existing classes (`VAE`, `TransitionModel`, `ContextInferenceEngine`).
2.  Modifies the `VAE` class or its loss function `vae_loss_function` to reflect the core idea.
3.  Ensures the new script is syntactically correct and complete.
4.  Includes all necessary imports.

Provide only the complete, raw Python code for the new file. Do not include any explanations or markdown formatting.
"""

        current_code = ""
        error_history = []

        for i in range(max_retries):
            logging.info(f"  -> Attempt {i+1}/{max_retries} to generate and verify challenger {challenger_id}...")

            try:
                message = self.llm_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                current_code = message.content[0].text

                # Clean up potential markdown code blocks
                if current_code.startswith("```python"):
                    current_code = current_code[9:]
                if current_code.endswith("```"):
                    current_code = current_code[:-3]

                is_valid, error_output = self._verify_challenger_code(current_code, challenger_id)

                if is_valid:
                    challenger_filename = f"world_model_challenger_{challenger_id}.py"
                    challenger_filepath = os.path.join(CHALLENGER_DIR, challenger_filename)
                    with open(challenger_filepath, 'w') as f:
                        f.write(current_code)

                    logging.info(f"  -> Successfully created and verified challenger: {challenger_filepath}")
                    return {
                        "challenger_id": challenger_id,
                        "origin_paper": origin_paper,
                        "challenger_module": f"{CHALLENGER_DIR}.{os.path.splitext(challenger_filename)[0]}"
                    }
                else:
                    error_history.append(error_output)
                    prompt += f"\n\nThe previous attempt failed with this compilation error:\n{error_output}\nPlease fix the code and provide a new, complete script."

            except Exception as e:
                logging.error(f"  -> An unexpected error occurred during generation attempt {i+1}: {e}")
                error_history.append(str(e))

        logging.error(f"  -> Failed to create a valid challenger for paper '{origin_paper}' after {max_retries} attempts.")
        logging.error(f"  -> Error history: {error_history}")
        return None

    def _verify_challenger_code(self, code, challenger_id):
        """Verify the generated code snippet by trying to compile it."""
        temp_filepath = os.path.join(CHALLENGER_DIR, f"temp_verify_{challenger_id}.py")
        with open(temp_filepath, "w") as f:
            f.write(code)

        try:
            # Use py_compile to check for syntax errors without executing
            result = subprocess.run(
                ["python", "-m", "py_compile", temp_filepath],
                capture_output=True, text=True, check=True
            )
            logging.info(f"  -> Verification successful for challenger {challenger_id}.")
            return True, None
        except subprocess.CalledProcessError as e:
            logging.warning(f"  -> Verification failed for challenger {challenger_id}.")
            return False, e.stderr # Return the error message
        finally:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    def _save_manifest(self, manifest):
        """Saves the list of generated challengers."""
        with open(CHALLENGER_MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    architect = Architect()
    architect.generate_challengers()
