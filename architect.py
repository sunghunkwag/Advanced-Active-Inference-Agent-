import os
import json
import ast
import logging

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

    def generate_challengers(self):
        """Main method to generate new experimental code versions."""
        logging.info("Architect starting generation of challenger architectures...")

        if not os.path.exists(KNOWLEDGE_DB_PATH):
            logging.warning("Knowledge database not found. Cannot generate challengers.")
            return

        with open(KNOWLEDGE_DB_PATH, 'r') as f:
            knowledge_db = json.load(f)

        challenger_manifest = []
        challenger_count = 0

        for entry in knowledge_db:
            logging.info(f"Considering knowledge from paper: {entry['paper_title']}")
            for snippet in entry['core_code_snippets']:
                # For this PoC, we focus on replacing the VAE loss function
                if "world_model.py" in snippet["file_path"] or "vae" in snippet["file_path"]:
                    for loss_name in snippet["losses"]:
                        logging.info(f"  -> Found potential loss function '{loss_name}' to integrate.")

                        challenger_metadata = self._try_create_challenger(
                            target_file="world_model.py",
                            target_function="vae_loss_function",
                            foreign_code=snippet["code"],
                            foreign_function_name=loss_name,
                            origin_paper=entry["paper_title"],
                            challenger_id=challenger_count
                        )

                        if challenger_metadata:
                            challenger_manifest.append(challenger_metadata)
                            challenger_count += 1

        self._save_manifest(challenger_manifest)
        logging.info(f"Architect finished. Generated {len(challenger_manifest)} challengers.")

    def _try_create_challenger(self, target_file, target_function, foreign_code, foreign_function_name, origin_paper, challenger_id):
        """Attempts to create a single challenger file by replacing a function."""
        try:
            # 1. Parse foreign code and find the function node
            foreign_ast = ast.parse(foreign_code)
            new_func_node = None
            for node in ast.walk(foreign_ast):
                if isinstance(node, ast.FunctionDef) and node.name == foreign_function_name:
                    new_func_node = node
                    break
            if not new_func_node:
                logging.warning(f"Could not find function '{foreign_function_name}' in snippet AST.")
                return None

            # 2. Parse our own code
            our_code_path = os.path.join(OUR_CODEBASE_ROOT, target_file)
            with open(our_code_path, 'r') as f:
                our_code_content = f.read()
            our_ast = ast.parse(our_code_content)

            # 3. Transform our AST by replacing the function
            transformer = FunctionReplacer(target_function, new_func_node)
            transformed_ast = transformer.visit(our_ast)

            if not transformer.replaced:
                logging.warning(f"Target function '{target_function}' not found in '{target_file}'. Cannot create challenger.")
                return None

            # 4. Collect and merge imports
            our_imports_collector = ImportCollector()
            our_imports_collector.visit(our_ast)
            our_imports = our_imports_collector.get_import_statements_as_strings()

            foreign_imports_collector = ImportCollector()
            foreign_imports_collector.visit(foreign_ast)
            foreign_imports = foreign_imports_collector.get_import_statements_as_strings()

            # Combine imports, removing duplicates
            all_imports = sorted(list(our_imports.union(foreign_imports)))

            # 5. Unparse the transformed AST back to code
            new_code_body = ast.unparse(transformed_ast)

            # 6. Prepend the merged imports to the new code
            final_code = "\n".join(all_imports) + "\n\n" + new_code_body

            # 7. Save the new challenger code
            challenger_filename = f"{os.path.splitext(target_file)[0]}_challenger_{challenger_id}.py"
            challenger_filepath = os.path.join(CHALLENGER_DIR, challenger_filename)
            with open(challenger_filepath, 'w') as f:
                f.write(final_code)

            logging.info(f"  -> Successfully created challenger: {challenger_filepath}")

            return {
                "challenger_id": challenger_id,
                "origin_paper": origin_paper,
                "base_file": target_file,
                "modified_function": target_function,
                "source_function": foreign_function_name,
                "challenger_module": f"{CHALLENGER_DIR}.{os.path.splitext(challenger_filename)[0]}"
            }

        except Exception as e:
            logging.error(f"  -> Failed to create challenger. Reason: {e}")
            return None

    def _save_manifest(self, manifest):
        """Saves the list of generated challengers."""
        with open(CHALLENGER_MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)

if __name__ == '__main__':
    architect = Architect()
    architect.generate_challengers()
