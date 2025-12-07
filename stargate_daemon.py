import os
import json
import subprocess
import time
import logging
import importlib

# --- Configuration ---
STATE_FILE = "daemon_state.json"
LOG_LEVEL = logging.INFO
# --- Loop Timings (in seconds) ---
INNER_LOOP_INTERVAL = 60 * 10 # Run a training cycle every 10 minutes
OUTER_LOOP_INTERVAL = 60 * 60 * 12 # Search for new knowledge every 12 hours

# --- Setup Logging ---
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [StargateDaemon] %(message)s',
    filename='stargate.log',
    filemode='a' # Append to the log file
)

class StargateDaemon:
    """
    The orchestrator for the entire self-improving and evolving system.
    Manages the dual loops of gradual improvement and innovative leaps.
    """

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self):
        """Loads the daemon's state from a file, or creates a default state."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        else:
            # Initial state for a fresh start
            return {
                "champion_version": 0,
                "champion_architecture": "baseline", # 'baseline' or a challenger module path
                "champion_performance": {"avg_reward": -float('inf')},
                "last_outer_loop_time": 0,
                "consecutive_failures": 0
            }

    def _save_state(self):
        """Saves the current state to the file."""
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def run_forever(self):
        """The main infinite loop that drives the Stargate system."""
        logging.info("Stargate Daemon activated. Initiating infinite loop of self-improvement.")
        while True:
            try:
                # --- OUTER LOOP: KNOWLEDGE-DRIVEN EVOLUTION ---
                if time.time() - self.state["last_outer_loop_time"] > OUTER_LOOP_INTERVAL:
                    self._run_outer_loop()
                    self.state["last_outer_loop_time"] = time.time()
                    self._save_state()

                # --- INNER LOOP: GRADUAL SELF-IMPROVEMENT ---
                self._run_inner_loop()

                logging.info(f"Inner loop complete. Waiting {INNER_LOOP_INTERVAL / 60:.1f} minutes for next cycle.")
                time.sleep(INNER_LOOP_INTERVAL)

            except Exception as e:
                logging.critical(f"An unrecoverable error occurred in the main loop: {e}", exc_info=True)
                self.state["consecutive_failures"] += 1
                self._save_state()
                if self.state["consecutive_failures"] > 3:
                    logging.critical("Too many consecutive failures. Shutting down to prevent damage.")
                    break
                time.sleep(60) # Wait a minute before retrying

    def _run_inner_loop(self):
        """Runs one cycle of training and evaluation for the current champion."""
        logging.info(f"--- Starting Inner Loop: Gradual Improvement ---")
        logging.info(f"Training current champion (Version {self.state['champion_version']}, Arch: {self.state['champion_architecture']})")
        self._run_subprocess(["python3", "main.py"])

        logging.info("Evaluating improved champion...")
        eval_result = self._run_evaluation(self.state['champion_architecture'])

        if eval_result and eval_result["avg_reward"] > self.state["champion_performance"]["avg_reward"]:
            logging.info(f"Performance improved! New avg reward: {eval_result['avg_reward']:.2f}")
            self.state["champion_performance"] = eval_result
        else:
            logging.info("No performance improvement in this cycle.")

        self.state["champion_version"] += 1
        self._save_state()

    def _run_outer_loop(self):
        """Runs the knowledge acquisition and competitive evolution cycle."""
        logging.info("--- Starting Outer Loop: Knowledge-driven Evolution ---")

        logging.info("Running Knowledge Curator...")
        self._run_subprocess(["python3", "curator.py"])

        logging.info("Running Architect...")
        self._run_subprocess(["python3", "architect.py"])

        challenger_manifest_path = os.path.join("generated_challengers", "challengers.json")
        if not os.path.exists(challenger_manifest_path):
            logging.info("No new challengers were generated. Ending outer loop.")
            return

        with open(challenger_manifest_path, 'r') as f:
            challengers = json.load(f)

        best_challenger = None
        best_challenger_performance = self.state["champion_performance"]["avg_reward"]

        for challenger in challengers:
            logging.info(f"Evaluating challenger: {challenger['challenger_module']}")
            eval_result = self._run_evaluation(challenger['challenger_module'])
            if eval_result and eval_result["avg_reward"] > best_challenger_performance:
                best_challenger_performance = eval_result["avg_reward"]
                best_challenger = challenger

        if best_challenger:
            logging.info(f"**Apotheosis!** New champion has emerged from paper '{best_challenger['origin_paper']}'!")
            logging.info(f"New performance: {best_challenger_performance:.2f} | Old performance: {self.state['champion_performance']['avg_reward']:.2f}")
            self.state["champion_architecture"] = best_challenger["challenger_module"]
            self.state["champion_performance"] = {"avg_reward": best_challenger_performance}
            self.state["champion_version"] = 0
        else:
            logging.info("Champion defended its title. No new champion promoted.")

    def _run_evaluation(self, architecture_module):
        """Runs evaluate.py for a specific architecture and returns the results."""
        result_file = "temp_eval_result.json"
        env = os.environ.copy()
        env["EVAL_MODULE"] = architecture_module

        try:
            self._run_subprocess(["python3", "evaluate.py"], env=env)
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    return json.load(f)
        except subprocess.CalledProcessError as e:
            logging.error(f"Evaluation failed for module {architecture_module}.")
        finally:
            if os.path.exists(result_file):
                os.remove(result_file)
        return None

    def _run_subprocess(self, command, env=None):
        """Helper to run a subprocess and log its output, raising an error on failure."""
        try:
            result = subprocess.run(
                command,
                check=True, capture_output=True, text=True, env=env
            )
            if result.stdout:
                logging.info(f"Subprocess output:\n{result.stdout}")
            if result.stderr:
                logging.warning(f"Subprocess error output:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Subprocess '{' '.join(command)}' failed with exit code {e.returncode}.")
            logging.error(f"STDOUT:\n{e.stdout}")
            logging.error(f"STDERR:\n{e.stderr}")
            raise e

if __name__ == '__main__':
    daemon = StargateDaemon()
    daemon.run_forever()
