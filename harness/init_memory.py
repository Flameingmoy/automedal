"""
AutoMedal — Memory Initializer
=================================
Creates (or resets) the file-based memory artifacts used by the v2 harness:
  - knowledge.md          (curated KB, rewritten by strategist)
  - experiment_queue.md   (pending hypotheses, written by strategist)
  - research_notes.md     (arxiv findings, appended by researcher)
  - journal/              (directory with .gitkeep)

Usage:
    python harness/init_memory.py               # create only missing files
    python harness/init_memory.py --force       # overwrite existing files
"""

import argparse
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWLEDGE_HEADER = """# AutoMedal Knowledge Base
_Last curated: (none)_

## Open questions
- (Strategist will populate this on the first planning pass.)
"""

QUEUE_HEADER = """# Experiment Queue
_Empty — awaiting first Strategist run._
"""

RESEARCH_HEADER = """# Research Notes
"""


def _write(path, content, force):
    if os.path.exists(path) and not force:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def init_memory(project_root=PROJECT_ROOT, force=False):
    """Create (or reset) the file-based memory artifacts.

    Args:
        project_root: Repo root. Defaults to the parent of this file.
        force: When True, overwrite existing files. When False, only create
               files that don't exist yet.

    Returns:
        dict mapping artifact name → "created" | "reset" | "kept".
    """
    results = {}

    knowledge_path = os.path.join(project_root, "knowledge.md")
    existed = os.path.exists(knowledge_path)
    if _write(knowledge_path, KNOWLEDGE_HEADER, force):
        results["knowledge.md"] = "reset" if existed else "created"
    else:
        results["knowledge.md"] = "kept"

    queue_path = os.path.join(project_root, "experiment_queue.md")
    existed = os.path.exists(queue_path)
    if _write(queue_path, QUEUE_HEADER, force):
        results["experiment_queue.md"] = "reset" if existed else "created"
    else:
        results["experiment_queue.md"] = "kept"

    research_path = os.path.join(project_root, "research_notes.md")
    existed = os.path.exists(research_path)
    if _write(research_path, RESEARCH_HEADER, force):
        results["research_notes.md"] = "reset" if existed else "created"
    else:
        results["research_notes.md"] = "kept"

    journal_dir = os.path.join(project_root, "journal")
    os.makedirs(journal_dir, exist_ok=True)
    gitkeep = os.path.join(journal_dir, ".gitkeep")
    if not os.path.exists(gitkeep):
        with open(gitkeep, "w") as f:
            f.write("")
        results["journal/"] = "created"
    else:
        results["journal/"] = "kept"

    return results


def main():
    parser = argparse.ArgumentParser(description="Initialize AutoMedal memory files")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing memory files (used on fresh competition bootstrap)",
    )
    args = parser.parse_args()

    results = init_memory(force=args.force)
    for artifact, state in results.items():
        print(f"  {state:>7}  {artifact}")


if __name__ == "__main__":
    main()
