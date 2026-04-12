"""
AutoMedal — Template Renderer
===============================
Renders AGENTS.md and program.md from Jinja2 templates using
configs/competition.yaml as the data source.

Usage:
    python scout/render.py
    python -m scout.render
"""

import os
import sys
import jinja2

# Ensure project root is on sys.path for config_loader import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config_loader import load_config

TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# Template → output mapping
RENDER_TARGETS = {
    "AGENTS.md.j2": "AGENTS.md",
    "program.md.j2": os.path.join("agent", "program.md"),
}


def render_templates(config=None):
    """Render all Jinja2 templates using competition config.

    Args:
        config: Optional config dict. Loads from YAML if not provided.

    Returns:
        List of rendered output file paths.
    """
    if config is None:
        config = load_config()

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    rendered = []
    for template_name, output_name in RENDER_TARGETS.items():
        template_path = os.path.join(TEMPLATES_DIR, template_name)
        if not os.path.exists(template_path):
            print(f"  [SKIP] Template not found: {template_path}")
            continue

        template = env.get_template(template_name)
        output = template.render(**config)
        output_path = os.path.join(PROJECT_ROOT, output_name)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"  Rendered: {template_name} -> {output_name}")
        rendered.append(output_path)

    return rendered


def render_prepare_starter(config=None):
    """Render the starter prepare.py from template.

    Args:
        config: Optional config dict. Loads from YAML if not provided.

    Returns:
        Path to rendered prepare.py, or None if skipped.
    """
    if config is None:
        config = load_config()

    template_name = "prepare_starter.py.j2"
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    output_path = os.path.join(PROJECT_ROOT, "agent", "prepare.py")

    if not os.path.exists(template_path):
        print(f"  [SKIP] Template not found: {template_path}")
        return None

    if os.path.exists(output_path):
        print(f"  [SKIP] agent/prepare.py already exists — not overwriting.")
        print(f"         Delete it first if you want a fresh starter.")
        return None

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )

    template = env.get_template(template_name)
    output = template.render(**config)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"  Rendered: {template_name} -> agent/prepare.py")
    return output_path


def main():
    print("=" * 60)
    print("AutoMedal — Rendering templates")
    print("=" * 60)

    config = load_config()
    comp = config.get("competition", {})
    print(f"  Competition: {comp.get('slug', '?')} — {comp.get('subtitle', '?')}")
    print()

    rendered = render_templates(config)
    print(f"\n  Rendered {len(rendered)} template(s).")
    print("Done.")


if __name__ == "__main__":
    main()
