"""Paper2Code line: the vendored DeepCode engine driven as a DeepEvol artifact line.

Layout (see PLAN.md §2): ``config.py`` / ``provider.py`` / ``runner.py`` /
``agent.py`` implement the engine's seams; ``tools/`` exposes the engine's
seven "MCP servers" as in-process tools; ``execution/`` is the remote
execution port; ``intake.py`` / ``gates.py`` / ``phases.py`` / ``driver.py``
are the file-backed run driver. Nothing in ``apps/v2/agent_engine/paper2code``
imports from here.
"""
