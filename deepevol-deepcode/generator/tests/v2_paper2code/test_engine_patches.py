"""C2: the validation-repo patch subset applied to the vendored engine.

Covers the engine-side hunks (VENDOR.md entries 2, 4, 6). The denylist,
repeat-fetch ledger and tool-name sanitizing live in the line's tools and
are tested in test_tools_registry.py / test_fetch_policy.py.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from apps.v2.agent_engine.paper2code.seams import compat
from apps.v2.agent_engine.paper2code.seams.config import (
    AgentDefaults,
    AgentsConfig,
    KernelConfig,
    KernelRuntime,
    use_runtime,
)
from apps.v2.agent_engine.paper2code.seams.agent_runtime import Tool, ToolRegistry
from apps.v2.agent_engine.paper2code.workflows import agent_orchestration_engine as engine
from apps.v2.agent_engine.paper2code.workflows import code_implementation_workflow as impl


class _Named(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"test tool {self._name} with a long enough description"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        return "ok"


def test_tool_filter_matches_sanitized_server_prefix() -> None:
    registry = ToolRegistry()
    for name in (
        "mcp_github_downloader_git_clone",
        "mcp_github_downloader_download_github_repo",
        "mcp_filesystem_read_text_file",
    ):
        registry.register(_Named(name))

    kept = compat.apply_tool_filter(
        registry,
        {"github-downloader": {"git_clone"}},
        agent_name="t",
        agent_server_names=["github-downloader"],
    )
    assert kept.tool_names == ["mcp_github_downloader_git_clone"]


def test_acquisition_fails_fast_when_nothing_was_cloned(tmp_path, monkeypatch) -> None:
    asyncio.run(_acquisition_scenario(tmp_path, monkeypatch))


async def _acquisition_scenario(tmp_path, monkeypatch) -> None:
    async def _prose_only(search_result: str, paper_dir: str, logger) -> str:
        return "I would clone the repositories now."

    monkeypatch.setattr(engine, "github_repo_download", _prose_only)
    monkeypatch.setattr(engine.asyncio, "sleep", _no_sleep)
    paper_dir = tmp_path / "paper_x"
    paper_dir.mkdir()
    dir_info = {"paper_dir": str(paper_dir), "download_path": str(paper_dir / "github_download.txt")}

    with pytest.raises(RuntimeError, match="Code base directory was not created"):
        await engine.automate_repository_acquisition_agent("report", dir_info, _Logger())

    (paper_dir / "code_base").mkdir()
    with pytest.raises(RuntimeError, match="produced no repositories"):
        await engine.automate_repository_acquisition_agent("report", dir_info, _Logger())

    (paper_dir / "code_base" / "some_repo").mkdir()
    await engine.automate_repository_acquisition_agent("report", dir_info, _Logger())
    assert (paper_dir / "github_download.txt").read_text() == "I would clone the repositories now."


async def _no_sleep(_: float) -> None:
    return None


class _Logger:
    def info(self, *a: Any, **k: Any) -> None: ...
    def warning(self, *a: Any, **k: Any) -> None: ...
    def error(self, *a: Any, **k: Any) -> None: ...


def test_wall_clock_and_stall_threshold_read_env_at_call_time(monkeypatch) -> None:
    monkeypatch.delenv("DEEPCODE_MAX_WALL_SECONDS", raising=False)
    assert impl._max_wall_seconds() == impl._MAX_WALL_SECONDS
    monkeypatch.setenv("DEEPCODE_MAX_WALL_SECONDS", "21600")
    assert impl._max_wall_seconds() == 21600.0

    config = KernelConfig(agents=AgentsConfig(defaults=AgentDefaults(model="test-model")))
    with use_runtime(KernelRuntime(config)):
        monkeypatch.delenv("DEEPCODE_STALL_THRESHOLD", raising=False)
        assert impl.CodeImplementationWorkflow().loop_detector.stall_threshold == 300
        monkeypatch.setenv("DEEPCODE_STALL_THRESHOLD", "7200")
        assert impl.CodeImplementationWorkflow().loop_detector.stall_threshold == 7200


def test_indexer_token_knobs_have_upstream_defaults() -> None:
    src = open(os.path.join(os.path.dirname(engine.__file__), "..", "tools", "code_indexer.py")).read()
    for knob, default in (
        ("DEEPCODE_PREFILTER_MAX_TOKENS", "2000"),
        ("DEEPCODE_ANALYSIS_MAX_TOKENS", "1000"),
        ("DEEPCODE_RELATIONSHIP_MAX_TOKENS", "1500"),
    ):
        assert f'os.environ.get("{knob}", "{default}")' in src


# --- PLAN-3 item 1: the pre-filter must survive large repositories and must not carry a domain prior


class _IndexerProvider:
    """Scripted provider for CodeIndexer._call_llm: a list of (finish_reason, content) replies."""

    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.generation = type("G", (), {"max_tokens": 32000, "temperature": 0.1, "reasoning_effort": None})()

    def get_default_model(self) -> str:
        return "DeepSeek-V4-Flash-Vision-Exp"

    async def chat_with_retry(self, messages, tools=None, model=None, max_tokens=None, temperature=None, reasoning_effort=None, tool_choice=None, retry_mode="standard", on_retry_wait=None):
        from apps.v2.agent_engine.paper2code.seams.llm_runtime import LLMResponse

        self.calls += 1
        finish, content = self.replies.pop(0)
        return LLMResponse(content=content, finish_reason=finish)


def _indexer(tmp_path, provider):
    from apps.v2.agent_engine.paper2code.tools.code_indexer import CodeIndexer

    config = KernelConfig(agents=AgentsConfig(defaults=AgentDefaults(model="test-model")))
    with use_runtime(KernelRuntime(config)):
        indexer = CodeIndexer(code_base_path=str(tmp_path), target_structure="proj/\n├── main.py\n└── model.py\n", output_dir=str(tmp_path / "idx"))
    indexer.retry_delay = 0

    async def _init():
        return provider, None

    indexer._initialize_llm_client = _init  # type: ignore[method-assign]
    return indexer


def test_prefilter_prompt_is_domain_neutral_and_asks_for_paths_only(tmp_path) -> None:
    import inspect

    from apps.v2.agent_engine.paper2code.tools import code_indexer

    src = inspect.getsource(code_indexer.CodeIndexer.pre_filter_files)
    prompt = src.split('filter_prompt = f"""', 1)[1].split('"""', 1)[0]
    for leftover in ("recommendation systems", "graph neural networks", "diffusion models", "GCN", "relevance_reason", "expected_contribution"):
        assert leftover not in prompt
    assert '"confidence": 0.0-1.0' in prompt


def test_prefilter_parses_three_hundred_path_records(tmp_path) -> None:
    import json

    records = [{"file_path": f"pkg/mod_{i}.py", "confidence": 0.9 if i % 3 else 0.1} for i in range(300)]
    reply = json.dumps({"relevant_files": records, "summary": {"relevant_files_count": "120", "filtering_strategy": "x"}})
    assert len(reply) < 20_000  # paths-only stays far under a 32k-token budget
    provider = _IndexerProvider([("stop", reply)])
    indexer = _indexer(tmp_path, provider)
    selected = asyncio.run(indexer.pre_filter_files(tmp_path, "tree"))
    assert len(selected) == 200  # confidence > 0.3 only; the model's own "120" is not trusted
    assert provider.calls == 1


def test_truncated_prefilter_reply_is_retried_then_falls_back(tmp_path) -> None:
    truncated = '{"relevant_files": [{"file_path": "a.py", "confidence": 0.9}, {"file_path": "b.py", "conf'
    provider = _IndexerProvider([("length", truncated), ("length", truncated), ("length", truncated)])
    indexer = _indexer(tmp_path, provider)
    indexer.max_retries = 3
    selected = asyncio.run(indexer.pre_filter_files(tmp_path, "tree"))
    assert selected == []  # the documented fallback (analyse every file), reached only after the retries
    assert provider.calls == 3


def test_truncated_reply_recovers_on_retry(tmp_path) -> None:
    import json

    good = json.dumps({"relevant_files": [{"file_path": "a.py", "confidence": 0.9}], "summary": {}})
    provider = _IndexerProvider([("length", '{"relevant_files": [{"file_p'), ("stop", good)])
    indexer = _indexer(tmp_path, provider)
    indexer.max_retries = 3
    assert asyncio.run(indexer.pre_filter_files(tmp_path, "tree")) == ["a.py"]
    assert provider.calls == 2


# --- PLAN-3 items 7 / 7b (VENDOR 11): planning fan-out behind a switch; planner segment budget from the context window;
# the appendix as its own segment


SAPG_LIKE = """\\section*{1. Introduction}

We study policy gradients at scale. See Algorithm 1.

\\section*{4. Method}

The update rule is $\\theta \\leftarrow \\theta + \\alpha \\nabla J$ with the procedure in Algorithm 1.

\\section*{5. Experiments}

We run five tasks and report the results in Table 1.

\\section*{7. Conclusion}

We conclude that splitting and aggregating policy gradients scales to many environments; the follower policies
inherit the leader's data and the diversity terms keep exploration alive across all of the tasks we tried here.

\\section*{References}

Agarwal, A., Kumar, A., Malik, J., and Pathak, D. Legged locomotion in challenging terrains. In CoRL, 2022.

Watkins, C. and Dayan, P. Q-learning. Machine Learning, 8:279-292, 1992.

\\section*{A. Task and Environment Details}

Hard tasks are based on the Allegro-Kuka environments with a 16-dof hand.

\\section*{B. Training hyperparameters}

| Parameter | Value |
| horizon length | 16 |
| batch size | 24576 |
| gamma | 0.99 |
"""


def test_segmenter_gives_the_appendix_its_own_segment_and_ranks_it_for_planning() -> None:
    from apps.v2.agent_engine.paper2code.tools.document_segmentation_server import segmenter

    segments = segmenter._segment_academic_paper(SAPG_LIKE)
    by_type = {seg.content_type: seg for seg in segments}
    assert "appendix" in by_type, [(s.title, s.content_type) for s in segments]
    appendix = by_type["appendix"]
    assert appendix.title.startswith("A. Task and Environment Details")
    assert "24576" in appendix.content  # B follows A inside the appendix segment
    assert "Watkins" not in appendix.content
    references = next(seg for seg in segments if seg.title.lower().startswith("reference"))
    assert "Watkins" in references.content
    assert "24576" not in references.content  # no longer lumped into the references
    assert appendix.relevance_scores["code_planning"] >= 0.85
    assert appendix.relevance_scores["code_planning"] > references.relevance_scores["code_planning"]
    # the markdown-heading path classifies the same titles
    assert segmenter._classify_content_type("Appendix B: Hyperparameters", "") == "appendix"
    assert segmenter._classify_content_type("B.1. Shadow Hand", "") == "appendix"
    assert segmenter._classify_content_type("References", "") == "references"
    assert segmenter._classify_content_type("A method", "") == "methodology"


def _index(tmp_path, sizes: list[tuple[str, float, int]]):
    import json

    paper_dir = tmp_path / "paper"
    (paper_dir / "document_segments").mkdir(parents=True)
    segments = [
        {"id": f"s{i}", "title": title, "content": f"{title}: " + ("x" * (chars - len(title) - 2)), "content_type": "general",
         "keywords": [], "char_count": chars, "relevance_scores": {"code_planning": score}, "section_path": title}
        for i, (title, score, chars) in enumerate(sizes)
    ]
    (paper_dir / "document_segments" / "document_index.json").write_text(
        json.dumps({"document_type": "research_paper", "segmentation_strategy": "test", "total_segments": len(segments), "segments": segments})
    )
    return paper_dir


def test_planner_segment_budget_follows_the_context_window(tmp_path, monkeypatch) -> None:
    paper_dir = _index(tmp_path, [("Intro", 0.5, 10_000), ("Method", 0.9, 10_000), ("Results", 0.7, 10_000), ("Appendix", 0.95, 10_000)])

    # switch off: upstream's 8 segments / 24 000 chars, relevance order
    monkeypatch.delenv("DEEPCODE_PLANNER_CONTEXT_WINDOW", raising=False)
    assert engine._planner_segment_budget_chars(32768) is None
    upstream = engine._load_document_segments_context(str(paper_dir))
    order = [line.split("title: ")[1] for line in upstream.splitlines() if line.startswith("title: ")]
    assert order == ["Appendix", "Method"]  # 24k of 40k: the two best, Results (30k) does not fit
    assert upstream == engine._load_document_segments_context(str(paper_dir), budget_chars=None)

    # the whole paper fits: every segment, document order
    monkeypatch.setenv("DEEPCODE_PLANNER_CONTEXT_WINDOW", "1000000")
    budget = engine._planner_segment_budget_chars(32768)
    assert budget == int((1_000_000 - 32768 - 12_000) * 0.85 * 3.0)
    full = engine._load_document_segments_context(str(paper_dir), budget_chars=budget)
    assert [line.split("title: ")[1] for line in full.splitlines() if line.startswith("title: ")] == ["Intro", "Method", "Results", "Appendix"]

    # it does not fit: relevance ranking truncates, without the 8-segment cap
    truncated = engine._load_document_segments_context(str(paper_dir), budget_chars=31_000)
    assert [line.split("title: ")[1] for line in truncated.splitlines() if line.startswith("title: ")] == ["Appendix", "Method", "Results"]
    many = _index(tmp_path / "many", [(f"S{i:02d}", 0.5, 1_000) for i in range(12)])
    assert len([line for line in engine._load_document_segments_context(str(many)).splitlines() if line.startswith("title: ")]) == 8
    assert len([line for line in engine._load_document_segments_context(str(many), budget_chars=11_000).splitlines() if line.startswith("title: ")]) == 11

    monkeypatch.setenv("DEEPCODE_PLANNER_CONTEXT_WINDOW", "abc")
    assert engine._planner_segment_budget_chars(32768) is None
    monkeypatch.setenv("DEEPCODE_PLANNER_CONTEXT_WINDOW", "20000")
    assert engine._planner_segment_budget_chars(32768) is None  # no room beside max_tokens


def test_planning_fanout_switch_defaults_off(monkeypatch) -> None:
    monkeypatch.delenv("DEEPCODE_PLANNING_FANOUT", raising=False)
    assert engine._planning_fanout_enabled() is False
    monkeypatch.setenv("DEEPCODE_PLANNING_FANOUT", "0")
    assert engine._planning_fanout_enabled() is False
    monkeypatch.setenv("DEEPCODE_PLANNING_FANOUT", "1")
    assert engine._planning_fanout_enabled() is True


def test_fanout_appends_worker_outputs_to_the_planner_message() -> None:
    from apps.v2.agent_engine.paper2code.seams import compat as seams_compat
    from apps.v2.agent_engine.paper2code.seams.agent_runtime import AgentRunResult

    seen: list[tuple[str, str]] = []

    class FakeLLM:
        def __init__(self, agent):
            self.agent = agent

        async def generate(self, *, message, request_params):
            seen.append((self.agent.name, message))
            return AgentRunResult(final_content=f"{self.agent.name} says hi", messages=[], tools_used=[], usage={"total_tokens": 1}, stop_reason="completed")

    class FakeAgent:
        def __init__(self, name):
            self.name = name

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    async def fake_attach(agent, phase="planning"):
        return FakeLLM(agent)

    original = engine.attach_workflow_llm
    engine.attach_workflow_llm = fake_attach
    try:
        result = asyncio.run(
            engine._generate_plan_with_fanout(
                FakeAgent("CodePlannerAgent"), [FakeAgent("ConceptAnalysisAgent"), FakeAgent("AlgorithmAnalysisAgent")],
                message="PAPER", request_params=None, timeout_s=1, logger=type("L", (), {"info": lambda *a, **k: None, "warning": lambda *a, **k: None})(),
            )
        )
    finally:
        engine.attach_workflow_llm = original
    assert result.final_content == "CodePlannerAgent says hi"
    assert [name for name, _ in seen] == ["ConceptAnalysisAgent", "AlgorithmAnalysisAgent", "CodePlannerAgent"]
    assert seen[0][1] == "PAPER"
    planner_message = seen[2][1]
    assert planner_message.startswith("PAPER\n\n---\n# Worker outputs\n## ConceptAnalysisAgent\nConceptAnalysisAgent says hi\n\n## AlgorithmAnalysisAgent\n")
    assert seams_compat is not None


def test_implement_max_tokens_reads_the_env_with_upstream_default(monkeypatch) -> None:
    monkeypatch.delenv("DEEPCODE_IMPLEMENT_MAX_TOKENS", raising=False)
    assert impl._implement_max_tokens() == 8192
    monkeypatch.setenv("DEEPCODE_IMPLEMENT_MAX_TOKENS", "32768")
    assert impl._implement_max_tokens() == 32768
    monkeypatch.setenv("DEEPCODE_IMPLEMENT_MAX_TOKENS", "x")
    assert impl._implement_max_tokens() == 8192
    from apps.v2.agent.paper2code.config import ENV_DEFAULTS

    assert ENV_DEFAULTS["DEEPCODE_IMPLEMENT_MAX_TOKENS"] == "65536"  # thinking shares the budget (fre-t17)


# --- VENDOR 13: paper fidelity (formulas verbatim in the plan, read_paper in the coding loop) -----------------------

_PAPER = """\\title{
SAPG: Split and Aggregate Policy Gradients
}

\\section*{1. Introduction}

Prose about the problem.

\\section*{4. Split and Aggregate Policy Gradients}

Overview of the method.

\\subsection*{4.1. Aggregating data using off-policy updates}

The on-policy loss is
\\[
L_{on}(\\pi_\\theta)=\\mathbb{E}[\\min(r_t A_t, \\operatorname{clip}(r_t, 1-\\epsilon, 1+\\epsilon) A_t)]
\\]
and the combined loss is
\\[
L(\\pi_i)=L_{on}(\\pi_i)+\\lambda \\cdot L_{off}(\\pi_i; \\mathcal{X})
\\]
with $\\lambda = 1$ for symmetric aggregation.

\\subsection*{4.2. Symmetric aggregation}

Set $\\lambda$ to 1.

\\section*{References}

[1] Something.

\\section*{A. Hyperparameters}

$$
\\gamma = 0.99
$$
"""


def test_paper_index_resolves_section_pointers_and_lists_equations() -> None:
    from apps.v2.agent_engine.paper2code.workflows.paper_readback import PaperIndex

    idx = PaperIndex.from_markdown(_PAPER)
    numbers = [s.number for s in idx.sections]
    assert numbers == ["", "1", "4", "4.1", "4.2", "", "A"]  # front matter, then the headings in order
    s41 = idx.find(section="4.1")[0]
    assert s41.title == "Aggregating data using off-policy updates"
    assert len(s41.equations) == 2
    assert s41.equations[1].startswith("L(\\pi_i)=L_{on}(\\pi_i)+\\lambda")
    assert idx.find(section="§4.1") == [s41]
    assert idx.find(section="Section 4.1.") == [s41]
    assert idx.find(section="4.1. Aggregating data") == [s41]
    assert idx.find(section="symmetric aggregation")[0].number == "4.2"
    assert [s.number for s in idx.find(section="4")] == ["4"]  # exact number wins over its children
    assert idx.find(section="A")[0].equations == ["\\gamma = 0.99"]
    assert idx.find(query="combined loss lambda")[0].number == "4.1"
    assert idx.find(section="9.9") == []
    rendered = idx.render(s41)
    assert "§4.1 Aggregating data using off-policy updates" in rendered
    assert "Display equations in this section" in rendered
    assert "(2) L(\\pi_i)" in rendered
    long = PaperIndex.from_markdown("# One\n" + ("word " * 2000))
    first = long.render(long.sections[0], part=1, part_chars=1000)
    assert "part 1/" in first
    assert "part=2" in first


def test_read_paper_tool_pages_counts_and_falls_back_to_the_outline() -> None:
    import asyncio

    from apps.v2.agent_engine.paper2code.workflows.paper_readback import PaperIndex, ReadPaperTool

    calls: list[dict] = []
    tool = ReadPaperTool(PaperIndex.from_markdown(_PAPER), on_call=lambda args, text: calls.append(args))
    assert tool.name == "read_paper"
    assert tool.read_only
    assert "section" in tool.parameters["properties"]
    text = asyncio.run(tool.execute(section="4.1"))
    assert "L_{on}(\\pi_\\theta)" in text
    outline = asyncio.run(tool.execute())
    assert outline.startswith("Paper outline")
    assert "§4.2 Symmetric aggregation" in outline
    missing = asyncio.run(tool.execute(section="7.3"))
    assert missing.startswith("No section matches '7.3'")
    assert tool.calls == 3
    assert len(calls) == 3


def test_paper_fidelity_switch_gates_every_prompt_and_the_tool(monkeypatch, tmp_path) -> None:
    from apps.v2.agent_engine.paper2code.workflows import code_implementation_workflow as ciw
    from apps.v2.agent_engine.paper2code.workflows import paper_readback as pr
    from apps.v2.agent_engine.paper2code.workflows.agent_orchestration_engine import _planner_instruction

    monkeypatch.delenv(pr.FIDELITY_ENV, raising=False)
    base = "PLANNER PROMPT"
    assert _planner_instruction(base, use_segmentation=False) == base  # upstream byte-identical when unset
    assert "PAPER FIDELITY" not in ciw._indexed_success_guidance(3)
    assert pr.load_paper_index(tmp_path) is None
    monkeypatch.setenv(pr.FIDELITY_ENV, "1")
    assert _planner_instruction(base, use_segmentation=False) == base + pr.PLANNING_FIDELITY_ADDENDUM
    assert "Source: §" in _planner_instruction(base, use_segmentation=True)  # section pointers (ADR 0004)
    assert "read_paper(file_path=" in ciw._indexed_success_guidance(3)
    (tmp_path / "paper.md").write_text(_PAPER, encoding="utf-8")
    idx = pr.load_paper_index(tmp_path)
    assert idx is not None
    assert len(idx.sections) == 7


def test_plan_fidelity_stats_counts_equations_and_resolves_source_refs() -> None:
    from apps.v2.agent_engine.paper2code.workflows.paper_readback import plan_fidelity_stats

    plan = (
        "implementation_components: |\n"
        "  Loss (Source: §4.1 Aggregating data): \\[ L(\\pi_i)=L_{on}(\\pi_i)+\\lambda L_{off} \\]\n"
        "  lambda = 1 (Source: §4.2)\n  gamma $\\gamma = 0.99$ (Source: §A)\n  invented (Source: §9.9)\n"
    )
    stats = plan_fidelity_stats(plan, _PAPER)
    assert stats["plan_display_equations"] == 1
    assert stats["plan_inline_math"] == 1
    assert stats["source_refs"] == 4
    assert stats["source_refs_resolved"] == 3
    assert stats["paper_display_equations"] == 3
    assert stats["paper_sections"] == 5
    assert plan_fidelity_stats("nothing here") == {"plan_display_equations": 0, "plan_inline_math": 0, "source_refs": 0}
