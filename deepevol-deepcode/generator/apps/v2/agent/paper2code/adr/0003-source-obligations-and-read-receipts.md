# ADR 0003 · Source obligations close the paper-fidelity chain

Date: 2026-09-20  
Status: accepted

## Context

DeepCode's CodeMem keeps generated files coherent across clean-slate coding
rounds. It records each file's purpose, interface and dependencies, but it does
not prove that a formula-bearing file used the paper's original wording. T14
adds `read_paper(section, part)` for that purpose.

A section pointer and a global `paper_reads` count are insufficient evidence. A
model could write a formula file without reading its cited section, read the
wrong section, or read the right section for a different file. PaperBench
Code-Dev is an outcome measure and cannot close this provenance gap.

## Decision

Each independently implementable paper requirement is a **Source obligation**:
a formula, loss, update rule, sampling distribution, hyper-parameter group,
architecture detail or experiment setting. One sentence may support several
obligations, and several files may carry the same obligation.

The planner emits structured obligations with a stable id, paper digest,
section and heading, raw quote, quote digest, section digest and offsets when
available. Blueprint files reference obligations with `source_ids`.

Reading remains on demand, but writing is strict for files that carry source
obligations. The `write_file` wrapper checks the current file round's receipts;
if any required source has not been read, it returns a bounded
`SOURCE_READ_REQUIRED` result and does not write. A normal file round is:

```text
read unique cited sections → optional reference lookup → write_file
```

The same section is read once per file round after deduplication. Long sections
are paged. Repair rereads only the source obligations of changed files;
unchanged files may reuse their receipts. The final Fidelity gate repeats the
checks against paper, section and quote digests.

Historical runs without structured source obligations are discarded from the
strict T14 comparison rather than silently upgraded.

CodeMem remains the code-continuity layer. Source obligations and receipts are
the paper-authority layer. Neither replaces the other.

PaperBench remains the external code-quality measure. Its paired protocol must
hold the paper set, model, thinking setting, token and wall-clock budgets,
execution policy and judge version fixed; it does not replace the local
Fidelity gate.

## Alternatives rejected

1. Prompt-only compliance: a nonzero read count does not identify the file or
   prove the section was the one the file needed.
2. Soft writes followed by ordinary compliance rounds: this lets an ungrounded
   tree exist and adds avoidable phases. The write boundary gives immediate,
   bounded feedback instead.
3. Full-paper injection for every file: it raises context cost and weakens
   locality. Section-scoped, paged reads preserve the source without recreating
   the full-document prompt.
4. PaperBench as trace validation: it measures reviewed-code outcome, not source
   provenance.

## Consequences

- Formula-bearing files pay one read per unique cited section, with paging for
  long sections; ordinary glue files pay no paper-read cost.
- Missing reads fail at the write boundary rather than becoming a later silent
  drift.
- The implementation report must retain per-file Read receipts, not only a
  global `paper_reads` counter.
- Quote matching must preserve raw source text and support common LaTeX display
  environments; equation counts and `Source` counts alone are insufficient.
- Quality reports include mechanical fidelity metrics and the paired PaperBench
  result, keeping provenance benefit separate from benchmark outcome.
