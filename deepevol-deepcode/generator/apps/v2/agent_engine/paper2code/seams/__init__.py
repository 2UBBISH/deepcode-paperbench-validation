"""The seams: every symbol the business layer needs from outside itself.

Each module says at the top which of its names are REAL (copied, working
code) and which are CONTRACT (raise ``NotImplementedError`` until the
integrator provides them). The contract set, in full:

    seams.config.KernelRuntime.provider_for
    seams.llm_runtime.LLMProvider.chat_with_retry
    seams.llm_runtime.LLMProvider.get_default_model
    seams.agent_runtime.AgentRunner.run
    seams.compat.Agent.__aenter__
    seams.compat.Agent.__aexit__
    seams.compat.Agent.attach_llm
    seams.compat.AugmentedLLM.generate
    seams.harness.build_permission_engine
    seams.harness.TerminalApprover.__call__

Ten methods. ``docs/INTEGRATION.md`` describes each one's obligations and
gives a reference loop for ``AgentRunner.run``.

The business layer imports these modules directly (``from seams.compat
import Agent``); this package file only documents the set.
"""
