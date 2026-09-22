"""BBOX-ADAPTER: Lightweight Adapting for Black-Box Large Language Models.

Reference implementation of the method described in
"BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models"
(Sun, Zhuang, Wei, Zhang, Dai; ICML 2024).

The package is organised around the four algorithmic pieces of the paper:

* :mod:`bbox_adapter.adapter`   -- the energy based adapter ``g_theta`` and the
  ranking based NCE loss (Sections 3.1/3.2).
* :mod:`bbox_adapter.inference` -- adapted inference: sentence level beam search
  where the black-box LLM proposes and the adapter evaluates (Section 3.3).
* :mod:`bbox_adapter.online`    -- the online adaptation framework with positive /
  negative sample banks (Section 3.4, Algorithm 1).
* :mod:`bbox_adapter.llm`       -- thin clients for the black-box LLMs (GPT-3.5-Turbo
  and GPT-4 through Azure OpenAI, davinci-002 through the OpenAI API and
  Mixtral-8x7B-v0.1 through HuggingFace).
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
