"""NetHack experiments (Section 3-5, Appendix B.1, D).

Setting
-------
* The pre-trained policy ``pi_*`` is the 30M/33M-parameter LSTM model of
  Tuyls et al. (2023), trained with behavioral cloning on 115B transitions of
  the AutoAscend agent (Human Monk).  It scores over 5K points.
* ``pi_*`` rarely leaves the first dungeon level, so the **state coverage gap**
  is instantiated as: CLOSE = first level, FAR = subsequent levels.  Because
  ``pi_*`` is a *cloned* expert rather than the expert itself, the problem is
  also an instance of the **imperfect cloning gap** (Figure 4).
* Fine-tuning uses Asynchronous PPO (APPO) (Petrenko et al., 2020) with
  knowledge retention: kickstarting (best), behavioral cloning, EWC.
* ``pi_*`` reaches 10 588 +- 672 points with fine-tuning + KS, doubling the
  previous state of the art (Table 5).

The module implements the architecture (:mod:`fpc.nethack.model`), the APPO
training loop with retention (:mod:`fpc.nethack.appo`), the NLD-AA dataset
plumbing and Fisher estimation (:mod:`fpc.nethack.dataset`) and the evaluations
(:mod:`fpc.nethack.eval`).
"""
