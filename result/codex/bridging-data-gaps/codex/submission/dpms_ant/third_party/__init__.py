"""Third-party, permissively licensed code vendored for reproducibility.

``guided_diffusion`` holds an unmodified copy of the OpenAI *guided-diffusion*
model definitions (MIT licence, see ``guided_diffusion/LICENSE``).  The paper
uses exactly this architecture: the DDPM backbone is "similar to DDPM-PA" and
the domain classifier is "a model pre-trained on the ImageNet dataset provided
by (Dhariwal & Nichol, 2021)" -- i.e. the released
``256x256_diffusion_uncond.pt`` and ``256x256_classifier.pt`` checkpoints.
Vendoring the (tiny, MIT-licensed) model definitions lets us load those
checkpoints bit-exactly without depending on the paper's own repository.

https://github.com/openai/guided-diffusion  (commit of the ``main`` branch)
"""

