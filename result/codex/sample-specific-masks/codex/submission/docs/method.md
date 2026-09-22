# Method: sample-specific multi-channel masks (SMM)

Paper reference: Section 3 (framework, mask generator, patch-wise interpolation,
learning strategy) and Section 4 (theory).

## Input transformation

```
f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)
```

* `r` – resizing of the target image to the pre-trained input size
  (224x224 for ResNet-18/50, 384x384 for ViT-B/32); in the paper's code this is
  also the data transform of the addendum (`Resize((imgsize, imgsize))`).
* `delta` – the noise pattern that is *shared* by all samples, initialised to
  zeros and optimised together with the generator (Algorithm 1).
* `f_mask` – a lightweight CNN followed by the patch-wise interpolation module.
  It produces a **three-channel** mask of the same spatial size as the image,
  and it varies from sample to sample, which is the novelty over the
  pre-defined shared binary mask `M` of previous VR methods.

`smm/reprogram.py` implements this for `variant="smm"`; the same module also
implements the shared-mask baselines by replacing `f_mask(r(x))` with a fixed
`M` (full watermark, narrow/medium borders of width 28/56, padding-based
reprogramming which additionally centres the resized image on a zero canvas).

## Mask generator (`smm/mask_generator.py`)

* 5-layer CNN (ResNet-18/50) or 6-layer CNN (ViT-B/32);
* 3x3 convolutions, stride 1, padding 1; 2x2 max pooling;
* 3 max-pooling layers by default, so the mask has spatial size
  `floor(H/8) x floor(W/8)` and the patch size is `2**l = 8` (Section 5);
* the last convolution has 3 output channels and **no activation** – the paper
  needs an affine last layer for the constructive proof of Proposition 4.3;
* the number of pooling layers `l` is a hyper-parameter (Figure 4); `l = 0`
  means no pooling and no interpolation module.

Parameter budget (Table 4): 26,499 parameters for the 5-layer CNN (17.6% of the
3 x 224 x 224 pattern parameters, 0.23% of ResNet-18) and 102,339 for the
6-layer CNN (23.13% / 0.12% of ViT-B/32).

## Patch-wise interpolation (Section 3.3)

The mask produced at resolution `floor(H/2**l) x floor(W/2**l)` is expanded back
to `H x W` by repeating every pixel over a `2**l x 2**l` patch, so the mask is
constant inside each patch; when the patch grid does not divide the image size
the closest patches are mirrored and the result is cropped.  Implemented with
`repeat_interleave` (a copy), which avoids the floating-point derivations of
bilinear/bicubic interpolation (Appendix A.3, Table 5) while remaining
differentiable so that `phi` can be optimised.

## Learning strategy (Algorithm 1)

```
delta <- 0; initialise phi randomly
for j in 1..E:
    (Ilm) refresh f_out^(j) from the frequency matrix of the current f_in
    f_in(x_i) = r(x_i) + delta * f_mask(r(x_i) | phi)
    L = mean_i CE( f_out^(j)( f_P( f_in(x_i) ) ), y_i )
    delta <- delta - alpha_1 * grad_delta L
    phi   <- phi   - alpha_2 * grad_phi   L
```

Settings used for every experiment: 200 epochs, SGD with momentum 0.9,
`alpha_1 = alpha_2 = 0.01` with decay 0.1 at epochs 100 and 145 for the ResNets
(Section 5 / Table 9); `alpha_1 = alpha_2 = 0.001` with no decay for ViT-B/32,
the unified setting found by the learning-rate search of Appendix C (Table 7).
Batch size 256, except 64 for DTD and OxfordPets (Table 9).

## Output mapping (`smm/label_mapping.py`)

`f_out` maps a subset `Y_sub^P` of the pre-trained label space to the target
labels injectively.  The trainer restricts the ImageNet logits to
`Y_sub^P`, re-indexes them to the target label space and applies the
cross-entropy loss there.

* **Rlm** – random injective mapping, drawn once before training.
* **Flm** – the mapping that maximises the frequency of
  `(predicted pre-trained class, target class)` pairs under the identity
  `f_in`, computed once (Algorithms 2-3).
* **Ilm** – the same frequency matrix recomputed with the current `f_in` before
  every epoch and rematched greedily (Algorithm 4).  `--ilm-every N` refreshes
  it every N epochs instead (a cost/accuracy trade-off: the pass over the whole
  training set is what makes Ilm expensive).

## Theory (`smm/theory.py`)

`Err_D^apx(F) = inf_{f in F} E[l(f(X), Y)] - R_D^*` (Definition 4.1) and
`F1 subseteq F2 => Err^apx(F1) >= Err^apx(F2)` (Theorem 4.2) imply, since
`F_shr subseteq F_smm` (Proposition 4.3) and `F_sp subseteq F_smm`
(Proposition B.1), that SMM has the smallest approximation error of the three
hypothesis spaces.  The module constructs the inclusion explicitly (zero
weights, last-layer affine term equal to the shared mask / `delta = J`) and the
accompanying script optimises the three families on a synthetic deterministic
task to compare their empirical approximation errors.
