\subsection*{4.3. Evaluation of Zero-Shot Classification}

We evaluate clean and robust accuracy of the CLIP models on ImageNet and 13 zero-shot datasets (details in App. B.10), similar to Mao et al. (2023).
For each dataset, class names are combined with a predefined set of prompt templates. The resulting prompts are encoded with the CLIP text-encoder and averaged for each class (Radford et al., 2021), giving a latent embedding for each class. Zero-shot classification is then performed as described in Sec. 3.

Attack setup. To evaluate the adversarial robustness of the models, we employ the first two attacks of AutoAttack (Croce & Hein, 2020), namely APGD with cross-entropy and APGD with DLR loss (100 iterations each). Note that we use the targeted DLR loss (similar to AutoAttack) in contrast to Mao et al. (2023), where the weaker untargeted version is used.

Results.
On ImageNet, TeCoA models perform best in clean and robust evaluations, as they have undergone supervised training on this dataset. FARE models are also trained on ImageNet but do not take labels into account.
On the other zero-shot datasets, the undefended CLIP model expectedly has the best performance on clean data, while TeCoA models suffer significant decrease of clean performance.
In contrast, the FARE models, especially FARE$^{2}$, maintain much better clean accuracy.
On adversarial inputs, CLIP breaks completely at both radii. FARE$^{4}$ performs best in this scenario, outperforming TeCoA$^{4}$ and TeCoA$^{2}$ across threat models.
FARE is thus also in this setting the only method that provides high-performing and robust models.
