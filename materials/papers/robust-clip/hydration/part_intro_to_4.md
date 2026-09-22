Given the flexibility and effectiveness of such large foundation models, in particular LVLMs, it is foreseeable that they will be used in the near future in many real-world applications.
This likely large-scale deployment raises questions on the safety and alignment of these systems, and how to prevent the abuse of their abilities and weaknesses by malicious actors.
Therefore it becomes extremely important to test and improve the robustness of these models.
Recent works (Zhao et al., 2023; Zou et al., 2023) have shown that LVLMs are highly vulnerable to adversarial attacks
on either text or image inputs.
In particular, the vision modality is argued to be the easier one to fool (Carlini et al., 2023):
even commercial LVLMs like BARD could be attacked successfully with large perturbations (Dong et al., 2023).
Moreover, Schlarmann & Hein (2023) show that imperceptible changes of an image can be used for targeted attacks on LVLMs.
This allows malicious third parties to spread such images on the web for defrauding users or spreading misinformation on a massive scale.
In this paper, we tackle the vulnerability of the vision modality of LVLMs as well as generic adversarial robustness of zero-shot classification using CLIP.
To this end, we propose FARE (Fine-tuning for
Adversarially Robust Embeddings), an unsupervised fine-tuning scheme for the vision embedding of CLIP to make it robust to adversarial perturbations while also preserving the features
of the original CLIP model as much as possible.
In this way, simultaneously two objectives are achieved: (i) we can readily replace the original CLIP with our robust CLIP in all down-stream tasks without retraining or fine-tuning since the features on clean inputs are (approximately) preserved. (ii) all down-stream tasks, e.g. zero-shot classification or zero-shot tasks of LVLMs, become robust to attacks on the vision modality (see an example in Fig. 2).

The only existing method, TeCoA (Mao et al., 2023), for a robust CLIP vision encoder performs supervised adversarial fine-tuning (using ImageNet) on the zero-shot classifier derived from CLIP (see Sec. 3.2).
However, the resulting fine-tuned CLIP model shows significant degradation of zero-shot classification accuracy on datasets different from ImageNet, and on integration into LVLMs is detrimental to their performance.
In extensive experiments we show that FARE-CLIP preserves much better the clean performance of CLIP on down-stream tasks such as zero-shot classification or
when used inside LVLMs like OpenFlamingo or LLaVA, while having better robustness to $\ell_\infty$-bounded attacks (see summary in Fig. 1). In particular, we show that using our FARE-CLIP makes LLaVA robust against imperceptible targeted attacks, see Fig. 2.
FARE also demonstrates robustness to jailbreak attacks, leads to lower hallucination rate of LLaVA, and can better solve chain-of-thoughts tasks compared to TeCoA.

\section*{2. Related Work}

Multi-modal models.
Many LVLMs such as Flamingo (Alayrac et al., 2022), OpenFlamingo (OF) (Awadalla et al., 2023), Fromage (Koh et al., 2023), Mini-GPT-4 (Zhu et al., 2023), LLaVA (Liu et al., 2023b; Liu et al., 2023a) and more (Laurençon et al., 2023; Li et al., 2023a; Chen et al., 2023) have recently appeared. Most of them use a pre-trained large language model (LLM) as well as a large vision encoder such as CLIP. The vision encoder is frozen during training, and only the interaction e.g. via a projection layer or cross-attention is learnt. We focus our evaluation on OF (Awadalla et al., 2023) and LLaVA-1.5 (Liu et al., 2023a) as they both use the original ViT-L/14 CLIP model as vision encoder, similar to Chen et al. (2023); Li et al. (2023a), but are based on different LLMs: OF on MPT-7B (MosaicML, 2023) and LLaVA on Vicuna-7B (Chiang et al., 2023), a fine-tuned version of Llama (Touvron et al., 2023).

General adversarial robustness.
The vulnerability of machine learning models to adversarial attacks is well known and has been extensively studied (Szegedy et al., 2014; Goodfellow et al., 2015).
Adversarial training (Madry et al., 2018) is the most prominent defense against adversarial examples.
Most existing attacks focus on uni-modal models, especially those working on image data (Croce & Hein, 2020)
or text (Jia & Liang, 2017; Ebrahimi et al., 2018; Zou et al., 2023; Shen et al., 2023).
Ban & Dong (2022) propose adversarial perturbations that transfer from pre-trained to fine-tuned models.
Moreover, adversarial attacks and defenses for deep metric learning models have also been investigated (Mao et al., 2019; Zhou & Patel, 2022; Zhou et al., 2024).

Adversarial robustness of LVLMs.
In the realm of large vision-language models, multiple works have begun to investigate their vulnerability to adversarial attacks (Qi et al., 2023; Carlini et al., 2023; Schlarmann & Hein, 2023; Shayegani et al., 2023; Zhao et al., 2023; Bagdasaryan et al., 2023; Dong et al., 2023; Bailey et al., 2023; Gu et al., 2024). In Schlarmann & Hein (2023) it is shown that an attacker can use imperceptible perturbations of input images to force the model to produce exact outputs of their choice.
In Carlini et al. (2023) and Qi et al. (2023) visual adversarial attacks that allow jailbreaking of LVLMs are proposed. In contrast to our setting, these attacks grant adversaries a large perturbation-radius.
Supervised adversarial fine-tuning of CLIP has been investigated by Mao et al. (2023), which is the baseline for our work.

Unsupervised adversarial fine-tuning.
It has been investigated for SimCLR (Chen et al., 2020) models in (Kim et al., 2020; Jiang et al., 2020; Fan et al., 2021; Luo et al., 2023; Xu et al., 2023), whose methods are based on a contrastive loss formulation.
Gowal et al. (2020) propose a self-supervised adversarial training scheme based on BYOL (Grill et al., 2020). Robust classifiers are obtained by adding linear heads to their model. Zhang et al. (2022) propose a two-stage training procedure for SimCLR, with clean training done in the first stage and cosine similarity based adversarial training in the second.
In contrast, our method focuses on CLIP and ensures robustness of down-stream tasks even in a zero-shot setting by preserving the original embedding.

\section*{3. Unsupervised Adversarial Fine-Tuning for CLIP}

Similar to supervised image classifiers, CLIP is not robust against adversarial attacks when used for zero-shot image classification (Mao et al., 2023).
In the following we first formalize how adversarial attacks on CLIP are built in this context, then review the adversarial fine-tuning method of Mao et al. (2023) and finally introduce our proposed scheme.

\subsection*{3.1. Robustness of CLIP as Zero-Shot Classifier}

The CLIP model provides an image encoder $\phi: I \rightarrow \mathbb{R}^D$ and a text encoder $\psi: T \rightarrow \mathbb{R}^D$ which map inputs from different modalities into a joint $D$-dimensional space.
Zero-shot classification of an image $x$ on $K$ classes can then be carried out by forming the text prompts
$t_k=$```A photo of <class $k$>`''
for all classes $k=1,\ldots,K$, and then choosing the class with the highest cosine similarity to the image embedding, i.e.
$$
\operatorname{argmax}_{k=1,\ldots,K}\; \cos(\phi(x),\psi(t_k)).
$$

Since in this case the text prompts $t_k$ are fixed, an image embedding function $\phi$ defines a classifier $f$ via its logits

$$
f_k(\phi,x)=\cos(\phi(x),\psi(t_k))=\left\langle\frac{\phi(x)}{\left\|\phi(x)\right\|_2},\frac{\psi(t_k)}{\left\|\psi(t_k)\right\|_2}\right\rangle.
$$

Given an image $x$ with label $y$, an adversarial image $z$ for the classifier $f(\phi, \cdot)$ in the $\ell_p$-norm threat model satisfies:

$$
\operatorname{argmax}_{k=1,\ldots,K}\; f_k(\phi, z) \neq y, \quad \left\|z - x\right\|_p\leq \epsilon, \quad z\in I,
$$

where $\epsilon$ is the perturbation size.
We focus on the $\ell_\infty$-threat model, and $z$ can be found by standard attacks on image classifiers such as AutoAttack (Croce & Hein, 2020).

\subsection*{3.2. Supervised Adversarial Fine-Tuning}

Mao et al. (2023) suggest to make the vision encoder of CLIP robust by fine-tuning it with adversarial training (Madry et al., 2018)
on ImageNet.
Since the cross-entropy loss is used, the training objective of the approach of Mao et al. (2023), called TeCoA (text-guided contrastive adversarial training), is given by

$$
L_\mathrm{TeCoA}(y,f(\phi,x))=-\log\left(\frac{e^{f_y(\phi,x)}}{\sum_{k=1}^K e^{f_k(\phi,x)}}\right)
$$

Let $(x_i,y_i)_{i=1}^n$ denote the training set, then
this can be written in the standard adversarial training
formulation as

$$
\phi_{FT}=\operatorname{argmin}_{\phi} \sum_{i=1}^n \max_{\left\|z-x_i\right\|_\infty\leq \epsilon} L_\mathrm{TeCoA}\left(y_i,f(\phi,z)\right),
$$

where the inner problem is approximately solved with projected gradient descent (PGD) during training and $\phi_{FT}$ indicates the weights of the robust CLIP vision encoder.

This approach has two main problems.
First, adversarial training is done with respect to the fixed set of text embeddings of the classes of ImageNet
. This does not take into account the effect on other text embeddings, e.g.
of categories which are not part of ImageNet, and thus the fine-tuning can lead to heavy distortions with respect to unseen classes, which explains the high losses in standard performance for other down-stream zero-shot classification tasks,
see Table 4.
Second, the loss uses the cosine similarity, which effectively means that it only cares about the projection of the embedding on the hypersphere: one could multiply each $\phi(x)$ by a different scalar factor $\alpha(x)$ and the cosine similarity would be unaffected. Thus during fine-tuning it can happen that the embedding is changed along the radial direction in an arbitrary fashion.
As other down-stream tasks of CLIP, e.g. LVLMs (Alayrac et al., 2022; Liu et al., 2023b; Li et al., 2023a), use the unnormalized embedding this can again lead to huge performance losses.
While for the first problem there is no easy solution, the second problem could be solved by retraining the part of the LVLM that connects the vision and language components. However,
our approach solves both problems at the same time, so that we can get the benefits of our robust CLIP model and maintain good clean performance on all down-stream tasks without the need of fine-tuning or retraining.

\subsection*{3.3. Unsupervised Adversarial Fine-Tuning of the Image Embedding}

The CLIP embedding has been trained on 400M image-text pairs on the WIT dataset (Srinivasan et al., 2021) and provides very good zero-shot performance. Moreover, down-stream tasks like LVLMs have been tuned using this embedding.
Therefore, our goal is to make the vision encoder robust to adversarial attacks while preserving its output on clean points so that it retains clean zero-shot performance and does not require re-training or fine-tuning of components of down-stream tasks, like LVLMs.
As discussed in the previous section, the supervised fine-tuning is not suited for this.
Instead, we introduce an unsupervised adversarial fine-tuning scheme which is not bound to any specific dataset, and does not rely on the text encoder. In the following we denote with $\phi_{org}$ the original CLIP encoder. Given an image $x$, we propose the following embedding loss:

$$
L_{\mathrm{FARE}}(\phi,x)= \max_{\left\|z-x\right\|_\infty \leq \epsilon} \left\|\phi(z)-\phi_{org}(x)\right\|^2_2.
$$

This loss enforces that the features of perturbed points $\phi(z)$ stay close to the unperturbed ones $\phi_{org}(x)$ of the original CLIP model.
Moreover, as $L_\mathrm{FARE}$ goes to zero, the embedding given by the fine-tuned model for clean images is the same as the one by the original model, that is $\left\|\phi(x)-\phi_{org}(x)\right\|^2_2 \rightarrow 0$: this implies that the fine-tuned CLIP vision encoder can be plugged into LVLMs without influencing their performance.
For a set of images $(x_i)_{i=1}^n$,
our proposed fine-tuning scheme consists in optimizing

$$
\phi_{FT} = \operatorname{argmin}_{\phi}\sum_{i=1}^n L_{\mathrm{FARE}}(\phi,x_i).
$$

The inner maximization problem in Eq. (3) of this feature-based variant of adversarial training can be solved by PGD.
We call our proposed method Fine-tuning for Adversarially Robust Embeddings (FARE).

While we focus here on CLIP and its down-stream tasks, our approach can be applied to any foundation model which has an intermediate embedding layer linking modalities.

The following result shows that preserving the image embedding, that is keeping the $\ell_2$-distance between original $\phi_{org}$ and fine-tuned embedding $\phi_{FT}$ small, also preserves the cosine similarities between image and text embeddings, thereby maintaining zero-shot classification performance.

**Theorem 3.1.**
Let $\phi_{org},\phi_{FT}$ be the original and fine-tuned image embeddings and $\psi$ the text embedding of CLIP. Then

$$
|\cos\left(\phi_{FT}(x),\psi(t)\right)-\cos\left(\phi_{org},\psi(t)\right)| \leq
\min\left(\frac{2}{\left\|\phi_{org}(x)\right\|_2},\frac{2}{\left\|\phi_{FT}(x)\right\|_2}\right)
\left\|\phi_{FT}(x)-\phi_{org}(x)\right\|_2.
$$

Proof. See App. App. A.

\section*{4. Experiments}

We conduct experiments for our robust CLIP models on various down-stream tasks such as zero-shot classification as well as using them in LVLMs by replacing their vision encoder. We use OpenFlamingo 9B (OF) (Awadalla et al., 2023) and LLaVA-1.5 7B (Liu et al., 2023b) as LVLMs.

Setting. As the LVLMs OpenFlamingo and LLaVA use the ViT-L/14 vision encoder of CLIP, we focus on this model.
While FARE requires no labels for training and could thus be trained on any image dataset, we use ImageNet in order to stay comparable to TeCoA.
For adversarial training we use 10 steps of PGD for the inner maximization in Eqs. (2), (3).
Notably, we only use two epochs of adversarial fine-tuning on ImageNet (FARE uses no labels) which is only about $0.2%$ of the computational cost of training the original CLIP model ($32$ epochs for 400M images).
We note that there is no additional task-specific training performed for the tasks shown in this paper. In particular, projection layers and language models of LVLMs are fixed.

We compare the clean vision encoder of CLIP from Radford et al. (2021) and two robust fine-tuned versions of it: TeCoA (Mao et al., 2023) and FARE. For a detailed comparison to TeCoA (ViT-B), an ablation of hyperparameters (ViT-B) leading to our chosen parameters for the ViT-L models and training details we refer to
App. B.

Controlling the clean vs robust accuracy trade-off.
A well-known drawback of robust models obtained with adversarial training/fine-tuning is the degradation of clean performance.
In order to control the trade-off, we use $\epsilon=\frac{4}{255}$ and $\epsilon=\frac{2}{255}$ for fine-tuning and denote the CLIP-models as FARE$^{4}$ and FARE$^{2}$ (resp. TeCoA$^{4}$ and TeCoA$^{2}$). The larger radius is standard for ImageNet. We observe that the smaller radius is sufficient to get non-trivial robustness even when testing at $\frac{4}{255}$ while maintaining a clean performance close to the the original CLIP model. However, only the models
trained for $\epsilon=\frac{4}{255}$ are fully robust against targeted imperceptible attacks on LVLMs, see Table Table 3 and Fig. 3.
