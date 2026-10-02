**Title:** Which LIME should I trust? Concepts,
Challenges, and Solutions

**Link:** https://arxiv.org/pdf/2503.24365

**Key Takeaways:** This paper addresses the different adaptations of LIME and provides a holistic overview of advancements in LIME.

**Abstract:** As neural networks become dominant in essential systems,
Explainable Artificial Intelligence (XAI) plays a crucial role in fostering trust and detecting potential misbehavior of opaque models. LIME
(Local Interpretable Model-agnostic Explanations) is among the most
prominent model-agnostic approaches, generating explanations by approximating the behavior of black-box models around specific instances.
Despite its popularity, LIME faces challenges related to fidelity, stability, and applicability to domain-specific problems. Numerous adaptations
and enhancements have been proposed to address these issues, but the
growing number of developments can be overwhelming, complicating efforts to navigate LIME-related research. To the best of our knowledge,
this is the first survey to comprehensively explore and collect LIME’s
foundational concepts and known limitations. We categorize and compare its various enhancements, offering a structured taxonomy based
on intermediate steps and key issues. Our analysis provides a holistic
overview of advancements in LIME, guiding future research and helping practitioners identify suitable approaches. Additionally, we provide a
continuously updated interactive website, Which LIME Should I Trust?,
offering a concise and accessible overview of the survey.

**Model:** Claude Sonnet 5.5

**Prompt:**
*Title:* Which LIME should I trust? Concepts,
Challenges, and Solutions

*Link:* https://arxiv.org/pdf/2503.24365 Please provide an AI summary of this paper, rate out of 10 this papers relevance to my topic and state how it relates. My topic is LIME cost/quality scaling on High-Dimensional Tabular Data on different models. 


**AI Summary:** 
This is a survey of LIME and its many variants, not an empirical paper. It does not propose a method or run experiments. The authors reviewed over 20,000 LIME-related papers and built a taxonomy along two axes:

- The issue addressed: locality, fidelity, interpretability, stability, or efficiency.
- Where in the LIME pipeline the variant intervenes: feature generation, sample generation, feature attribution (the surrogate model and kernel), or explanation representation.

Table 1 maps about 45 variants onto these axes, and Table 2 groups them by modality (tabular, image, text, time series, and so on). The discussion section makes three main points:

Reproducibility is poor. About 50% of the surveyed methods have no code.
Evaluation is inconsistent. Most papers compare only against vanilla LIME and use cherry-picked metrics.
The field needs a standard LIME evaluation framework and help selecting variants.

- Relevance: 6/10

It is useful as a map of the field, but it won't give you direct evidence for your question.

- How it relates to your topic:
It names the trade-offs you're studying. Section 4.1 notes that the issues interact: increasing locality can hurt efficiency, while reducing efficiency (more computation) can improve stability. That is the cost/quality tension at the core of your topic, though the paper only states it qualitatively.
Its metrics table gives you vocabulary. Table 3 lists Efficiency and Scalability ("how well the method performs as the model or dataset size increases") alongside Correctness, Consistency and Continuity. It is a ready-made menu for operationalising "cost" and "quality".
It catalogues tabular-specific variants. Table 2 lists about 19 universal tabular methods, including US-LIME, GMM-LIME, BMB-LIME, UnRAvEL-LIME, LSLIME, QLIME, GuidedLIME and LIMEtree. Several target fidelity or stability on tabular data, which is the "quality" side of your question.
It covers different model types. The paper notes LIME has been applied to NNs, CNNs, LSTMs, transformers, decision trees and random forests. It also cites Klein et al. (2024) on pairing evaluation metrics with model architectures, which fits your "different models" angle.
It highlights directly relevant cost/quality methods. S-LIME estimates how many samples are needed for stable explanations, which is the clearest cost/quality scaling idea in the survey. OptiLIME tunes kernel width to balance stability against fidelity.

- Why it isn't higher:
There are no scaling results. It contains no experiments, no runtime or sample-count curves, and no analysis of how cost or quality changes with dimensionality d. The dimensionality d appears only in the notation.
The efficiency literature it covers is thin and off-target. The only efficiency-focused methods are Attention-LIME (text), survLIME and survLIME-inf (survival models). None address high-dimensional tabular data.
It treats model dependence lightly. LIME is described as model-agnostic, but the paper does not compare how explanation cost or quality varies across model families.
It excludes SHAP and other methods. It looks only at LIME variants, so there is no cross-method cost comparison.
Suggested follow-ups

- Chase these from the reference list:
S-LIME (Zhou et al., KDD 2021), for sample-size and stability.
OptiLIME (Visani et al.), for the stability/fidelity trade-off.
US-LIME, UnRAvEL-LIME and GLIME (Tan et al., NeurIPS 2023), for sampling and stability improvements.
Garreau & von Luxburg, "Looking deeper into tabular LIME", for theory on how tabular LIME behaves.
Klein et al. (2024) and OpenXAI (Agarwal et al.), for benchmarking explanation methods across models.
Sahakyan et al. (2021), a survey of explainable AI for tabular data.

**Paper's Value:** This paper poses more relevance as it discusses the computational inefficieny's  of LIME and limitations in the handling of certain types of data as one of five issue categories. The paper mentions that evaluation practice is inconsistent and about half the methods lack code, a possible reason why scaling comparisons are hard to find.

**Source Validation:**
*Source of discovery:* 
arxiv.org

This is a conference paper published in 2025, available on Springer journals which provides a portfolio of peer-reviewed titles.