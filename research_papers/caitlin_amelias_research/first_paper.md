**Title:** Towards Robust and Accurate Stability Estimation of Local Surrogate Models in Text-based Explainable AI


**Link:** https://arxiv.org/html/2501.02042

**Key Takeaways:** This paper discusses the stability and robustness of XAI methods under adversarial attacks (deliberate manipulations that trick or distort the interpretability tools), and proposes a weighting scheme for text-based data that provides improved estimates of actual weakness of XAI methods to adversarial examples. The paper states the existence of LIME instability was already shown in earlier work and this paper observes how LIME's sampliong noise can make over-sensitive measures.

**Abstract:** Recent work has investigated the concept of adversarial attacks on explainable AI (XAI) in the NLP domain with a focus on examining the vulnerability of local surrogate methods such as Lime to adversarial perturbations or small changes on the input of a machine learning (ML) model. In such attacks, the generated explanation is manipulated while the meaning and structure of the original input remain similar under the ML model. Such attacks are especially alarming when XAI is used as a basis for decision making (e.g., prescribing drugs based on AI medical predictors) or for legal action (e.g., legal dispute involving AI software). Although weaknesses across many XAI methods have been shown to exist, the reasons behind why remain little explored. Central to this XAI manipulation is the similarity measure used to calculate how one explanation differs from another. A poor choice of similarity measure can lead to erroneous conclusions about the stability or adversarial robustness of an XAI method. Therefore, this work investigates a variety of similarity measures designed for text-based ranked lists referenced in related work to determine their comparative suitability for use. We find that many measures are overly sensitive, resulting in erroneous estimates of stability. We then propose a weighting scheme for text-based data that incorporates the synonymity between the features within an explanation, providing more accurate estimates of the actual weakness of XAI methods to adversarial examples. Source codes will be made publicly available.

**Model:** Claude Sonnet 5.5

**Prompt:**
"Towards Robust and Accurate Stability Estimation of Local Surrogate Models in Text-based Explainable AI" (arXiv 2501.02042) https://arxiv.org/html/2501.02042 Please provide an AI summary of this paper, rate out of 10 this papers relevance to my topic and state how it relates. My topic is LIME cost/quality scaling on High-Dimensional Tabular Data on different models. 

**AI Summary:** 
This paper (Burger et al., 2025) asks a narrow question: when researchers attack LIME with small adversarial perturbations to measure how stable its explanations are, how much does the choice of similarity measure change the conclusion? LIME's output is a ranked list of features with weights, so "did the explanation change?" has to be answered by a ranked-list similarity metric.

- Measures tested: Jaccard, Kendall's Tau distance, Spearman's footrule, and Rank-Biased Overlap (RBO, with p = 0.5, 0.7, 0.9), plus weighted variants of the first three.
- Setup: Greedy word-substitution attacks on LIME explanations of DistilBERT classifiers, on two short-text datasets (a Twitter gender-bias set and a symptoms-to-diagnosis set). Each attack counts as "successful" at similarity thresholds of 30-60%.
- Findings: The metric choice can swing attack success rates enormously.
Kendall is almost always "successful" (~95-100%), so it overstates instability.
Jaccard and Spearman become overly sensitive at looser thresholds.
Weighted Jaccard and RBO₀.₉ are too coarse and rarely register a change.
RBO gives the best balance, but you have to tune p.
- Contribution: They propose synonymity weighting, which gives partial credit when a feature is swapped for a near-synonym (cosine similarity on GloVe embeddings). This sharply cuts the false instability for Jaccard and Spearman, but barely changes RBO or Kendall.
- Caveat on the extremes: Much of the apparent instability is LIME's own sampling randomness, not the perturbation. They also note the measure conclusions may not transfer to other XAI methods.
- Relevance: 3/10

Why it scores low: It is a text/NLP paper, not tabular, and it never varies dimensionality or LIME's sample budget. Only one model (DistilBERT) is used, so there is no cross-model comparison. The research question is adversarial robustness, not how explanation quality scales with computational cost.

Where it does connect to your topic:
- Quality metrics: If you measure LIME "quality" as stability (re-running LIME with different seeds or sample sizes and comparing top-k rankings), this paper is a useful guide to choosing the metric. Its core warning is that Kendall and Jaccard can give misleading results and RBO is the safest default. If you use rank-similarity to track convergence as the number of perturbation samples grows, this directly affects your conclusions.
- Inherent LIME instability: They show that a large share of explanation variation comes from LIME's sampling process, independent of any attack. That is your quality-versus-cost trade-off in a nutshell: fewer samples means cheaper but noisier explanations. You could use this to justify a seed-to-seed baseline in your experiments.
- Cost evidence: Their statement that explanation generation, not the metric, is the computational bottleneck, and that IMDB was infeasible (~125 days on an A6000), is a data point supporting cost scaling with input dimensionality. It is anecdotal rather than systematic, though.
- Related work: Their background cites Garreau & von Luxburg's "Looking deeper into tabular LIME" and Mardaoui & Garreau on text LIME. Both are likely closer to your tabular focus and worth reading.

**Paper Value:** This paper is not of much value given our topic as it compares similarity measures for ranked explanations and proposes synonymity weighting which doesn't deal with cost-scaling for tabular data.

**Source Validation:** This paper is a pre-print. Christopher Burger, Lingwei Chen, and Thai Le have written papers together for the 2023, Conference on Empirical Methods in Natural Language Processing (EMNLP 2023), this paper was subsequently cited 28 times. Christopher Burger has 13 other publications which are pre-prints or conference papers available on arxiv.org.