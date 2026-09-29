****Research Paper Links****


**Research as to why there are no studies on LIME cost-scaling for tabular data.**
1. **arXiv.org** *'Conditional Local Importance by Quantile Expectations'* : https://arxiv.org/abs/2411.08821 
- It compares runtimes of CLIQUE, SHAP, ICI and LIME as features and observations increase, but on MNIST reduced to 6x6 pixels and restricted to two digit classes. This is a side experiment on a small feature range, not a dedicated scaling study.
2. **arXiv.org** *'Towards Robust and Accurate Stability Estimation of Local Surrogate Models in Text-based Explainable AI'* : https://arxiv.org/html/2501.02042 
- This paper notes that local surrogate models have been shown to lack stability across image, tabular and text data. We could use it to argue that research attention has gone to fidelity and stability rather than runtime.
  
3. Ross, Hughes & Doshi-Velez — Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations
- Directly identifies computational scalability as a limitation of LIME. The authors explain that LIME's per-example perturbation and fitting process can become computationally prohibitive when explanations are required across an entire dataset.    
- Provides useful runtime evidence: LIME took 0.03s for 34 features, 1.03s for 75, 1.54s for 784 and 2.59s for 5,000, while their gradient method remained much faster. However, this was a comparison with input gradients rather than a dedicated investigation of LIME's scaling behaviour. 
- Also notes that increasing LIME's number of samples could improve coverage for high-dimensional data, but doing so increases computational cost because more model evaluations are required. 
4. Dieber & Kirrane — Why model why? Assessing the strengths and limitations of LIME
- Particularly relevant because it specifically evaluates LIME on tabular data. The researchers use LIME to explain four ML models trained on an Australian weather dataset. 
- Shows a different type of scaling problem: LIME works well for individual/local explanations, but doesn't provide an easy way of scaling these into a global understanding of the model. The researchers had to manually aggregate multiple LIME outputs in Excel. 
- They describe global analysis as hours of repetitive manual work, while noting that individual LIME explanations themselves were quick to compute. 
- This is strong evidence for your research gap because the paper specifically studies tabular LIME and discusses scalability/usability, but does not experimentally investigate how computational cost changes as the number of rows/features increases.

**Background information on LIME**
1. **geeksForGeeks** : https://www.geeksforgeeks.org/artificial-intelligence/introduction-to-explainable-aixai-using-lime/
2. **arXiv.org** *'Which LIME should I trust? Concepts, Challenges, and Solutions '* : https://arxiv.org/html/2501.02042

**Potential Papers to Look at** 
ChatGPT recommend these 
 - Garreau & von Luxburg (2020) — Looking Deeper into Tabular LIME https://arxiv.org/pdf/2008.11092 [69 Pages Long]
 - Tan, Tian & Li (2023) — GLIME: General, Stable and Local LIME Explanation - https://arxiv.org/pdf/2311.15722 [28 Pages Long]
