## Zhao et al. (2021) — *BayLIME: Bayesian Local Interpretable Model-Agnostic Explanations*

### Link
- Paper: https://proceedings.mlr.press/v161/zhao21a/zhao21a.pdf
- PMLR publication page: https://proceedings.mlr.press/v161/zhao21a.html

### My Key Takeaways
- BayLIME was developed mainly to address the **instability of standard LIME explanations** caused by random perturbation sampling.
- The paper shows that increasing the number of LIME perturbation samples improves explanation stability but also increases computational cost.
- The authors experimentally observe an approximately **linear relationship between the number of perturbation samples and LIME runtime**.
- The paper studies cost scaling with the number of samples, not with the number of input features, making it closely related but distinct from our proposed experiment.

### Abstract
BayLIME extends the original LIME method using Bayesian inference. Standard LIME explanations may change between repeated runs because the method relies on randomly generated perturbation samples. BayLIME attempts to improve the consistency and robustness of explanations by combining information from these new samples with prior knowledge.

The authors introduce measures for evaluating explanation consistency and robustness and investigate different sources of prior information. Their experiments compare BayLIME against existing explanation methods and indicate that appropriate prior information can improve consistency, robustness to kernel settings, and explanation fidelity.

### Summary
Zhao et al. focus on one of LIME's major weaknesses: the fact that explanations can vary depending on the randomly generated perturbation samples used to train the local surrogate model. One way to reduce this variation is to generate more perturbation samples. However, doing so increases the amount of computation required. BayLIME introduces a Bayesian local surrogate model that incorporates prior knowledge, allowing the explanation to rely less heavily on large numbers of newly generated samples while attempting to preserve or improve explanation quality.

The paper is particularly relevant to our research because it explicitly investigates LIME's computational behaviour. The authors vary the number of perturbation samples and observe that runtime increases approximately linearly as the sample count increases. This provides evidence that LIME contains measurable scaling behaviour and that the parameters used to generate the local explanation can significantly affect cost. However, the independent variable in BayLIME is mainly the **number of perturbation samples**, whereas our research focuses on the **number of input features in tabular data**. Therefore, BayLIME provides a useful methodological and computational comparison but does not answer our specific research question.

### Paper Value
This paper is **highly relevant** because it directly investigates a computational trade-off in LIME and provides experimental evidence that runtime grows with perturbation sample count.

Its limitation is that it studies scaling with **sample count rather than feature-space dimensionality**. This actually strengthens the motivation for our work because we can investigate another major dimension of LIME's computational cost that was not isolated in this paper.

### Proof Paper is Valid
Published at UAI 2021, a recognised peer-reviewed conference focused on AI, uncertainty, and machine learning.

The official paper is published through Proceedings of Machine Learning Research (PMLR): https://proceedings.mlr.press/v161/zhao21a.html

The publication is also independently listed by university research repositories, confirming the authors, venue, and peer-reviewed status.

The authors have research backgrounds in areas including Explainable AI, trustworthy AI, robustness, and machine-learning verification.

The paper has been cited by later XAI research and builds directly on the established LIME framework.

It includes both theoretical analysis and empirical experiments rather than relying only on conceptual arguments.