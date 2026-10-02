## Ribeiro et al. (2016) — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*

### Link
- arXiv: https://arxiv.org/pdf/1602.04938
- DOI / Published version: https://doi.org/10.1145/2939672.2939778

### My Key Takeaways
- This is the original paper that introduced **LIME (Local Interpretable Model-Agnostic Explanations)**.
- LIME explains an individual prediction by generating perturbed samples around the instance, querying the original black-box model, and fitting a simpler interpretable model locally.
- The paper states that LIME's computational cost depends strongly on the **number of perturbation samples** and the time required to evaluate the original model.
- The paper does not directly investigate how LIME's runtime changes as the **number of input features increases**, which creates a useful research gap for our project.

### Abstract
The paper introduces LIME, a model-agnostic technique for explaining individual predictions produced by machine-learning models. The main motivation is that users may be reluctant to trust predictions made by complex or opaque models without understanding why those predictions were produced. LIME generates an interpretable local approximation of a complex model around the instance being explained.

The authors also introduce SP-LIME, which selects a representative set of explanations to help users understand the broader behaviour of a model. Experiments involving simulated tasks and human participants show that explanations can help people identify unreliable models, compare classifiers, and detect undesirable model behaviour.

### Summary
Ribeiro et al. introduce LIME as a method for explaining predictions from essentially any classifier without requiring access to the model's internal structure. For each instance being explained, LIME creates multiple modified or perturbed versions of that instance and sends them through the original model. These samples are weighted according to how close they are to the original observation. LIME then trains a simple local surrogate model, normally a sparse linear model, to approximate the black-box model's behaviour near that observation. The coefficients of this local model are presented as the explanation.

For our research, the computational structure of LIME is particularly important. Each explanation requires perturbation generation, repeated calls to the original prediction model, distance calculations, weighting, feature selection, and fitting a local surrogate. Ribeiro et al. specifically mention that computational cost is influenced by the number of perturbation samples and the cost of evaluating the original model. However, they do not systematically vary the dimensionality of tabular data to determine how runtime changes as the number of input features grows. This makes the paper essential background for our experiment while also exposing a gap that our study can investigate.

### Paper Value
This paper is **essential** for our topic because it is the original LIME paper and explains the algorithm whose computational behaviour we are investigating.

Its main limitation for our research is that it does **not directly analyse runtime scaling with tabular feature-space size**. It provides the theoretical and algorithmic foundation for our study rather than answering our exact research question.

### Proof Paper is Valid
Published at KDD 2016, a well-established peer-reviewed conference in data mining and machine learning.

The paper has an official ACM DOI: https://doi.org/10.1145/2939672.2939778

The authors — Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin — are established machine-learning researchers with strong publication histories.

The paper has received thousands of citations, showing that it has had major influence in Explainable AI research.

LIME has also been widely used and extended in later research, including methods such as Anchors and BayLIME.