link to paper: https://arxiv.org/pdf/2501.10774

------------------------------------------------------------------------

Paper AI Overview:

Model: Claude Opus 5.5 Medium
Promt: can you give me an overview of the Lime paper we were discussing (https://arxiv.org/pdf/2501.10774) and the section that relates to my work please. Ensure you detail the context of the paper, the problem the related section was trying to solve, the approach of the LIME scaling research and the core findings

This paper is Carlos Mougan Navarro's PhD thesis, Model Monitoring in the Absence of Labeled Data via Feature Attributions Distributions (University of Southampton, January 2025). It isn't a LIME paper. LIME comes up only as an alternative the author tests and decides against. The paper considers that once a model is deployed, you usually don't have the true outcomes yet. A loan model's predictions can't be checked until loans are repaid or not. So how do you tell whether the model is behaving badly?

---

Related Work:
The Equal Treatment method is "parametric in the explanation function": in principle any per-feature attribution method can be plugged in. So the section asks whether LIME could replace SHAP, and what it would cost. This matters because LIME is model-agnostic and widely used, while TreeSHAP only works on tree models. The method needs an explanation for every row in the dataset, not just a few. Whatever LIME costs per explanation is multiplied by the whole dataset size. As he puts it, the cost problem "is exacerbated when producing explanations for multiple instances

------------------------------------------------------------------------

My Key Takeaways:

-On pg 14 (or 30 on browser pdf)section 2.2.2 describes the tabular LIME pipeline step by step
-On pg 98 (or 114 on the browser pdf) section: 4.5.1 Runtime - this discusses how they ran the experiments for Lime and SHAP 
-On pg 53 (69) good info on whether LIME is fair on models

---

Book's Lime Pipeline Explanation:
Instead of attempting to interpret the entire model globally, LIME focuses on understanding predictions at a local level. It achieves this by
perturbing or sampling data points in the vicinity of the prediction of interest and fitting a simple, interpretable model to approximate the complex model’s behavior in that local region. The interpretable model, often a linear model, decision tree or a decision rule, can be easily understood and analyzed (Guidotti et al., 2018a).

This approach effectively ”explains” a model’s prediction by providing insight into which features were most influential in the local context. LIME quantifies the feature importance, enabling users to grasp why the model made a particular prediction for a specific data point. This local approximation is crucial because it reflects the model’s behavior regarding the instance in question, which may differ from its global behavior.

In this thesis we use the implementation based on the Euclidean version of LIME, called “tabular LIME” and it involves the following steps:

• Selection of Data Point: Choose the data point for which you want an explanation.

• Data Perturbation: Generate a dataset of similar data points by perturbing the chosen instance. This step is crucial for approximating local behavior.

• Prediction Collection: Obtain model predictions for the perturbed data points.

• Surrogate Model Creation: Fit an interpretable model, such as a linear regression
model, to the perturbed data and their corresponding model predictions. This
surrogate model approximates the complex model’s behavior locally.

• Explanation Generation: The surrogate model’s coefficients are used to estimate the importance of each feature for the chosen prediction, resulting in an explanation of the model’s decision for that instance.

LIME is a versatile and model-agnostic technique, which means it can be applied to virtually any machine learning model without requiring knowledge of the model’s internal architecture. This makes it a valuable tool for interpretable AI in diverse domains.

---

SHAP vs LIME experiment details:
"The experiments were run on a server with 4 vCPUs and 32 GB of RAM. We used shap version 0.41.0 and lime version 0.2.0.1 as software
packages. In order to define the local neighborhood for both methods in this example we use all the data provided as background data. As an fθ model, we use an xgboost
and compare the results of TreeShap against LIME. When varying the number of samples we use 5 features and while varying the number of features we use 1000 samples."

---

the Equal Treatment method doesn't care which explanation tool you use. SHAP is the default, but any method that gives per-feature contributions could slot in. So he checks what changes if you use LIME. The Taylor/Maclaurin comparison is an intuition pump: a Taylor series approximates a curve near one point with a straight line (plus extra terms), and LIME does the same for a model near one row, fitting a straight line that's only valid close by.

Drawback 1: Computationally expensive. TreeSHAP reads the model's internal structure directly, so it's fast. LIME treats the model as a black box: it generates fake neighbours, queries the model thousands of times and fits a regression, for every single row. "Exacerbated when producing explanations for multiple instances" is the point I made before: his method needs an explanation for every row, so LIME's per-row cost multiplies across the whole dataset.

Drawback 2: Local neighbourhood. The fake neighbours are random, so running LIME twice on the same row can give different answers. The papers he cites show explanations can vary a lot between runs. For his method that's bad, because random noise in the explanations could make two groups look different (or the same) when they aren't.

Drawback 3: Dimensionality. This is the one most relevant to us.

LIME makes you choose how many features the local linear model uses (the num_features setting, default 10), and reports only those.
His method needs all features explained, so he'd have to set it to the full feature count.
A linear model with many features, fitted on a limited number of noisy fake neighbours, gets unreliable. That's the curse of dimensionality: more features means you need far more samples to pin down each coefficient.
Then comes the admission: his datasets are low-dimensional, so this problem never showed up in his experiments.

------------------------------------------------------------------------

Paper Value:

I consider this paper to have meduim to high value for our research topic. Although the core paper topic is not related, the work related to deciding if LIME would be better to use over SHAP for the papers problem reflects the concepts of our idea

------------------------------------------------------------------------

Proof of Credibilty:

Carlos Mougan Navarro has had 5 other publications
