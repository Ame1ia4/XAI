**Title:** Conditional Local Importance by Quantile Expectations

**Link:** https://arxiv.org/html/2411.08821

**Key Takeaways:** This paper proposes a new model-agnostic XAI method called CLIQUE, that could highlight locally dependent relationships and improve stability over perturbation based methods. The paper establishes that hyperparameter tuning can be needed to detect localised interaction effects with LIME, which can be difficult.

**Abstract:** Global variable importance measures are commonly used to interpret the results of machine learning models. Local variable importance techniques assess how variables contribute to individual observations. Current, popular methods, including LIME and SHAP, provide useful measures of feature contribution in the prediction space, while leaving opportunities for improved characterization of local structure in the model loss space. Additionally, they are not natively adapted for multi-class classification problems. We propose a new model-agnostic method for calculating local variable importance, CLIQUE, that highlights locally dependent relationships, provides improved stability over permutation-based methods, and can be directly applied to multi-class classification problems. Simulated and real-world examples show that CLIQUE emphasizes locally dependent information, captures interaction behavior beyond what can be evaluated by correlations, and assigns zero importance in regions where the response is invariant to changes in variables.

**Model:** Claude Sonnet 5.5

**Prompt:**
*Title:* Conditional Local Importance by Quantile Expectations

*Link:* https://arxiv.org/html/2411.08821 Please provide an AI summary of this paper, rate out of 10 this papers relevance to my topic and state how it relates. My topic is LIME cost/quality scaling on High-Dimensional Tabular Data on different models. 


**AI Summary:**
The paper (Bladen, Cutler, Cutler & Moon) proposes CLIQUE, a model-agnostic local variable importance method. Unlike LIME and SHAP, which explain predictions, CLIQUE measures importance in terms of cross-validated error. For each observation and feature, it replaces the feature with M quantile-grid values, predicts with a CV model that never saw that point, and averages the resulting loss change.

The authors claim it:
- Gives exactly zero importance where the model output is locally invariant to a feature (proved in Proposition 1).
- Has lower variance than random-permutation approaches (their "Local Permute" variant).
- Extends to multi-class problems without one-vs-all decompositions.
Is competitive on runtime.

Experiments use simulated data with known interaction structure (AND gate, Corners, regression interaction, a 101-feature correlated dataset) and real data (Concrete, Lichen, MNIST digits). Models are Random Forest, XGBoost and ANN. CLIQUE beats LIME, SHAP and ICI on a "mean absolute error from true zero importance" metric, often by an order of magnitude. Limitations include out-of-distribution inputs from the quantile replacement and sensitivity to the CV design.

- Relevance: 6/10
LIME is a comparison baseline here, not the subject, so the paper doesn't answer your question directly. It does give you useful material on both halves of cost/quality.

- Quality:
Appendix A shows LIME's explanations depend heavily on kernel width and number of bins. Smaller kernels recover conditional structure better, but there is no principled way to tune them without ground truth.
Default LIME mostly captures marginal effects and gives non-zero importance to features that have no local influence.
Table 1 gives a ground-truth quality metric you could borrow: standardised MAE from zero on regions where a feature is known to be irrelevant. LIME scores 0.28 to 0.87 against CLIQUE's 0.01 to 0.04.
In the 101-feature setting (Section 3.4, Appendix C), LIME's default settings assign larger importance to both variable groups when z=1, so it doesn't reflect the structure cleanly. At ρ=0.25 it does reflect the structure but with much higher variance, and at ρ=0 only CLIQUE gets it right. This is a useful data point on how LIME behaves as dimensionality and correlation vary.

- Cost:
Appendix F times LIME, SHAP, ICI and CLIQUE as observations and features grow. LIME and ICI are significantly slower than SHAP and CLIQUE.
Their tree-SHAP runtime is roughly flat in feature count but grows quadratically with observations. CLIQUE is linear in both.

- Models:
Different models are covered: RF, XGBoost and ANN, including a direct RF vs XGBoost comparison on the Lichen data (Appendix D). The comparison is qualitative, though, and not about LIME's cost or quality scaling per model.

- Why it isn't higher:
The timing study is limited. It uses at most 36 features (6x6 MNIST), binary classification only, one LIME configuration (10 bins) and a single machine. It doesn't vary LIME's number of samples or kernel width against runtime, which is the heart of a cost/quality trade-off.
"Quality" here means detecting conditional or interaction structure and assigning zeros. It doesn't cover LIME's fidelity or stability across repeated runs, which you may care about more.
The high-dimensional case is a single synthetic ANN experiment with 101 features, not a systematic scaling study.


**Paper's Value:** Valuable in understanding LIME and SHAP from the perspective 0f feature contribution, and slightly relevant to the cost-effectiveness of LIME when dealing with high dimensional tabular data. The paper has a 101-feature experiment and a runtime comparison that includes LIME. LIME and ICI were noticeably slower than SHAP and CLIQUE, and the timing was measured as both features and observations increased. The limits are that the timing tops out at 36 features (6x6 pixels, two digit classes), it uses a Random Forest, and LIME is not the focus. So it is limited evidence, but not irrelevant.

**Source Validation:** *Source of discovery:* arxiv.org

This paper was recently published (2026) in the journal "Transactions on Machine Learning Research (TMLR)"