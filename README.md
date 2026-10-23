# XAI Research

A GitHub repository tracking the research, development, decisions and contributions throughout our Explainable AI project.

## Title of the Report

**LIME Cost/Quality Scaling on High-Dimensional Tabular Data Across Multiple Models**

## Scope

Our research investigates how LIME's computational cost and explanation quality change across:

- **Models:** Logistic Regression, Random Forest and XGBoost.
- **Dataset:** UCI Human Activity Recognition (561 features).
- **Dimensionality:** 50, 100, 300 and 500 perturbed features.
- **Sample Sizes:** 500, 1,000, 2,500, 5,000 and 10,000.
- **Measurements:** Runtime, explanation stability and fidelity.

## Links

- [Final Report Submitted](LIME_Scaling_Paper_Final.pdf)
- [Latest Report Draft](research_paper_draft.md)
- [Research Journal](log_entries/)
- [Research Papers](papers_researched/)
- [Experimental Code and Results](lime_study/)
- [Dataset Research](dataset_sources/)
- [AI Usage and Chat Logs](chat_logs/)
- [Presentation](Presentation-LIME_CostQuality_Scaling_on_High_Dimensional_Tabular_Data_Across_Multiple_Models.pdf)

## Research Log Overview

The following entries document the project's development, discussions and research decisions.

- **[18 September – Initial Meeting](log_entries/18-09-2026.md):** Discussed project expectations, source validation, documentation and next steps.
- **[21 September – Topic Exploration](log_entries/21-09-2026.md):** Considered studying LIME's computational scaling with feature dimensionality.
- **[25 September – Alternative Approaches](log_entries/25-09-2026.md):** Investigated SHAP, LIME and other explainability approaches when considering possible topics.
- **[29 September – Research Planning](log_entries/29-09-2026.md):** Assigned literature research, dataset selection and experimental planning responsibilities.
- **[1 October – Literature Review](<log_entries/01-10-2026(David%20Paper%20Summary).md>):** Reviewed research discussing LIME's computational cost and limitations.
- **[4 October – Topic Justification](log_entries/04-10-2026.md):** Established the research motivation and identified gaps in LIME cost/quality scaling studies.
- **[7 October – Evaluation Metrics](log_entries/07-10-2026.md):** Planned how runtime, stability and fidelity would be measured.
- **[16 October – Report Structure](log_entries/16-10-2026.md):** Discussed IEEE formatting and the structure of the research paper.
- **[17 October – Experimental Revisions](log_entries/17-10-2026.md):** Planned dataset improvements, rerunning experiments and individual writing responsibilities.
- **[18 October – Methodology Planning](log_entries/18-10-2026.md):** Outlined the methodology section, experimental parameters and reproducibility requirements.
- **[20 October – Dataset and Metric Changes](log_entries/20-10-2026.md):** Documented the transition to UCI HAR and the decision to remove Spearman correlation.
- **[21 October – Methodology Corrections](log_entries/21-10-2026.md):** Revised fidelity, statistical testing, reference comparisons and dimensionality settings.
- **[22 October – Final Task List](log_entries/22-10-2026.md):** Final taskes to do before submission.

## Changes Since the Presentation

Following our initial presentation, several significant changes were made to improve the research methodology and experimental reliability.

- **Dataset:** Replaced Covertype (54 features) with UCI HAR (561 features).
  - This provided a more suitable high-dimensional dataset and allowed meaningful feature-scaling experiments.

- **Dimensionality:** Introduced experiments across 50, 100, 300 and 500 features instead of a fixed 54-feature dataset.
  - This allowed us to investigate how increasing dimensionality affected LIME's runtime, stability and fidelity.

- **Stability:** Removed Spearman correlation in favour of top-5 Jaccard agreement.
  - This provided a more appropriate measurement of agreement between the most influential features across repeated explanations.

- **Fidelity:** Replaced LIME's in-sample explanation score with held-out, kernel-weighted R².
  - This allowed us to evaluate how accurately the surrogate approximated the original model on previously unseen perturbations.

- **Models:** Finalised Logistic Regression, Random Forest and XGBoost, excluding the previously discussed TabFM model.
  - The final experiments focused on comparing LIME across these three distinct classification models.

- **Experimental Reliability:** Increased the number of explained cases, introduced reference explanations and improved statistical testing.
  - These changes strengthened the consistency, reproducibility and statistical support for the final results.

These revisions mean that the original presentation results are not directly comparable with the final study. Further explanations of the decisions are documented in the [20 October](log_entries/20-10-2026.md) and [21 October](log_entries/21-10-2026.md) research logs.
