# Dataset justification: UCI Human Activity Recognition (HAR)

## Decision

The study uses **Human Activity Recognition Using Smartphones** (UCI dataset 240, DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K)) in place of Covertype.

The paper is about LIME on *high-dimensional* tabular data. Covertype has only 54 predictors, 44 of which are binary indicators. HAR has **561 real, dense, numeric features**, so it puts the "hundreds of features" requirement from the 17 Oct meeting at the centre of the study.

### Suggested wording for the paper

> We use the UCI Human Activity Recognition Using Smartphones dataset [Anguita et al., 2013], which contains 10,299 windows of smartphone accelerometer and gyroscope signals from 30 volunteers, each described by 561 time- and frequency-domain features and labelled with one of six activities. We use the dataset's official subject-independent partition (21 volunteers / 7,352 windows for training; 9 volunteers / 2,947 windows for testing), so no individual contributes to both sets. The dataset is part of the OpenML-CC18 curated benchmark suite [Bischl et al., 2021].

## Why HAR

| Requirement | HAR |
|---|---|
| Hundreds of features | 561 numeric features |
| Real (not synthetic) tabular data | Statistics computed from real sensor recordings: mean, std, energy, entropy, correlation, frequency-band energy, and so on |
| Clean | No missing values; all features continuous |
| Fair train/test split | Official split by volunteer, so there is no person-level leakage |
| Balanced classes for 3 explained cases per class | 6 classes; the largest is 18.9% and the smallest 13.7% |
| Feasible for every model | 7,352 training rows, so every model trains on the full training set with no subsampling |
| Recognised benchmark | One of the 72 datasets in OpenML-CC18 |
| Compatible with TabFM later | The 50/100/300/500 perturbed-feature sweep stays within TabFM's `max_num_features=500` default |

**Why the split is 70/30 rather than our old 80/20.** It is HAR's own split, made by volunteer rather than by row. Windows recorded from the same person are highly correlated, so a random row split would put the same people in both train and test and inflate accuracy. Using the published split also keeps our accuracy figures comparable with the original paper (multiclass SVM, 96%). Covertype's 80/20 was only our own choice, because that dataset had no official split. The ratio is supported for high-dimensional classifiers: Dobbin & Simon (source 9) found that about 2/3 for training is close to optimal when n ≥ 100 and accuracy is 85% or higher, and HAR meets both conditions. Splitting by subject is supported by Saeb et al. (source 10), who show that record-wise splits often massively overestimate accuracy.

**Why 50 / 100 / 300 / 500 perturbed features.** The models always see all 561 features. Only the number of features LIME perturbs changes, so model quality is held fixed while explanation dimensionality varies.

- **500 (upper bound)** is TabFM's documented `max_num_features` (source 5). Above it, TabFM would subsample features while the other models wouldn't, breaking the controlled comparison.
- **100, 300, 500** are the high-dimensional conditions: the "hundreds of features" regime the study is about, in equal 200-feature steps.
- **50** is a deliberate low-dimensional baseline (added 21-10-2026, fix 13 in `claim_safe_fixes.md`), not a high-dimensional condition:
  - a fourth d value gives 20 design points for the runtime power-law fit (Goldsmith et al., 2007);
  - it shows quality degrading *from* a setting where LIME should work well;
  - it never has a low sample-to-feature ratio (at least 10 samples per feature);
  - it halves 100 and is still 5× the 10 reported features.

Crossed with the 500–10,000 sample budgets, the grid spans sample-to-feature ratios from 1:1 to 200:1 for LIME's ridge surrogate. It also spans default kernel widths of 0.75·√d = 5.3, 7.5, 13.0 and 16.8. Garreau & von Luxburg (source 6) show this kind of parameter choice affects which features LIME recovers.

The subsets are nested (50 ⊂ 100 ⊂ 300 ⊂ 500) and seeded. No source prescribes these exact values. 500 is fixed by TabFM, the 100–500 range by the study's focus, and 50 by the reasons above. In the paper, keep "high-dimensional" claims to 100–500 and describe 50 as the baseline.

## Verified sources

Every source below was opened and checked on 20-10-2026. The "Status" column says how.

| # | Source | What it supports | Status |
|---|---|---|---|
| 1 | UCI dataset page: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>. DOI 10.24432/C54S4K, licence CC BY 4.0 | 10,299 instances, 561-feature vectors, no missing values, 6 activities. Creators: Jorge Reyes-Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra. 70% of the 30 volunteers were used for training and 30% for testing | Page fetched |
| 2 | D. Anguita, A. Ghio, L. Oneto, X. Parra, J. L. Reyes-Ortiz. "A Public Domain Dataset for Human Activity Recognition Using Smartphones." *ESANN 2013*, Bruges, pp. 437–442, ISBN 978-2-87419-081-0. PDF: <https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf> | 30 volunteers aged 19–48 wearing a waist-mounted Samsung Galaxy S II. 50 Hz accelerometer and gyroscope, 2.56 s windows with 50% overlap. 561 features. 70/30 split with 2,947 test patterns. Multiclass SVM reaches 96% accuracy | Full PDF read |
| 3 | OpenML dataset 1478 "har": <https://www.openml.org/d/1478> | 10,299 × 561 numeric features, 6 classes, 0 missing values, class balance 13.7–18.9%. The OpenML copy has **no subject column**, which is why we load the UCI original | API fetched |
| 4 | B. Bischl, G. Casalicchio, M. Feurer, P. Gijsbers, F. Hutter, M. Lang, R. G. Mantovani, J. N. van Rijn, J. Vanschoren. "OpenML Benchmarking Suites." *NeurIPS 2021 Datasets and Benchmarks Track*. arXiv:[1708.03731](https://arxiv.org/abs/1708.03731). Suite: <https://www.openml.org/s/99> | HAR (ID 1478) is one of the 72 curated OpenML-CC18 classification datasets | Suite API and arXiv fetched |
| 5 | TabFM, google-research/tabfm: `tabfm/README.md` (FAQ) and `tabfm/tabfm/src/classifier_and_regressor.py` | `max_num_features` defaults to 500, which is the reason for the 500 cap | Code read in our local clone |
| 6 | D. Garreau, U. von Luxburg. "Explaining the Explainer: A First Theoretical Analysis of LIME." *AISTATS 2020*, PMLR 108:1287–1296. <https://proceedings.mlr.press/v108/garreau20a.html> | LIME's parameter choices (e.g. kernel width, which grows with the number of features) can cause it to miss important features. This motivates studying quality against dimension | Search-verified (PMLR, arXiv:2001.03447) |
| 7 | G. Visani, E. Bagli, F. Chesani, A. Poluzzi, D. Capuzzo. "Statistical stability indices for LIME: obtaining reliable explanations for machine learning models." *Journal of the Operational Research Society* 73(1):91–101. DOI 10.1080/01605682.2020.1865846. arXiv:[2001.11757](https://arxiv.org/abs/2001.11757) | Repeated LIME runs give different explanations, which motivates our stability metric | arXiv verified. **Still to do: confirm volume and pages via the DOI** |
| 8 | Secondary confirmation of the subject split: arXiv:[1811.00170](https://arxiv.org/abs/1811.00170), arXiv:[2006.00367](https://arxiv.org/abs/2006.00367) | 21 training volunteers with 7,352 windows, and 9 test volunteers with 2,947 windows | Search-verified. Also checked in code: `load_har()` asserts the shapes and that the subject sets are disjoint |
| 9 | K. K. Dobbin, R. M. Simon. "Optimally splitting cases for training and testing high dimensional classifiers." *BMC Medical Genomics* 4:31, 2011. DOI 10.1186/1755-8794-4-31. <https://pmc.ncbi.nlm.nih.gov/articles/PMC3090739/> | Giving about 2/3 of cases to training "was close to optimal for reasonable sized datasets (n ≥ 100) with strong signals (85% or greater full dataset accuracy)". Supports the ≈70/30 ratio | Abstract read (PMC) |
| 10 | S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, K. P. Kording. "The need to approximate the use-case in clinical machine learning." *GigaScience* 6(5):1–9, 2017. DOI 10.1093/gigascience/gix019. <https://pmc.ncbi.nlm.nih.gov/articles/PMC5441397/> | Record-wise splitting "often massively overestimates the prediction accuracy", and almost half of the studies reviewed made this mistake. Supports the subject-based split | Abstract read (PMC) |
| 11 | D. Gholamiangonabadi, N. Kiselov, K. Grolinger. "Deep Neural Networks for Human Activity Recognition With Wearable Sensors: Leave-One-Subject-Out Cross-Validation for Model Selection." *IEEE Access* 8:133982–133994, 2020 | Argues for subject-held-out evaluation in HAR specifically | Citation verified via DBLP; **full text not read** |

## Alternatives considered

| Dataset | Size | Why not |
|---|---|---|
| Isolet (UCI 54) | 7,797 × 617, 26 classes | 26 classes × 3 explained cases is 78 cases, about 4× the LIME cost |
| Madelon (UCI 171) | 4,400 × 500, 2 classes | **Synthetic**: 480 of its 500 features are deliberate noise probes, so it isn't realistic tabular data |
| Bioresponse (OpenML 4134) | 3,751 × 1,776, 2 classes | Exceeds TabFM's 500-feature limit |
| cnae-9 (OpenML 1468) | 1,080 × 856, 9 classes | Too few rows for a 2,000-row LIME background plus a held-out test set |

## How to verify (for the team)

1. Open the DOI <https://doi.org/10.24432/C54S4K> and check it resolves to the UCI page with the five creators listed above.
2. Open the ESANN PDF. Page 1 lists the authors and affiliations: University of Genova (DITEN), Italy, and Universitat Politècnica de Catalunya (CETpD), Spain. Section 2.2 states the 561 features, and Section 3 states the 2,947 test patterns and the 96% SVM result.
3. Check the authors on Google Scholar or DBLP, for example "Davide Anguita Genova" and "Jorge Luis Reyes-Ortiz".
4. Open <https://www.openml.org/d/1478> for the dataset statistics, and <https://www.openml.org/s/99> to confirm HAR is in CC18.
5. Confirm the JORS volume and pages for source 7 via its DOI before citing it.
6. Our loader records the SHA-256 of the downloaded archive in every `results/*_run_meta.json`, so anyone can confirm they used the same file.
