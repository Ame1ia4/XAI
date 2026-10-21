"""Run the controlled LIME benchmark for LogisticRegression.

Usage (from the repository root): python -m lime_study.models.run_logistic_regression [--quick] [--variant NAME]
Debug only: runs this model on its own. Paper results must come from run_all, which
interleaves all models in one run; plot_results refuses to mix runs.
"""

from lime_study.protocol import cli

if __name__ == "__main__":
    cli(["LogisticRegression"])
