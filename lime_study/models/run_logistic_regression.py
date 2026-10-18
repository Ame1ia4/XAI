"""Run the controlled LIME benchmark for LogisticRegression.

Usage (from the repository root): python -m lime_study.models.run_logistic_regression [--quick]
The model definition lives in lime_study/config.py; everything else in protocol.py.
"""

from lime_study.protocol import cli

if __name__ == "__main__":
    cli("LogisticRegression")
