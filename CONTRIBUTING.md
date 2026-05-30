# Contributing

Thank you for your interest in contributing to the Reservoir Performance Proxy project.

## Ways to Contribute

- **Bug reports:** Open an issue describing the problem and steps to reproduce it.
- **New simulation data:** Additional OPM Flow runs can extend the training set. See `dataset_timeseries_lstm.csv` for the expected column format.
- **Model improvements:** New architectures or training strategies are welcome — open a pull request with your R² results compared against the existing benchmarks in the README.
- **Documentation:** Corrections, clarifications, and translations are always appreciated.

## Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-change`
3. Make your changes and verify that existing notebooks still produce R² values consistent with those reported in the README
4. Open a pull request with a clear description of what changed and why

## Reporting Issues

Open a GitHub Issue and include:
- Which model/notebook you were using
- The full error message or unexpected output
- Your Python and TensorFlow versions (`python --version`, `pip show tensorflow`)

## Code Style

Follow the conventions already present in each notebook. Do not reformat unrelated code in a pull request.
