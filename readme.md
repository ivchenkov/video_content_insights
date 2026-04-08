# Video Content Insights

This repository contains an exploratory analytics project built around multi-platform video content performance data. The focus is not on publishing raw datasets, but on extracting practical insights about what drives reach, how long content keeps accumulating views, and how performance differs across platforms and formats.

The repo combines a notebook, reusable analysis modules, and a presentation deck:

- `insights.ipynb` is the main exploration notebook.
- `src/popularity_insights.py` groups content into popularity tiers and compares platform mix and cross-platform spread.
- `src/half_life.py` estimates content life time from cohort trajectories using a regression-based half-life approach.
- `src/bayesian_uplift.py` fits a hierarchical Bayesian model to quantify weekday and format uplift in early views.
- `insights_slides.pdf` is the presentation version of the analysis.

## What the project looks at

The analysis is built around anonymized video content data from several social platforms. Typical questions covered by the project include:

- Which platforms over-index in low, mid, high, and viral view tiers.
- Whether broader cross-platform distribution is associated with stronger outcomes.
- How quickly different videos saturate after publication.
- Which publishing weekdays and content formats show the strongest expected uplift.

## Project structure

```text
.
├── insights.ipynb
├── insights_slides.pdf
├── pyproject.toml
└── src/
    ├── bayesian_uplift.py
    ├── half_life.py
    └── popularity_insights.py
```

## Setup

Dependency metadata lives in `pyproject.toml`.

Create an environment and install the project dependencies with your preferred tool, then open the notebook or import the modules from `src/` for reuse in further analysis.

## Notes

- The repository is designed for analysis code and derived insights.
- Source data is assumed to be anonymized and kept outside the repo unless explicitly added later.
