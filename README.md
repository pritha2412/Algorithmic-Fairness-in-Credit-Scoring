# Algorithmic Fairness & Bias in Credit Scoring

A comparative machine-learning study investigating fairness and bias in automated credit-scoring models, with a focus on how different bias-mitigation strategies affect both fairness and predictive performance.

The project evaluates conventional machine-learning models on two datasets, audits disparities across protected groups, and compares three different mitigation approaches:

- Reweighing
- Adversarial Debiasing
- Calibrated Equalized Odds (CE-Odds)

Rather than assuming that a mitigation technique is universally beneficial, the project examines the fairness–utility trade-off across datasets, models, protected attributes, and fairness metrics.

---

## Project Objective

The objective of this project is to:

1. Train standard classification models on the Adult Income and German Credit datasets.
2. Establish a baseline fairness profile before mitigation.
3. Measure disparities across multiple protected attributes.
4. Apply different bias-mitigation techniques.
5. Compare the effect of mitigation on predictive performance and fairness.
6. Study whether fairness improvements come at the cost of predictive performance.
7. Compare how mitigation behaves across different datasets and protected groups.

---

## Research Questions

### RQ1 — Baseline Bias

Do standard machine-learning models exhibit measurable disparities across protected demographic groups?

### RQ2 — Mitigation Effectiveness

How effectively do Reweighing, Adversarial Debiasing, and Calibrated Equalized Odds reduce measured disparities?

### RQ3 — Fairness vs Performance

Do improvements in fairness lead to reductions in predictive performance?

### RQ4 — Dataset Dependence

Does the effectiveness of a mitigation method depend on the underlying dataset?

### RQ5 — Protected-Attribute Dependence

Does a mitigation technique behave differently when targeting different protected attributes?

### RQ6 — Intersectional Fairness

Are disparities more pronounced when protected attributes are considered jointly rather than independently?

---

## Datasets

### 1. Adult Income

The Adult Income dataset is used to study income-classification decisions and demographic disparities.

The project evaluates fairness with respect to:

- Age
- Sex
- Age × Sex intersection

### 2. German Credit

The German Credit dataset contains credit-related information used for binary credit-risk classification.

The project evaluates fairness with respect to:

- Age
- Sex
- Age × Sex intersection

The intersectional analysis considers combinations of age and sex groups, including:

- Female + under 25
- Female + 25 and above
- Male + under 25
- Male + 25 and above

---

## Protected Attributes

| Protected dimension | Description |
|---|---|
| Age | Age-based group comparison |
| Sex | Sex-based group comparison |
| Age × Sex | Intersectional comparison of age and sex |

The intersectional analysis is important because evaluating age and sex separately can hide disparities affecting specific combinations of demographic groups.

---

## Machine Learning Models

Three standard classification models were evaluated:

### Logistic Regression

A linear baseline model used as an interpretable reference point.

### Random Forest

An ensemble tree-based classifier using:

```text
n_estimators = 100
```

### XGBoost

A gradient-boosted tree model using:

```text
n_estimators = 100
```

---

## Experimental Methodology

### Baseline Evaluation

The baseline experiments use:

- 70/30 stratified train-test split
- Multiple random seeds
- Logistic Regression
- Random Forest
- XGBoost

The test set remains untouched during model training and fairness mitigation.

The experiments establish the predictive and fairness characteristics of the unmitigated models.

---

## Evaluation Metrics

### Predictive Performance

- **Accuracy** — proportion of correctly classified observations.
- **F1-score** — balance between precision and recall.
- **ROC-AUC** — discrimination/ranking performance using continuous prediction scores.

### Fairness Metrics

#### Demographic Parity Difference (DPD)

Measures the difference in positive prediction rates between protected groups.

**Ideal value: 0**

#### Equalized Odds Difference (EOD)

Measures disparity in prediction behavior across groups considering true-positive and false-positive related behavior.

**Ideal value: 0**

#### Equal Opportunity Difference (EOPD)

Measures differences in true-positive rates between protected groups.

**Ideal value: 0**

#### Disparate Impact (DI)

Measures the ratio of favorable outcome rates between groups.

**Ideal value: 1**

Values substantially below 1 indicate stronger disparity in favorable outcomes.

---

## Bias-Mitigation Methods

### 1. Reweighing

Reweighing is a **pre-processing** approach.

Different training samples are assigned different weights based on their protected-group and outcome combinations. The goal is to modify the influence of observations during training without changing the underlying test set.

Reweighing was evaluated while targeting:

- Age
- Sex

across the three machine-learning models.

### 2. Adversarial Debiasing

Adversarial Debiasing is an **in-processing** approach.

The model is trained while an adversarial component attempts to extract protected-attribute information from the learned representation/predictions.

Three adversarial strengths were evaluated:

| Strength | α |
|---|---:|
| Weak | 0.1 |
| Medium | 1 |
| Strong | 10 |

Adversarial Debiasing was evaluated for both Age-targeted and Sex-targeted mitigation.

### 3. Calibrated Equalized Odds

Calibrated Equalized Odds is a **post-processing** approach.

The final implementation uses:

- Sigmoid probability calibration
- `CalibratedClassifierCV`
- 5-fold calibration
- AIF360 `CalibratedEqOddsPostprocessing`
- `cost_constraint="fnr"`

The calibration/post-processing procedure was designed so that the **test set remains untouched until final evaluation**, avoiding test-data leakage.

Because CE-Odds operates on post-processed outputs rather than directly providing comparable continuous scores, ROC-AUC is reported as unavailable (`NaN`) for the post-processed results rather than being artificially reconstructed.

---

## Experimental Pipeline

```text
Datasets
   ↓
Data Loading & EDA
   ↓
Protected Attributes
   ├── Age
   ├── Sex
   └── Age × Sex
   ↓
Baseline Models
   ├── Logistic Regression
   ├── Random Forest
   └── XGBoost
   ↓
Baseline Performance + Fairness Audit
   ↓
Bias Mitigation
   ├── Reweighing
   ├── Adversarial Debiasing
   │      ├── α = 0.1
   │      ├── α = 1
   │      └── α = 10
   └── Calibrated Equalized Odds
   ↓
Comparative Evaluation
   ├── Accuracy
   ├── F1
   ├── ROC-AUC
   ├── DPD
   ├── EOD
   ├── EOPD
   └── DI
   ↓
Tables + Figures
   ↓
Fairness–Utility Analysis
```

---

## Code Structure

### `config.py`

Contains project configuration used by the experiments.

### `load_data.py`

Handles dataset loading and common data-loading functionality.

### Phase 1 — Exploratory Data Analysis

- `phase1a_eda_german.py`
- `phase1b_eda_adult.py`

These scripts perform exploratory analysis and generate visualizations for the datasets.

### Phase 2 — Baseline Models

`phase2_models.py`

Trains and evaluates:

- Logistic Regression
- Random Forest
- XGBoost

The baseline establishes the reference point against which all mitigation methods are compared.

### Phase 3 — Bias Mitigation

`phase3_reweighing.py`

Runs the Reweighing experiments.

`phase3_adversarial_debiasing.py`

Evaluates Adversarial Debiasing at α = 0.1, 1, and 10.

`phase3_calibrated_equalized_odds.py`

Runs the final CE-Odds implementation using calibration followed by AIF360 post-processing.

`generate_figures.py`

Generates the main research figures from the final summary CSV files and saves them to:

```text
outputs/figures/
```

---

## Results Organization

Each experiment produces:

### Raw results

```text
*_results.csv
```

These contain the individual experimental results.

### Summary results

```text
*_summary.csv
```

These aggregate the experimental results and are used for the final tables and figures.

The project contains summary files for:

- Phase 2 baseline
- Phase 3 Reweighing
- Phase 3 Adversarial Debiasing
- Phase 3 Calibrated Equalized Odds

---

## Figures

Five main figures were generated for the final analysis.

### Figure 1 — Baseline Disparate Impact

Compares baseline Disparate Impact across Age, Sex, and Age × Sex for the Adult Income and German Credit datasets.

The ideal DI reference is:

```text
DI = 1
```

### Figure 2 — Fairness–Accuracy Positioning

Plots predictive accuracy against a descriptive distance from the fairness ideal.

For this summary:

```text
DPD / EOD / EOPD → ideal = 0
DI → ideal = 1
```

Lower distance indicates closer proximity to the fairness ideals.

**Important:** this is a descriptive summary and is **not treated as an optimized fairness–utility score**.

### Figure 3 — Intersectional Disparate Impact

Compares intersectional DI before and after Reweighing, Adversarial Debiasing, and CE-Odds for both datasets.

### Figure 4 — Adversarial Debiasing Strength

Examines the effect of α = 0.1, 1, and 10 on Accuracy and intersectional DI for Adult and German, under Age-targeted and Sex-targeted mitigation.

### Figure 5 — Predictive Performance Comparison

Compares Accuracy and F1-score across Baseline, Reweighing, Adversarial Debiasing, and CE-Odds for Adult and German.

---

## Main Findings

### 1. Adult Income exhibited substantially larger baseline disparities

The Adult dataset showed considerably stronger baseline fairness disparities than German Credit, particularly for the intersectional **Age × Sex** dimension.

The intersectional DI values were especially far from the ideal value of 1.

### 2. German Credit was substantially fairer at baseline

The German Credit models generally started with much smaller fairness disparities, particularly for the Sex dimension.

This means that the need for mitigation was not uniform across datasets.

### 3. Reweighing produced modest fairness improvements

Reweighing generally improved selected fairness measures while preserving predictive performance relatively well.

However, improvements were not uniformly observed across every protected attribute and metric.

### 4. CE-Odds produced particularly strong fairness improvements on German Credit

CE-Odds was especially effective for German Credit.

Strong improvements were observed across Age and intersectional fairness, with several models moving close to the ideal fairness values.

These improvements were accompanied by some reduction in predictive performance in certain configurations.

### 5. Adversarial Debiasing was sensitive to its strength

The adversarial experiments demonstrated that the choice of α can substantially affect both fairness and predictive performance.

The Adult Age-targeted experiment at **α = 1** was particularly unstable, showing a major reduction in predictive performance.

Therefore, simply increasing adversarial fairness strength cannot be assumed to produce a universally better model.

### 6. Fairness improvements can come with a utility cost

Some mitigation configurations substantially reduced disparities while also reducing Accuracy or F1-score.

This highlights the central **fairness–utility trade-off** in algorithmic decision-making.

### 7. There is no universally best mitigation method

There is no single mitigation technique that dominates across every dataset, protected attribute, model, and fairness metric.

The effectiveness of a mitigation strategy depends on:

- Dataset
- Model
- Protected attribute
- Fairness metric
- Mitigation strength

Therefore, fairness mitigation should be evaluated empirically rather than assuming one technique is always superior.

---

## Fairness–Utility Interpretation

The project does **not** define a single mathematical "best model" based on an arbitrary combination of fairness and accuracy.

Instead, the results are interpreted using the individual metrics:

```text
Accuracy
F1
ROC-AUC

DPD
EOD
EOPD
DI
```

This avoids hiding important trade-offs behind a single composite score.

A model may improve one fairness metric while worsening another, so the analysis considers the full metric profile rather than selecting a winner using one subjective score.

---

## Experimental Considerations

### Multiple Seeds

The baseline methodology uses repeated runs with the following random seeds:

```text
42
123
456
789
1000
```

This reduces dependence on a single train/test split.

### Test-Set Isolation

The experiments were designed to prevent test-set leakage.

In particular, the CE-Odds pipeline separates model training/calibration from final evaluation on the untouched test set.

### AUC Handling

The baseline models provide continuous prediction scores, allowing ROC-AUC to be calculated.

However:

- Adversarial Debiasing results have AUC unavailable.
- CE-Odds post-processed outputs do not provide a directly comparable continuous score.

These values are therefore retained as `NaN` rather than replacing them with baseline AUC values or fabricated estimates.

---

## Limitations

### Dataset limitations

The analysis is based on two datasets and therefore cannot establish that the observed behavior generalizes to every credit-scoring dataset.

### Protected-attribute scope

The fairness analysis focuses on Age, Sex, and Age × Sex. Other protected characteristics are outside the scope of this study.

### Metric dependence

Different fairness definitions can produce different conclusions. A model that performs well according to one fairness metric may perform differently according to another.

### Mitigation trade-offs

Fairness improvements can reduce predictive performance, and the magnitude of this trade-off varies across configurations.

### Adversarial instability

Adversarial Debiasing can be sensitive to the selected fairness strength, and some configurations exhibited substantial instability.

### Post-processing and AUC

Because CE-Odds operates through post-processing, directly comparable ROC-AUC values are not available for those final post-processed predictions.

---

## Reproducibility

The project separates the workflow into:

```text
Data
   ↓
EDA
   ↓
Baseline
   ↓
Mitigation
   ↓
Evaluation
   ↓
Tables
   ↓
Figures
```

Generated results are stored in:

```text
outputs/tables/
outputs/figures/
```

To regenerate the figures:

```bash
python3 src/generate_figures.py
```

The figures are automatically saved to:

```text
outputs/figures/
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- AIF360
- Matplotlib
- Seaborn

---

## Conclusion

This project demonstrates that algorithmic fairness is not simply a matter of selecting a mitigation technique and observing whether one fairness metric improves.

Across Adult Income and German Credit, different mitigation methods behaved differently depending on the:

- Dataset
- Model
- Protected attribute
- Fairness metric
- Mitigation strength

Adult Income exhibited considerably stronger baseline disparities, particularly at the intersection of Age and Sex. German Credit was comparatively fairer at baseline, and CE-Odds produced particularly strong fairness improvements on this dataset.

At the same time, stronger fairness constraints could introduce predictive-performance costs or instability, particularly in adversarial settings.

Overall, the experiments support a **multi-dimensional evaluation of fairness**, where mitigation methods are compared using both fairness and predictive-performance metrics rather than being reduced to a single "best" score.

---

## Author

**Pritha Ranjan**

B.Tech — Electronics & Communication Engineering  
Indira Gandhi Delhi Technical University for Women (IGDTUW)
