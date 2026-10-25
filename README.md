# Ames Housing — Data Cleaning and Quality Audit

This is my Week 1 data-quality task at NeuroFive Solutions using the Kaggle **House Prices — Advanced Regression Techniques** (Ames Housing) dataset. The work covers missing values, duplicates, outliers, data types, and cleaning decisions. The regression target is `SalePrice`; I did not use it for the cleaning decisions.

## Main results

- Starting dataset: **1,460 rows × 81 columns**
- Original missing cells: **6,965**, across **19 columns**
- Exact duplicate records: **0**
- Possible near-duplicate records: **85**, in **36 groups** (kept for review)
- IQR outliers: **31** for `GrLivArea`, **69** for `LotArea`, and **21** for `GarageArea`
- Cleaned dataset: **1,460 rows × 89 columns**
- Missing values left: **81**, all in `GarageYrBlt`, intentionally preserved for houses without garages

See [DATA_QUALITY_REPORT.md](DATA_QUALITY_REPORT.md) for the full explanation and justifications.

## Project files

| File | Purpose |
|---|---|
| `Ames_Housing_Simple.ipynb` | Main notebook: cleaning and quality audit in Google Colab |
| `DATA_QUALITY_REPORT.md` | Written report with the decisions and actual results |
| `ames_housing_cleaned.csv` | Final cleaned dataset |
| `missing_values_report.csv` | Missing-value counts and percentages across all original columns |
| `near_duplicate_candidates.csv` | Candidate similar records for review |
| `outlier_report.csv` | IQR thresholds and flag counts |
| `data_types_report.csv` | Final Pandas data types |
| `ames_housing_simple.py` | Optional standalone Python version of the notebook |

## How to run it in Google Colab

1. Download `train.csv` from the [official Kaggle competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
2. Open `Ames_Housing_Simple.ipynb` in [Google Colab](https://colab.research.google.com/).
3. In the left **Files** pane, choose **Upload**, then upload `train.csv` to the Colab session. Keep the name exactly `train.csv`.
4. Select **Runtime → Run all**.
5. Download the generated CSV files from the left Files pane. Colab's session files are temporary, so download anything you want to keep.

The notebook uses only Python and Pandas, available in Colab. It does not require scaling or encoding. The original `train.csv` is **not included** here; download it from Kaggle using the link above.

## Notes

- The report describes the output files included in this repository.
- Outliers are retained and flagged rather than automatically deleted.
- The `GarageYrBlt` missing values are intentional. Do not fill them with an invented construction year.
- When building a model later, first split the dataset and compute imputation and outlier thresholds using only the training portion to avoid leakage.
- CSV files do not retain Pandas categorical dtypes, so see `data_types_report.csv` when reloading the data.
