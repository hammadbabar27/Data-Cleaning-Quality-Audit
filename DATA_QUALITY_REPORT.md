# Data Quality Report — Ames Housing Dataset

**Dataset:** House Prices — Advanced Regression Techniques (Ames Housing)  
**Source:** [Kaggle competition dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)  
**Task:** Week 1 — Data Cleaning & Data Quality Audit (Regression Track) at NeuroFive Solutions ML Internship
**Tools:** Python, Pandas, Google Colab

## 1. Introduction

The aim of this task was to examine the housing data before using it for machine learning. I looked for missing values, duplicate records, possible outliers, and columns stored with the wrong data type. My main goal was to clean the data without losing useful information or changing the house sale prices.

I worked with the Kaggle **training dataset**, which contains 1,460 houses and 81 original columns. These include 79 input features, an `Id` column, and the target `SalePrice`. I did not use `SalePrice` to fill missing values or decide which houses were outliers, because this first task is about data quality rather than prediction.

## 2. Initial data check

| Item | Result |
|---|---:|
| Original rows | 1,460 |
| Original columns | 81 |
| Input features (excluding `Id` and `SalePrice`) | 79 |
| Columns with at least one missing value | 19 |
| Total missing cells in the raw dataset | 6,965 |
| Columns with no missing values | 62 |
| Rows after cleaning | 1,460 |
| Columns after cleaning | 89 |
| Missing cells remaining | 81 |

The column count increased from 81 to 89 because I added **five missing-value indicator columns** and **three outlier indicator columns**. No original rows or columns were deleted.

## 3. Missing-value analysis

Missing data was the biggest issue in this dataset. Some columns had only one missing value, while others were empty for nearly every house. However, I found that a blank cell does not always mean the data was recorded incorrectly. For example, a missing `PoolQC` value normally means that the house does not have a pool.

The columns with missing values were:

| Column | Missing | Percentage |
|---|---:|---:|
| `PoolQC` | 1,453 | 99.52% |
| `MiscFeature` | 1,406 | 96.30% |
| `Alley` | 1,369 | 93.77% |
| `Fence` | 1,179 | 80.75% |
| `FireplaceQu` | 690 | 47.26% |
| `LotFrontage` | 259 | 17.74% |
| `GarageType` | 81 | 5.55% |
| `GarageFinish` | 81 | 5.55% |
| `GarageQual` | 81 | 5.55% |
| `GarageCond` | 81 | 5.55% |
| `GarageYrBlt` | 81 | 5.55% |
| `BsmtExposure` | 38 | 2.60% |
| `BsmtFinType2` | 38 | 2.60% |
| `BsmtQual` | 37 | 2.53% |
| `BsmtCond` | 37 | 2.53% |
| `BsmtFinType1` | 37 | 2.53% |
| `MasVnrType` | 8 | 0.55% |
| `MasVnrArea` | 8 | 0.55% |
| `Electrical` | 1 | 0.07% |

The remaining **62 original columns had no missing values**. The complete 81-column list, including their counts and percentages, is provided in **Appendix A** and in `missing_values_report.csv`.

### Why the missing values needed different treatment

A simple approach like replacing every missing cell with the mean would not make sense here. The mean cannot describe the absence of a swimming pool or garage, and it can also be affected by unusually large numerical values.

In statistics, *MCAR* means a value is missing for reasons unrelated to the data, while *MNAR* means the reason for missingness is related to the missing information itself. I cannot prove which mechanism applies to every column using this dataset alone. Many blanks are more clearly **structural missing values**: the feature does not exist for that house. I handled these separately from values that are genuinely unknown.

## 4. Cleaning decisions and reasons

I used the following approaches rather than dropping all rows with blanks.

| Column or group | What I did | Reason |
|---|---|---|
| `Alley`, `Fence`, `MiscFeature` | Replaced missing values with `None`. | The blank usually means that the property has no such feature. |
| `PoolQC` | Used `None` if `PoolArea` was zero; otherwise `Unknown`. | No pool and unknown pool quality should not be treated as the same thing. |
| `FireplaceQu` | Used `None` when `Fireplaces` was zero; otherwise `Unknown`. | Missing quality is expected when a house has no fireplace. |
| `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` | Used `None` when `GarageArea` was zero; otherwise `Unknown`. | These details are not applicable to a house without a garage. |
| `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` | Used `None` when `TotalBsmtSF` was zero; otherwise `Unknown`. | This distinguishes a house with no basement from one with incomplete basement information. |
| `LotFrontage` | Filled missing values with the median of the same neighborhood, with an overall median fallback. Added `LotFrontage_missing`. | Street frontage depends on the area; the median is less sensitive to extreme lot sizes. |
| `Electrical` | Filled the one missing value with the most common category. Added `Electrical_missing`. | There was just one unknown value, so keeping that house was reasonable. |
| `MasVnrType` | Kept genuine text `None`; used `None` where missing type was paired with zero veneer area, otherwise `Unknown`. Added a missing flag. | An absent veneer is different from an unrecorded veneer type. |
| `MasVnrArea` | Used zero for an established no-veneer case, otherwise the median of positive veneer areas. Added a missing flag. | Zero should mean no veneer, not simply a measurement that was not recorded. |
| `GarageYrBlt` | Left its missing years empty, and added `GarageYrBlt_missing`. | A house without a garage cannot have a real garage construction year. |

I deliberately **kept** the original rows instead of dropping houses whenever a field was missing. Deleting rows or highly incomplete columns such as `PoolQC` would remove information that may be useful later.

Two small details also appeared in the results: `BsmtExposure` and `BsmtFinType2` each ended up with **one `Unknown` value** where a basement exists but the particular detail was not recorded. That is more accurate than labeling those houses as having no basement.

The missing indicator columns show how many values were originally absent:

| Indicator | Number flagged |
|---|---:|
| `LotFrontage_missing` | 259 |
| `Electrical_missing` | 1 |
| `MasVnrType_missing` | 8 |
| `MasVnrArea_missing` | 8 |
| `GarageYrBlt_missing` | 81 |

After cleaning, **only the 81 values in `GarageYrBlt` remain missing**. I checked that these 81 records all have `GarageArea = 0`, so leaving the year empty is intentional and not an unfinished cleaning step.

## 5. Duplicate and near-duplicate check

I checked exact duplicates, duplicates after ignoring the `Id` column, and repeated IDs. I also looked for houses that shared the same values in four features: `Neighborhood`, `YearBuilt`, `OverallQual`, and `GrLivArea`.

| Check | Result |
|---|---:|
| Fully identical records | 0 |
| Identical records excluding `Id` | 0 |
| Repeated `Id` values | 0 |
| Possible near-duplicate rows | 85 |
| Groups of similar records | 36 |

The 85 possible near-duplicates are **candidates for review**, not 85 proven duplicate houses. For example, six houses in the `CollgCr` neighborhood have `YearBuilt = 2004`, `OverallQual = 6`, and `GrLivArea = 848`. Their IDs differ, their lot areas are not all the same, and their sale prices range from **141,000 to 155,900**. They could be similar houses built as part of the same development.

**Decision:** I kept all 85 candidates. Matching a few characteristics is not enough evidence that two records describe the same property. The file `near_duplicate_candidates.csv` contains the records for further manual checking.

## 6. Outlier detection using IQR

I checked three numerical columns with the interquartile range (IQR) method:

- **IQR = Q3 − Q1**
- **Lower bound = Q1 − 1.5 × IQR**
- **Upper bound = Q3 + 1.5 × IQR**

Values outside these limits were marked as possible outliers. I added a separate flag for each column instead of deleting those houses.

| Feature | Lower bound | Upper bound | Flagged rows | Decision |
|---|---:|---:|---:|---|
| `GrLivArea` | 158.62 | 2,747.62 | 31 | Keep and flag |
| `LotArea` | 1,481.50 | 17,673.50 | 69 | Keep and flag |
| `GarageArea` | -27.75 | 938.25 | 21 | Keep and flag |

**`GrLivArea` (above-ground living area):** The largest value is **5,642 sq ft**, and 31 houses were above the calculated upper bound. Large houses are unusual, but their size alone does not show an entry mistake. I kept them and marked them with `GrLivArea_outlier`.

**`LotArea` (lot size):** This column has **69 flagged values**, including 67 above the upper bound and two below the lower bound. The largest lot is **215,245 sq ft**. Some properties genuinely have much larger or smaller lots than others, so I did not remove them without additional evidence. These were marked with `LotArea_outlier`.

**`GarageArea` (garage size):** There are **21 flagged values**, all above the upper bound; the largest garage area is **1,418 sq ft**. The negative lower bound is only a result of the IQR formula, not evidence of negative garage sizes. Zero garage area is valid for a home with no garage. These records were marked with `GarageArea_outlier`.

A total of **104 different houses** have at least one of these three outlier flags. This is less than 31 + 69 + 21 because some houses are unusual in more than one feature.

### Are these errors or genuine extreme values?

Based on the available checks, I found **no evidence strong enough to label these values as data-entry errors**. The cleaned data contains no nonpositive `LotArea` values, no nonpositive `GrLivArea` values, and no negative `GarageArea` values. That does not prove every large value is correct; it means a statistical outlier should be reviewed before being changed or removed.

## 7. Data-type corrections

Some columns had numbers that were actually labels. I corrected these types so they would not be misunderstood during later analysis.

| Column(s) | Final type | Why |
|---|---|---|
| `MSSubClass` | Categorical | Its numeric codes describe building classes, not a measurable quantity. |
| `MoSold` | Categorical | Month numbers identify a month rather than a measurement. |
| `MSZoning`, `Neighborhood`, and other text descriptions | Categorical | These values represent groups or labels. |
| `GarageYrBlt` | Nullable integer (`Int64`) | Years are whole numbers, but some properties have no garage year. |
| `LotArea`, `GrLivArea`, `GarageArea` | Numeric | These are actual measurements. |
| `OverallQual`, `OverallCond` | Integer | These are ordered ratings; I did not encode or scale them in this task. |
| `SalePrice` | Integer | The regression target was not transformed. |

The final type report contains **45 categorical columns, 41 standard integer columns, two floating-point columns, and one nullable integer column**. The full list is in `data_types_report.csv`.

The cleaned CSV keeps the values, but a CSV file does not preserve Pandas' `category` and nullable-integer metadata. Those types must be reapplied when loading the CSV for future modeling.

## 8. Final verification

The outputs show that the cleaned dataset has **1,460 rows and 89 columns**. There are still 81 intentional missing construction-year values. The cleaning code checks that `Id` and `SalePrice` match the original data, and it does not modify or use the target in its cleaning calculations.

The outputs also show that no exact duplicates were found and no rows were removed. No scaling, one-hot encoding, or predictive model was applied because those steps belong to the next stage.

| Final check | Outcome |
|---|---|
| All original house rows retained | Yes, 1,460 rows |
| Eight audit flags added | Yes, 5 missingness + 3 outlier flags |
| Duplicate rows removed | None; no exact duplicates found |
| Suspected outliers removed | None; flagged for review |
| Original target transformed | No; `SalePrice` was left unchanged by the cleaning code |
| Missing construction years invented | No |
| Final dataset saved | `ames_housing_cleaned.csv` |

## 9. What I learned and what comes next

The main lesson from this audit is that missing data needs context. A blank garage field is not the same problem as an unknown street-frontage measurement. I also learned that two records can look similar without being duplicates, and an unusually large house is not automatically a mistake.

For this exercise I used the full raw dataset, as required. **Before training a model**, I will split the data first and calculate median, mode, and outlier thresholds using the training set only. Applying values calculated from all the data to a validation or test set would introduce data leakage. Encoding and scaling will also be handled in the modeling stage rather than in this data-quality audit.

## 10. Files produced

- `ames_housing_cleaned.csv` — cleaned records with indicator columns.
- `missing_values_report.csv` — missing-value counts and percentages for every original column.
- `near_duplicate_candidates.csv` — 85 records sharing selected characteristics.
- `outlier_report.csv` — IQR bounds and flagged counts.
- `data_types_report.csv` — data types of all 89 final columns.

## Appendix A — Full missing-value report for all 81 original columns

This table is the original missing-value audit, before any imputation.

| Column | Missing count | Missing (%) |
|---|---:|---:|
| `PoolQC` | 1,453 | 99.52% |
| `MiscFeature` | 1,406 | 96.30% |
| `Alley` | 1,369 | 93.77% |
| `Fence` | 1,179 | 80.75% |
| `FireplaceQu` | 690 | 47.26% |
| `LotFrontage` | 259 | 17.74% |
| `GarageFinish` | 81 | 5.55% |
| `GarageQual` | 81 | 5.55% |
| `GarageYrBlt` | 81 | 5.55% |
| `GarageType` | 81 | 5.55% |
| `GarageCond` | 81 | 5.55% |
| `BsmtExposure` | 38 | 2.60% |
| `BsmtFinType2` | 38 | 2.60% |
| `BsmtCond` | 37 | 2.53% |
| `BsmtFinType1` | 37 | 2.53% |
| `BsmtQual` | 37 | 2.53% |
| `MasVnrArea` | 8 | 0.55% |
| `MasVnrType` | 8 | 0.55% |
| `Electrical` | 1 | 0.07% |
| `Condition2` | 0 | 0.00% |
| `BldgType` | 0 | 0.00% |
| `Neighborhood` | 0 | 0.00% |
| `LandSlope` | 0 | 0.00% |
| `LotConfig` | 0 | 0.00% |
| `Condition1` | 0 | 0.00% |
| `LandContour` | 0 | 0.00% |
| `LotShape` | 0 | 0.00% |
| `Street` | 0 | 0.00% |
| `LotArea` | 0 | 0.00% |
| `MSSubClass` | 0 | 0.00% |
| `MSZoning` | 0 | 0.00% |
| `Id` | 0 | 0.00% |
| `Utilities` | 0 | 0.00% |
| `HouseStyle` | 0 | 0.00% |
| `Foundation` | 0 | 0.00% |
| `ExterQual` | 0 | 0.00% |
| `ExterCond` | 0 | 0.00% |
| `BsmtUnfSF` | 0 | 0.00% |
| `TotalBsmtSF` | 0 | 0.00% |
| `Heating` | 0 | 0.00% |
| `BsmtFinSF1` | 0 | 0.00% |
| `Exterior2nd` | 0 | 0.00% |
| `Exterior1st` | 0 | 0.00% |
| `RoofMatl` | 0 | 0.00% |
| `RoofStyle` | 0 | 0.00% |
| `YearRemodAdd` | 0 | 0.00% |
| `OverallQual` | 0 | 0.00% |
| `YearBuilt` | 0 | 0.00% |
| `OverallCond` | 0 | 0.00% |
| `BsmtHalfBath` | 0 | 0.00% |
| `BsmtFullBath` | 0 | 0.00% |
| `GrLivArea` | 0 | 0.00% |
| `LowQualFinSF` | 0 | 0.00% |
| `2ndFlrSF` | 0 | 0.00% |
| `1stFlrSF` | 0 | 0.00% |
| `CentralAir` | 0 | 0.00% |
| `HeatingQC` | 0 | 0.00% |
| `BsmtFinSF2` | 0 | 0.00% |
| `Fireplaces` | 0 | 0.00% |
| `TotRmsAbvGrd` | 0 | 0.00% |
| `KitchenQual` | 0 | 0.00% |
| `KitchenAbvGr` | 0 | 0.00% |
| `Functional` | 0 | 0.00% |
| `FullBath` | 0 | 0.00% |
| `HalfBath` | 0 | 0.00% |
| `GarageCars` | 0 | 0.00% |
| `GarageArea` | 0 | 0.00% |
| `BedroomAbvGr` | 0 | 0.00% |
| `PavedDrive` | 0 | 0.00% |
| `EnclosedPorch` | 0 | 0.00% |
| `3SsnPorch` | 0 | 0.00% |
| `OpenPorchSF` | 0 | 0.00% |
| `WoodDeckSF` | 0 | 0.00% |
| `PoolArea` | 0 | 0.00% |
| `ScreenPorch` | 0 | 0.00% |
| `MiscVal` | 0 | 0.00% |
| `MoSold` | 0 | 0.00% |
| `YrSold` | 0 | 0.00% |
| `SaleType` | 0 | 0.00% |
| `SaleCondition` | 0 | 0.00% |
| `SalePrice` | 0 | 0.00% |


## Reference

Kaggle. *House Prices — Advanced Regression Techniques (Ames Housing).*  
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
