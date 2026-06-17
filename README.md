# 📊 SandeepStician — Data Analysis Toolkit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)
![Plotly](https://img.shields.io/badge/Charts-Plotly-56B4E9)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

> **SandeepStician** is a user-friendly, modular, and visually appealing data analysis toolkit built with **Python**, **Streamlit**, and **Plotly**.  
> It helps users explore datasets, generate descriptive statistics, visualize data, test assumptions, run hypothesis tests, fit probability distributions, and understand the Central Limit Theorem through an interactive dashboard.

---

## 🌐 Public Toolkit Link

GitHub Repository:  
**https://github.com/hidella-sand/python-statistics-dashboard.git**

Live Application
**https://sandeepstician.streamlit.app/**

---

## 🎯 Assignment Objective

This project was developed for the **Tools and Methods of Data Analysis** assignment.

The main objective is to design and implement a reusable **Data Analysis Toolkit** that helps users perform common statistical and exploratory analysis tasks efficiently and intuitively.

The toolkit is designed for users with limited programming experience. Instead of writing code manually, users can upload a dataset, select columns, choose analysis methods, and interpret results through a clean interactive interface.

---

## ✨ Key Features

### 1. 📁 Dataset Import and Preparation

Users can upload datasets directly into the toolkit.

Supported file formats:

- `.csv`
- `.xlsx`

The app automatically displays:

- Number of rows
- Number of columns
- Missing values
- Duplicate rows
- Dataset preview
- Column summary

The toolkit also provides column selection so users can remove unnecessary variables such as IDs, names, emails, or other identifier columns before analysis.

---

### 2. 🔍 Dataset Overview

After selecting useful columns, the toolkit gives a clear overview of the cleaned working dataset.

It displays:

- Total selected rows
- Number of selected columns
- Numerical column count
- Categorical column count
- Missing-value percentage
- Duplicate row count
- Data types
- Unique values per column
- Selected dataset preview

This helps users understand the structure and quality of the dataset before performing statistical analysis.

---

### 3. 📌 Descriptive Statistics

The toolkit automatically detects whether a selected variable is:

- Continuous numerical
- Discrete numerical
- Binary categorical
- Categorical
- Identifier-like
- Text-like

For numerical variables, the app calculates:

- Mean
- Median
- Mode
- Minimum
- Maximum
- Range
- Variance
- Standard deviation
- Skewness
- Excess kurtosis
- Missing-value percentage

For categorical variables, the app calculates:

- Frequency counts
- Percentages
- Number of categories
- Mode category
- Mode percentage
- Missing values

Each result includes simple interpretation notes to help users understand what the numbers mean.

---

### 4. 📈 Interactive Data Visualization

The toolkit uses **Plotly** to create clean and interactive visualizations.

Available visualizations include:

- Histogram
- Boxplot
- Estimated PDF / KDE
- CDF
- Q-Q plot
- PMF for discrete numerical variables
- Count bar charts
- Percentage bar charts
- Donut charts

The charts use a soft professional color palette:

```python
["#56B4E9", "#D55E00", "#009E73", "#E69F00"]
```

This makes the interface more readable, pleasant, and presentation-friendly.

---

### 5. 🧪 Normality Testing

The toolkit includes normality testing to help users check whether a numerical variable is approximately normally distributed.

Included tests:

- Shapiro-Wilk test
- Kolmogorov-Smirnov test
- Anderson-Darling test

The app provides:

- Null and alternative hypotheses
- Test statistic
- p-value
- Decision
- Plain-English conclusion
- Histogram
- Q-Q plot

This is useful before choosing between parametric and non-parametric tests.

---

### 6. 📊 Hypothesis Testing

The toolkit includes several common hypothesis tests learned during the course.

#### T-tests

Included t-tests:

- One-sample t-test
- Independent two-sample t-test
- Paired t-test

Each t-test includes:

- Assumption checks
- Diagnostic plots
- Test result table
- p-value interpretation
- Plain-English conclusion

#### Z-tests

Included z-tests:

- One-sample mean z-test
- Two-sample mean z-test
- One-proportion z-test
- Two-proportion z-test

The z-test module helps users test means and proportions using interactive input fields and result visualizations.

#### ANOVA

Included ANOVA tests:

- One-way ANOVA
- Two-way ANOVA with interaction

The ANOVA section includes:

- Assumption checks
- Group comparison plots
- ANOVA result tables
- Tukey HSD post-hoc test for significant one-way ANOVA results
- Interpretation of main effects and interaction effects

#### Chi-square Tests

Included chi-square tests:

- Chi-square test of independence
- Chi-square goodness-of-fit test

The toolkit displays:

- Observed frequencies
- Expected frequencies
- Contingency tables
- Heatmaps
- Goodness-of-fit bar charts
- Cramer's V for association strength

---

### 7. 🔁 Non-parametric Tests

The toolkit includes non-parametric alternatives for situations where normality assumptions are weak.

Included tests:

- Mann-Whitney U test
- Wilcoxon signed-rank test
- Kruskal-Wallis test
- Friedman test

These tests are useful for:

- Skewed data
- Ordinal/rank-based data
- Outlier-heavy data
- Small samples
- Non-normal distributions

---

### 8. 📉 Probability Distribution Fitting

The toolkit includes probability distribution fitting using least-squares comparison.

Supported distributions:

- Normal distribution
- Exponential distribution
- Uniform distribution

The module calculates:

- SSE
- MSE
- RMSE
- Kolmogorov-Smirnov statistic
- KS p-value

It also provides:

- Histogram with fitted PDF curves
- Single-distribution fit plot
- Q-Q plot against selected distribution
- Interpretation of best-fitting distribution

---

### 9. 📚 Central Limit Theorem Simulation

The Central Limit Theorem module allows users to simulate repeated sampling.

Users can choose:

- Numerical column
- Sample size
- Number of samples
- Number of bins

The toolkit shows:

- Original data distribution
- Sampling distribution of the mean
- Normal curve overlay
- Standard error comparison
- Sample-size comparison plots
- Normality check on sample means

This helps users visually understand how sample means become more normally distributed as sample size increases.

---

### 10. 📄 Export Report

The toolkit includes a report export feature that generates a Markdown report summarizing the dataset and completed analyses.

The report can include:

- Dataset overview
- Selected columns
- Distribution fitting results
- CLT simulation results
- Key interpretation notes

---

## 🧱 Project Structure

```text
python-statistics-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
│
└── utils/
    ├── data_loader.py
    ├── descriptive_stats.py
    ├── smart_descriptive.py
    ├── visualizations.py
    ├── normality_tests.py
    ├── assumption_checks.py
    ├── t_tests.py
    ├── z_tests.py
    ├── anova_tests.py
    ├── chi_square_tests.py
    ├── nonparametric_tests.py
    ├── distribution_fitting.py
    ├── clt_simulation.py
    ├── export_report.py
    └── ui_components.py
```

---

## ⚙️ How to Run the Toolkit

### Step 1: Clone the Repository

```bash
git clone https://github.com/hidella-sand/python-statistics-dashboard.git
cd python-statistics-dashboard
```

### Step 2: Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main packages manually:

```bash
pip install streamlit pandas numpy scipy statsmodels plotly openpyxl
```

### Step 4: Run the App

```bash
streamlit run app.py
```

The app will open in your browser. If it does not open automatically, copy the local URL shown in the terminal and paste it into your browser.

---

## 🧭 How to Use the Toolkit

### Step 1: Import Dataset

Upload a CSV or Excel file.

The app will display:

- Dataset size
- Missing values
- Duplicate rows
- Dataset preview

### Step 2: Select Columns

Choose the columns you want to keep for analysis.

Remove columns such as:

- ID numbers
- Names
- Emails
- Ticket numbers
- Unnecessary text columns

### Step 3: Explore Dataset

Go to **Dataset Overview** to understand selected columns, data types, missing values, and overall dataset quality.

### Step 4: Run Descriptive Statistics

Open **Descriptive Statistics**, select a column, and review summary measures and interpretations.

### Step 5: Create Visualizations

Use the **Visualizations** page to generate interactive plots such as histograms, boxplots, KDE curves, CDFs, Q-Q plots, and PMFs.

### Step 6: Perform Statistical Tests

Use the hypothesis testing pages based on your analysis goal:

| Goal | Recommended Page |
|---|---|
| Check normality | Normality Tests |
| Compare one mean to a known value | T-tests or Z-tests |
| Compare two independent groups | T-tests or Non-parametric Tests |
| Compare paired before/after data | T-tests or Wilcoxon signed-rank |
| Compare three or more groups | ANOVA or Kruskal-Wallis |
| Test categorical association | Chi-square Tests |
| Compare proportions | Z-tests |
| Fit probability distributions | Distribution Fitting |
| Understand sampling distributions | Central Limit Theorem |

### Step 7: Export Report

After completing selected analyses, use the export option to download a Markdown report.

---

## 🧪 Example Analyses

### Example 1: Titanic Dataset

Possible analysis questions:

- Is survival related to passenger class?
- Is survival related to sex?
- Are fare values normally distributed?
- What is the distribution of passenger ages?
- Can a chi-square test show association between survival and passenger class?

Useful toolkit pages:

- Dataset Overview
- Descriptive Statistics
- Visualizations
- Normality Tests
- Chi-square Tests
- T-tests

### Example 2: Housing Dataset

Possible analysis questions:

- What is the distribution of median house value?
- Is median income normally distributed?
- Does ocean proximity affect median house value?
- Which probability distribution fits house value best?
- How does the Central Limit Theorem behave for median income?

Useful toolkit pages:

- Descriptive Statistics
- Visualizations
- ANOVA
- Distribution Fitting
- Central Limit Theorem

---

## 🖼️ Suggested Screenshots for Submission

Screenshots are optional but recommended for the assignment.

Recommended screenshots:

```text
screenshots/
├── 01_import_dataset.png
├── 02_column_selection.png
├── 03_dataset_overview.png
├── 04_descriptive_statistics.png
├── 05_visualizations.png
├── 06_normality_tests.png
├── 07_t_tests.png
├── 08_anova.png
├── 09_chi_square.png
├── 10_distribution_fitting.png
└── 11_clt_simulation.png
```

You can add screenshots to the README like this:

```markdown
![Dataset Overview](screenshots/03_dataset_overview.png)
![Visualization Example](screenshots/05_visualizations.png)
![Distribution Fitting](screenshots/10_distribution_fitting.png)
```

---

## 🎨 Design and User Experience

The toolkit was designed with the following goals:

- Easy to use
- Clean and professional layout
- Sidebar-based navigation
- Clear modular structure
- Interactive Plotly charts
- Plain-English statistical interpretations
- Accessible for users with limited programming experience
- Light visual theme with a pleasant scientific color palette

The color palette used throughout the application is:

| Color | Hex Code | Usage |
|---|---:|---|
| Sky Blue | `#56B4E9` | Main plots, highlights |
| Vermillion | `#D55E00` | Contrast, warnings, comparison |
| Bluish Green | `#009E73` | Success, means, fitted curves |
| Warm Orange | `#E69F00` | Median, reference lines, caution |

---

## 📦 Technologies Used

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web app dashboard |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| SciPy | Statistical tests and distributions |
| Statsmodels | ANOVA and Tukey HSD |
| Plotly | Interactive visualizations |
| OpenPyXL | Excel file support |

---

## ✅ Assignment Requirement Coverage

| Assignment Requirement | Toolkit Implementation |
|---|---|
| Data import/export | CSV/XLSX upload and Markdown report export |
| Data exploration | Dataset preview, column summary, selected overview |
| Data cleaning/preprocessing | Column selection and missing-value overview |
| Missing-value analysis | Missing count and missing percentage |
| Descriptive statistics | Numerical and categorical summaries |
| Summary tables | Descriptive, test, and distribution result tables |
| Data visualization | Interactive Plotly charts |
| Probability distributions | Normal, Exponential, Uniform fitting |
| PDF/CDF | KDE/PDF, CDF, fitted PDF curves |
| Goodness-of-fit | MSE, RMSE, SSE, KS test |
| Statistical inference | T-tests, Z-tests, ANOVA, Chi-square, non-parametric tests |
| Normality testing | Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling |
| Easy to use | Sidebar navigation and no-code interface |
| Well documented | README and in-app explanations |
| Visually appealing | Soft professional UI and interactive charts |
| Modular | Separate utility modules in `utils/` |

---

## 🚀 Future Improvements

Possible future enhancements:

- Add confidence interval calculators
- Add correlation and regression analysis
- Add automated cleaning options
- Add downloadable PDF reports
- Add more probability distributions
- Add effect-size visualizations
- Add sample datasets inside the repository
- Deploy the app publicly using Streamlit Community Cloud

---

## 👤 Author

**Sandeep Hidellarachchi**

MSc Big Data & Artificial Intelligence  
SRH University Leipzig

GitHub: [hidella-sand](https://github.com/hidella-sand)

---

## 📌 Final Note

SandeepStician is designed as a practical learning and analysis toolkit. It combines statistical methods, clean visualizations, and user-friendly explanations into one interactive application.

The goal is not only to calculate results, but also to help users understand what each result means.
