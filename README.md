# NSDUH Mental Health Access Explorer

This project explores mental health treatment access patterns using SAMHSA NSDUH public-use data. The dashboard focuses on treatment rate differences across age, income, insurance coverage, poverty level, county type, race/ethnicity, and reported barriers to treatment.

# Research Question

How do mental health treatment rates vary across demographic, socioeconomic, insurance, and geographic groups in the NSDUH dataset?

# How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

3. The app runs with the included first-50-lines sample by default. To use the full NSDUH file, upload the full tab-delimited `.txt` file in the dashboard sidebar.

# Dashboard Features

- Sidebar filters for age, income, insurance, county type, and race/ethnicity
- Summary metric cards for rows, treatment rate, insured rate, and reported barriers
- Interactive Plotly distribution explorer
- Interactive Plotly treatment comparison explorer
- Barrier explorer for treatment access obstacles
- Drill-down view comparing treatment rate across two concepts at once
- Data preview for reproducibility and checking filtered records