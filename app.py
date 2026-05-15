# -----------------------------------------------------------------------------
# NSDUH Mental Health Access Explorer
# Isaac Moua || dpn34@txstate.edu 
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------------------------------
# Page Title Config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NSDUH Mental Health Access Explorer",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Constants and mappings
# -----------------------------------------------------------------------------
#DEFAULT_PATH = Path("data/NSDUH_2021_First50_Tab.txt")
DEFAULT_PATH = Path("data/NSDUH_2021_2024_Tab.txt")


COLUMNS_NEEDED = [
    "YEAR",
    "AGE3",
    "CATAGE",
    "IRSEX",
    "NEWRACE2",
    "IRPINC3",
    "IRINSUR4",
    "ANYHLTI2",
    "ADSEEDOC",
    "ADRX12MO",
    "MHTSEEKTX",
    "MHTUNCOST",
    "MHTUNINSCV",
    "MHTUNNOFND",
    "MHTUNTIME",
    "MHTUNPRIV",
    "MHTUNMED",
    "MHTUNCARE",
    "COUTYP4",
    "ANALWT_C",
    "ANALWT2_C1",
]

MISSING_CODES = [-9, -8, -7, -1, 85, 89, 94, 97, 98, 99, 991, 998, 999, 9991, 9998, 9999]

AGE_MAP = {
    1: "12-17",
    2: "18-20",
    3: "21-25",
    4: "26-34",
    5: "35-49",
    6: "50-64",
    7: "65+",
    8: "12-13",
    9: "14-15",
    10: "16-17",
    11: "18-25",
}
AGE_ORDER = ["12-13", "14-15", "16-17", "18-20", "21-25", "26-34", "35-49", "50-64", "65+", "12-17", "18-25"]

SEX_MAP = {1: "Male", 2: "Female"}
SEX_ORDER = ["Male", "Female"]

RACE_MAP = {
    1: "White",
    2: "Black/African American",
    3: "Native American/Alaska Native",
    4: "Native Hawaiian/Pacific Islander",
    5: "Asian",
    6: "More than one race",
    7: "Hispanic",
}
RACE_ORDER = [
    "White",
    "Black/African American",
    "Native American/Alaska Native",
    "Native Hawaiian/Pacific Islander",
    "Asian",
    "More than one race",
    "Hispanic",
]

INCOME_MAP = {
    1: "Less than $10,000",
    2: "$10,000-$19,999",
    3: "$20,000-$29,999",
    4: "$30,000-$39,999",
    5: "$40,000-$49,999",
    6: "$50,000-$74,999",
    7: "$75,000 or more",
}
INCOME_ORDER = [
    "Less than $10,000",
    "$10,000-$19,999",
    "$20,000-$29,999",
    "$30,000-$39,999",
    "$40,000-$49,999",
    "$50,000-$74,999",
    "$75,000 or more",
]

INSURANCE_MAP = {1: "Insured", 2: "Uninsured"}
INSURANCE_ORDER = ["Insured", "Uninsured"]

YES_NO_MAP = {1: "Yes", 2: "No"}
YES_NO_ORDER = ["Yes", "No"]

COUNTY_MAP = {
    1: "Large metro",
    2: "Small metro",
    3: "Nonmetro",
}
COUNTY_ORDER = ["Large metro", "Small metro", "Nonmetro"]

BARRIER_COLS = [
    "MHTUNCOST",
    "MHTUNINSCV",
    "MHTUNNOFND",
    "MHTUNTIME",
    "MHTUNPRIV",
    "MHTUNMED",
    "MHTUNCARE",
]
BARRIER_LABELS = {
    "MHTUNCOST": "Could not afford cost",
    "MHTUNINSCV": "Insurance did not cover care",
    "MHTUNNOFND": "Could not find services",
    "MHTUNTIME": "Did not have time",
    "MHTUNPRIV": "Privacy concerns",
    "MHTUNMED": "Did not want medication",
    "MHTUNCARE": "Did not think care was needed",
}

DISTRIBUTION_OPTIONS = {
    "Age distribution": ("age_group", AGE_ORDER, "Age group"),
    "Income distribution": ("income_group", INCOME_ORDER, "Income group"),
    "Insurance coverage": ("insurance_status", INSURANCE_ORDER, "Insurance status"),
    "Treatment rate": ("saw_doctor_mh", YES_NO_ORDER, "Saw doctor for mental health"),
    "Sex distribution": ("sex", SEX_ORDER, "Sex"),
    "Race/ethnicity distribution": ("race_ethnicity", RACE_ORDER, "Race/ethnicity"),
    "County type distribution": ("county_type", COUNTY_ORDER, "County type"),
}

COMPARISON_OPTIONS = {
    "Age vs Treatment": ("age_group", AGE_ORDER, "Age group"),
    "Income vs Treatment": ("income_group", INCOME_ORDER, "Income group"),
    "Insurance vs Treatment": ("insurance_status", INSURANCE_ORDER, "Insurance status"),
    "County Type vs Treatment": ("county_type", COUNTY_ORDER, "County type"),
    "Race/Ethnicity vs Treatment": ("race_ethnicity", RACE_ORDER, "Race/ethnicity"),
}

# -----------------------------------------------------------------------------
# Data loading and cleaning helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    """Load uploaded or default tab-delimited NSDUH file.

    usecols keeps the app scalable by avoiding hundreds of unused columns.
    If a needed column is missing, pandas simply ignores it through the callable.
    """
    source = uploaded_file if uploaded_file is not None else DEFAULT_PATH
    if not uploaded_file and not DEFAULT_PATH.exists():
        raise FileNotFoundError(DEFAULT_PATH)

    return pd.read_csv(
        source,
        sep="\t",
        usecols=lambda col: col in COLUMNS_NEEDED,
        low_memory=False,
    )


def to_numeric_if_present(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create readable analysis variables from NSDUH coded columns."""
    available = [col for col in COLUMNS_NEEDED if col in raw.columns]
    missing = [col for col in COLUMNS_NEEDED if col not in raw.columns]

    eda = raw[available].copy()
    eda = to_numeric_if_present(eda, available)

    if "AGE3" in eda.columns:
        eda["age_group"] = eda["AGE3"].map(AGE_MAP)
    elif "CATAGE" in eda.columns:
        eda["age_group"] = eda["CATAGE"].map(AGE_MAP)

    if "IRSEX" in eda.columns:
        eda["sex"] = eda["IRSEX"].map(SEX_MAP)
    if "NEWRACE2" in eda.columns:
        eda["race_ethnicity"] = eda["NEWRACE2"].map(RACE_MAP)
    if "IRPINC3" in eda.columns:
        eda["income_group"] = eda["IRPINC3"].map(INCOME_MAP)
    if "IRINSUR4" in eda.columns:
        eda["insurance_status"] = eda["IRINSUR4"].map(INSURANCE_MAP)
    elif "ANYHLTI2" in eda.columns:
        eda["insurance_status"] = eda["ANYHLTI2"].map(INSURANCE_MAP)
    if "ADSEEDOC" in eda.columns:
        eda["saw_doctor_mh"] = eda["ADSEEDOC"].map(YES_NO_MAP)
    if "ADRX12MO" in eda.columns:
        eda["mh_rx_past_year"] = eda["ADRX12MO"].map(YES_NO_MAP)
    if "MHTSEEKTX" in eda.columns:
        eda["sought_mh_treatment"] = eda["MHTSEEKTX"].map(YES_NO_MAP)
    if "COUTYP4" in eda.columns:
        eda["county_type"] = eda["COUTYP4"].map(COUNTY_MAP)

    for col in BARRIER_COLS:
        if col in eda.columns:
            eda[col] = eda[col].replace(MISSING_CODES, np.nan)

    return eda, missing


def apply_filters(eda: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
    filtered = eda.copy()
    for col, values in filters.items():
        if values and col in filtered.columns:
            filtered = filtered[filtered[col].isin(values)]
    return filtered


def distribution_table(eda: pd.DataFrame, label_col: str, order: List[str]) -> pd.DataFrame:
    counts = eda[label_col].value_counts().reindex(order).dropna().reset_index()
    counts.columns = [label_col, "count"]
    total = counts["count"].sum()
    counts["percent"] = np.where(total > 0, counts["count"] / total * 100, 0)
    return counts


def treatment_table(eda: pd.DataFrame, group_col: str, order: List[str]) -> pd.DataFrame:
    temp = eda.dropna(subset=[group_col, "saw_doctor_mh"])
    grouped = temp.groupby([group_col, "saw_doctor_mh"], observed=False).size().reset_index(name="count")
    totals = grouped.groupby(group_col)["count"].transform("sum")
    grouped["percent"] = np.where(totals > 0, grouped["count"] / totals * 100, 0)
    grouped[group_col] = pd.Categorical(grouped[group_col], categories=order, ordered=True)
    grouped["saw_doctor_mh"] = pd.Categorical(grouped["saw_doctor_mh"], categories=YES_NO_ORDER, ordered=True)
    return grouped.sort_values([group_col, "saw_doctor_mh"])


def treatment_rate_by_group(eda: pd.DataFrame, group_col: str, order: List[str]) -> pd.DataFrame:
    temp = eda.dropna(subset=[group_col, "saw_doctor_mh"])
    summary = (
        temp.assign(treated=temp["saw_doctor_mh"].eq("Yes"))
        .groupby(group_col, observed=False)
        .agg(respondents=("treated", "size"), treatment_rate=("treated", "mean"))
        .reset_index()
    )
    summary["treatment_rate"] = summary["treatment_rate"] * 100
    summary[group_col] = pd.Categorical(summary[group_col], categories=order, ordered=True)
    return summary.sort_values(group_col)


def barrier_table(eda: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    usable = [col for col in BARRIER_COLS if col in eda.columns]
    if group_col and group_col in eda.columns:
        for group_value, group_df in eda.dropna(subset=[group_col]).groupby(group_col, observed=False):
            for col in usable:
                rows.append(
                    {
                        "group": group_value,
                        "barrier": BARRIER_LABELS[col],
                        "count": int((group_df[col] == 1).sum()),
                    }
                )
    else:
        for col in usable:
            rows.append({"barrier": BARRIER_LABELS[col], "count": int((eda[col] == 1).sum())})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Header and Title
# -----------------------------------------------------------------------------
st.title("NSDUH Mental Health Access Explorer")
st.markdown(
    "This dashboard explores **mental health treatment access** using NSDUH-style variables, "
    "with a focus on age, income, insurance coverage, barriers to care, and geographic context."
)

st.divider()
st.markdown("### Interpretation Guide")
st.write(
    "Start with the distribution tab to describe the filtered population, then use the treatment comparison "
    "tab to identify differences across age, income, insurance, county type, or race/ethnicity. "
    "Use the barrier and Insurance tabs to support the final narrative about possible structural barriers "
    "and limitations."
)

st.divider()


# -----------------------------------------------------------------------------
# Load data SideBar Tab
# -----------------------------------------------------------------------------
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload full NSDUH tab-delimited file",
    type=["txt", "tsv", "csv"],
    help="The included sample file runs by default. Upload the full public-use file here for final analysis.",
)

try:
    raw = load_data(uploaded_file)
except FileNotFoundError:
    st.error(
        "Default sample file was not found. Put data/NSDUH_2021_First50_Tab.txt next to app.py "
        "or upload the NSDUH tab-delimited file in the sidebar."
    )
    st.stop()

eda, missing_cols = clean_data(raw)


# -----------------------------------------------------------------------------
# Filters SideBar Tab
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")
filters = {}
if "age_group" in eda.columns:
    filters["age_group"] = st.sidebar.multiselect("Age group", AGE_ORDER, default=[])
if "income_group" in eda.columns:
    filters["income_group"] = st.sidebar.multiselect("Income group", INCOME_ORDER, default=[])
if "insurance_status" in eda.columns:
    filters["insurance_status"] = st.sidebar.multiselect("Insurance status", INSURANCE_ORDER, default=[])
if "county_type" in eda.columns:
    filters["county_type"] = st.sidebar.multiselect("County type", COUNTY_ORDER, default=[])
if "race_ethnicity" in eda.columns:
    filters["race_ethnicity"] = st.sidebar.multiselect("Race/ethnicity", RACE_ORDER, default=[])

filtered = apply_filters(eda, filters)

# -----------------------------------------------------------------------------
# Dashboard Summary metrics
# -----------------------------------------------------------------------------
st.subheader("Dashboard Summary")
metric_cols = st.columns(5)
metric_cols[0].metric("Rows after filters", f"{len(filtered):,}")
metric_cols[1].metric("Raw columns loaded", f"{len(raw.columns):,}")

if "saw_doctor_mh" in filtered.columns:
    valid = filtered["saw_doctor_mh"].dropna()
    treatment_rate = valid.eq("Yes").mean() * 100 if len(valid) else 0
    metric_cols[2].metric("Treatment rate", f"{treatment_rate:.1f}%")
else:
    metric_cols[2].metric("Treatment rate", "N/A")

if "insurance_status" in filtered.columns:
    insured_valid = filtered["insurance_status"].dropna()
    insured_rate = insured_valid.eq("Insured").mean() * 100 if len(insured_valid) else 0
    metric_cols[3].metric("Insured rate", f"{insured_rate:.1f}%")
else:
    metric_cols[3].metric("Insured rate", "N/A")

barriers = barrier_table(filtered)
barrier_total = int(barriers["count"].sum()) if not barriers.empty and "count" in barriers else 0
metric_cols[4].metric("Reported barriers", f"{barrier_total:,}")

if filtered.empty:
    st.warning("No rows match the current filters. Clear one or more filters to continue.")
    st.stop()

# -----------------------------------------------------------------------------
# Model Type Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Distributions",
        "Treatment Comparisons",
        "Barrier Explorer",
        "Gap & Drill-Down",
    ]
)

with tab1:
    st.subheader("Interactive Distribution Explorer")
    available_distribution_options = {
        name: spec for name, spec in DISTRIBUTION_OPTIONS.items() if spec[0] in filtered.columns
    }

    if not available_distribution_options:
        st.info("No mapped distribution variables are available.")
    else:
        selected_dist = st.selectbox("Choose a subject to visualize", list(available_distribution_options.keys()))
        label_col, order, axis_label = available_distribution_options[selected_dist]
        dist_df = distribution_table(filtered, label_col, order)
        y_choice = st.radio("Display as", ["Count", "Percent"], horizontal=True)
        y_col = "count" if y_choice == "Count" else "percent"

        fig = px.bar(
            dist_df,
            x=label_col,
            y=y_col,
            text=dist_df[y_col].round(1),
            title=selected_dist,
            labels={label_col: axis_label, y_col: y_choice},
        )
        fig.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Use this view to understand who is represented in the filtered dataset before comparing treatment access.")
    
        st.header("Summarized Table")
        st.dataframe(dist_df)

with tab2:
    st.subheader("Treatment Access Comparison")
    available_comparisons = {
        name: spec
        for name, spec in COMPARISON_OPTIONS.items()
        if spec[0] in filtered.columns and "saw_doctor_mh" in filtered.columns
    }

    if not available_comparisons:
        st.info("Treatment comparisons require a treatment variable and at least one grouping variable.")
    else:
        selected_comparison = st.selectbox("Compare treatment by", list(available_comparisons.keys()))
        group_col, group_order, group_label = available_comparisons[selected_comparison]
        mode = st.radio("Chart mode", ["100% stacked bar", "Treatment-rate bar"], horizontal=True)

        if mode == "100% stacked bar":
            treat_df = treatment_table(filtered, group_col, group_order)
            fig2 = px.bar(
                treat_df,
                x=group_col,
                y="percent",
                color="saw_doctor_mh",
                barmode="stack",
                text=treat_df["percent"].round(1),
                title=f"{selected_comparison}: Percent Within Each Group",
                labels={group_col: group_label, "percent": "Percent within group", "saw_doctor_mh": "Treatment"},
            )
            fig2.update_layout(xaxis_tickangle=-35, yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("This view compares the proportion of respondents who did and did not receive mental health treatment within each group.")
            st.header("Summarized Table")
            st.dataframe(treat_df)
        else:
            rate_df = treatment_rate_by_group(filtered, group_col, group_order).dropna()
            fig2 = px.bar(
                rate_df,
                x=group_col,
                y="treatment_rate",
                text=rate_df["treatment_rate"].round(1),
                hover_data=["respondents"],
                title=f"Treatment Rate by {group_label}",
                labels={group_col: group_label, "treatment_rate": "Treatment rate (%)", "respondents": "Respondents"},
            )
            fig2.update_layout(xaxis_tickangle=-35, yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("This view focuses only on the percentage of each group that reported receiving mental health treatment.")
            st.header("Summarized Table")
            st.dataframe(rate_df)

with tab3:
    st.subheader("Barriers to Mental Health Treatment")
    compare_barriers = st.checkbox("Compare barriers by group", value=False)
    group_choice = None
    if compare_barriers:
        valid_groups = {
            "Income group": ("income_group", INCOME_ORDER),
            "Insurance status": ("insurance_status", INSURANCE_ORDER),
            "Age group": ("age_group", AGE_ORDER),
            "County type": ("county_type", COUNTY_ORDER),
        }
        valid_groups = {k: v for k, v in valid_groups.items() if v[0] in filtered.columns}
        if valid_groups:
            group_choice = st.selectbox("Group barriers by", list(valid_groups.keys()))

    if compare_barriers and group_choice:
        group_col, group_order = valid_groups[group_choice]
        bdf = barrier_table(filtered, group_col=group_col)
        bdf[group_col] = pd.Categorical(bdf["group"], categories=group_order, ordered=True)
        bdf = bdf[bdf["count"] > 0].sort_values([group_col, "count"])
        if bdf.empty:
            st.info("No reported barrier responses were found after filters were applied.")
        else:
            fig3 = px.bar(
                bdf,
                x="count",
                y="barrier",
                color="group",
                orientation="h",
                barmode="group",
                title=f"Reported Barriers by {group_choice}",
                labels={"count": "Number of respondents", "barrier": "Barrier", "group": group_choice},
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.header("Summarized Table")
            st.dataframe(bdf[["group", "barrier", "count"]])
    else:
        bdf = barrier_table(filtered)
        if bdf.empty or bdf["count"].sum() == 0:
            st.info("No reported barrier responses were found after filters were applied.")
        else:
            bdf = bdf.sort_values("count", ascending=True)
            fig3 = px.bar(
                bdf,
                x="count",
                y="barrier",
                orientation="h",
                text="count",
                title="Reported Barriers to Mental Health Treatment",
                labels={"count": "Number of respondents", "barrier": "Barrier"},
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("This view identifies which access barriers appear most often in the filtered dataset.")
            st.header("Summarized Table")
            st.dataframe(bdf.sort_values("count", ascending=False))

with tab4:
    st.subheader("Access Gap and Drill-Down || Insurance Comparison")
    st.write("This section highlights differences in treatment rates across variables and Insurance.")

    drill_options = {
        "Income by Insurance": ("income_group", "insurance_status", INCOME_ORDER, "Income group"),
        "Age by Insurance": ("age_group", "insurance_status", AGE_ORDER, "Age group"),
        "County Type by Insurance": ("county_type", "insurance_status", COUNTY_ORDER, "County type"),
        "Race/Ethnicity by Insurance": ("race_ethnicity", "insurance_status", RACE_ORDER, "Race/ethnicity"),
    }
    drill_options = {
        name: spec
        for name, spec in drill_options.items()
        if spec[0] in filtered.columns and spec[1] in filtered.columns and "saw_doctor_mh" in filtered.columns
    }

    if not drill_options:
        st.info("Drill-down requires treatment, insurance, and at least one demographic or socioeconomic grouping variable.")
    else:
        selected_drill = st.selectbox("Choose drill-down comparison", list(drill_options.keys()))
        x_col, color_col, x_order, x_label = drill_options[selected_drill]
        temp = filtered.dropna(subset=[x_col, color_col, "saw_doctor_mh"]).copy()
        temp["treated"] = temp["saw_doctor_mh"].eq("Yes")
        drill_df = (
            temp.groupby([x_col, color_col], observed=False)
            .agg(respondents=("treated", "size"), treatment_rate=("treated", "mean"))
            .reset_index()
        )
        drill_df["treatment_rate"] = drill_df["treatment_rate"] * 100
        drill_df[x_col] = pd.Categorical(drill_df[x_col], categories=x_order, ordered=True)
        drill_df = drill_df.sort_values([x_col, color_col])

        fig4 = px.bar(
            drill_df,
            x=x_col,
            y="treatment_rate",
            color=color_col,
            barmode="group",
            text=drill_df["treatment_rate"].round(1),
            hover_data=["respondents"],
            title=f"Treatment Rate Drill-Down: {selected_drill}",
            labels={x_col: x_label, color_col: color_col.replace("_", " ").title(), "treatment_rate": "Treatment rate (%)"},
        )
        fig4.update_layout(xaxis_tickangle=-35, yaxis_range=[0, 100])
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Use this view to compare whether the treatment access gap changes when two concepts are examined together.")
        st.header("Summarized Table")
        st.dataframe(drill_df)

