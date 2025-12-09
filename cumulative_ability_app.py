import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Cumulative Ability Explorer", layout="wide")

TEAM_COLORS = {
    "Los Angeles Lakers": {"primary": "#552583", "secondary": "#FDB927"},
    "Boston Celtics": {"primary": "#007A33", "secondary": "#BA9653"},
    "Golden State Warriors": {"primary": "#1D428A", "secondary": "#FFC72C"},
    "Chicago Bulls": {"primary": "#CE1141", "secondary": "#000000"},
    "Miami Heat": {"primary": "#98002E", "secondary": "#F9A01B"},
    "Dallas Mavericks": {"primary": "#00538C", "secondary": "#002B5E"},
    "New York Knicks": {"primary": "#006BB6", "secondary": "#F58426"},
}

@st.cache_data
def load_data():
    df = pd.read_csv(
        "cumulative_ability_EB_era_pos_rho_agecurve_nok_with_teams.csv",
        low_memory=False,
    )
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    if "minutes" in df.columns:
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    return df

df = load_data()

tab_players, tab_leaderboards, tab_teams, tab_methodology = st.tabs(
    ["Players", "Leaderboards", "Teams", "Methodology"]
)

# -----------------------
# PLAYER TAB
# -----------------------
with tab_players:
    st.title("Player View")
    players = df["player_name"].dropna().sort_values().unique()
    selected_player = st.selectbox("Select player", players, key="player_select")
    pdf = df[df["player_name"] == selected_player].copy()
    pdf = pdf.sort_values("season")

    cols = [
        "season",
        "team_name",
        "team_name2",
        "primary_pos",
        "minutes",
        "spm_total",
        "rapm_centered",
        "ca_offense",
        "ca_defense",
        "ca_total",
    ]
    existing_cols = [c for c in cols if c in pdf.columns]
    table = pdf[existing_cols].copy()

    # weights for career row
    if "minutes" in pdf.columns and pdf["minutes"].notna().any():
        w = pdf["minutes"].fillna(0).values
        w = np.where(w > 0, w, 0)
    else:
        w = np.ones(len(pdf)) if len(pdf) > 0 else np.array([])

    career = {}
    for c in existing_cols:
        if c == "season":
            career[c] = "Career"
        elif c in ["team_name", "team_name2", "primary_pos"]:
            career[c] = ""
        elif c == "minutes":
            career[c] = float(w.sum())
        else:
            if len(pdf) > 0:
                career[c] = float(np.average(pdf[c].fillna(0).values, weights=w))
            else:
                career[c] = np.nan

    if len(existing_cols) > 0:
        career_row = pd.DataFrame([career])[existing_cols]
        table_with_career = pd.concat([table, career_row], ignore_index=True)
    else:
        table_with_career = table

    num_cols = table_with_career.select_dtypes(include="number").columns
    table_with_career[num_cols] = table_with_career[num_cols].round(2)

    if "season" in table_with_career.columns:
        table_with_career["season"] = table_with_career["season"].astype(str)

    display_table = table_with_career.copy()
    display_table.columns = [c.replace("_", " ").title() for c in display_table.columns]

    st.subheader("Season-by-season stats")
    st.dataframe(display_table, width="stretch")

    if len(pdf) > 0 and "ca_total" in pdf.columns:
        pdf["season"] = pdf["season"].astype(int)

        primary_color = "#1D428A"
        if "team_name" in pdf.columns and pdf["team_name"].notna().any():
            team_key = pdf["team_name"].iloc[-1]
            if isinstance(team_key, str) and team_key in TEAM_COLORS:
                primary_color = TEAM_COLORS[team_key]["primary"]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=pdf["season"],
                y=pdf["ca_total"],
                mode="lines+markers",
                name="CA total",
                line=dict(width=3, color=primary_color),
            )
        )

        if "rapm_centered" in pdf.columns:
            fig.add_trace(
                go.Scatter(
                    x=pdf["season"],
                    y=pdf["rapm_centered"],
                    mode="lines+markers",
                    name="RAPM",
                )
            )

        if "spm_total" in pdf.columns:
            fig.add_trace(
                go.Scatter(
                    x=pdf["season"],
                    y=pdf["spm_total"],
                    mode="lines+markers",
                    name="SPM",
                )
            )

        fig.update_layout(
            xaxis_title="Season",
            yaxis_title="Value",
            title=f"{selected_player} CA, RAPM, and SPM by season",
        )
        fig.update_xaxes(type="category")

        st.subheader("Career progression")
        st.plotly_chart(fig, width="stretch")

# -----------------------
# LEADERBOARDS TAB
# -----------------------
with tab_leaderboards:
    st.title("Leaderboards")

    lb_total_tab, lb_season_tab = st.tabs(["Overall leaderboard", "Season leaderboard"])

    base_cols = [
        "season",
        "player_name",
        "team_name",
        "team_name2",
        "primary_pos",
        "minutes",
        "spm_total",
        "rapm_centered",
        "ca_offense",
        "ca_defense",
        "ca_total",
    ]
    existing_lb_cols = [c for c in base_cols if c in df.columns]

    with lb_total_tab:
        st.subheader("Overall player-season leaderboard (by CA total)")

        overall = df[existing_lb_cols].copy()
        overall = overall.dropna(subset=["ca_total"])
        overall = overall.sort_values("ca_total", ascending=False).reset_index(drop=True)

        overall.insert(0, "rank", np.arange(1, len(overall) + 1))

        num_cols = [c for c in overall.select_dtypes(include="number").columns if c != "rank"]
        overall[num_cols] = overall[num_cols].round(2)

        overall_display = overall.copy()
        overall_display.columns = [c.replace("_", " ").title() for c in overall_display.columns]

        st.dataframe(overall_display, width="stretch")

    with lb_season_tab:
        st.subheader("Season leaderboard (by CA total)")

        seasons = (
            df["season"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .sort_values(ascending=False)
            .tolist()
        )

        selected_season = st.selectbox(
            "Select season",
            seasons,
            format_func=lambda x: str(x),
            key="lb_season_select",
        )

        season_df = df[df["season"] == selected_season][existing_lb_cols].copy()
        season_df = season_df.dropna(subset=["ca_total"])
        season_df = season_df.sort_values("ca_total", ascending=False).reset_index(drop=True)

        season_df.insert(0, "rank", np.arange(1, len(season_df) + 1))

        num_cols_s = [c for c in season_df.select_dtypes(include="number").columns if c != "rank"]
        season_df[num_cols_s] = season_df[num_cols_s].round(2)

        season_display = season_df.copy()
        season_display.columns = [c.replace("_", " ").title() for c in season_display.columns]

        st.dataframe(season_display, width="stretch")

# -----------------------
# TEAMS TAB
# -----------------------
with tab_teams:
    st.title("Teams")

    teams_primary = df["team_name"].dropna().unique()
    teams_secondary = df["team_name2"].dropna().unique()
    teams = sorted(set(teams_primary).union(set(teams_secondary)))

    selected_team = st.selectbox("Select team", teams, key="team_selectbox")

    team_seasons = (
        df[
            (df["team_name"] == selected_team)
            | (df["team_name2"] == selected_team)
        ]["season"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )

    if len(team_seasons) == 0:
        st.write("No seasons found for this team.")
    else:
        selected_team_season = st.selectbox(
            "Select season",
            team_seasons,
            format_func=lambda x: str(x),
            key="team_season_select",
        )

        tdf = df[
            (
                (df["team_name"] == selected_team)
                | (df["team_name2"] == selected_team)
            )
            & (df["season"] == selected_team_season)
        ].copy()

        if tdf.empty:
            st.write("No players found for this team and season.")
        else:
            st.write(f"{selected_team} – {selected_team_season} season")

            cols_team = [
                "player_name",
                "primary_pos",
                "minutes",
                "spm_total",
                "rapm_centered",
                "ca_offense",
                "ca_defense",
                "ca_total",
            ]
            existing_team_cols = [c for c in cols_team if c in tdf.columns]
            team_table = tdf[existing_team_cols].copy()

            if "ca_total" in team_table.columns:
                team_table = team_table.sort_values("ca_total", ascending=False)

            avg_row = {}
            for c in existing_team_cols:
                if c == "player_name":
                    avg_row[c] = "Team Average"
                elif c == "primary_pos":
                    avg_row[c] = ""
                else:
                    avg_row[c] = team_table[c].mean()

            team_table = pd.concat(
                [team_table, pd.DataFrame([avg_row])[existing_team_cols]],
                ignore_index=True,
            )

            num_cols_team = team_table.select_dtypes(include="number").columns
            team_table[num_cols_team] = team_table[num_cols_team].round(2)

            team_display = team_table.copy()
            team_display.columns = [c.replace("_", " ").title() for c in team_display.columns]

            primary_color = TEAM_COLORS.get(selected_team, {}).get("primary", "#1D428A")

            header_vals = list(team_display.columns)
            cell_vals = [team_display[c] for c in team_display.columns]

            fig_team = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=header_vals,
                            fill_color=primary_color,
                            font_color="white",
                            align="left",
                        ),
                        cells=dict(
                            values=cell_vals,
                            fill_color="white",
                            align="left",
                        ),
                    )
                ]
            )

            fig_team.update_layout(
                title_text=f"{selected_team} {selected_team_season} roster stats",
                title_x=0.0,
            )

            st.plotly_chart(fig_team, width="stretch")

# -----------------------
# METHODOLOGY TAB
# -----------------------
with tab_methodology:
    st.title("Cumulative Ability Methodology")

    st.markdown("""
This app is built on an empirical Bayes **cumulative ability (CA)** model that combines
season-level SPM and RAPM into a single, smoothed talent estimate for each player-season.
The model assumes that every player has an underlying ability that evolves over time,
while SPM and RAPM are noisy signals of that ability. CA pools information across seasons
and shrinks noisy years toward reasonable prior expectations. Essentially, it measures a 
player's true talent, not just how good they played in one season, but combining previous 
season play as well to get an overall picture of player ability at a point in time. Lebron
was considered the best player of the 2010s, but didn't win MVP every year because he didn't
necessarily have the best season every year. However, he is at the top of or near the top of
CA still even in those years where he didn't have the best individual season. This is due to the knowledge
of past seasons reinforcing what we as fans know, which is that he was one of if not the best player
in basketball almost every year.

### Age curve and residualization

Before fitting CA, the model removes the average age and experience pattern from SPM so
that CA captures “above or below expectation” instead of raw performance. On pre-test
seasons (before 2018), total SPM is regressed on years in league, years in league squared,
position group, and era (early vs modern). This gives a predicted SPM for each
age/position/era combination. The model then forms an age-adjusted SPM residual:
actual SPM minus predicted SPM. A residual of +2 means “two points better than a typical
player of this age, position, and era.” These residuals are used inside the CA recursion,
and at the end the age curve is added back so CA remains on an interpretable
total-impact scale.

### Empirical Bayes priors by era and position

Prior distributions for player ability are estimated using only pre-test seasons.
For total, offense, and defense the model:

1. Computes each player’s possession-weighted average SPM.
2. Decomposes variation into between-player variance and within-player noise using a simple variance regression.
3. Repeats this separately for each era × position cell.

This produces, for every era/position, a prior mean and a prior “weight per possession.”
Modern guards and early-era bigs, for example, start from different priors, and the
amount of shrinkage is learned from the data: when players in a cell are very similar
and single seasons are very noisy, the model shrinks harder toward the era/position mean.

### Dynamic CA with AR(1) persistence

Within each player, CA is treated as a latent state that follows an AR(1) process
across seasons. A rookie’s prior is their era × position prior. Each season, the model:

- Discounts the previous precision by a persistence parameter **rho** raised to the number
  of missed seasons, and
- Updates with that season’s age-adjusted SPM residual.

The residual is weighted by an effective sample size equal to possessions played divided
by a baseline possession constant. High-minute seasons move CA more, while low-minute
seasons act as weak evidence that stays closer to the prior or to previous years.

### Using RAPM as a second signal

For total CA, RAPM is used as an additional noisy measurement of the same underlying
ability. When RAPM is available in a season, it is treated as another observation with
an effective weight equal to:

- a scaling factor **lambda_rapm** times
- possessions divided by a **baseline_rapm_poss** constant.

These two hyperparameters control how much the model trusts RAPM per possession
relative to SPM. There are no ad-hoc recency boosts; recency enters only through the
AR(1) persistence and the fact that recent seasons often have more possessions.

### Hyperparameters and tuning

The main hyperparameters are:

- **rho**: the AR(1) persistence of ability (how quickly the model “forgets” older seasons).
- **baseline_rapm_poss**: the possession scale that converts RAPM into an effective sample size.
- **lambda_rapm**: the relative weight of RAPM vs SPM for a given number of possessions.

Rho is estimated by regressing next-season RAPM on current-season RAPM in the pre-test
window and then searching over a small grid around that value. For the two RAPM
weighting parameters, the model defines small grids of plausible values and runs a grid
search. For each combination, it runs the CA recursion on the full dataset and evaluates
only on pre-test seasons by correlating CA in season *t* with RAPM in season *t + 1*.
Those pre-test seasons are split into train and validation blocks, and the final
hyperparameters are chosen to maximize the out-of-sample correlation.
""")
