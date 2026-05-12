"""
app.py  —  Streamlit Dashboard
Employee–Project Fit Recommendation System
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import (load_data, greedy_allocate, ilp_allocate,
                   SKILL_MAP, WEIGHTS, MAX_PROJECTS)

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HR Fit Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1c1f2e;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label {
        color: #8b8fa8 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e8eaf6 !important;
        font-size: 28px !important;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #5c6bc0;
        margin: 24px 0 8px 0;
    }

    /* Score bar card */
    .score-card {
        background: #1c1f2e;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #5c6bc0;
        margin-bottom: 10px;
    }
    .score-card h4 { margin: 0 0 6px 0; color: #e8eaf6; font-size: 14px; }
    .score-card p  { margin: 2px 0; color: #8b8fa8; font-size: 12px; }

    /* Employee chip */
    .emp-chip {
        display: inline-block;
        background: #26294a;
        color: #9fa8da;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        margin: 3px;
        border: 1px solid #3c4080;
    }

    /* Tag badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-green  { background: #1b2e22; color: #66bb6a; border: 1px solid #2e5c36; }
    .badge-blue   { background: #1a2340; color: #64b5f6; border: 1px solid #2a3a6e; }
    .badge-orange { background: #2e2010; color: #ffa726; border: 1px solid #5e3a10; }

    /* Comparison differ highlight */
    .differ { background: #2e1e10; border-left: 3px solid #ffa726; padding: 4px 10px; border-radius: 4px; }
    .same   { background: #1b2e22; border-left: 3px solid #66bb6a; padding: 4px 10px; border-radius: 4px; }

    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE + DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_all(emp, proj, roles):
    return load_data(emp, proj, roles)


def init_state():
    for k, v in [("employees", None), ("projects", None), ("roles", None),
                 ("ilp_result", None), ("ilp_status", None), ("ilp_score", None),
                 ("data_loaded", False)]:
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 HR Fit Recommender")
    st.markdown("---")

    st.markdown('<p class="section-header">📂 Upload your datasets</p>', unsafe_allow_html=True)
    emp_file   = st.file_uploader("Employee CSV",  type="csv", key="emp_up")
    proj_file  = st.file_uploader("Projects CSV",  type="csv", key="proj_up")
    roles_file = st.file_uploader("Roles CSV",     type="csv", key="roles_up")

    if emp_file and proj_file and roles_file:
        try:
            employees, projects, roles = load_all(emp_file, proj_file, roles_file)
            st.session_state.employees   = employees
            st.session_state.projects    = projects
            st.session_state.roles       = roles
            st.session_state.data_loaded = True
            st.success(f"✓ {len(employees):,} employees · {len(projects)} projects · {len(roles)} roles")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown('<p class="section-header">⚡ ILP Optimizer</p>', unsafe_allow_html=True)
        st.caption("Runs global optimization across all projects at once")
        if st.button("🚀 Run ILP Optimization", use_container_width=True):
            with st.spinner("Solving integer linear program…"):
                try:
                    asgn, status, score = ilp_allocate(
                        st.session_state.projects,
                        st.session_state.roles,
                        st.session_state.employees,
                    )
                    st.session_state.ilp_result = asgn
                    st.session_state.ilp_status = status
                    st.session_state.ilp_score  = score
                    st.success(f"✓ {status} | Score: {score:.2f}")
                except Exception as e:
                    st.error(f"ILP error: {e}")

        st.markdown("---")
        st.markdown('<p class="section-header">🔍 Greedy — select project</p>', unsafe_allow_html=True)
        proj_names = st.session_state.projects["ProjectName"].tolist()
        selected_proj_name = st.selectbox("Project", proj_names, label_visibility="collapsed")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if not st.session_state.data_loaded:
    # ── Welcome screen ──────────────────────────────────────────
    st.markdown("# 🎯 Employee–Project Fit Recommender")
    st.markdown("#### A data-driven HR decision support system")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1**\n\nUpload your 3 CSV files using the sidebar on the left.")
    with c2:
        st.info("**Step 2**\n\nExplore the workforce on the Overview and Explorer tabs.")
    with c3:
        st.info("**Step 3**\n\nRun Greedy or ILP allocation and compare results.")
    st.stop()


# ── Data references ──────────────────────────────────────────────
employees = st.session_state.employees
projects  = st.session_state.projects
roles     = st.session_state.roles

# ── Tabs ─────────────────────────────────────────────────────────
tab_ov, tab_explorer, tab_greedy, tab_ilp, tab_compare, tab_analytics = st.tabs([
    "📊 Overview",
    "👥 Employee Explorer",
    "⚡ Greedy Allocation",
    "🧠 ILP Optimization",
    "🔄 Compare",
    "📈 Analytics",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab_ov:
    st.markdown("### Workforce at a Glance")
    st.markdown("")

    available = employees[employees["availability_projects"] < MAX_PROJECTS]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees",    f"{len(employees):,}")
    c2.metric("Available Now",      f"{len(available):,}")
    c3.metric("Active Projects",    f"{len(projects)}")
    c4.metric("Open Role Slots",    f"{int(roles['RequiredEmployees'].sum()) if 'RequiredEmployees' in roles.columns else len(roles)}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Career Level Distribution")
        level_map  = {0: "Junior", 1: "Mid-Level", 2: "Senior"}
        level_data = employees["current_level"].map(level_map).value_counts().reset_index()
        level_data.columns = ["Level", "Count"]
        fig = px.pie(level_data, names="Level", values="Count",
                     color_discrete_sequence=["#5c6bc0", "#42a5f5", "#26a69a"],
                     hole=0.4)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#c5cae9", legend_font_size=13,
                          margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Primary Skills Distribution (Top 12)")
        skill_counts = employees["primary_skill"].value_counts().head(12).reset_index()
        skill_counts.columns = ["Skill", "Count"]
        fig2 = px.bar(skill_counts, x="Count", y="Skill", orientation="h",
                      color="Count", color_continuous_scale="Blues")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c5cae9", yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Skill Proficiency Distribution")
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=employees["primary_skill_proficiency"],
                                    name="Primary", marker_color="#5c6bc0", opacity=0.8, nbinsx=20))
        fig3.add_trace(go.Histogram(x=employees["secondary_skill_proficiency"],
                                    name="Secondary", marker_color="#26a69a", opacity=0.7, nbinsx=20))
        fig3.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", font_color="#c5cae9",
                           legend=dict(bgcolor="rgba(0,0,0,0)"),
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("#### Workload (Active Projects per Employee)")
        wl = employees["availability_projects"].value_counts().sort_index().reset_index()
        wl.columns = ["Active Projects", "Employees"]
        fig4 = px.bar(wl, x="Active Projects", y="Employees",
                      color="Active Projects",
                      color_continuous_scale=["#26a69a", "#ffa726", "#ef5350"])
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c5cae9", coloraxis_showscale=False,
                           margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — EMPLOYEE EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab_explorer:
    st.markdown("### Employee Explorer")

    f1, f2, f3 = st.columns(3)
    with f1:
        dept_filter = st.multiselect("Department",
                                     options=employees["Department"].unique().tolist(),
                                     default=[])
    with f2:
        level_opts  = {0: "Junior", 1: "Mid-Level", 2: "Senior"}
        level_filter = st.multiselect("Career Level",
                                      options=[0, 1, 2],
                                      format_func=lambda x: level_opts[x],
                                      default=[])
    with f3:
        avail_filter = st.multiselect("Availability (active projects)",
                                      options=sorted(employees["availability_projects"].unique().tolist()),
                                      default=[])

    filtered = employees.copy()
    if dept_filter:
        filtered = filtered[filtered["Department"].isin(dept_filter)]
    if level_filter:
        filtered = filtered[filtered["current_level"].isin(level_filter)]
    if avail_filter:
        filtered = filtered[filtered["availability_projects"].isin(avail_filter)]

    st.caption(f"Showing **{len(filtered):,}** of {len(employees):,} employees")

    display_cols = ["EmployeeNumber", "Department", "JobRole", "current_level",
                    "primary_skill", "primary_skill_proficiency",
                    "secondary_skill", "secondary_skill_proficiency",
                    "TotalWorkingYears", "availability_projects"]
    existing = [c for c in display_cols if c in filtered.columns]

    rename_map = {
        "EmployeeNumber": "Emp #", "Department": "Dept", "JobRole": "Job Role",
        "current_level": "Level", "primary_skill": "Primary Skill",
        "primary_skill_proficiency": "P.Prof",
        "secondary_skill": "Secondary Skill", "secondary_skill_proficiency": "S.Prof",
        "TotalWorkingYears": "Exp (yrs)", "availability_projects": "Active Proj",
    }
    disp = filtered[existing].rename(columns=rename_map)
    disp["Level"] = disp["Level"].map({0: "Junior", 1: "Mid-Level", 2: "Senior"})

    st.dataframe(disp, use_container_width=True, height=420)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — GREEDY
# ══════════════════════════════════════════════════════════════════
with tab_greedy:
    st.markdown("### ⚡ Greedy Role Allocation")

    proj_row = projects[projects["ProjectName"] == selected_proj_name].iloc[0]
    dept     = proj_row["Department"]

    st.markdown(f"""
    <div style='background:#1c1f2e;border-radius:10px;padding:16px 20px;margin-bottom:20px;'>
      <span style='color:#5c6bc0;font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase'>Selected project</span><br>
      <span style='color:#e8eaf6;font-size:20px;font-weight:700'>{proj_row['ProjectName']}</span>
      &nbsp;&nbsp;<span class='badge badge-blue'>{dept}</span>
    </div>
    """, unsafe_allow_html=True)

    project_roles = roles[roles["ProjectID"] == proj_row["ProjectID"]]
    if project_roles.empty:
        st.warning("No roles defined for this project in the roles dataset.")
        st.stop()

    team = greedy_allocate(proj_row, roles, employees)

    if not team:
        st.warning("No employees could be matched to any role in this project.")
    else:
        for role_name, data in team.items():
            st.markdown(f"#### Role: {role_name}")
            selected_df  = data["selected"]
            top_df       = data["top_candidates"]

            col_sel, col_top = st.columns([1, 1])

            with col_sel:
                st.markdown("**✅ Selected Employee(s)**")
                for _, emp in selected_df.iterrows():
                    ex = emp["explanation"]
                    score_pct = int(ex["FinalScore"] * 100)
                    bar_html  = f"""
                    <div class='score-card'>
                      <h4>Employee #{int(emp['EmployeeNumber'])}
                        &nbsp;<span class='badge badge-green'>Score: {ex['FinalScore']}</span>
                      </h4>
                      <div style='background:#0f1117;border-radius:6px;height:8px;margin:8px 0;'>
                        <div style='background:linear-gradient(90deg,#5c6bc0,#42a5f5);
                             width:{score_pct}%;height:8px;border-radius:6px;'></div>
                      </div>
                      <p>🎯 Skill match: <b>{ex['SkillMatch']}</b> &nbsp;|&nbsp;
                         📅 Experience: <b>{ex['ExperienceScore']}</b> &nbsp;|&nbsp;
                         🏠 Stability: <b>{ex['StabilityScore']}</b> &nbsp;|&nbsp;
                         🟢 Availability: <b>{ex['AvailabilityScore']}</b></p>
                    </div>"""
                    st.markdown(bar_html, unsafe_allow_html=True)

            with col_top:
                st.markdown("**🏆 Top 5 Candidates**")
                top_display = top_df[["EmployeeNumber", "score",
                                       "TotalWorkingYears", "primary_skill",
                                       "availability_projects"]].copy()
                top_display.columns = ["Emp #", "Score", "Exp(yrs)", "Primary Skill", "Active Proj"]
                top_display["Score"] = top_display["Score"].round(3)
                st.dataframe(top_display.reset_index(drop=True), use_container_width=True)

            # Mini radar chart for selected employees
            if len(selected_df) > 0:
                st.markdown("**Score Breakdown (Radar)**")
                cats = ["Skill Match", "Experience", "Stability", "Availability"]
                fig = go.Figure()
                for _, emp in selected_df.iterrows():
                    ex = emp["explanation"]
                    vals = [ex["SkillMatch"], ex["ExperienceScore"],
                            ex["StabilityScore"], ex["AvailabilityScore"]]
                    fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]], theta=cats + [cats[0]],
                        fill="toself", name=f"Emp #{int(emp['EmployeeNumber'])}",
                        opacity=0.75,
                    ))
                fig.update_layout(
                    polar=dict(bgcolor="rgba(0,0,0,0)",
                               radialaxis=dict(visible=True, range=[0, 1],
                                               color="#8b8fa8", gridcolor="#2e3250")),
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#c5cae9",
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    margin=dict(t=20, b=20, l=20, r=20), height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — ILP
# ══════════════════════════════════════════════════════════════════
with tab_ilp:
    st.markdown("### 🧠 ILP Global Optimization")

    if st.session_state.ilp_result is None:
        st.info("👈 Click **Run ILP Optimization** in the sidebar to solve the global allocation.")
    else:
        asgn   = st.session_state.ilp_result
        status = st.session_state.ilp_status
        score  = st.session_state.ilp_score

        c1, c2, c3 = st.columns(3)
        c1.metric("Solver Status",       status)
        c2.metric("Total Suitability",   f"{score:.3f}")
        total_assigned = sum(len(v) for proj in asgn.values() for v in proj.values())
        c3.metric("Total Assignments",   total_assigned)

        st.markdown("---")

        for _, proj in projects.iterrows():
            pid = proj["ProjectID"]
            if pid not in asgn:
                continue

            with st.expander(f"📁 {proj['ProjectName']}  [{proj['Department']}]", expanded=False):
                for role_name, emp_list in asgn[pid].items():
                    st.markdown(f"**Role: {role_name}** — {len(emp_list)} assigned")
                    rows = []
                    for eid, s, ex in emp_list:
                        rows.append({
                            "Emp #": eid,
                            "Final Score": ex["FinalScore"],
                            "Skill Match": ex["SkillMatch"],
                            "Experience": ex["ExperienceScore"],
                            "Stability": ex["StabilityScore"],
                            "Availability": ex["AvailabilityScore"],
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                    st.markdown("")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🔄 Greedy vs ILP — Side-by-Side")

    if st.session_state.ilp_result is None:
        st.info("Run ILP optimization first (sidebar button), then come back here.")
    else:
        proj_row_c = projects[projects["ProjectName"] == selected_proj_name].iloc[0]
        pid_c      = proj_row_c["ProjectID"]

        greedy_team = greedy_allocate(proj_row_c, roles, employees)
        ilp_asgn    = st.session_state.ilp_result

        roles_in_proj = roles[roles["ProjectID"] == pid_c]["RequiredRole"].tolist()

        if not roles_in_proj:
            st.warning("No roles for this project.")
        else:
            for role_name in roles_in_proj:
                st.markdown(f"#### Role: {role_name}")
                col_g, col_i = st.columns(2)

                # ── Greedy side ──────────────────────────────────────────
                with col_g:
                    st.markdown('<span class="badge badge-blue">⚡ Greedy</span>', unsafe_allow_html=True)
                    greedy_data = greedy_team.get(role_name)
                    if greedy_data is None:
                        st.caption("No match found")
                        greedy_eids = []
                    else:
                        greedy_eids = greedy_data["selected"]["EmployeeNumber"].tolist()
                        for _, emp in greedy_data["selected"].iterrows():
                            ex = emp["explanation"]
                            st.markdown(f"""
                            <div class='score-card'>
                              <h4>Employee #{int(emp['EmployeeNumber'])}</h4>
                              <p>Final Score: <b>{ex['FinalScore']}</b></p>
                              <p>Skill {ex['SkillMatch']} | Exp {ex['ExperienceScore']}
                                 | Stab {ex['StabilityScore']} | Avail {ex['AvailabilityScore']}</p>
                            </div>""", unsafe_allow_html=True)

                # ── ILP side ─────────────────────────────────────────────
                with col_i:
                    st.markdown('<span class="badge badge-orange">🧠 ILP</span>', unsafe_allow_html=True)
                    ilp_role_data = (ilp_asgn.get(pid_c) or {}).get(role_name)
                    if not ilp_role_data:
                        st.caption("No assignment made")
                        ilp_eids = []
                    else:
                        ilp_eids = [t[0] for t in ilp_role_data]
                        for eid, s, ex in ilp_role_data:
                            st.markdown(f"""
                            <div class='score-card'>
                              <h4>Employee #{eid}</h4>
                              <p>Final Score: <b>{ex['FinalScore']}</b></p>
                              <p>Skill {ex['SkillMatch']} | Exp {ex['ExperienceScore']}
                                 | Stab {ex['StabilityScore']} | Avail {ex['AvailabilityScore']}</p>
                            </div>""", unsafe_allow_html=True)

                # ── Agreement indicator ───────────────────────────────────
                if set(greedy_eids) == set(ilp_eids) and greedy_eids:
                    st.markdown('<div class="same">✅ Both approaches selected the same employee(s)</div>',
                                unsafe_allow_html=True)
                elif greedy_eids or ilp_eids:
                    st.markdown('<div class="differ">⚠️ Approaches disagree — ILP considers global workload constraints</div>',
                                unsafe_allow_html=True)

                st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — ANALYTICS
# ══════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📈 Workforce Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Experience vs Proficiency")
        fig = px.scatter(
            employees,
            x="TotalWorkingYears",
            y="primary_skill_proficiency",
            color="Department",
            hover_data=["EmployeeNumber", "primary_skill"],
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#c5cae9", legend=dict(bgcolor="rgba(0,0,0,0)"),
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Availability Heatmap by Department & Level")
        heat_data = employees.groupby(["Department", "current_level"])["availability_projects"].mean().reset_index()
        heat_data["current_level"] = heat_data["current_level"].map({0: "Junior", 1: "Mid", 2: "Senior"})
        fig2 = px.density_heatmap(heat_data, x="current_level", y="Department",
                                   z="availability_projects",
                                   color_continuous_scale="Blues",
                                   text_auto=".1f")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c5cae9", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Department Weight Comparison")
        dept_opts = list(WEIGHTS.keys())
        sel_dept  = st.selectbox("Select department", dept_opts)
        w = WEIGHTS[sel_dept]
        fig3 = go.Figure(go.Bar(
            x=list(w.values()),
            y=list(w.keys()),
            orientation="h",
            marker_color=["#5c6bc0", "#42a5f5", "#26a69a", "#ffa726"],
        ))
        fig3.update_layout(xaxis_range=[0, 0.6],
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c5cae9", margin=dict(t=10, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### Performance Rating Distribution")
        if "PerformanceRating" in employees.columns:
            pr = employees["PerformanceRating"].value_counts().sort_index().reset_index()
            pr.columns = ["Rating", "Count"]
            pr["Rating"] = pr["Rating"].map({3: "Good (3)", 4: "Excellent (4)"})
            fig4 = px.bar(pr, x="Rating", y="Count",
                          color="Rating",
                          color_discrete_sequence=["#42a5f5", "#5c6bc0"])
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#c5cae9", showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.caption("PerformanceRating column not found.")

    st.markdown("---")
    st.markdown("#### Top 20 Employees by Primary Skill Proficiency")
    top20 = employees.nlargest(20, "primary_skill_proficiency")[
        ["EmployeeNumber", "Department", "primary_skill", "primary_skill_proficiency",
         "secondary_skill", "secondary_skill_proficiency", "current_level", "TotalWorkingYears"]
    ].copy()
    top20["current_level"] = top20["current_level"].map({0: "Junior", 1: "Mid-Level", 2: "Senior"})
    st.dataframe(top20.reset_index(drop=True), use_container_width=True)
