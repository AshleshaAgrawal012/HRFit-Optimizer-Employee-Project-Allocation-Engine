"""
utils.py  —  All scoring & allocation logic
Extracted from matching.ipynb (both Greedy and ILP approaches)
"""

import pandas as pd
from pulp import (LpProblem, LpMaximize, LpVariable, LpStatus,
                  lpSum, value, PULP_CBC_CMD)

# ─────────────────────────────────────────────
# SKILL ONTOLOGY
# ─────────────────────────────────────────────
SKILL_MAP = {
    "data_science":         ["python", "machine learning", "statistics", "data analysis", "r", "deep learning"],
    "sales":                ["sales", "negotiation", "crm", "business development", "account management", "lead generation", "cold calling"],
    "software_engineering": ["java", "python", "software development", "c++", "sql", "cloud", "devops"],
    "marketing":            ["seo", "digital marketing", "analytics", "content marketing", "brand management"],
    "management":           ["manager", "leadership", "project management", "team management", "operations", "strategy", "planning", "decision making"],
    "human_resources":      ["human resources", "hr", "recruitment", "talent acquisition", "employee relations", "payroll", "performance management", "training"],
    "research":             ["research", "research scientist", "clinical research", "data collection", "hypothesis testing", "laboratory", "lab techniques", "experimental design"],
    "laboratory":           ["laboratory technician", "lab technician", "chemistry", "biology", "quality control", "testing", "calibration", "microscopy"],
    "healthcare":           ["healthcare representative", "healthcare", "medical sales", "pharmacology", "clinical", "patient management", "medical devices"],
    "manufacturing":        ["manufacturing", "production", "supply chain", "lean manufacturing", "six sigma", "process optimization", "operations management"],
}

SKILL_TO_CATEGORY = {skill: cat for cat, skills in SKILL_MAP.items() for skill in skills}

# ─────────────────────────────────────────────
# DEPARTMENT WEIGHTS
# ─────────────────────────────────────────────
WEIGHTS = {
    "Research & Development": {"skill": 0.50, "experience": 0.25, "stability": 0.15, "availability": 0.10},
    "Sales":                  {"skill": 0.40, "experience": 0.20, "stability": 0.10, "availability": 0.30},
    "Human Resources":        {"skill": 0.35, "experience": 0.25, "stability": 0.25, "availability": 0.15},
}

MAX_PROJECTS = 3


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_category(skill):
    return SKILL_TO_CATEGORY.get(skill, None)


def skill_match(emp_p, emp_s, req_p, req_s):
    score = 0.0
    if emp_p == req_p:
        score += 1.0
    elif emp_s == req_s:
        score += 0.5
    if emp_p != req_p:
        ec, rc = get_category(emp_p), get_category(req_p)
        if ec is not None and ec == rc:
            score += 0.5
    return min(score, 1.0)


def scale_experience(y):   return min(y / 40, 1.0)
def scale_stability(y):    return min(y / 20, 1.0)
def scale_availability(p): return 1 / (1 + p)


def score_employee(emp, role_row, dept):
    w  = WEIGHTS.get(dept, WEIGHTS["Research & Development"])
    sk = skill_match(emp["primary_skill"], emp["secondary_skill"],
                     role_row["PrimarySkill"], role_row["SecondarySkill"])
    final = (w["skill"]        * sk
           + w["experience"]   * scale_experience(emp["TotalWorkingYears"])
           + w["stability"]    * scale_stability(emp["YearsInCurrentRole"])
           + w["availability"] * scale_availability(emp["availability_projects"]))
    exp = {
        "SkillMatch":        round(sk, 2),
        "ExperienceScore":   round(scale_experience(emp["TotalWorkingYears"]), 2),
        "StabilityScore":    round(scale_stability(emp["YearsInCurrentRole"]), 2),
        "AvailabilityScore": round(scale_availability(emp["availability_projects"]), 2),
        "FinalScore":        round(final, 3),
    }
    return final, exp


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data(emp_path, proj_path, roles_path):
    employees = pd.read_csv(emp_path)
    projects  = pd.read_csv(proj_path)
    roles     = pd.read_csv(roles_path)

    employees["primary_skill"]   = employees["primary_skill"].str.lower().str.strip()
    employees["secondary_skill"] = employees["secondary_skill"].str.lower().str.strip()
    roles["PrimarySkill"]        = roles["PrimarySkill"].str.lower().str.strip()
    roles["SecondarySkill"]      = roles["SecondarySkill"].str.lower().str.strip()

    if "EmployeesRequired" in roles.columns:
        roles = roles.rename(columns={"EmployeesRequired": "RequiredEmployees"})
    elif "RequiredEmployees" not in roles.columns:
        roles["RequiredEmployees"] = 1

    return employees, projects, roles


# ─────────────────────────────────────────────
# APPROACH 1  —  GREEDY
# ─────────────────────────────────────────────
def greedy_allocate(project_row, roles_df, employees_df):
    """
    Returns dict:  role_name → {"selected": DataFrame, "top_candidates": DataFrame}
    Does NOT mutate employees_df.
    """
    emp = employees_df.copy()
    project_roles = roles_df[roles_df["ProjectID"] == project_row["ProjectID"]]
    team = {}

    for _, role in project_roles.iterrows():
        eligible = emp[
            (emp["TotalWorkingYears"] >= role["MinExperienceYears"]) &
            (emp["availability_projects"] < MAX_PROJECTS)
        ].copy()

        if eligible.empty:
            continue

        scores, explanations = [], []
        for _, e in eligible.iterrows():
            s, ex = score_employee(e, role, project_row["Department"])
            scores.append(s)
            explanations.append(ex)

        eligible["score"]       = scores
        eligible["explanation"] = explanations
        ranked = eligible.sort_values("score", ascending=False)

        needed = int(role.get("RequiredEmployees", role.get("EmployeesRequired", 1)))
        team[role["RequiredRole"]] = {
            "selected":       ranked.head(needed),
            "top_candidates": ranked.head(5),
        }

    return team


# ─────────────────────────────────────────────
# APPROACH 2  —  ILP
# ─────────────────────────────────────────────
def ilp_allocate(projects_df, roles_df, employees_df):
    """
    Returns (assignments dict, status str, total_score float)
    assignments:  project_id → {role_name → [(eid, score, explanation), …]}
    """
    emp = employees_df.copy()

    # Build role list
    role_list = []
    for _, r in roles_df.iterrows():
        pid = r["ProjectID"]
        dept_rows = projects_df.loc[projects_df["ProjectID"] == pid, "Department"]
        if dept_rows.empty:
            continue
        needed = int(r.get("RequiredEmployees", r.get("EmployeesRequired", 1)))
        role_list.append({
            "role_id":    f"{pid}__{r['RequiredRole']}",
            "project_id": pid,
            "role_name":  r["RequiredRole"],
            "req_count":  needed,
            "min_exp":    r["MinExperienceYears"],
            "dept":       dept_rows.iloc[0],
            "role_row":   r,
        })

    # Score matrix
    score_matrix, explanation_matrix = {}, {}
    for _, e in emp.iterrows():
        eid = e["EmployeeNumber"]
        for role in role_list:
            if e["TotalWorkingYears"] >= role["min_exp"]:
                s, ex = score_employee(e, role["role_row"], role["dept"])
                score_matrix[(eid, role["role_id"])]       = s
                explanation_matrix[(eid, role["role_id"])] = ex

    # Build ILP
    prob = LpProblem("Workforce_Allocation", LpMaximize)
    x    = LpVariable.dicts("assign", score_matrix.keys(), cat="Binary")

    prob += lpSum(score_matrix[k] * x[k] for k in score_matrix)

    for role in role_list:
        rid      = role["role_id"]
        eligible = [k for k in score_matrix if k[1] == rid]
        needed   = role["req_count"]
        assigned = lpSum(x[k] for k in eligible)
        if len(eligible) >= needed:
            prob += assigned == needed, f"exact_{rid}"
        else:
            prob += assigned <= len(eligible), f"max_{rid}"

    for _, e in emp.iterrows():
        eid      = e["EmployeeNumber"]
        capacity = max(MAX_PROJECTS - int(e["availability_projects"]), 0)
        keys     = [k for k in score_matrix if k[0] == eid]
        if keys:
            prob += lpSum(x[k] for k in keys) <= capacity, f"cap_{eid}"

    prob.solve(PULP_CBC_CMD(msg=0))
    status      = LpStatus[prob.status]
    total_score = value(prob.objective) or 0.0

    # Extract
    per_role = {}
    for (eid, rid), var in x.items():
        if value(var) is not None and value(var) > 0.5:
            per_role.setdefault(rid, []).append(
                (eid, score_matrix[(eid, rid)], explanation_matrix[(eid, rid)])
            )

    assignments = {}
    for role in role_list:
        rid      = role["role_id"]
        pid      = role["project_id"]
        rname    = role["role_name"]
        selected = sorted(per_role.get(rid, []), key=lambda t: t[1], reverse=True)[:role["req_count"]]
        if selected:
            assignments.setdefault(pid, {})[rname] = selected

    return assignments, status, total_score
