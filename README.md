# HRFit Optimizer

A workforce allocation and employee-project fit recommendation system that helps HR teams identify the most suitable employees for project roles using multi-factor scoring, Greedy Allocation, and Integer Linear Programming (ILP) optimization.

## Features
- Employee-project fit scoring
- Greedy and ILP-based allocation comparison
- Interactive Streamlit dashboard
- Workforce analytics and visualizations
- Department-specific scoring weights
- Skill-based recommendation engine

## Tech Stack
- Python
- Streamlit
- Pandas
- Plotly
- PuLP

## Run Locally

Install dependencies:

```bash
pip install streamlit pandas plotly pulp
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Dataset Disclaimer

This project uses three datasets:

- `HR_Employee_Attrition_COMPLETE_D...`
  - Based on a real IBM HR employee attrition dataset.
  - The dataset was slightly modified and extended to better align with the workforce allocation use case of this project.

- `project_role_requirements (2) (1).csv`
- `projects_dataset (3).csv`
  - These two datasets were synthetically generated to simulate realistic enterprise project allocation scenarios and to complement the real-world employee dataset.

The synthetic datasets were designed to preserve practical workforce planning patterns while enabling controlled experimentation and optimization analysis.

## Documentation

For complete project explanation, methodology, architecture, scoring logic, limitations, and future improvements, please refer to the detailed project documentation included in this repository.

## Authors
- Ashlesha Agrawal
- Aashi Soni