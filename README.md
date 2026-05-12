# 🎯 HR Fit Recommender — Dashboard Setup Guide
Complete beginner-friendly instructions

---

## What you will get
A fully working web dashboard in your browser that looks like a real product.
No coding knowledge needed beyond copy-pasting commands.

---

## Step 1 — Make sure Python is installed

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```
python --version
```

You should see something like `Python 3.10.x` or higher.
If not, download Python from https://python.org/downloads  (choose version 3.10 or newer)

---

## Step 2 — Create a project folder

Make a new folder anywhere on your computer. For example on Windows:
```
mkdir C:\my_dashboard
cd C:\my_dashboard
```

On Mac/Linux:
```
mkdir ~/my_dashboard
cd ~/my_dashboard
```

---

## Step 3 — Copy the files into that folder

Copy these two files into your folder:
- `app.py`
- `utils.py`
- `requirements.txt`

Also copy your three CSV data files into the SAME folder:
- `HR_Employee_Attrition_COMPLETE_DATASET.csv`
- `projects_dataset.csv`
- `project_role_requirements (2).csv`

---

## Step 4 — Install required libraries

In your terminal (make sure you are inside your folder), run:

```
pip install streamlit pandas plotly pulp
```

Wait for it to finish. This downloads everything needed. It only takes a minute.

---

## Step 5 — Run the dashboard

In the same terminal, type:

```
streamlit run app.py
```

Your browser will automatically open at http://localhost:8501 and the dashboard will appear!

---

## Step 6 — Using the dashboard

1. **Upload your CSVs** using the sidebar on the left
   - Click "Employee CSV" and select your employee file
   - Click "Projects CSV" and select your projects file
   - Click "Roles CSV" and select your roles file

2. **Explore Overview tab** — see charts about your workforce

3. **Employee Explorer tab** — filter and browse employees

4. **Greedy Allocation tab** — select a project from the sidebar dropdown, see who gets picked

5. **Run ILP Optimization** — click the button in the sidebar to run the global optimizer

6. **Compare tab** — see Greedy vs ILP side by side for the selected project

7. **Analytics tab** — deeper charts and insights

---

## If something goes wrong

**Error: "streamlit not found"**
→ Run: `pip install streamlit` again

**Error: "pulp not found"**  
→ Run: `pip install pulp`

**Error: "No module named plotly"**
→ Run: `pip install plotly`

**The browser doesn't open automatically**
→ Manually go to: http://localhost:8501

**To stop the dashboard**
→ Press Ctrl+C in the terminal

---

## File structure (what your folder should look like)

```
my_dashboard/
├── app.py                                    ← main dashboard
├── utils.py                                  ← scoring & ILP logic
├── requirements.txt                          ← library list
├── HR_Employee_Attrition_COMPLETE_DATASET.csv
├── projects_dataset.csv
└── project_role_requirements (2).csv
```

---

## Dashboard tabs explained

| Tab | What it shows |
|-----|--------------|
| 📊 Overview | KPI cards + 4 charts about the workforce |
| 👥 Employee Explorer | Filterable table of all 1,470 employees |
| ⚡ Greedy Allocation | Role-by-role recommendations for selected project |
| 🧠 ILP Optimization | Global optimal assignment across ALL projects |
| 🔄 Compare | Side-by-side Greedy vs ILP for one project |
| 📈 Analytics | Scatter plots, heatmaps, proficiency rankings |
