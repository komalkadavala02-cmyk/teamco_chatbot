import os
import re
import json
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Catbot - User Projects", layout="wide")

st.title("🐱 Catbot - User Projects Details")
st.caption("Enter ID or username to fetch projects and tasks")

@st.cache_data(show_spinner=False)
def load_df():
    csv_path = os.path.join(os.path.dirname(__file__), "users_large_dataset.csv")
    return pd.read_csv(csv_path)


df = load_df()

@st.cache_data(show_spinner=False)
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")
    return joblib.load(model_path)


def parse_projects(projects_raw):
    if isinstance(projects_raw, str):
        try:
            return json.loads(projects_raw)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(projects_raw)
            except Exception:
                return []
    if isinstance(projects_raw, list):
        return projects_raw
    return []


def extract_user_features(projects):
    total_tasks = 0
    progress_values = []
    for project in projects:
        tasks = project.get("tasks") or []
        total_tasks += len(tasks)
        for task in tasks:
            progress_values.append(task.get("progress", 0) or 0)
    avg_progress = float(np.mean(progress_values)) if progress_values else 0.0
    return {"total_tasks": total_tasks, "avg_progress": avg_progress}


def completion_label_to_text(label):
    mapping = {
        "<=14": "1-14 days",
        "15-30": "15-30 days",
        "31-60": "31-60 days",
        ">60": "more than 60 days",
    }
    return mapping.get(label, str(label))


def estimate_remaining_days(label, avg_progress):
    midpoints = {
        "<=14": 7.0,
        "15-30": 22.0,
        "31-60": 45.0,
        ">60": 75.0,
    }
    total_estimate = midpoints.get(label, 30.0)
    remaining = total_estimate * max(0.0, 1.0 - avg_progress / 100.0)
    return round(remaining)


user_tab, output_tab = st.tabs(["Lookup", "Result"])

with user_tab:
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Search by", ["ID", "Username"], horizontal=True)
    with col2:
        st.write("")

    if mode == "ID":
        raw = st.text_input("Enter ID", placeholder="e.g. 1")
    else:
        raw = st.text_input("Enter username", placeholder="e.g. priya_gajjar_0001")

    btn = st.button("Get Details", type="primary")

with output_tab:
    if not btn:
        st.info("Enter details and click **Get Details**.")
    else:
        if not raw.strip():
            st.warning("Input is empty.")
            st.stop()

        row = None
        if mode == "ID":
            try:
                input_id = int(raw.strip())
            except ValueError:
                st.error("ID must be a number.")
                st.stop()
            matches = df[df["id"] == input_id]
            if len(matches) > 0:
                row = matches.iloc[0].to_dict()
        else:
            username = raw.strip()
            if "username" not in df.columns:
                st.error("CSV does not contain 'username' column.")
                st.stop()
            matches = df[df["username"].astype(str) == username]
            if len(matches) > 0:
                row = matches.iloc[0].to_dict()

        if row is None:
            st.error("No user found for that input.")
            st.stop()

        # Basic user fields
        st.markdown("### User Details")
        cols = st.columns(2)
        with cols[0]:
            st.write(f"**ID:** {row.get('id')}")
            st.write(f"**Username:** {row.get('username')}")
            st.write(f"**Email:** {row.get('email')}")
            st.write(f"**DOB:** {row.get('dob')}")
        with cols[1]:
            st.write(f"**isVerified:** {row.get('isVerified')}")
            st.write(f"**isDeleted:** {row.get('isDeleted')}")
            st.write(f"**createdAt:** {row.get('createdAt')}")
            st.write(f"**updatedAt:** {row.get('updatedAt')}")

        profile_url = row.get("profileUrl")
        if profile_url and isinstance(profile_url, str) and profile_url.strip():
            st.markdown("#### Profile")
            st.image(profile_url, width=220)

        projects_raw = row.get("projects")
        projects = parse_projects(projects_raw)

        st.markdown("### Projects")
        if not projects:
            st.warning("No projects found for this user.")
            st.stop()

        # Prediction section using saved logistic regression model
        features = extract_user_features(projects)
        model = load_model()
        user_features = [[features["total_tasks"], features["avg_progress"]]]
        prediction_index = int(model.predict(user_features)[0])
        prediction_labels = ["15-30", "31-60", "<=14", ">60"]
        predicted_label = prediction_labels[prediction_index] if 0 <= prediction_index < len(prediction_labels) else "Unknown"
        predicted_days = completion_label_to_text(predicted_label)
        remaining_days = estimate_remaining_days(predicted_label, features["avg_progress"])

        st.markdown("### Prediction")
        st.write(f"- Total tasks: {features['total_tasks']}")
        st.write(f"- Average progress: {features['avg_progress']:.1f}%")
        st.write(f"- Predicted completion window: {predicted_days}")
        st.write(f"- Estimated remaining days: {remaining_days} days")

        st.markdown("---")

        # Render project cards
        for p in projects:
            project_name = p.get("projectName", "(no projectName)")
            project_image = p.get("projectImage")
            description = p.get("description")
            is_deleted = p.get("isDeleted")

            st.markdown("---")
            card = st.container()
            with card:
                c1, c2 = st.columns([1, 2])
                with c1:
                    if project_image:
                        st.image(project_image, use_container_width=True)
                with c2:
                    st.markdown(f"#### {project_name}")
                    if description:
                        st.write(description)
                    st.write(f"**projectId:** {p.get('id')}")
                    if is_deleted is not None:
                        st.write(f"**isDeleted:** {is_deleted}")

                tasks = p.get("tasks") or []
                if tasks:
                    st.markdown("##### Tasks")
                    for t in tasks:
                        st.write(
                            f"- **{t.get('name','(no task name)')}** "
                            f"(progress: {t.get('progress')}, isDeleted: {t.get('isDeleted')})"
                        )
                        tdesc = t.get("description")
                        if tdesc:
                            st.caption(tdesc)
                        # Dates
                        sd = t.get("startDate")
                        ed = t.get("endDate")
                        if sd or ed:
                            st.caption(f"Dates: {sd} → {ed}")
                else:
                    st.write("No tasks for this project.")

