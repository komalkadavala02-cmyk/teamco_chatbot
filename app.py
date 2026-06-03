import os
import re
import json
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import urllib.parse
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="User Projects", page_icon="🏗️", layout="wide")

st.title("🏗️ User Projects Details")
st.caption("Enter ID or username to fetch projects and tasks, then ask questions in Chat.")

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


def build_chat_knowledge(row, projects, features, predicted_label, remaining_days):
    sentences = []
    sentences.append(f"User ID is {row.get('id')}.")
    sentences.append(f"Username is {row.get('username')}.")
    sentences.append(f"Email is {row.get('email')}.")
    sentences.append(f"DOB is {row.get('dob')}.")
    sentences.append(f"Verified status is {row.get('isVerified')}.")
    sentences.append(f"Deleted status is {row.get('isDeleted')}.")
    sentences.append(f"Created at {row.get('createdAt')}.")
    sentences.append(f"Updated at {row.get('updatedAt')}.")
    sentences.append(f"The prediction window is {completion_label_to_text(predicted_label)}.")
    sentences.append(f"Estimated remaining days are {remaining_days}.")
    sentences.append(f"Total tasks are {features['total_tasks']}.")
    sentences.append(f"Average progress is {features['avg_progress']:.1f} percent.")

    for idx, project in enumerate(projects, start=1):
        project_name = project.get("projectName") or f"project {idx}"
        tasks = project.get("tasks") or []
        sentences.append(
            f"Project {project_name} has {len(tasks)} tasks and isDeleted is {project.get('isDeleted')}."
        )
        if project.get("description"):
            sentences.append(f"Project {project_name} description: {project['description']}.")

        for task in tasks:
            task_name = task.get("name") or "unnamed task"
            sentences.append(
                f"Task {task_name} in project {project_name} has progress {task.get('progress')} "
                f"and isDeleted {task.get('isDeleted')}.")
            if task.get("description"):
                sentences.append(f"Task {task_name} description: {task['description']}.")
            if task.get("startDate") or task.get("endDate"):
                sentences.append(
                    f"Task {task_name} dates: {task.get('startDate')} to {task.get('endDate')}.")

    return sentences


def answer_chat_question(question, knowledge_sentences):
    question = str(question).strip()
    if not question:
        return "Please type your question."

    docs = knowledge_sentences + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(docs)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    best_index = int(np.argmax(similarity))
    best_score = float(similarity[best_index])

    if best_score < 0.1:
        # Question is outside user data - search Google
        return search_google(question)

    return knowledge_sentences[best_index]


def search_google(query):
    """Search Google and return a brief answer using DuckDuckGo API (no auth required)."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_redirect=1"
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        if data.get("AbstractText"):
            answer = data["AbstractText"]
            if data.get("AbstractURL"):
                answer += f"\n\n[Learn more]({data['AbstractURL']})"
            return answer
        
        if data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
            result = data["RelatedTopics"][0]
            if "Text" in result:
                return result["Text"]
        
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        return f"[🔍 Search Google for '{query}']({search_url})"
    
    except Exception as e:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        return f"[🔍 Search Google for this question]({search_url})"


if 'selected_user' not in st.session_state:
    st.session_state.selected_user = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_user_id' not in st.session_state:
    st.session_state.selected_user_id = None

user_tab, output_tab, chat_tab = st.tabs(["Lookup", "Result", "Chat"])

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
    if not btn and st.session_state.selected_user is None:
        st.info("Enter details and click **Get Details**.")
    else:
        if btn:
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

            projects_raw = row.get("projects")
            projects = parse_projects(projects_raw)
            features = extract_user_features(projects)
            model = load_model()
            user_features = [[features["total_tasks"], features["avg_progress"]]]
            prediction_index = int(model.predict(user_features)[0])
            prediction_labels = ["15-30", "31-60", "<=14", ">60"]
            predicted_label = prediction_labels[prediction_index] if 0 <= prediction_index < len(prediction_labels) else "Unknown"
            predicted_days = completion_label_to_text(predicted_label)
            remaining_days = estimate_remaining_days(predicted_label, features["avg_progress"])

            st.session_state.selected_user = {
                "row": row,
                "projects": projects,
                "features": features,
                "predicted_label": predicted_label,
                "remaining_days": remaining_days,
                "knowledge_sentences": build_chat_knowledge(row, projects, features, predicted_label, remaining_days),
            }
            if st.session_state.selected_user_id != row.get("id"):
                st.session_state.chat_history = []
                st.session_state.selected_user_id = row.get("id")
        else:
            row = st.session_state.selected_user["row"]
            projects = st.session_state.selected_user["projects"]
            features = st.session_state.selected_user["features"]
            predicted_label = st.session_state.selected_user["predicted_label"]
            remaining_days = st.session_state.selected_user["remaining_days"]
            predicted_days = completion_label_to_text(predicted_label)

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

        st.markdown("### Projects")
        if not projects:
            st.warning("No projects found for this user.")
            st.stop()

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
                        sd = t.get("startDate")
                        ed = t.get("endDate")
                        if sd or ed:
                            st.caption(f"Dates: {sd} → {ed}")
                else:
                    st.write("No tasks for this project.")

with chat_tab:
    st.markdown("### Chat Assistant")
    
    if st.session_state.selected_user is None:
        st.info("First find a user in the Lookup tab, then ask questions about that user here.")
    else:
        st.markdown("**Conversation:**")
        if st.session_state.chat_history:
            chat_container = st.container()
            with chat_container:
                for chat in st.session_state.chat_history:
                    st.markdown(f"**You:** {chat['question']}")
                    st.markdown(f"**Bot:** {chat['answer']}")
                    st.divider()
        else:
            st.info("No messages yet. Start a conversation below!")

        st.markdown("---")
        st.write(
            "Ask questions about the selected user or anything else. "
            "If we don't have the answer in user data, we'll search Google for you and provide a link."
        )
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input("Your question", key="chat_question_user", label_visibility="collapsed")
        with col2:
            if st.button("Send", type="primary", key="send_chat_user"):
                if question.strip():
                    answer = answer_chat_question(question, st.session_state.selected_user["knowledge_sentences"])
                    st.session_state.chat_history.append({
                        "question": question, 
                        "answer": answer
                    })
                    st.rerun()
