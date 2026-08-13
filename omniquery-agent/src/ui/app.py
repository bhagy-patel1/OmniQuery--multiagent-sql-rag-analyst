import os
import streamlit as st
import requests
import time
import uuid

# Reads from cloud environment, defaults to localhost for testing
API_URL = os.getenv("API_URL", "http://localhost:8000") 

st.set_page_config(page_title="OmniQuery — Enterprise Data Intelligence", page_icon="🤖", layout="wide")

# Unique User Session Generation
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

# Initialize Messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: DATA CONNECTORS ---
with st.sidebar:
    st.header("🔌 Data Connectors")
    st.caption(f"Session ID: `{session_id[:8]}`")
    
    # 1. Dynamic Database Connection Inputs
    st.subheader("1. PostgreSQL Database")
    use_custom_db = st.checkbox("Connect Custom Database", value=False)
    
    db_config_payload = None
    if use_custom_db:
        db_host = st.text_input("Host", value="localhost")
        db_port = st.number_input("Port", value=5432)
        db_name = st.text_input("Database Name", value="enterprise_hub")
        db_user = st.text_input("User", value="admin")
        db_pass = st.text_input("Password", type="password")
        
        db_config_payload = {
            "host": db_host, "port": db_port, "dbname": db_name, 
            "user": db_user, "password": db_pass
        }

    if st.button("Test DB Connection"):
        with st.spinner("Pinging database..."):
            try:
                res = requests.post(f"{API_URL}/api/db/test", json=db_config_payload, timeout=5).json()
                if res.get("status") == "success":
                    st.success("Connected successfully!")
                else:
                    st.error(f"DB Error: {res.get('message')}")
            except Exception:
                st.error("API Server Offline.")

    st.divider()

    # 2. Session CSV Upload
# 2. Session CSV Upload
    st.subheader("2. CSV Spreadsheet")
    # 🚨 FIX: Allow multiple files
    csv_files = st.file_uploader("Upload Ad-Hoc Data", type=["csv"], accept_multiple_files=True)
    
    if csv_files and st.button("Update CSV Database"):
        with st.spinner("Uploading datasets..."):
            try:
                # 🚨 FIX: Loop through each file and upload it
                for csv_file in csv_files:
                    files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
                    data = {"session_id": session_id}
                    res = requests.post(f"{API_URL}/api/upload/csv", data=data, files=files, timeout=30)
                    
                    if res.status_code != 200:
                        st.error(f"Failed to upload {csv_file.name}: {res.text}")
                        
                st.success(f"Successfully uploaded {len(csv_files)} CSV file(s)!")
            except Exception as e:
                st.error(f"Upload error: {str(e)}")

    st.divider()

    # 3. Session PDF Upload
    st.subheader("3. PDF Documents")
    pdf_file = st.file_uploader("Upload Enterprise Reports", type=["pdf"])
    if pdf_file and st.button("Ingest to Vector Store"):
        with st.spinner("Chunking and Embedding..."):
            try:
                files = {"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")}
                data = {"session_id": session_id}
                res = requests.post(f"{API_URL}/api/upload/pdf", data=data, files=files, timeout=120)
                if res.status_code == 200:
                    st.success(f"PDF Embedded! {res.json().get('message')}")
                else:
                    st.error(f"Ingestion failed: {res.text}")
            except Exception as e:
                st.error(f"Ingestion error: {str(e)}")

    st.divider()

    # 4. Session Management
    st.subheader("4. Privacy & Session")
    if st.button("🗑️ Clear Session Data & Restart", type="primary"):
        with st.spinner("Erasing all files and vectors..."):
            try:
                res = requests.delete(f"{API_URL}/api/session/{session_id}", timeout=10)
                if res.status_code == 200:
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4()) # Generate fresh ID
                    st.success("Session wiped! Restarting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to clear session.")
            except Exception as e:
                st.error("Server offline. Cannot clear session.")

# --- MAIN CHAT INTERFACE ---
st.title("🤖 OmniQuery — Enterprise AI Analyst")
st.caption("Tri-Modal Autonomous Reasoning across PostgreSQL, CSV Spreadsheets, and PDF Reports")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            has_evidence = False
            if meta.get("sql_query") or meta.get("csv_query") or meta.get("rag_context"):
                has_evidence = True
            
            if has_evidence:
                with st.expander("🔍 Inspection & Source Evidence"):
                    st.write(f"**Route Taken:** `{meta.get('route')}`")
                    if meta.get("sql_query"):
                        st.markdown("**PostgreSQL Executed:**")
                        st.code(meta["sql_query"], language="sql")
                    if meta.get("csv_query"):
                        st.markdown("**DuckDB CSV Executed:**")
                        st.code(meta["csv_query"], language="sql")
                    if meta.get("rag_context"):
                        st.markdown("**PDF Excerpts Used:**")
                        for ctx in meta["rag_context"]:
                            st.caption(f"📄 **{ctx['document_name']}** (Score: {ctx['rerank_score']})")
                            st.text(ctx["content"])

if prompt := st.chat_input("Ask a question about sales metrics, budgets, or strategy reports..."):
    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing queries across data stores..."):
            start_time = time.time()
            try:
                # 🚨 Pass full session history to the backend with increased timeout
                payload = {
                    "query": prompt,
                    "session_id": session_id,
                    "history": st.session_state.messages,
                    "db_config": db_config_payload if use_custom_db else None
                }
                response = requests.post(f"{API_URL}/api/chat", json=payload, timeout=180)
                elapsed = round(time.time() - start_time, 2)

                if response.status_code == 200:
                    data = response.json()
                    final_res = data["final_response"]
                    route = data["route"]

                    st.markdown(final_res)
                    st.caption(f"⏱️ Executed in {elapsed}s | 🔀 Route: `{route}`")

                    has_evidence = False
                    if data.get("sql_query") or data.get("csv_query") or data.get("rag_context"):
                        has_evidence = True
                        
                    if has_evidence:
                        with st.expander("🔍 Inspection & Source Evidence"):
                            st.write(f"**Route Taken:** `{route}`")
                            if data.get("sql_query"):
                                st.markdown("**PostgreSQL Executed:**")
                                st.code(data["sql_query"], language="sql")
                            if data.get("csv_query"):
                                st.markdown("**DuckDB CSV Executed:**")
                                st.code(data["csv_query"], language="sql")
                            if data.get("rag_context"):
                                st.markdown("**PDF Excerpts Used:**")
                                for ctx in data["rag_context"]:
                                    st.caption(f"📄 **{ctx['document_name']}** (Score: {ctx['rerank_score']})")
                                    st.text(ctx["content"])

                    st.session_state.messages.append({"role": "assistant", "content": final_res, "metadata": data})
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.Timeout:
                st.error("Query timed out. The backend server took too long to respond.")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend. Details: {str(e)}")