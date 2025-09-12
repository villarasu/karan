# app.py
import os
import io
import json
import datetime
import tempfile
import requests
import streamlit as st
import pandas as pd
import boto3
import psycopg2
from pgvector.psycopg2 import register_vector
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from sentence_transformers import SentenceTransformer
from botocore.client import Config
from typing import List, Tuple, Optional

# ================== CONFIG ==================
#


# Load your credentials securely (e.g., from st.secrets)
# Load secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

AWS_REGION = st.secrets["AWS_REGION"]
S3_BUCKET = st.secrets["S3_BUCKET"]
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]

TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

SMTP_EMAIL = st.secrets["SMTP_EMAIL"]
SMTP_PASS = st.secrets["SMTP_PASS"]

RDS_HOST = st.secrets["RDS_HOST"]
RDS_DB = st.secrets["RDS_DB"]
RDS_USER = st.secrets["RDS_USER"]
RDS_PASS = st.secrets["RDS_PASS"]



LOCAL_MODEL_PATH =  "best.pt"
TABLE_NAME       =  "detections"



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Make sure you have a local model file (e.g., best.pt)



# Option: if True, DO NOT store detections labeled "helmet" (class 0)
SKIP_HELMET_DETECTIONS = True

# ================== INIT ==================
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)

@st.cache_resource
def load_local_embedder():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load local embedder: {e}")
        return None

local_embedder = load_local_embedder()

# =============== HELPERS ===============
def presign_s3(bucket: str, key: str, expires_in: int = 3600) -> str:
    if not bucket or not key:
        return ""
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
    except Exception:
        return ""


def send_telegram_alert(camera_id, location, timestamp, confidence, proof_url):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    message = (
        f"🚨 Accident Detected!\n"
        f"Camera: {camera_id}\n"
        f"Location: {location}\n"
        f"Timestamp: {timestamp}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Proof: {proof_url}"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=5,
        )
    except Exception:
        # don't crash the app if Telegram fails
        st.warning("Telegram alert failed (check token/chat id).")

def make_pdf(df: pd.DataFrame, filename="detection_report.pdf") -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", size=16)
    pdf.cell(200, 10, "Helmet & Accident Detection Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.ln(10)

    for _, row in df.iterrows():
        # Ensure timestamp is a string
        ts = str(row.get("timestamp", "N/A"))
        cam = str(row.get("camera_id", "N/A"))
        cls = str(row.get("class_label", "N/A"))
        conf = row.get("confidence", 0.0)
        
        pdf.set_font("Arial", "B", 10)
        pdf.multi_cell(0, 7, f"Timestamp: {ts} | Camera: {cam} | Class: {cls} | Confidence: {conf:.2f}")
        
        pdf.set_font("Arial", "", 9)
        proof = row.get("proof_url", "") or row.get("s3_image_url", "")
        pdf.multi_cell(0, 5, f"Proof URL: {proof}")
        pdf.ln(4) # Add a small space between entries
        
    pdf.output(filename)
    return filename

def send_email(to_email: str, subject: str, body: str, attachment_path: str):
    if not SMTP_EMAIL or not SMTP_PASS:
        st.error("❌ SMTP credentials are not configured.")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        with open(attachment_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)
        
        # Connect to Gmail's SMTP server
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        # Login using your email and the 16-character App Password
        server.login(SMTP_EMAIL, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError:
        st.error("❌ SMTP Authentication Failed: Check your email and App Password. Ensure you're using a 16-character App Password, not your regular Google password.")
        return False
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False


def get_embedding(text: str) -> List[float]:
    text = text[:1500]
    if HF_API_KEY:
        try:
            resp = requests.post(
                "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": text},
                timeout=20,
            )
            if resp.status_code == 200:
                emb = resp.json()
                return emb[0] if isinstance(emb[0], list) else emb
        except Exception:
            st.warning("HF embedding failed, using local embedder.")
    if local_embedder:
        return local_embedder.encode(text, normalize_embeddings=True).tolist()
    raise RuntimeError("No embedding available.")

def get_conn():
    return psycopg2.connect(host=RDS_HOST, database=RDS_DB, user=RDS_USER, password=RDS_PASS)

# ============= SQL TOOLS ==============
def sql_accidents_last_week(conn) -> pd.DataFrame:
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT timestamp, camera_id, location, class_label, confidence, s3_image_url
        FROM {TABLE_NAME}
        WHERE class_label = 'accident' AND timestamp >= NOW() - INTERVAL '7 days'
        ORDER BY timestamp DESC
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

def sql_most_helmet_violations(conn, limit=5) -> pd.DataFrame:
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT camera_id, COUNT(*) AS violations
        FROM {TABLE_NAME}
        WHERE class_label = 'no_helmet'
        GROUP BY camera_id
        ORDER BY violations DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["camera_id", "violations"])

def sql_most_detections_by_camera(conn) -> Optional[Tuple[str,int]]:
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT camera_id, COUNT(*) AS total_detections
        FROM {TABLE_NAME}
        GROUP BY camera_id
        ORDER BY total_detections DESC
        LIMIT 1
    """)
    return cur.fetchone()

def sql_violations_by_location(conn) -> pd.DataFrame:
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT location, COUNT(*) AS violations
        FROM {TABLE_NAME}
        WHERE class_label = 'no_helmet'
        GROUP BY location
        ORDER BY violations DESC
    """)
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["location", "violations"])

def sql_accidents_by_day(conn, days=14) -> pd.DataFrame:
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT DATE(timestamp) AS day, COUNT(*) AS accidents
        FROM {TABLE_NAME}
        WHERE class_label = 'accident' AND timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY day
        ORDER BY day ASC
    """, (days,))
    rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["day", "accidents"])

# ============= VECTOR SEARCH ==============
def vector_search(query: str, top_k: int = 10) -> pd.DataFrame:
    q_emb = get_embedding(query)
    conn = get_conn()
    register_vector(conn)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT id, timestamp, camera_id, location, class_label, confidence, s3_image_url
        FROM {TABLE_NAME}
          ORDER BY embedding <-> %s::vector
        LIMIT %s
    """, (q_emb, top_k))
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    if "s3_image_url" in df.columns:
        df["proof_url"] = df["s3_image_url"].apply(lambda k: presign_s3(S3_BUCKET, k))
    conn.close()
    return df

# ============= RAG LOGIC ==============
DATA_KEYWORDS = ["accident","helmet","violation","violations","camera","location","detection","timestamp","report","no_helmet","accidents","helmet_violation"]

def is_data_related(q: str) -> bool:
    ql = q.lower()
    if any(k in ql for k in DATA_KEYWORDS):
        return True
    for token in ["who","what","when","which","show","list","how many","count","top"]:
        if token in ql and any(k in ql for k in DATA_KEYWORDS):
            return True
    return False

def analyze_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        return "No matching records found."
    total = len(df)
    by_class = df["class_label"].value_counts().to_dict() if "class_label" in df.columns else {}
    by_camera = df["camera_id"].value_counts().head(5).to_dict() if "camera_id" in df.columns else {}
    earliest = pd.to_datetime(df["timestamp"]).min() if "timestamp" in df.columns else None
    latest = pd.to_datetime(df["timestamp"]).max() if "timestamp" in df.columns else None
    
    lines = [f"Found {total} matching events."]
    if by_class:
        lines.append("\nBreakdown by class:")
        for cls,cnt in by_class.items():
            lines.append(f"  • {cls}: {cnt}")
    if by_camera:
        lines.append("\nTop cameras:")
        for cam,cnt in by_camera.items():
            lines.append(f"  • {cam}: {cnt}")
    if earliest and latest:
        lines.append(f"\nTime range: {earliest.strftime('%Y-%m-%d %H:%M')} → {latest.strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


def run_rag_pipeline(user_query: str):
    ql = user_query.lower().strip()
    if not is_data_related(ql):
        return ("⚠️ This assistant only answers data-related questions about detections, cameras, helmet violations, timestamps and reports.", pd.DataFrame())
    conn = get_conn()
    register_vector(conn)
    try:
        if "accidents from last week" in ql or ("accidents" in ql and "last week" in ql):
            df = sql_accidents_last_week(conn)
            if not df.empty:
                df["proof_url"] = df["s3_image_url"].apply(lambda k: presign_s3(S3_BUCKET, k))
            conn.close()
            return (f"Accidents from last 7 days: {len(df)} records.", df)
        if "most helmet violations" in ql or ("helmet" in ql and "most" in ql):
            df = sql_most_helmet_violations(conn, limit=10)
            conn.close()
            return ("Top cameras by helmet violations.", df)
        if "most" in ql and "camera" in ql:
            res = sql_most_detections_by_camera(conn)
            conn.close()
            if res:
                cam,count = res
                return (f"📸 Camera *{cam}* recorded the most detections: {count} events.", pd.DataFrame([{"camera_id":cam,"total_detections":count}]))
            else:
                return ("No detection data available.", pd.DataFrame())
        if "helmet violations by location" in ql or ("helmet" in ql and "location" in ql):
            df = sql_violations_by_location(conn)
            conn.close()
            return ("Helmet violations grouped by location.", df)
        if "accidents by day" in ql or ("accidents" in ql and "by day" in ql):
            df = sql_accidents_by_day(conn, days=14)
            conn.close()
            return ("Accident counts by day (last 14 days).", df)
        df = vector_search(user_query, top_k=50)
        summary = analyze_dataframe(df)
        return (f"Results retrieved by semantic search:\n\n{summary}", df)
    except Exception as e:
        st.error(f"Database query failed: {e}")
        if conn:
            conn.close()
        return (f"Error while handling query: {e}", pd.DataFrame())

# ============= STREAMLIT APP ==============
st.set_page_config(page_title="Helmet & Accident RAG Chatbot", layout="wide", page_icon="🚦")
st.title("🚦 Helmet & Accident Detection System — RAG Chatbot")
mode = st.sidebar.radio("Mode", ["Detect", "Query", "RAG Chatbot"])

# label mapping for your YOLO model
LABELS_MAP = {0: "helmet", 1: "no_helmet", 2: "accident"}

if mode == "Detect":
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4"])
    camera_id = st.text_input("Camera ID","cam_001")
    location = st.text_input("Location","unknown")
    if uploaded_file:
        ext = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            tmp_path = tmp.name

        # load model and run inference
        model = YOLO(LOCAL_MODEL_PATH)
        results = model(tmp_path)

        conn = get_conn()
        register_vector(conn)
        cursor = conn.cursor()

        for r in results:
            st.image(r.plot(), caption="Detection Result", use_column_width=True)
            for i, box in enumerate(r.boxes):
                try:
                    cls_id = int(box.cls[0])
                except Exception:
                    continue # Skip if class ID is invalid

                class_label = LABELS_MAP.get(cls_id, f"cls_{cls_id}")
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()

                if SKIP_HELMET_DETECTIONS and class_label == "helmet":
                    st.info(f"Detected helmet (class {cls_id}) — skipped storage.")
                    continue

                s3_key = f"detections/{class_label}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i}{ext}"
                try:
                    s3.upload_file(tmp_path, S3_BUCKET, s3_key)
                    proof_url = presign_s3(S3_BUCKET, s3_key)
                except Exception as e:
                    st.error(f"S3 upload failed: {e}")
                    proof_url = ""

                desc = f"{class_label} at {location}, conf {confidence:.2f}, camera {camera_id}"
                try:
                    embedding = get_embedding(desc)
                except Exception as e:
                    st.warning(f"Embedding failed: {e}")
                    embedding = None

                st.write({"camera_id": camera_id, "class_label": class_label, "confidence": confidence, "s3_key": s3_key})

                try:
                    cursor.execute(f"""
                        INSERT INTO {TABLE_NAME}
                        (timestamp, camera_id, location, class_label, confidence, bbox_coordinates, s3_image_url, embedding)
                        VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
                    """, (camera_id, location, class_label, confidence, json.dumps(bbox), s3_key, embedding))
                    conn.commit()
                    st.success(f"Saved detection: {class_label} (conf {confidence:.2f})")
                except Exception as e:
                    conn.rollback()
                    st.error(f"DB insert failed: {e}")
                    continue

                if class_label == "accident":
                    send_telegram_alert(camera_id, location, datetime.datetime.now().isoformat(), confidence, proof_url)
                    st.warning("‼️ Accident detected! Telegram alert sent.")

        cursor.close()
        conn.close()
        os.remove(tmp_path)

elif mode == "Query":
    st.header("Query Logs (Structured Queries)")
    query = st.text_input("Enter your question:")
    if st.button("Search"):
        if not is_data_related(query):
            st.warning("❌ Only data-related queries are supported.")
            st.stop()
        conn = get_conn()
        ql = query.lower()
        try:
            if "accidents from last week" in ql:
                df = sql_accidents_last_week(conn)
                if not df.empty and "s3_image_url" in df.columns:
                    df["proof_url"] = df["s3_image_url"].apply(lambda k: presign_s3(S3_BUCKET,k))
                st.dataframe(df)
            elif "most helmet violations" in ql:
                st.dataframe(sql_most_helmet_violations(conn,limit=10))
            elif "most" in ql and "camera" in ql:
                res = sql_most_detections_by_camera(conn)
                if res:
                    st.success(f"📸 Camera **{res[0]}** recorded the **most detections**: {res[1]} events.")
                else:
                    st.info("No detection data found.")
            else:
                st.info("Try: 'accidents from last week' or 'most helmet violations'.")
        finally:
            conn.close()

elif mode == "RAG Chatbot":
    st.header("RAG Chatbot — Ask data-related questions")

    # Initialize session state to hold data across reruns
    if 'rag_df' not in st.session_state:
        st.session_state.rag_df = pd.DataFrame()
    if 'summary_text' not in st.session_state:
        st.session_state.summary_text = ""

    user_query = st.text_input("Ask anything about detection data:", key="rag_query")

    if st.button("Ask", key="rag_ask"):
        if user_query:
            with st.spinner("Searching for answers..."):
                summary, df = run_rag_pipeline(user_query)
                # Store results in session state to persist them
                st.session_state.summary_text = summary
                st.session_state.rag_df = df
        else:
            st.warning("Please enter a question.")

    # Always display results from session state
    if st.session_state.summary_text:
        st.markdown("### Answer")
        st.text_area("", value=st.session_state.summary_text, height=150, disabled=True)

    # Check if the DataFrame in session state has data
    if not st.session_state.rag_df.empty:
        df_results = st.session_state.rag_df
        st.markdown("### Retrieved Records")
        st.dataframe(df_results.head(50))

        # --- Visualizations ---
        col1, col2 = st.columns(2)
        with col1:
            if "class_label" in df_results.columns:
                st.markdown("##### Detections by Class")
                st.bar_chart(df_results["class_label"].value_counts())
        with col2:
            if "camera_id" in df_results.columns:
                st.markdown("##### Detections by Camera")
                st.bar_chart(df_results["camera_id"].value_counts().head(10))

        # --- PDF and Email Section ---
        st.markdown("---")
        st.subheader("Generate & Send Report")
        
        # Define a stable path for the PDF in a temporary directory
        temp_dir = tempfile.gettempdir()
        pdf_filename = "detection_report.pdf"
        pdf_file_path = os.path.join(temp_dir, pdf_filename)
        
        # Generate the PDF from the dataframe stored in session state
        make_pdf(df_results, filename=pdf_file_path)

        # Create the download button
        with open(pdf_file_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f.read(),
                file_name="detection_report.pdf",
                mime="application/pdf",
            )
        
        # Email functionality
        with st.form("email_form"):
            email_recipient = st.text_input("Recipient Email Address")
            submitted = st.form_submit_button("📧 Email Report")
            if submitted:
                if email_recipient:
                    with st.spinner("Sending email..."):
                        if send_email(email_recipient, "Detection Report", "Please find the attached detection report.", pdf_file_path):
                            st.success(f"✅ Report successfully emailed to {email_recipient}.")
                        # Error messages are handled inside the send_email function
                else:
                    st.warning("⚠️ Please enter an email address.")
