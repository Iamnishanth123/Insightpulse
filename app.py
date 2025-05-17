import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# --- Set up Gemini ---
genai.configure(api_key="Api_key")  # Replace with your real key

st.set_page_config(page_title="InsightPulse 📊", layout="wide")
st.title("📊 InsightPulse - GenAI Business Data Analyzer")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Data Preview ---
    st.subheader("📌 Preview of Your Data")
    st.dataframe(df.head())

    # --- Data Summary ---
    st.subheader("📊 Data Summary")
    st.write(df.describe())

    # --- Charts ---
    st.subheader("📈 Visualizations")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols[:3]:
        st.markdown(f"**Distribution of {col}**")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    for col in cat_cols[:2]:
        st.markdown(f"**Top Categories in {col}**")
        top_categories = df[col].value_counts().nlargest(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_categories.index, y=top_categories.values, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if len(numeric_cols) >= 2:
        st.markdown("**📌 Correlation Heatmap**")
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- GenAI Summary ---
    st.subheader("🧠 GenAI Summary Insights")

    with st.spinner("Generating insights using Gemini..."):
        sample_data = df.head(10).to_csv(index=False)
        stats_summary = df.describe().to_string()

        summary_prompt = f"""
You are a business analyst AI.

Here is a sample of uploaded business data:
{sample_data}

And here are some numerical summaries:
{stats_summary}

Please provide 5 concise and insightful observations about this data. Use simple business language.
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(summary_prompt)
        st.markdown("### 🧠 Gemini's Insight Summary:")
        st.markdown(response.text)

    # --- Chat with Your Data ---
    st.subheader("💬 Chat with Your Data")

    # Session State Initialization
    if "chat_ended" not in st.session_state:
        st.session_state.chat_ended = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Exit Chat Button
    if not st.session_state.chat_ended:
        if st.button("🛑 Exit Chat"):
            st.session_state.chat_ended = True

    # Question Input Form (if chat not ended)
    if not st.session_state.chat_ended:
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about your data:")
            submit_button = st.form_submit_button("Send")

        if submit_button and user_question:
            chat_prompt = f"""
You are a smart business analyst AI.

Here is a sample of the business data in CSV:
{df.head(15).to_csv(index=False)}

User asked:
{user_question}

Answer clearly using this data.
"""
            chat_response = model.generate_content(chat_prompt)
            answer = chat_response.text
            st.session_state.chat_history.append((user_question, answer))

    # Chat History
    if st.session_state.chat_history:
        st.markdown("### 🤝 Chat History:")
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"**🧑 You:** {q}")
            st.markdown(f"**🤖 Gemini:** {a}")

    # Exit Message
    if st.session_state.chat_ended:
        st.success("🔚 Chat ended. Refresh the page to start a new session.")


    # --- Generate PDF Report ---
    def generate_pdf(summary, chat_history):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        y = height - 40

        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "📊 InsightPulse Report")
        y -= 30

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "🧠 Gemini Summary:")
        y -= 20

        c.setFont("Helvetica", 10)
        for line in summary.split("\n"):
            c.drawString(50, y, line.strip())
            y -= 15
            if y < 50:
                c.showPage()
                y = height - 40

        if chat_history:
            y -= 20
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "💬 Chat History:")
            y -= 20

            c.setFont("Helvetica", 10)
            for q, a in chat_history:
                c.drawString(50, y, f"User: {q}")
                y -= 15
                c.drawString(50, y, f"Gemini: {a}")
                y -= 25
                if y < 50:
                    c.showPage()
                    y = height - 40

        c.save()
        buffer.seek(0)
        return buffer

# ✅ PLACE THIS HERE (OUTSIDE THE FUNCTION)
if st.session_state.chat_ended and response and st.session_state.chat_history:
    pdf = generate_pdf(response.text, st.session_state.chat_history)
    st.download_button(
        label="📥 Download Summary & Chat as PDF",
        data=pdf,
        file_name="insightpulse_report.pdf",
        mime="application/pdf"
    )

