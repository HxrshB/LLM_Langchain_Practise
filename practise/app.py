import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
st.title("AI Dietitian 🥗")
st.write("Get AI-generated health insights and diet recommendations from blood_work.txt.")

# Sidebar Configuration
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox(
    "Choose an AI model",
    ["OpenAI GPT-4o", "Google Gemini 3 Pro", "Anthropic Claude 4.5"]
)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.0, 1.0, 0.6)
st.sidebar.markdown("---")
st.sidebar.info("This app uses AI to analyze blood reports and suggest diet plans.")

# Input Section
st.header("1. Blood Report Analysis")
st.info("Fetching data from blood_work.txt (as in the Jupyter Notebook)...")

# Processing Logic
if st.button("Run AI Dietitian Analysis"):
    try:
        with open("blood_work.txt", "r") as f:
            raw_text = f.read()
        st.success("Read blood_work.txt successfully!")
    except FileNotFoundError:
        st.error("Error: blood_work.txt not found in the current directory.")
        st.stop()

    st.markdown("---")

    with st.spinner(f"Analyzing blood report using {model_choice}..."):
        # Initialize LLM
        # Using gemma-4-31b-it as configured in the notebook
        llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=temperature)

        # --- STAGE 1: Extract Values ---
        extraction_prompt = f"""
        You are a medical data extraction assistant.

        From the blood work report provided below, extract each value and classify each one as Low, Normal and High 
        based on the reference ranges provided in the report.

        Response format:
        Test name: Value | Status: High / Low / Normal | Reference: Range 

        Blood Report:
        {raw_text}
        """
        
        extraction_response = llm.invoke(extraction_prompt)
        # Using .content if available (standard LangChain), otherwise fallback to .text (as in notebook)
        extraction_text = extraction_response.text

        # --- STAGE 2: Health Summary ---
        diet_prompt = f"""
        You are a clinical nutritionist expert in Indian dietary habits.

        Based on the blood work report provided below, write:
        1. A short summary in 2-3 lines explaining the patient's condition in simple language.
        2. A short practical Indian dietary plan with 2 sections: a) Foods to avoid b) Foods to consume more

        Do not include any other sections in the diet plan.

        Blood Work Report:
        {extraction_text}
        """

        diet_response = llm.invoke(diet_prompt)
        diet_text = diet_response.text

    st.success("Analysis Complete!")

    st.markdown("---")

    # Display Results
    st.header("2. AI Analysis")
    with st.expander("View Extracted Values", expanded=True):
        st.text(extraction_text)

    with st.expander("View Health Summary & Diet Plan", expanded=True):
        st.markdown(diet_text)

else:
    st.info("Click the button above to start the analysis.")