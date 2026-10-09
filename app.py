import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px



st.set_page_config(
    page_title="DataGuard AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 DataGuard AI")
st.subheader("Smart Data Analysis Platform")

st.write(
    "Upload your dataset and discover meaningful insights."
)

file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.success("Dataset uploaded successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Data Summary")
    st.dataframe(df.describe(include="all"))

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    
    st.subheader("📊 Interactive Data Visualization")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
    )

    columns = df.columns.tolist()

    x_col = st.selectbox("Select X-axis", columns)

    y_col = None

    if chart_type != "Histogram":
        y_col = st.selectbox(
            "Select Y-axis",
            columns,
            index=min(1, len(columns) - 1)
        )

    if st.button("Generate Chart"):
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col)

        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col)

        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col)

        else:
            fig = px.histogram(df, x=x_col)

        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("🧹 Smart Data Cleaning")

    st.write("Missing values in each column")
    st.dataframe(df.isnull().sum().rename("Missing Values"))

    cleaning_option = st.selectbox(
        "Choose Cleaning Action",
        [
            "Fill Missing Values",
            "Remove Rows with Missing Values",
            "Remove Duplicate Rows"
        ]
    )

    if cleaning_option == "Fill Missing Values":
        fill_method = st.selectbox(
            "Choose Fill Method",
            ["Mean", "Median", "Mode", "Fill with 0"]
        )

        if st.button("Clean Missing Values"):
            cleaned_df = df.copy()

            for col in cleaned_df.columns:
                if cleaned_df[col].isnull().any():
                    if fill_method == "Mean" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(
                            cleaned_df[col].mean()
                        )
                    elif fill_method == "Median" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(
                            cleaned_df[col].median()
                        )
                    elif fill_method == "Mode":
                        modes = cleaned_df[col].mode()
                        if not modes.empty:
                            cleaned_df[col] = cleaned_df[col].fillna(
                                modes.iloc[0]
                            )
                    elif fill_method == "Fill with 0":
                        cleaned_df[col] = cleaned_df[col].fillna(0)

            st.session_state["cleaned_df"] = cleaned_df
            st.success("Missing values processed!")

    elif cleaning_option == "Remove Rows with Missing Values":
        if st.button("Remove Missing Rows"):
            cleaned_df = df.dropna().copy()
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Rows with missing values removed!")

    else:
        if st.button("Remove Duplicate Rows"):
            cleaned_df = df.drop_duplicates().copy()
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Duplicate rows removed!")

    if "cleaned_df" in st.session_state:
        cleaned_df = st.session_state["cleaned_df"]

        st.write("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(10))

        st.write("Rows remaining:", len(cleaned_df))

        st.download_button(
            label="Download Cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv"
        )
        
    from google import genai

    st.subheader("🤖 AI Data Assistant")

    question = st.text_input(
        "Ask a question about your dataset",
        placeholder="Example: What are the main insights?"
    )

    if st.button("Ask AI"):
        if question.strip():
            try:
                client = genai.Client(
                api_key = st.secrets["GEMINI_API_KEY"]                )

                # Send a small data summary, not the entire dataset
                summary = df.head(10).to_string(index=False)
                stats = df.describe(include="all").to_string()

                prompt = f"""
                You are a data analysis assistant.
                Answer using the dataset information below.
                Clearly mention when the information is insufficient.

                Dataset columns: {list(df.columns)}
                First 10 rows:
                {summary}

                Summary statistics:
                {stats}

                User question: {question}
                """

                with st.spinner("Analyzing your question..."):
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )

                st.markdown("### 💡 AI Answer")
                st.write(response.text)

            except Exception as e:
                st.error(f"Error: {e}")