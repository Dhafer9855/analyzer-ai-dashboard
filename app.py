import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# --- UI Setup ---
st.set_page_config(layout="wide")

st.markdown("""
    <style>
    input[type='password'] {
        autocomplete: off !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load config ---
def load_config():
    if os.path.exists("config.json"):
        with open("config.json") as f:
            return json.load(f)
    return {}

config = load_config()

# --- Sidebar ---
st.sidebar.header("🤖 Analysis Settings")
llm_provider = st.sidebar.selectbox("Choose Analysis Provider", ["local", "openai", "groq"])

available_models = {
    "local": ["None"],
    "openai": ["gpt-4", "gpt-3.5-turbo"],
    "groq": ["llama3-70b-8192"]
}

use_local_llm = False
local_model_name = "llama3"
if llm_provider == "local":
    use_local_llm = st.sidebar.checkbox("✅ Use Local LLM", value=False)
    if use_local_llm:
        local_model_name = st.sidebar.text_input(
            "Local model name", 
            value="llama3", 
            help="Enter the name of your local Ollama model"
        )
        st.sidebar.info("💡 **Recommendation:** We suggest using Llama3 for best results.")
        st.sidebar.markdown("📖 [Full instructions to download free LLM (Llama3)](https://ollama.com/library/llama3)")
        st.sidebar.markdown("*You can use any local LLM through Ollama: Llama3, DeepSeek, Qwen, Mistral, CodeLlama, etc.*")
        st.sidebar.markdown("⚡ **Note:** Ollama must be running on port 11434 (default port).")
        
        # Test Connection Button
        if st.sidebar.button("🔗 Test Connection", help="Test if Ollama is running and model is available"):
            with st.sidebar:
                with st.spinner("Testing connection..."):
                    try:
                        # Test Ollama connection
                        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
                        if test_response.status_code == 200:
                            ollama_models = test_response.json().get("models", [])
                            model_names = [model.get("name", "").split(":")[0] for model in ollama_models]
                            
                            if local_model_name in model_names or any(local_model_name in name for name in model_names):
                                st.success(f"✅ Connected! Model '{local_model_name}' is available.")
                            else:
                                st.warning(f"⚠️ Ollama connected, but model '{local_model_name}' not found.")
                                if model_names:
                                    st.info(f"📋 Available models: {', '.join(model_names[:5])}")
                                st.info(f"💡 To install: `ollama pull {local_model_name}`")
                        else:
                            st.error("❌ Ollama is running but API not responding properly.")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to Ollama. Make sure it's running on port 11434.")
                        st.info("💡 Start Ollama with: `ollama serve`")
                    except requests.exceptions.Timeout:
                        st.error("❌ Connection timeout. Ollama might be slow to respond.")
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {str(e)}")
                        st.info("💡 Check if Ollama is installed and running.")

model_options = available_models.get(llm_provider, ["unknown"])
if llm_provider != "local" or not use_local_llm:
    model_name = st.sidebar.selectbox("Choose model name", model_options)
else:
    model_name = local_model_name

api_key = None
if llm_provider in ["openai", "groq"]:
    api_key = st.sidebar.text_input("Enter API key", type="password", key="api_key_input")
    if llm_provider == "openai":
        st.sidebar.markdown("🔑 [Get OpenAI API Key - Sign up here](https://platform.openai.com/api-keys)")
    elif llm_provider == "groq":
        st.sidebar.markdown("🔑 [Get Groq API Key - Sign up here](https://console.groq.com/keys)")

# --- Upload dataset ---
st.title("📊 Analyzer AI Dashboard")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.reset_index(drop=True, inplace=True)  # Ensure clean index

    # ✅ ENHANCED FIX: Convert all columns to Arrow-safe native types
    for col in df.columns:
        try:
            if df[col].dtype.name in ['Int64', 'Int32', 'Int16', 'Int8']:
                # Handle nullable integer types
                df[col] = df[col].fillna(0).astype('int64')
            elif pd.api.types.is_integer_dtype(df[col]):
                df[col] = df[col].fillna(0).astype('int64')
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].fillna(0.0).astype('float64')
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(bool)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype(str)
        except Exception as e:
            st.warning(f"⚠️ Column '{col}' fallback to string due to: {e}")
            df[col] = df[col].astype(str)

    st.dataframe(df.head(), use_container_width=True)
    st.success("✅ File uploaded and previewed successfully.")

    label_column = st.selectbox("Select the label (target) column", df.columns, index=len(df.columns) - 1, key="label_column")
    
    # Get unique values count for the selected label column
    unique_values = df[label_column].nunique()
    unique_vals_list = df[label_column].unique().tolist()
    
    # Determine the type of target variable and handle accordingly
    if pd.api.types.is_numeric_dtype(df[label_column]):
        # Case 1: Numeric target - Always use Maximize/Minimize regardless of unique values
        goal = st.radio("What kind of outcome do you want to optimize?", ["Maximize", "Minimize"], key="goal")
        desired_value = "high" if goal == "Maximize" else "low"
        
    else:
        # Case 2 & 3: Non-numeric (categorical) targets
        if unique_values == 1:
            st.error("❌ This label column has only 1 unique value. Please select a different label column.")
            st.stop()
        elif unique_values == 2:
            # Case 2: Binary target (2 choices) - use radio buttons
            st.info(f"📊 Binary target detected with values: {unique_vals_list}")
            desired_value = st.radio("Which value do you want more often?", unique_vals_list, key="binary_target")
        elif unique_values <= 10:
            # Case 3: Multi-class target (3-10 choices) - use dropdown
            st.info(f"🎯 Multi-class target detected with {unique_values} values: {unique_vals_list}")
            desired_value = st.selectbox("Which value do you want to occur more often?", unique_vals_list, key="multiclass_target")
        else:
            # Too many categorical values
            st.error(f"❌ This categorical column has {unique_values} unique values, which exceeds the maximum allowed (10). Please choose another label column.")
            st.stop()

    if st.button("🚀 Start Analyzing"):
        st.subheader("🧠 Auto Summary")
        st.markdown(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write(df.dtypes)
        st.markdown("**Missing Values:**")
        st.write(df.isnull().sum())
        
        # Show target distribution
        st.subheader("🎯 Target Variable Distribution")
        target_counts = df[label_column].value_counts()
        target_percentages = df[label_column].value_counts(normalize=True) * 100
        unique_count = len(target_counts)
        
        # Handle different cases based on number of unique values
        if unique_count <= 10:
            # Case 1: Few categories - show combined table and chart
            st.markdown("**Distribution:**")
            
            # Create combined dataframe with counts and percentages
            distribution_df = pd.DataFrame({
                'Value': target_counts.index,
                'Count': target_counts.values,
                'Percentage': [f"{pct:.1f}%" for pct in target_percentages.values]
            })
            
            # Display the combined table
            st.dataframe(distribution_df, use_container_width=True, hide_index=True)
            
            # Show bar chart
            fig, ax = plt.subplots(figsize=(max(8, unique_count*0.8), 5))
            target_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {label_column}')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        elif unique_count <= 50:
            # Case 2: Medium number - show summary stats and histogram
            st.info(f"📊 {unique_count} unique values detected. Showing summary statistics and histogram.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Summary Statistics:**")
                if pd.api.types.is_numeric_dtype(df[label_column]):
                    st.write(f"**Count:** {len(df[label_column])}")
                    st.write(f"**Mean:** {df[label_column].mean():.2f}")
                    st.write(f"**Median:** {df[label_column].median():.2f}")
                    st.write(f"**Min:** {df[label_column].min():.2f}")
                    st.write(f"**Max:** {df[label_column].max():.2f}")
                    st.write(f"**Std:** {df[label_column].std():.2f}")
                else:
                    st.write(f"**Total unique values:** {unique_count}")
                    st.write(f"**Most common:** {target_counts.index[0]} ({target_counts.iloc[0]} times)")
            
            with col2:
                st.markdown("**Top 6 Values:**")
                top_6 = target_counts.head(6)
                for val, count in top_6.items():
                    pct = (count / len(df)) * 100
                    st.write(f"{val}: {count} ({pct:.1f}%)")
            
            # Show histogram for numeric data
            if pd.api.types.is_numeric_dtype(df[label_column]):
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df[label_column], bins=min(30, unique_count), color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {label_column}')
                ax.set_xlabel(label_column)
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
            
        else:
            # Case 3: Too many values - summary only
            st.warning(f"⚠️ {unique_count} unique values detected. Showing summary statistics only.")
            
            if pd.api.types.is_numeric_dtype(df[label_column]):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Statistical Summary:**")
                    st.write(f"**Count:** {len(df[label_column])}")
                    st.write(f"**Mean:** {df[label_column].mean():.2f}")
                    st.write(f"**Median:** {df[label_column].median():.2f}")
                    st.write(f"**Min:** {df[label_column].min():.2f}")
                    st.write(f"**Max:** {df[label_column].max():.2f}")
                    st.write(f"**Std Dev:** {df[label_column].std():.2f}")
                
                with col2:
                    st.markdown("**Distribution Info:**")
                    q25 = df[label_column].quantile(0.25)
                    q75 = df[label_column].quantile(0.75)
                    st.write(f"**25th percentile:** {q25:.2f}")
                    st.write(f"**75th percentile:** {q75:.2f}")
                    st.write(f"**Unique values:** {unique_count}")
                    
                    # Show optimization info
                    if 'desired_value' in locals():
                        if desired_value == "high":
                            st.success(f"🎯 **Goal:** Maximize values (currently {df[label_column].max():.2f})")
                        else:
                            st.success(f"🎯 **Goal:** Minimize values (currently {df[label_column].min():.2f})")
                
                # Show histogram with more bins for very distributed data
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.hist(df[label_column], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {label_column}')
                ax.set_xlabel(label_column)
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write(f"**Total unique categories:** {unique_count}")
                st.write("**Top 10 most frequent values:**")
                top_10 = target_counts.head(10)
                for val, count in top_10.items():
                    pct = (count / len(df)) * 100
                    st.write(f"• {val}: {count} times ({pct:.1f}%)")

        st.subheader("📈 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

        if pd.api.types.is_numeric_dtype(df[label_column]):
            st.subheader("📊 Feature Correlation with Target")
            correlations = df.corr(numeric_only=True)[label_column].drop(label_column)
            corr_df = correlations.sort_values(ascending=False).reset_index()
            corr_df.columns = ["Feature", "Correlation"]
            st.dataframe(corr_df, use_container_width=True)

        # --- Rule-Based Recommendations (No LLM) ---
        if llm_provider == "local" and not use_local_llm:
            st.subheader("📌 Rule-Based Data Insights & Decisions")

            if pd.api.types.is_numeric_dtype(df[label_column]):
                st.markdown("**📊 Feature Impact Suggestions:**")
                for col in df.select_dtypes(include=np.number).columns:
                    if col == label_column:
                        continue
                    corr = df[[col, label_column]].corr().iloc[0, 1]
                    if corr > 0.7:
                        st.success(f"✅ Feature **{col}** has strong **positive** correlation with the target. Consider increasing it.")
                    elif corr < -0.7:
                        st.warning(f"⚠️ Feature **{col}** has strong **negative** correlation with the target. Consider reducing or investigating it.")

            st.markdown("**🧼 Data Cleaning Recommendations:**")
            nulls = df.isnull().mean()
            for col, pct in nulls.items():
                if pct > 0.3:
                    st.error(f"Column **{col}** has **{int(pct*100)}% missing** values. Consider dropping or imputing.")

            # Updated imbalance detection for all target types
            if unique_values <= 10:  # For categorical targets
                counts = df[label_column].value_counts(normalize=True)
                majority_class = counts.idxmax()
                if counts.max() > 0.75:
                    st.warning(f"⚠️ Target is imbalanced. **{majority_class}** makes up **{int(counts.max()*100)}%** of values.")
                
                # Show information about desired class
                if not pd.api.types.is_numeric_dtype(df[label_column]):
                    desired_pct = counts.get(desired_value, 0) * 100
                    st.info(f"🎯 Your desired outcome **'{desired_value}'** currently occurs **{desired_pct:.1f}%** of the time.")

            st.markdown("**✅ Rule-Based Recommendation Summary:**")
            recommendations = []
            if pd.api.types.is_numeric_dtype(df[label_column]):
                recommendations.append("- Analyze high-impact features")
            else:
                recommendations.append(f"- Focus on increasing '{desired_value}' occurrences")
            recommendations.extend([
                "- Clean or drop high-null columns",
                "- Handle class imbalance before modeling",
                "- Consider feature engineering for better predictions"
            ])
            st.markdown("\n".join(recommendations))

        # --- Local LLM (Ollama) ---
        elif llm_provider == "local" and use_local_llm:
            # Updated prompt for multi-class support
            target_info = f"Target column: {label_column}"
            if pd.api.types.is_numeric_dtype(df[label_column]):
                goal_info = f"Desired outcome: {'maximize' if desired_value == 'high' else 'minimize'}"
            else:
                goal_info = f"Desired outcome: Increase occurrences of '{desired_value}'"
                
            prompt = f"""
            Based on the following dataset summary, provide a possible solution, a data-driven recommendation, and a data-driven decision.

            Column types:
            {df.dtypes.to_string()}

            Summary statistics:
            {df.describe().to_string()}

            {target_info}
            {goal_info}
            Target distribution: {df[label_column].value_counts().to_dict()}
            """
            st.subheader("💬 AI Response (Local LLM)")
            
            # Show spinner while processing
            with st.spinner("🔄 Connecting to local LLM via Ollama... This may take 1-2 minutes"):
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": model_name, "prompt": prompt, "stream": False}
                    )
                    output = response.json().get("response", "").strip()
                    if output:
                        st.markdown(output)
                    else:
                        st.error("❌ No response from local model.")
                except Exception as e:
                    st.error(f"Error calling local LLM: {e}")

        # --- Online (Groq / OpenAI) ---
        elif llm_provider in ["openai", "groq"]:
            # Updated prompt for multi-class support
            target_info = f"Target column is: {label_column}"
            if pd.api.types.is_numeric_dtype(df[label_column]):
                goal_info = f"Goal is to {'maximize' if desired_value == 'high' else 'minimize'} it."
            else:
                goal_info = f"Goal is to increase the frequency of '{desired_value}' occurrences."
                
            prompt = f"""
            Here is a summary of the dataset:

            Variables:
            {df.dtypes.to_string()}

            Summary Statistics:
            {df.describe().to_string()}

            {target_info}
            {goal_info}
            Current target distribution: {df[label_column].value_counts().to_dict()}
            
            Provide a possible solution, a data-driven recommendation, and a data-driven decision.
            """
            st.subheader("💬 AI Response")
            st.info(f"LLM provider: {llm_provider.upper()}")

            # Add spinner for online providers
            spinner_text = f"🔄 Connecting to {llm_provider.upper()} API... Please wait"
            with st.spinner(spinner_text):
                try:
                    if llm_provider == "openai":
                        import openai
                        openai.api_key = api_key or config.get("openai_api_key", "")
                        response = openai.ChatCompletion.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.3,
                        )
                        st.markdown(response.choices[0].message.content)

                    elif llm_provider == "groq":
                        headers = {"Authorization": f"Bearer {api_key or config.get('groq_api_key', '')}"}
                        response = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.3,
                            }
                        )
                        data = response.json()
                        if "choices" in data and data["choices"]:
                            st.markdown(data["choices"][0]["message"]["content"])
                        else:
                            st.error("❌ Unexpected response from GROQ.")
                except Exception as e:
                    st.error(f"LLM API error: {e}")