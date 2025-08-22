import os
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from dotenv import load_dotenv
import google.generativeai as genai
from fpdf import FPDF  
 
def setup(key: str):
    if "STREAMLIT_RUNTIME" in os.environ:
        try:
            return st.secrets[key]
        except:
            st.error("GOOGLE_API_KEY not found in api.env")
            st.stop()
    else:
        try:
            load_dotenv("app.env")
            google_api_key = os.getenv(key)
            return google_api_key
        except:
            st.error("GOOGLE_API_KEY not found in api.env")
            st.stop()
api_key = setup("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file. Please add it before running.")
    st.stop()
genai.configure(api_key=api_key)
 
st.set_page_config(page_title="📊 Demand Forecasting & AI Insights", layout="wide")
st.title("📊 Demand Forecasting Dashboard with AI Insights")
 
@st.cache_data(show_spinner=False)
def fit_arima(train_series, order):
    model = ARIMA(train_series, order=order)
    return model.fit()
 
def highlight_forecast_row(row):
    if pd.isna(row.get("Actual Sales")):
        return ["background-color: lightgreen"] * len(row)
    else:
        return [""] * len(row)
 
uploaded_file = st.file_uploader("📂 Upload your sales dataset (CSV)", type=["csv"])
 
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
 
        date_col = None
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
        if date_col is None:
            st.error("❌ No date column found. Please ensure dataset has a date column.")
            st.stop()
 
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
 
        df = df.groupby(df.index).sum(numeric_only=True)
 
        sales_col = None
        for col in df.columns:
            if "sale" in col.lower() or "revenue" in col.lower() or "price" in col.lower():
                sales_col = col
                break
        if sales_col is None:
            st.error("❌ No sales column found. Please ensure dataset has 'sales', 'price' or 'revenue' column.")
            st.stop()
 
        df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
        df = df.dropna(subset=[sales_col])
        if df.empty:
            st.error("❌ After cleaning, no numeric rows remain in the sales column.")
            st.stop()
 
        df['sales_log'] = np.log(df[sales_col].clip(lower=1))
        df['sales_smooth'] = df['sales_log'].rolling(window=3, min_periods=1).mean()
 
        st.subheader("📈 Forecasting - ARIMA (Actual vs Forecast Table)")
        forecast_steps = st.number_input("Enter forecast horizon (days)", min_value=1, max_value=365, value=30, step=1)
        forecast_steps = int(forecast_steps)
 
        arima_order = (5, 1, 0)
        model_fit = fit_arima(df['sales_smooth'], arima_order)
 
        forecast_log = model_fit.forecast(steps=forecast_steps)
        forecast = np.exp(forecast_log).astype(float)
 
        forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                       periods=forecast_steps, freq="D")
        forecast_df = pd.DataFrame({"Forecasted Sales": forecast}, index=forecast_index)
 
        actual_df = df[[sales_col]].rename(columns={sales_col: "Actual Sales"})
        full_df = pd.concat([actual_df, forecast_df], axis=1)
 
        st.markdown("### 📈 Chart: Actual (history) and Forecast (future)")
        st.line_chart(full_df)
 
        st.markdown("### 📌 Actual vs Forecast Table")
        styled_display = full_df.style.format("{:,.2f}").apply(highlight_forecast_row, axis=1)
        st.dataframe(styled_display, use_container_width=True)
 
        csv_bytes = full_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Actual vs Forecast CSV", data=csv_bytes, file_name="actual_vs_forecast.csv", mime="text/csv")
 
        st.markdown("### 🔮 Daily Forecasted Sales (Next Horizon)")
        cols = st.columns(min(5, forecast_steps))
        for i, (d, val) in enumerate(forecast_df.itertuples(index=True, name=None), start=1):
            with cols[(i-1) % len(cols)]:
                st.metric(label=f"Day {i} ({pd.to_datetime(d).date()})", value=f"{val:,.2f}")
 
        last_actual = df[sales_col].iloc[-1]
        last_forecast_value = forecast_df["Forecasted Sales"].iloc[-1]
        st.success(f"✅ Forecasted Sales after {forecast_steps} days: {last_forecast_value:,.2f}")
 
        st.subheader("💡 Generated Business Insights")
        growth = (last_forecast_value - last_actual) / max(last_actual, 1e-9) * 100
 
        prompt = f"""
        The last observed sales is {last_actual:.2f}, forecasted sales after {forecast_steps} days is {last_forecast_value:.2f}.
        That's a {growth:.2f}% change.
        Generate only a short paragraph summarizing the trend (no bullet points).
        """
 
        model_ai = genai.GenerativeModel("gemini-1.5-flash")
        response = model_ai.generate_content(prompt)
        insight_paragraph = response.text.strip()
 
        st.markdown("### 📝 Paragraph Insights")
        st.write(insight_paragraph)
        st.subheader("📑 Detailed Report")
        if st.button("📥 Generate & Download Report"):
            report_filename = "forecast_report.pdf"
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
 
            pdf.cell(200, 10, "Demand Forecasting Report", ln=True, align="C")
            pdf.ln(8)
            pdf.cell(200, 8, f"Last actual sales: {last_actual:.2f}", ln=True)
            pdf.cell(200, 8, f"Forecasted sales after {forecast_steps} days: {last_forecast_value:,.2f}", ln=True)
            pdf.cell(200, 8, f"Growth: {growth:.2f}%", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 8, "AI Insight (paragraph):", ln=True)
            pdf.set_font("Arial", size=12)
            safe_text = insight_paragraph.encode("latin-1", "replace").decode("latin-1")
            pdf.ln(4)
            pdf.multi_cell(0, 8, safe_text)
            pdf.ln(6)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 8, "Actual vs Forecast (sample rows):", ln=True)
            pdf.set_font("Arial", size=10)
            sample_table = full_df.tail(10).reset_index()
            for _, row in sample_table.iterrows():
                date_str = pd.to_datetime(row["index"]).strftime("%Y-%m-%d")
                actual_str = f"{row['Actual Sales']:,.2f}" if pd.notna(row["Actual Sales"]) else "-"
                forecast_str = f"{row['Forecasted Sales']:,.2f}" if pd.notna(row["Forecasted Sales"]) else "-"
                pdf.multi_cell(0, 6, f"{date_str} | Actual: {actual_str} | Forecast: {forecast_str}")
 
            pdf.output(report_filename)
 
            with open(report_filename, "rb") as f:
                st.download_button("⬇️ Download Report (PDF)", f, file_name=report_filename, mime="application/pdf")
 
    except Exception as e:
        st.error(f"❌ Error: {e}")