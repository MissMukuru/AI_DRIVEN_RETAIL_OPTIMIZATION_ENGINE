import streamlit as st
import pandas as pd
import os
from pathlib import Path

st.set_page_config(page_title="Retail Demand Pulse", page_icon="🛍️", layout="wide")

# Pink Theme
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #ffe4f3 0%, #f0d6ff 100%); }
    .main-header { font-size: 3rem; color: #d63384; text-align: center; }
    .stButton>button { background: #ff69b4; color: white; border-radius: 12px; padding: 12px 24px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🛍️ Retail Demand Pulse</h1>", unsafe_allow_html=True)
st.markdown("**AI-Powered Retail Replenishment Assistant**")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/pastel/512/shopping-cart.png", width=130)
    page = st.radio("Go to", ["🏠 Home", "🔮 Predict Replenishment", "📤 Upload Dataset", "📊 Reports"])

# ====================== HOME ======================
if page == "🏠 Home":
    st.success("🌸 Welcome! Your smart retail assistant is ready.")

# ====================== PREDICT ======================
elif page == "🔮 Predict Replenishment":
    st.subheader("🔮 Predict Today's Replenishment")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            product_name = st.text_input("Product Name", "Washing Powder (500g)")
            category = st.selectbox("Category", 
                ["Household", "Food & Beverages", "Personal Care", "Dairy", "Cleaning", "Snacks"])
            is_perishable = st.radio("Perishable?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            current_stock = st.number_input("Current Stock", min_value=0, value=225)
        with col2:
            avg_daily = st.number_input("Average Daily Sales", min_value=0.1, value=10.03, step=0.01)
            forecasted = st.number_input("Expected Sales Tomorrow", min_value=0.1, value=17.0, step=0.01)
        
        submitted = st.form_submit_button("🚀 Get Recommendation", type="primary")
        
        if submitted:
            with st.spinner("Calculating..."):
                try:
                    import requests
                    payload = {
                        "product_name": product_name,
                        "category": category,
                        "is_perishable": is_perishable,
                        "current_stock": current_stock,
                        "avg_daily": avg_daily,
                        "forecasted_demand_per_day": forecasted
                    }
                    response = requests.post("http://localhost:8000/predict/replenishment", data=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ Recommendation Ready!")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Stock Status", data.get("stock_status", "OK"))
                        with c2: st.metric("Days Remaining", f"{data.get('days_of_stock_remaining', 0):.1f}")
                        with c3: st.metric("Priority", data.get("replenishment_priority", "🟢 OK"))
                        st.metric("Recommended Order Qty", f"{data.get('recommended_order_qty', 0)} units")
                        st.metric("Estimated Cost", f"KES {data.get('estimated_order_cost_kes', 0):,.0f}")
                        with st.expander("Full Details"): st.json(data)
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Backend not reachable. Make sure FastAPI is running.\nError: {e}")

# ====================== UPLOAD ======================
elif page == "📤 Upload Dataset":
    st.subheader("📤 Upload Sales Dataset")
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file and st.button("Process Dataset & Generate Report", type="primary"):
        with st.spinner("Processing..."):
            try:
                import requests
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post("http://localhost:8000/upload-and-process", files=files)
                if response.status_code == 200:
                    st.success("✅ Processing Complete! Check Reports tab.")
                    st.json(response.json())
                else:
                    st.error(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

# ====================== REPORTS DASHBOARD ======================
elif page == "📊 Reports":
    st.subheader("📊 Replenishment Dashboard")
    
    # Direct file path from your logs
    report_path = r"C:\Users\USER\Desktop\AI_DRIVEN_OPTIMIZATION_ENGINE\retail_demand_pulse_project\data\processed\replenishment_report.csv"
    
    if os.path.exists(report_path):
        try:
            df = pd.read_csv(report_path)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Products", len(df))
            with col2: st.metric("🔴 Urgent", len(df[df['replenishment_priority'].astype(str).str.contains("URGENT", na=False)]))
            with col3: st.metric("Low Stock", len(df[df['stock_status'].astype(str).str.contains("LOW|CRITICALLY", na=False)]))
            with col4: st.metric("Total Cost", f"KES {df['estimated_order_cost_kes'].sum():,.0f}")

            # Filters
            st.markdown("### Filters")
            colf1, colf2 = st.columns(2)
            with colf1:
                status_filter = st.multiselect("Stock Status", options=df['stock_status'].unique().tolist(), default=df['stock_status'].unique().tolist())
            with colf2:
                priority_filter = st.multiselect("Priority", options=df['replenishment_priority'].unique().tolist(), default=df['replenishment_priority'].unique().tolist())

            filtered_df = df[
                df['stock_status'].isin(status_filter) & 
                df['replenishment_priority'].isin(priority_filter)
            ]

            # Charts
            st.markdown("### Charts")
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(df['replenishment_priority'].value_counts())
            with c2:
                if not filtered_df.empty:
                    top = filtered_df.nlargest(10, 'recommended_order_qty')
                    st.bar_chart(top.set_index('product_name')['recommended_order_qty'])

            # Table
            st.markdown("### Recommendations Table")
            st.dataframe(filtered_df, use_container_width=True)

            # Download
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Report as CSV",
                csv,
                "replenishment_report.csv",
                "text/csv",
                type="primary"
            )
            
        except Exception as e:
            st.error(f"Error reading report: {e}")
    else:
        st.warning("No report found yet.")
        st.info("1. Go to **Upload Dataset** tab\n2. Upload your Excel/CSV file\n3. Come back here")

st.caption("💖 Retail Demand Pulse | Soft Pink Edition")