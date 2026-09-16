"""Streamlit owner cockpit for model-driven retail replenishment."""

from pathlib import Path
import os
from datetime import datetime
from zoneinfo import ZoneInfo
        st.json({key: value for key, value in row.to_dict().items() if pd.notna(value)})


def _report_view(report: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Operations</div><div class="hero"><h1>Replenishment queue</h1><p>Filter, review, and export the current order recommendation.</p></div>', unsafe_allow_html=True)
    if report.empty:
        st.warning("No report found.")
        return
    status_options = sorted(report["stock_status"].dropna().unique())
    priority_options = sorted(report["replenishment_priority"].dropna().astype(int).unique())
    c1, c2, c3 = st.columns(3)
    statuses = c1.multiselect("Stock status", status_options, default=status_options)
    priorities = c2.multiselect("Priority", priority_options, default=priority_options, format_func=_priority_label)
    query = c3.text_input("Find product")
    filtered = report[report["stock_status"].isin(statuses) & report["replenishment_priority"].astype(int).isin(priorities)]
    if query:
        filtered = filtered[filtered["product_name"].str.contains(query, case=False, na=False)]
    st.dataframe(filtered.sort_values(["replenishment_priority", "days_of_stock_remaining"]), hide_index=True, use_container_width=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "replenishment_report.csv", "text/csv")


def _upload_view() -> None:
    st.markdown('<div class="eyebrow">Data refresh</div><div class="hero"><h1>Refresh the forecast</h1><p>Upload sales history to clean, engineer features, retrain models, and generate a new recommendation.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Sales history", type=["csv", "xlsx", "xls"])
    train = st.checkbox("Retrain ML models", value=True)
    generate = st.checkbox("Generate replenishment report", value=True)
    if uploaded and st.button("Run forecast pipeline", type="primary"):
        try:
            response = requests.post(f"{API_BASE_URL}/upload-and-process", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, data={"run_training": train, "generate_replenishment": generate}, timeout=600)
            response.raise_for_status()
            st.success("Pipeline completed. Reload the overview to see the new recommendations.")
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Pipeline request failed: {exc}")


health = _health()
report = _load_report()
with st.sidebar:
    st.markdown("## Retail Demand Pulse")
    st.caption(f"Kenya date: {_now_kenya().date().isoformat()}")
    page = st.radio("Workspace", ["Overview", "Product cockpit", "Replenishment queue", "Refresh forecast"])
    st.divider()
    st.caption(f"API: {health.get('status', 'offline')}")
    if st.button("Reload report"):
        st.cache_data.clear()
        st.rerun()

if page == "Overview":
    _overview(report, health)
elif page == "Product cockpit":
    _product_view(report)
elif page == "Replenishment queue":
    _report_view(report)
else:
    _upload_view()
import streamlit as st
import pandas as pd
"""Streamlit owner cockpit for model-driven retail replenishment."""

from datetime import date, datetime, timedelta
from pathlib import Path
import os
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "replenishment_report.csv"
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Retail Demand Pulse", page_icon="R", layout="wide")
st.markdown("""
<style>
:root { --ink:#17221f; --muted:#60706a; --paper:#f5f7f2; --line:#dbe3dc; --accent:#0e766e; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:#17221f; }
[data-testid="stSidebar"] * { color:#edf5ee !important; }
.eyebrow { color:var(--accent); font-size:.75rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero { border-bottom:1px solid var(--line); padding:1rem 0 1.25rem; margin-bottom:1.25rem; }
.hero h1 { color:var(--ink); font-size:2.5rem; margin:.2rem 0 .35rem; }
.hero p { color:var(--muted); margin:0; }
.context { background:#e7f3ed; border-left:4px solid var(--accent); padding:.8rem 1rem; margin:.5rem 0 1.25rem; }
div[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:1rem; }
</style>
""", unsafe_allow_html=True)


def _now_kenya() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(ZoneInfo("Africa/Nairobi")).date())


def _load_report() -> pd.DataFrame:
    try:
        response = requests.get(f"{API_BASE_URL}/replenishment/report/json", timeout=5)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except requests.RequestException:
        return pd.read_csv(REPORT_PATH) if REPORT_PATH.exists() else pd.DataFrame()


def _health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "offline", "model_ready": False}


def _format_kes(value: float) -> str:
    return f"KES {float(value):,.0f}"


def _priority_label(value) -> str:
    return {1: "URGENT", 2: "HIGH", 3: "MEDIUM", 4: "OK"}.get(int(value), str(value))


def _render_context(report: pd.DataFrame) -> None:
    if report.empty:
        return
    row = report.iloc[0]
    holiday = row.get("holiday_name")
    holiday_text = f"Holiday: {holiday}." if pd.notna(holiday) and holiday else "No public holiday detected."
    term = row.get("school_term") or "school holiday"
    st.markdown(
        f'<div class="context"><strong>Forecast context for {row.get("forecast_date", "tomorrow")}</strong> · '
        f'{holiday_text} School calendar: {term}. Expected weather: {row.get("expected_weather", "n/a")}. '
        f'Model: {row.get("forecast_model", "unavailable")}.</div>', unsafe_allow_html=True)


def _overview(report: pd.DataFrame, health: dict) -> None:
    st.markdown('<div class="eyebrow">Owner cockpit</div><div class="hero"><h1>Retail Demand Pulse</h1><p>Decide what to restock before tomorrow opens.</p></div>', unsafe_allow_html=True)
    _render_context(report)
    if report.empty:
        st.warning("No report is available. Run the forecast pipeline first.")
        return
    reorder = report[report["reorder_needed"].astype(bool)]
    urgent = reorder[reorder["replenishment_priority"].astype(int) == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products tracked", len(report)); c2.metric("Need replenishment", len(reorder))
    c3.metric("Urgent", len(urgent)); c4.metric("Recommended spend", _format_kes(report["estimated_order_cost_kes"].sum()))
    st.success("Demand model ready." if health.get("model_ready") else "Demand model is not ready.")
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("What needs attention")
        display = reorder.sort_values(["replenishment_priority", "days_of_stock_remaining"])[["product_name", "stock_status", "current_stock", "forecasted_demand_per_day", "recommended_order_qty", "estimated_order_cost_kes"]]
        display.columns = ["Product", "Status", "Stock", "Forecast / day", "Order qty", "Cost"]
        st.dataframe(display, hide_index=True, width="stretch")
    with right:
        st.subheader("Order quantities")
        st.bar_chart(reorder.nlargest(10, "recommended_order_qty").set_index("product_name")["recommended_order_qty"], color="#0e766e")


def _scenario_view(report: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">What-if forecasting</div><div class="hero"><h1>Owner scenario</h1><p>Enter the day\'s store details. The ML model supplies demand and the full restock decision.</p></div>', unsafe_allow_html=True)
    if report.empty:
        st.warning("Generate a report first so the product catalogue is available."); return
    products = report.sort_values("product_name")
    product_name = st.selectbox("Product", products["product_name"].tolist())
    product = products.loc[products["product_name"] == product_name].iloc[0]
    with st.form("owner_scenario"):
        left, right = st.columns(2)
        with left:
            target_date = st.date_input("Forecast date", value=date.today() + timedelta(days=1))
            current_stock = st.number_input("Current stock", min_value=0.0, value=float(product["current_stock"]), step=1.0)
            recent_sales = st.text_input("Recent daily sales, newest last", value=", ".join([str(round(product["avg_daily"], 1))] * 7))
        with right:
            weather = st.selectbox("Weather", ["Automatic seasonal weather", "Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Heavy Rain"])
            avg_daily = st.number_input("Average daily sales fallback", min_value=0.0, value=float(product["avg_daily"]), step=0.1)
            std_daily = st.number_input("Daily variability fallback", min_value=0.0, value=float(product["std_daily"]), step=0.1)
        submitted = st.form_submit_button("Run model scenario", type="primary")
    if not submitted:
        return
    payload = {"product_id": product["product_id"], "target_date": target_date.isoformat(), "current_stock": current_stock, "recent_sales": recent_sales, "avg_daily": avg_daily, "std_daily": std_daily}
    if weather != "Automatic seasonal weather": payload["weather_condition"] = weather
    try:
        response = requests.post(f"{API_BASE_URL}/predict/scenario", data=payload, timeout=30)
        response.raise_for_status(); result = response.json()
        st.success("Scenario completed using the trained demand model.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast / day", f"{result['forecasted_demand_per_day']:.1f} units")
        c2.metric("7-day demand", f"{result['forecasted_demand_7d']:,} units")
        c3.metric("Recommended order", f"{result['recommended_order_qty']:,} units")
        c4.metric("Estimated cost", _format_kes(result["estimated_order_cost_kes"]))
        st.write(f"**Status:** {result['stock_status']} · **Priority:** {result['replenishment_priority']} · **Days remaining:** {result['days_of_stock_remaining']:.1f}")
        st.write(f"**Forecast date:** {result['forecast_date']} · **Holiday:** {result['holiday_name'] or 'None'} · **School term:** {result['school_term'] or 'School holiday'} · **Weather:** {result['expected_weather']}")
        with st.expander("Full scenario output"): st.json(result)
    except requests.RequestException as exc:
        st.error(f"Scenario failed: {exc.response.text if exc.response is not None else exc}")


def _product_view(report: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Product intelligence</div><div class="hero"><h1>Product cockpit</h1><p>Inspect one item\'s forecast, stock position, and recommended action.</p></div>', unsafe_allow_html=True)
    if report.empty: st.warning("Generate a report first."); return
    selected = st.selectbox("Product", report["product_name"].tolist()); row = report.loc[report["product_name"] == selected].iloc[0]
    _render_context(pd.DataFrame([row])); c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tomorrow forecast", f"{row['forecasted_demand_per_day']:.1f} units"); c2.metric("7-day demand", f"{int(row['forecasted_demand_7d']):,} units")
    c3.metric("Stock remaining", f"{row['current_stock']:,.0f} units"); c4.metric("Recommended order", f"{row['recommended_order_qty']:,.0f} units")
    st.write(f"**Status:** {row['stock_status']} · **Days of stock:** {row['days_of_stock_remaining']:.1f} · **Priority:** {_priority_label(row['replenishment_priority'])} · **Estimated cost:** {_format_kes(row['estimated_order_cost_kes'])}")


def _queue_view(report: pd.DataFrame) -> None:
    st.markdown('<div class="eyebrow">Operations</div><div class="hero"><h1>Replenishment queue</h1><p>Filter, review, and export current order recommendations.</p></div>', unsafe_allow_html=True)
    if report.empty: st.warning("No report found."); return
    statuses = st.multiselect("Stock status", sorted(report["stock_status"].dropna().unique()), default=sorted(report["stock_status"].dropna().unique()))
    priorities = st.multiselect("Priority", sorted(report["replenishment_priority"].astype(int).unique()), default=sorted(report["replenishment_priority"].astype(int).unique()), format_func=_priority_label)
    query = st.text_input("Find product")
    filtered = report[report["stock_status"].isin(statuses) & report["replenishment_priority"].astype(int).isin(priorities)]
    if query: filtered = filtered[filtered["product_name"].str.contains(query, case=False, na=False)]
    st.dataframe(filtered.sort_values(["replenishment_priority", "days_of_stock_remaining"]), hide_index=True, width="stretch")
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "replenishment_report.csv", "text/csv")


def _refresh_view() -> None:
    st.markdown('<div class="eyebrow">Data refresh</div><div class="hero"><h1>Refresh the forecast</h1><p>Upload sales history to clean, retrain models, and generate a new recommendation.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Sales history", type=["csv", "xlsx", "xls"]); train = st.checkbox("Retrain ML models", value=True); generate = st.checkbox("Generate replenishment report", value=True)
    if uploaded and st.button("Run forecast pipeline", type="primary"):
        try:
            response = requests.post(f"{API_BASE_URL}/upload-and-process", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, data={"run_training": train, "generate_replenishment": generate}, timeout=600)
            response.raise_for_status(); st.success("Pipeline completed."); st.json(response.json())
        except requests.RequestException as exc: st.error(f"Pipeline request failed: {exc}")


health = _health(); report = _load_report()
with st.sidebar:
    st.markdown("## Retail Demand Pulse"); st.caption(f"Kenya date: {_now_kenya().date().isoformat()}")
    page = st.radio("Workspace", ["Overview", "Owner scenario", "Product cockpit", "Replenishment queue", "Refresh forecast"])
    st.divider(); st.caption(f"API: {health.get('status', 'offline')}")

if page == "Overview": _overview(report, health)
elif page == "Owner scenario": _scenario_view(report)
elif page == "Product cockpit": _product_view(report)
elif page == "Replenishment queue": _queue_view(report)
else: _refresh_view()
import os
from pathlib import Path
from datetime import date, timedelta

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

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
    page = st.radio("Go to", ["🏠 Home", "🧪 Owner Scenario", "🔮 Predict Replenishment", "📤 Upload Dataset", "📊 Reports"])

# ====================== HOME ======================
if page == "🏠 Home":
    st.success("🌸 Welcome! Your smart retail assistant is ready.")

# ====================== OWNER SCENARIO ======================
elif page == "🧪 Owner Scenario":
    st.subheader("🧪 Run a complete model scenario")
    st.caption("Enter the store details you know. The model derives tomorrow's demand, calendar context, and restock action.")
    report_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "replenishment_report.csv"
    if not report_path.exists():
        st.warning("Generate a replenishment report first.")
    else:
        catalogue = pd.read_csv(report_path)
        with st.form("owner_scenario_form"):
            product_name = st.selectbox("Product", catalogue["product_name"].tolist())
            product = catalogue[catalogue["product_name"] == product_name].iloc[0]
            col1, col2 = st.columns(2)
            with col1:
                target_date = st.date_input("Forecast date", value=date.today() + timedelta(days=1))
                current_stock = st.number_input("Current stock", min_value=0.0, value=float(product["current_stock"]), step=1.0)
                recent_sales = st.text_input("Recent daily sales, newest last", value=", ".join([str(round(product["avg_daily"], 1))] * 7))
            with col2:
                weather = st.selectbox("Weather", ["Automatic seasonal weather", "Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Heavy Rain"])
                avg_daily = st.number_input("Average daily sales fallback", min_value=0.0, value=float(product["avg_daily"]), step=0.1)
                std_daily = st.number_input("Daily variability fallback", min_value=0.0, value=float(product["std_daily"]), step=0.1)
            submitted = st.form_submit_button("Run model scenario", type="primary")

        if submitted:
            payload = {"product_id": product["product_id"], "target_date": target_date.isoformat(), "current_stock": current_stock, "recent_sales": recent_sales, "avg_daily": avg_daily, "std_daily": std_daily}
            if weather != "Automatic seasonal weather":
                payload["weather_condition"] = weather
            try:
                response = requests.post(f"{API_BASE_URL}/predict/scenario", data=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                st.success("Scenario completed using the trained demand model.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Forecast / day", f"{result['forecasted_demand_per_day']:.1f} units")
                c2.metric("7-day demand", f"{result['forecasted_demand_7d']:,} units")
                c3.metric("Recommended order", f"{result['recommended_order_qty']:,} units")
                c4.metric("Estimated cost", f"KES {result['estimated_order_cost_kes']:,.0f}")
                st.write(f"**Status:** {result['stock_status']} · **Priority:** {result['replenishment_priority']} · **Days remaining:** {result['days_of_stock_remaining']:.1f}")
                st.write(f"**Forecast date:** {result['forecast_date']} · **Holiday:** {result['holiday_name'] or 'None'} · **School term:** {result['school_term'] or 'School holiday'} · **Weather:** {result['expected_weather']}")
                with st.expander("Full scenario output"):
                    st.json(result)
            except requests.RequestException as exc:
                st.error(f"Scenario failed: {exc.response.text if exc.response is not None else exc}")

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
                    response = requests.post(f"{API_BASE_URL}/predict/replenishment", data=payload)
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
                response = requests.post(f"{API_BASE_URL}/upload-and-process", files=files)
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
    report_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "replenishment_report.csv"
    
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