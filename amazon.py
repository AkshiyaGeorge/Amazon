import pandas as pd
import streamlit as st
import plotly.express as px
import mysql.connector

# --- Page Config ---
st.set_page_config(page_title="📊 Amazon Decade of Sales Analysis Dashboard", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; font-size: 40px; color: #FF9900; margin-top: 10px;'>
        📊 Amazon Decade of Sales Analysis Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    button[role="tab"] {
        font-size: 18px !important;
        padding: 12px 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SQL Connection ---
@st.cache_resource
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Akshiya13",
        database="amazon"
    )

conn = get_connection()

# --- Load SQL Table ---
@st.cache_data
def load_table(table_name):
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, conn)

# --- Dashboard Sections ---
sections = {
    "📊 Executive Dashboard": {
        "📊 Executive Summary": ("transactions", "executive"),
        "📡 Real-Time Performance": ("transactions", "realtime"),
        "💰 Financial Analysis": ("transactions", "financial"),
        "🏢 Strategic Overview": ("transactions", "strategic"),
        "📈 Growth Analytics": ("customers", "growth"),
    },
    "💸 Revenue Analytics": {
        "📊 Revenue Trend Analysis": ("transactions", "revenue_trend"),
        "🗂️ Category Performance": ("products", "category_performance"),
        "🌍 Geographic Revenue": ("transactions", "geo_revenue"),
        "🎉 Festival Sales Analytics": ("transactions", "festival_sales"),
        "💸 Price Optimization": ("products", "price_optimization"),
    },
    "🧠 Customer Analytics": {
        "🧠 Customer Segmentation": ("customers", "segmentation"),
        "🚶 Customer Journey": ("transactions", "journey"),
        "👑 Prime Membership Analytics": ("transactions", "prime"),
        "🔁 Customer Retention": ("transactions", "retention"),
        "🌍 Demographics & Behavior": ("customers", "demographics"),
    },
    "🧪 Product & Brand Dashboards": {
        "📦 Product Performance": ("products", "product_performance"),
        "🏷️ Brand Analytics": ("products", "brand_analytics"),
        "📊 Inventory Optimization": ("products", "inventory_optimization"),
        "⭐ Rating & Review Analysis": ("products", "rating_review"),
        "🚀 New Product Launch": ("products", "new_launch"),
    },
    "🚚 Operations & Logistics": {
        "🚚 Delivery Performance": ("transactions", "delivery_performance"),
        "💳 Payment Analytics": ("transactions", "payment_analytics"),
        "🔁 Returns & Cancellations": ("transactions", "return_cancellation"),
        "📞 Customer Service": ("transactions", "customer_service"),
        "🔗 Supply Chain Insights": ("transactions", "supply_chain"),
    },
    "📡 Advanced Analytics": {
        "📈 Predictive Analytics": ("transactions", "predictive_analytics"),
        "🧠 Market Intelligence": ("products", "market_intelligence"),
        "🔗 Cross-sell & Upsell": ("products", "cross_upsell"),
        "📅 Seasonal Planning": ("transactions", "seasonal_planning"),
        "🧭 BI Command Center": ("transactions", "bi_command_center"),
    },
        "🔍 Raw Data Views": {
        "💰 View Transactions": ("transactions", "raw"),
        "📦 View Products": ("products", "raw"),
        "🧑 View Customers": ("customers", "raw"),
        "⏱️ View Time Dimension": ("time_dimension", "raw"),
    }
}

# --- Sample Limit ---
SAMPLE_LIMIT = 15000

# --- Create Tabs ---
tab_titles = list(sections.keys())
tabs = st.tabs(tab_titles)

for tab, section_title in zip(tabs, tab_titles):
    with tab:
        st.markdown(f"### {section_title}")
        view_options = list(sections[section_title].keys())
        selected_label = st.selectbox("📂 Choose a View", view_options, key=section_title)

        selected_table, view_type = sections[section_title][selected_label]
        df = load_table(selected_table)

        if len(df) > SAMPLE_LIMIT:
            df = df.sample(n=SAMPLE_LIMIT, random_state=42)

        st.markdown(
            f"""<div style='background-color:#f7f7f7;padding:20px;border-radius:10px'>
                <h1 style='color:#FF9900;text-align:center;'>📦 {selected_label}</h1>
            </div>""",
            unsafe_allow_html=True
        )

        # ✅ Show table only for raw views
        if view_type == "raw":
            with st.expander("📈 Summary Statistics"):
                st.write(df.describe(include="all"))

            st.markdown(f"### 🧾 Sampled Data Table (Showing up to {SAMPLE_LIMIT} rows)")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label="📥 Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_table}_{view_type}.csv",
                mime="text/csv"
            )

        # --- Dashboard Logic ---
        elif view_type == "executive":
            st.title("📈 Executive Summary Dashboard")

            # ✅ Load transaction and product data from SQL
            df = load_table("transactions")
            prod_df = load_table("products")
            df = df.merge(prod_df[["product_id", "category"]], on="product_id", how="left")

            # ✅ Total Revenue
            total_revenue = df["final_amount_inr"].sum()

            # ✅ Revenue by Year
            revenue_by_year = df.groupby("order_year")["final_amount_inr"].sum().sort_index()
            growth_rate = revenue_by_year.pct_change().fillna(0).round(2) * 100

            # ✅ Active Customers
            active_customers = df["customer_id"].nunique()

            # ✅ Average Order Value
            avg_order_value = df["final_amount_inr"].sum() / df["transaction_id"].nunique()

            # ✅ Top Performing Categories
            top_categories = df.groupby("category")["final_amount_inr"].sum().sort_values(ascending=False)

            # ✅ Metrics Summary
            st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
            st.metric("📈 Growth Rate (YoY)", f"{growth_rate.iloc[-1]:.2f}%")
            st.metric("👥 Active Customers", f"{active_customers:,}")
            st.metric("🛒 Avg Order Value", f"₹{avg_order_value:.2f}")

            # 🏆 Top Categories
            st.subheader("🏆 Top Performing Categories")
            st.bar_chart(top_categories.head(10))

            # 📉 Revenue Trend
            st.subheader("📉 Revenue Trend by Year")
            st.line_chart(revenue_by_year)

            # 📊 Growth Rate Trend
            st.subheader("📊 Growth Rate Trend")
            st.line_chart(growth_rate)

            # 👥 Customer Growth Over Time
            if "order_year" in df.columns:
                customer_growth = df.groupby("order_year")["customer_id"].nunique()
                st.subheader("👥 Active Customers by Year")
                st.line_chart(customer_growth)

            # 🧠 Category Contribution Over Time
            if "order_year" in df.columns:
                cat_year = df.groupby(["order_year", "category"])["final_amount_inr"].sum().unstack().fillna(0)
                st.subheader("🧠 Category Revenue Contribution Over Time")
                st.area_chart(cat_year)

        elif view_type == "realtime":
            st.title("📈 Realtime Performance Dashboard")

            # ✅ Define targets
            monthly_revenue_target = 5_000_000  # ₹5M
            customer_target = 1000

            # ✅ Get current date context
            today = pd.Timestamp.now()
            current_month = today.month
            current_year = today.year
            days_passed = today.day

            # ✅ Filter current month data
            monthly_data = df[(df["order_month"] == current_month) & (df["order_year"] == current_year)]

            # ✅ Metrics
            revenue_so_far = monthly_data["final_amount_inr"].sum()
            run_rate = (revenue_so_far / days_passed) * 30
            new_customers = monthly_data["customer_id"].nunique()
            avg_order_value = monthly_data["final_amount_inr"].sum() / monthly_data["transaction_id"].nunique()

            # ✅ Display metrics
            st.metric("📅 Current Month Revenue", f"₹{revenue_so_far:,.0f}", delta=f"{((revenue_so_far/monthly_revenue_target)*100):.1f}% of target")
            st.metric("🚀 Revenue Run-Rate", f"₹{run_rate:,.0f}", delta=f"{((run_rate/monthly_revenue_target)*100):.1f}% projected")
            st.metric("🧍‍♀️ New Customers", f"{new_customers:,}", delta=f"{((new_customers/customer_target)*100):.1f}% of target")
            st.metric("🛒 Avg Order Value", f"₹{avg_order_value:.2f}")

            # 📊 Daily Revenue Trend
            st.subheader("📊 Daily Revenue")
            if "order_date" in monthly_data.columns:
                monthly_data["order_date"] = pd.to_datetime(monthly_data["order_date"], errors="coerce")
                daily_rev = monthly_data.groupby(monthly_data["order_date"].dt.day)["final_amount_inr"].sum()
                st.line_chart(daily_rev)

            # ⚠️ Underperformance Alerts
            st.subheader("🚨 Performance Alerts")
            alerts = []

            expected_revenue = monthly_revenue_target * days_passed / 30
            if revenue_so_far < expected_revenue:
                alerts.append(
                    f"⚠️ Revenue is below expected pace.\n"
                    f"Current: ₹{revenue_so_far:,.0f} | Expected: ₹{expected_revenue:,.0f} (Target: ₹{monthly_revenue_target:,.0f})"
                )

            if run_rate < monthly_revenue_target:
                alerts.append(
                    f"⚠️ Projected revenue may miss target.\n"
                    f"Run Rate: ₹{run_rate:,.0f} | Monthly Target: ₹{monthly_revenue_target:,.0f}"
                )

            expected_customers = customer_target * days_passed / 30
            if new_customers < expected_customers:
                alerts.append(
                    f"⚠️ Customer acquisition is lagging.\n"
                    f"Current: {new_customers:,} | Expected: {expected_customers:,.0f} (Target: {customer_target:,})"
                )

            if alerts:
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ All key metrics are on track!")

            # 📦 Category Performance Snapshot
            if "category" in df.columns:
                top_cats = monthly_data.groupby("category")["final_amount_inr"].sum().sort_values(ascending=False)
                st.subheader("📦 Top Categories This Month")
                st.bar_chart(top_cats.head(5))

        elif view_type == "strategic":
            st.title("🏢 Strategic Overview Dashboard")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' exists and is datetime
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")

            # ✅ Enrich transactions with time dimensions
            transactions_df["order_year"] = transactions_df["order_date"].dt.year
            transactions_df["order_month"] = transactions_df["order_date"].dt.month
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # 📊 Market Share Analysis
            market_share = transactions_df.groupby("product_id")["final_amount_inr"].sum().reset_index()
            market_share = market_share.merge(products_df[["product_id", "category"]], on="product_id", how="left")
            category_share = market_share.groupby("category")["final_amount_inr"].sum()
            st.subheader("📊 Market Share by Product Category")
            st.bar_chart(category_share)

            # 🥇 Competitive Positioning
            top_products = market_share.sort_values(by="final_amount_inr", ascending=False).head(10)
            st.subheader("🥇 Top 10 Products by Revenue")
            st.dataframe(top_products[["product_id", "category", "final_amount_inr"]])

            # 🌍 Geographic Expansion Metrics
            if "region" in customers_df.columns:
                geo_metrics = customers_df.groupby("region")["customer_id"].nunique().sort_values(ascending=False)
                st.subheader("🌍 Active Customer Distribution by Region")
                st.bar_chart(geo_metrics)

            if {"latitude", "longitude"}.issubset(customers_df.columns):
                st.map(customers_df[["latitude", "longitude"]].dropna())

            # 📈 Business Health Indicators
            total_revenue = transactions_df["final_amount_inr"].sum()
            total_customers = customers_df["customer_id"].nunique()
            avg_order_value = transactions_df["final_amount_inr"].mean()

            st.subheader("📈 Business Health Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
            col2.metric("👥 Total Customers", f"{total_customers:,}")
            col3.metric("🧾 Avg Order Value", f"₹{avg_order_value:,.0f}")

            # 🕒 Time Trend: Monthly Revenue
            monthly_revenue = transactions_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
            monthly_revenue["date"] = pd.to_datetime(
                monthly_revenue["order_year"].astype(str) + "-" +
                monthly_revenue["order_month"].astype(str) + "-01"
            )
            monthly_revenue.set_index("date", inplace=True)

            st.subheader("📆 Monthly Revenue Trend")
            st.line_chart(monthly_revenue["final_amount_inr"])


        elif view_type == "financial":
            st.title("📊 Financial Performance Dashboard")

            # ✅ Load and merge product data
            prod_df = load_table("products")
            df = load_table("transactions")
            df = df.merge(prod_df[["product_id", "category"]], on="product_id", how="left")

            # ✅ Revenue Breakdown
            total_revenue = df["final_amount_inr"].sum()
            revenue_by_cat = df.groupby("category")["final_amount_inr"].sum().sort_values(ascending=False)

            # ✅ Discount & Delivery Cost
            avg_discount = df["discount_percent"].mean()
            total_delivery_cost = df["delivery_charges"].sum()

            # ✅ Profit Margin Analysis
            df["margin_inr"] = df["discounted_price_inr"] - df["delivery_charges"]
            avg_margin = df["margin_inr"].mean()
            margin_by_cat = df.groupby("category")["margin_inr"].mean().sort_values(ascending=False)

            # ✅ Cost Structure
            cost_structure = {
                "Delivery Costs": total_delivery_cost,
                "Discount Loss": (df["original_price_inr"] - df["discounted_price_inr"]).sum(),
                "Net Revenue": total_revenue
            }

            # ✅ Financial Forecasting
            df["order_year"] = pd.to_datetime(df["order_date"], errors="coerce").dt.year
            df["order_month"] = pd.to_datetime(df["order_date"], errors="coerce").dt.month
            monthly_rev = df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
            monthly_rev["date"] = pd.to_datetime(monthly_rev["order_year"].astype(str) + "-" + monthly_rev["order_month"].astype(str) + "-01")
            monthly_rev.set_index("date", inplace=True)
            forecast = monthly_rev["final_amount_inr"].rolling(window=3).mean()

            # ✅ Metrics Summary
            st.metric("📦 Total Revenue", f"₹{total_revenue:,.0f}")
            st.metric("🎯 Avg Discount", f"{avg_discount:.2f}%")
            st.metric("💰 Avg Profit Margin", f"₹{avg_margin:.2f}")
            st.metric("🚚 Total Delivery Cost", f"₹{total_delivery_cost:,.0f}")

            # 📊 Revenue Breakdown
            st.subheader("📊 Revenue Breakdown by Category")
            st.bar_chart(revenue_by_cat.head(10))

            # 📈 Profit Margin by Category
            st.subheader("📈 Avg Profit Margin by Category")
            st.bar_chart(margin_by_cat.head(10))

            # ⚙️ Cost Structure Visualization
            st.subheader("⚙️ Cost Structure Overview")
            st.dataframe(pd.DataFrame.from_dict(cost_structure, orient="index", columns=["Amount (INR)"]))

            # 📉 Revenue Forecasting
            st.subheader("📉 Revenue Forecast (3-Month Rolling Avg)")
            st.line_chart(forecast)

            # 📈 Revenue Trend
            st.subheader("📈 Monthly Revenue Trend")
            st.line_chart(monthly_rev["final_amount_inr"])


        elif view_type == "growth":
            st.title("📈 Growth Analysis")

            # ✅ Load data
            df = load_table("transactions")
            products_df = load_table("products")

            # ✅ Convert order_date to datetime
            if "order_date" in df.columns:
                df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
                df["order_year"] = df["order_date"].dt.year
                df["order_month"] = df["order_date"].dt.month

            # ✅ Monthly unique customer count
            monthly_customers = df.groupby(["order_year", "order_month"])["customer_id"].nunique().reset_index()
            monthly_customers["date"] = pd.to_datetime(
                monthly_customers["order_year"].astype(str) + "-" +
                monthly_customers["order_month"].astype(str) + "-01"
            )
            monthly_customers.set_index("date", inplace=True)

            # ✅ Product portfolio expansion
            new_products = products_df.groupby("category")["product_id"].nunique()

            # ✅ Prime member revenue
            prime_orders = df[df["is_prime_member"] == True]["final_amount_inr"].sum()

            # ✅ Display metrics and charts
            st.metric("📈 Prime Member Revenue", f"₹{prime_orders:,.0f}")
            st.subheader("👥 Monthly Customer Growth")
            st.line_chart(monthly_customers["customer_id"])

            st.subheader("📦 Product Portfolio Expansion")
            st.bar_chart(new_products)


        elif view_type == "revenue_trend":
            st.title("📊 Revenue Trend Analysis")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' exists and is datetime
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")

            # ✅ Enrich transactions with time dimensions
            transactions_df["order_year"] = transactions_df["order_date"].dt.year
            transactions_df["order_month"] = transactions_df["order_date"].dt.month
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # ✅ Interactive time period selection
            period = st.selectbox("📅 Select Time Period", ["Monthly", "Quarterly", "Yearly"])

            # ✅ Revenue aggregation
            if period == "Monthly":
                revenue_df = transactions_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
                revenue_df["date"] = pd.to_datetime(revenue_df["order_year"].astype(str) + "-" + revenue_df["order_month"].astype(str) + "-01")
            elif period == "Quarterly":
                if "quarter" not in transactions_df.columns:
                    st.warning("⚠️ 'quarter' column missing in time dimension data.")
                    st.stop()
                revenue_df = transactions_df.groupby(["order_year", "quarter"])["final_amount_inr"].sum().reset_index()
                revenue_df["date"] = pd.to_datetime(revenue_df["order_year"].astype(str) + "-01-01") + pd.to_timedelta((revenue_df["quarter"] - 1) * 3, unit="M")
            else:
                revenue_df = transactions_df.groupby("order_year")["final_amount_inr"].sum().reset_index()
                revenue_df["date"] = pd.to_datetime(revenue_df["order_year"].astype(str) + "-01-01")

            revenue_df.set_index("date", inplace=True)

            # 📈 Revenue Trend
            st.subheader(f"📈 {period} Revenue Trend")
            st.line_chart(revenue_df["final_amount_inr"])

            # 📊 Growth Rate Calculation
            revenue_df["growth_rate"] = revenue_df["final_amount_inr"].pct_change() * 100
            st.subheader(f"📊 {period} Growth Rate (%)")
            st.bar_chart(revenue_df["growth_rate"])

            # 🌤️ Seasonal Variation
            if period == "Monthly":
                seasonal_df = transactions_df.groupby("order_month")["final_amount_inr"].mean()
                st.subheader("🌤️ Average Monthly Revenue (Seasonality)")
                st.bar_chart(seasonal_df)

            # 🔮 Revenue Forecasting
            from sklearn.linear_model import LinearRegression
            import numpy as np

            forecast_df = revenue_df.reset_index()
            forecast_df["timestamp"] = forecast_df["date"].view("int64") // 10**9
            X = forecast_df[["timestamp"]]
            y = forecast_df["final_amount_inr"]

            model = LinearRegression()
            model.fit(X, y)

            future_dates = pd.date_range(start=forecast_df["date"].max(), periods=6, freq="M" if period == "Monthly" else "Q" if period == "Quarterly" else "Y")
            future_ts = (future_dates.view("int64") // 10**9).reshape(-1, 1)
            future_preds = model.predict(future_ts)

            forecast_plot = pd.Series(future_preds, index=future_dates)
            st.subheader("🔮 Revenue Forecast")
            st.line_chart(forecast_plot)


        elif view_type == "category_performance":
            st.title("🗂️ Category Performance Dashboard")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' exists and is datetime
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")

            # ✅ Enrich transactions with time dimensions
            transactions_df["order_year"] = transactions_df["order_date"].dt.year
            transactions_df["order_month"] = transactions_df["order_date"].dt.month
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # ✅ Merge product info
            merged_df = transactions_df.merge(products_df[["product_id", "category"]], on="product_id", how="left")

            # ✅ Interactive category selection
            selected_category = st.selectbox("📦 Select Product Category", sorted(merged_df["category"].dropna().unique()))
            category_df = merged_df[merged_df["category"] == selected_category]

            # 📊 Revenue Contribution
            total_revenue = merged_df["final_amount_inr"].sum()
            category_revenue = category_df["final_amount_inr"].sum()
            contribution_pct = (category_revenue / total_revenue) * 100
            st.metric("💰 Revenue Contribution", f"₹{category_revenue:,.0f} ({contribution_pct:.2f}%)")

            # 📈 Growth Trends
            growth_df = category_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
            growth_df["date"] = pd.to_datetime(growth_df["order_year"].astype(str) + "-" + growth_df["order_month"].astype(str) + "-01")
            growth_df.set_index("date", inplace=True)
            st.subheader("📈 Monthly Revenue Trend")
            st.line_chart(growth_df["final_amount_inr"])

            # 🥇 Market Share Changes
            market_df = merged_df.groupby(["order_year", "category"])["final_amount_inr"].sum().reset_index()
            pivot_market = market_df.pivot(index="order_year", columns="category", values="final_amount_inr").fillna(0)
            market_share = pivot_market.div(pivot_market.sum(axis=1), axis=0) * 100
            st.subheader("🥇 Market Share Over Time")
            st.line_chart(market_share[selected_category])

            # 📉 Category-wise Profitability
            if "cost_inr" in products_df.columns:
                category_df = category_df.merge(products_df[["product_id", "cost_inr"]], on="product_id", how="left")
                category_df["profit_inr"] = category_df["final_amount_inr"] - category_df["cost_inr"]
                profit_summary = category_df.groupby("product_id")[["final_amount_inr", "cost_inr", "profit_inr"]].sum().reset_index()
                top_profit = profit_summary.sort_values(by="profit_inr", ascending=False).head(10)
                st.subheader("📉 Top 10 Profitable Products")
                st.dataframe(top_profit)

            # 🔍 Drill-down: Product-Level Trend
            selected_product = st.selectbox("🔍 Drill Down to Product", sorted(category_df["product_id"].unique()))
            product_df = category_df[category_df["product_id"] == selected_product]
            product_trend = product_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
            product_trend["date"] = pd.to_datetime(product_trend["order_year"].astype(str) + "-" + product_trend["order_month"].astype(str) + "-01")
            product_trend.set_index("date", inplace=True)
            st.subheader(f"📦 Revenue Trend for Product {selected_product}")
            st.line_chart(product_trend["final_amount_inr"])


        elif view_type == "geo_revenue":
            st.title("🌍 Geographic Revenue Analysis")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' exists and is datetime
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")

            # ✅ Enrich transactions with time dimensions
            transactions_df["order_year"] = transactions_df["order_date"].dt.year
            transactions_df["order_month"] = transactions_df["order_date"].dt.month
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # ✅ Merge customer location info
            geo_df = transactions_df.merge(
                customers_df[["customer_id", "customer_state", "customer_city", "customer_tier"]],
                on="customer_id", how="left"
            )

            # 📍 State-wise Revenue
            state_revenue = geo_df.groupby("customer_state")["final_amount_inr"].sum().sort_values(ascending=False)
            st.subheader("📍 State-wise Revenue Performance")
            st.bar_chart(state_revenue)

            # 🏙️ City-wise Revenue
            top_cities = geo_df.groupby("customer_city")["final_amount_inr"].sum().sort_values(ascending=False).head(10)
            st.subheader("🏙️ Top 10 Cities by Revenue")
            st.dataframe(top_cities)

            # 🏘️ Tier-wise Growth Patterns
            tier_growth = geo_df.groupby(["order_year", "customer_tier"])["final_amount_inr"].sum().reset_index()
            tier_pivot = tier_growth.pivot(index="order_year", columns="customer_tier", values="final_amount_inr").fillna(0)
            st.subheader("🏘️ Tier-wise Revenue Growth")
            st.line_chart(tier_pivot)

            # 🧭 Market Penetration Opportunities
            penetration_df = geo_df.groupby(["customer_state", "customer_city"])["customer_id"].nunique().reset_index()
            penetration_df["revenue"] = geo_df.groupby(["customer_state", "customer_city"])["final_amount_inr"].sum().values
            penetration_df["avg_revenue_per_customer"] = penetration_df["revenue"] / penetration_df["customer_id"]

            st.subheader("🧭 Market Penetration Opportunities")
            st.dataframe(penetration_df.sort_values(by="avg_revenue_per_customer", ascending=False).head(10))

            # 📌 Regional Revenue Summary
            region_df = geo_df.groupby(["customer_state", "customer_city", "customer_tier"])["final_amount_inr"].sum().reset_index()
            st.subheader("📌 Regional Revenue Summary")
            st.dataframe(region_df.sort_values(by="final_amount_inr", ascending=False).head(10))

        elif view_type == "festival_sales":
            st.title("🎉 Festival Sales Analytics Dashboard")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' exists and is datetime
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # ✅ Festival selection
            festival_options = time_df["holiday_name"].dropna().unique()
            selected_festival = st.selectbox("🪔 Select Festival Period", sorted(festival_options))

            festival_df = transactions_df[transactions_df["holiday_name"] == selected_festival]

            # 🎯 Festival Period Performance
            total_festival_revenue = festival_df["final_amount_inr"].sum()
            total_orders = festival_df["transaction_id"].nunique()
            unique_customers = festival_df["customer_id"].nunique()

            st.subheader(f"🎯 Performance During {selected_festival}")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Revenue", f"₹{total_festival_revenue:,.0f}")
            col2.metric("🛍️ Orders", f"{total_orders:,}")
            col3.metric("👥 Unique Customers", f"{unique_customers:,}")

            # 📢 Campaign Effectiveness
            if "campaign_id" in festival_df.columns:
                campaign_perf = festival_df.groupby("campaign_id")["final_amount_inr"].sum().sort_values(ascending=False)
                st.subheader("📢 Campaign Revenue Impact")
                st.bar_chart(campaign_perf)

            # 🎁 Promotional Impact
            if "promotion_applied" in festival_df.columns:
                promo_df = festival_df.groupby("promotion_applied")["final_amount_inr"].sum()
                st.subheader("🎁 Revenue by Promotion Type")
                st.bar_chart(promo_df)

            # 📈 Seasonal Revenue Optimization
            seasonal_df = transactions_df.groupby("holiday_name")["final_amount_inr"].sum().reset_index()
            seasonal_df = seasonal_df.dropna(subset=["holiday_name"])
            st.subheader("📈 Revenue Across Festivals")
            st.line_chart(seasonal_df.set_index("holiday_name"))

            # 🔍 Product-Level Insights
            top_products = festival_df.groupby("product_id")["final_amount_inr"].sum().reset_index()
            top_products = top_products.merge(products_df[["product_id", "category"]], on="product_id", how="left")
            top_products = top_products.sort_values(by="final_amount_inr", ascending=False).head(10)

            st.subheader("🔍 Top Performing Products")
            st.dataframe(top_products[["product_id", "category", "final_amount_inr"]])


        elif view_type == "price_optimization":
            st.title("💸 Price Optimization Dashboard")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' is aligned
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # ✅ Merge product info
            transactions_df = transactions_df.merge(products_df[["product_id", "category", "brand"]], on="product_id", how="left")

            # 📈 Price Elasticity Analysis using original price
            elasticity_df = transactions_df.groupby("original_price_inr")["transaction_id"].count().reset_index()
            elasticity_df.rename(columns={"transaction_id": "units_sold"}, inplace=True)
            st.subheader("📈 Price Elasticity Curve")
            st.line_chart(elasticity_df.set_index("original_price_inr"))

            # ⚔️ Competitive Pricing by Brand
            if "brand" in transactions_df.columns and "discounted_price_inr" in transactions_df.columns:

                # Step 1: Clean and convert price column
                transactions_df["discounted_price_inr"] = pd.to_numeric(transactions_df["discounted_price_inr"], errors="coerce")

                # Step 2: Drop rows with missing values
                brand_df = transactions_df.dropna(subset=["brand", "discounted_price_inr"])

                # Step 3: Group and plot
                brand_pricing = brand_df.groupby("brand")["discounted_price_inr"].mean().sort_values(ascending=False)

                st.subheader("⚔️ Average Discounted Price by Brand")
                st.bar_chart(brand_pricing)

            # 🎯 Discount Effectiveness
            if all(col in transactions_df.columns for col in ["discount_percent", "final_amount_inr", "quantity", "customer_id"]):

                # Step 1: Clean and convert columns
                transactions_df["discount_percent"] = pd.to_numeric(transactions_df["discount_percent"], errors="coerce")
                transactions_df["final_amount_inr"] = pd.to_numeric(transactions_df["final_amount_inr"], errors="coerce")
                transactions_df["quantity"] = pd.to_numeric(transactions_df["quantity"], errors="coerce")

                # Step 2: Drop rows with missing or zero values
                clean_df = transactions_df.dropna(subset=["discount_percent", "final_amount_inr", "quantity", "customer_id"])
                clean_df = clean_df[clean_df["final_amount_inr"] > 0]

                # Step 3: Bin discount values to reduce clutter
                clean_df["discount_bin"] = clean_df["discount_percent"].round(1)

                # Step 4: Group metrics
                discount_metrics = clean_df.groupby("discount_bin").agg({
                    "final_amount_inr": "sum",
                    "quantity": "sum",
                    "customer_id": pd.Series.nunique
                }).rename(columns={
                    "final_amount_inr": "Total Revenue",
                    "quantity": "Units Sold",
                    "customer_id": "Unique Customers"
                }).sort_values(by="Total Revenue", ascending=False)

                # Step 5: Display metrics
                st.subheader("🎯 Discount Effectiveness Overview")
                st.dataframe(discount_metrics)


            # 💰 Revenue Impact of Pricing Strategies
            if "cost_inr" in transactions_df.columns:
                transactions_df["margin_inr"] = transactions_df["final_amount_inr"] - transactions_df["cost_inr"]
                strategy_df = transactions_df.groupby("original_price_inr")[["final_amount_inr", "margin_inr"]].sum().reset_index()
                strategy_df["margin_pct"] = (strategy_df["margin_inr"] / strategy_df["final_amount_inr"]) * 100

                st.subheader("💰 Revenue & Margin by Price Point")
                st.line_chart(strategy_df.set_index("original_price_inr")[["final_amount_inr", "margin_inr"]])

            # 🔍 Drill-down: Category-Level Pricing
            selected_category = st.selectbox("🔍 Select Category for Drill-down", sorted(products_df["category"].dropna().unique()))
            category_df = transactions_df[transactions_df["category"] == selected_category]
            cat_price_df = category_df.groupby("original_price_inr")["final_amount_inr"].sum().reset_index()

            st.subheader(f"📦 Revenue by Price Point in {selected_category}")
            st.bar_chart(cat_price_df.set_index("original_price_inr"))


        elif view_type == "segmentation":
            st.title("🧠 Customer Segmentation Dashboard")

            # ✅ Load data from SQL
            tx_df = load_table("transactions")
            products_df = load_table("products")
            cust_df = load_table("customers")

            tx_df["order_date"] = pd.to_datetime(tx_df["order_date"], errors="coerce")

            if "customer_id" in tx_df.columns:
                # 📊 RFM Analysis
                rfm = tx_df.groupby("customer_id").agg({
                    "order_date": lambda x: (pd.Timestamp.now() - x.max()).days,
                    "transaction_id": "count",
                    "final_amount_inr": "sum"
                }).rename(columns={
                    "order_date": "Recency",
                    "transaction_id": "Frequency",
                    "final_amount_inr": "Monetary"
                })

                st.subheader("📊 RFM Segmentation")
                st.dataframe(rfm)

                # 💎 Lifetime Value
                ltv = tx_df.groupby("customer_id")["final_amount_inr"].sum().sort_values(ascending=False)
                st.subheader("💎 Top Lifetime Value Customers")
                st.bar_chart(ltv.head(10))

                # 🧬 Behavioral Segmentation: Category diversity
                tx_df = tx_df.merge(products_df[["product_id", "category"]], on="product_id", how="left")
                behavior = tx_df.groupby("customer_id")["category"].nunique()
                st.subheader("🧬 Behavioral Segmentation (Category Diversity)")
                st.bar_chart(behavior)

                # 🌆 Geographic Segmentation
                city_counts = cust_df["customer_city"].value_counts().head(10)
                st.subheader("🌆 Top 10 Customer Cities")
                st.bar_chart(city_counts)

                state_counts = cust_df["customer_state"].value_counts()
                st.subheader("🗺️ Customer Distribution by State")
                st.bar_chart(state_counts)

                # 🏛️ Tier-Based Segmentation
                tier_counts = cust_df["customer_tier"].value_counts()
                st.subheader("🏛️ Customer Tier Distribution")
                st.bar_chart(tier_counts)

                # 💸 Spending Tier Segmentation
                if "customer_spending_tier" in cust_df.columns:
                    spending_tier_counts = cust_df["customer_spending_tier"].value_counts()
                    st.subheader("💸 Spending Tier Distribution")
                    st.bar_chart(spending_tier_counts)

                # 🎂 Age Group Segmentation
                if "customer_age_group" in cust_df.columns:
                    age_group_counts = cust_df["customer_age_group"].value_counts()
                    st.subheader("🎂 Age Group Distribution")
                    st.bar_chart(age_group_counts)

            else:
                st.warning("Customer ID not found in transactions data.")


        elif view_type == "journey":
            st.title("🚶 Customer Journey Analytics")

            # ✅ Load data from SQL
            df = load_table("transactions")

            # --- Ensure order_date is datetime ---
            if "order_date" in df.columns:
                df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

            # --- Loyalty Evolution ---
            if "customer_id" in df.columns and "order_date" in df.columns:
                loyalty = df.groupby("customer_id")["order_date"].agg(["min", "max", "count"]).reset_index()
                loyalty["duration_days"] = (loyalty["max"] - loyalty["min"]).dt.days
                loyalty = loyalty.rename(columns={"count": "total_orders"})

                st.subheader("📈 Customer Loyalty Evolution")
                st.dataframe(loyalty.sort_values("duration_days", ascending=False))

                # --- Loyalty Duration Histogram ---
                st.subheader("📉 Loyalty Duration Distribution")
                fig_duration = px.histogram(
                    loyalty,
                    x="duration_days",
                    nbins=30,
                    title="Customer Engagement Duration (Days)",
                    labels={"duration_days": "Days Active"},
                    color_discrete_sequence=["#FF9900"]
                )
                st.plotly_chart(fig_duration, use_container_width=True)

                # --- Total Orders Histogram ---
                st.subheader("🛒 Total Orders Distribution")
                fig_orders = px.histogram(
                    loyalty,
                    x="total_orders",
                    nbins=30,
                    title="Customer Order Frequency",
                    labels={"total_orders": "Number of Orders"},
                    color_discrete_sequence=["#3366CC"]
                )
                st.plotly_chart(fig_orders, use_container_width=True)

            else:
                st.warning("Insufficient data for loyalty analysis.")




        elif view_type == "prime":
            st.title("👑 Prime Membership Analytics")

            # ✅ Load transactions from SQL
            df = load_table("transactions")
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

            # ✅ Segment Prime vs Non-Prime
            prime_df = df[df["is_prime_member"] == True]
            non_prime_df = df[df["is_prime_member"] == False]

            # ✅ Revenue Comparison
            st.metric("Prime Revenue", f"₹{prime_df['final_amount_inr'].sum():,.0f}")
            st.metric("Non-Prime Revenue", f"₹{non_prime_df['final_amount_inr'].sum():,.0f}")

            # 🛍️ Avg Order Value Comparison
            st.subheader("🛍️ Avg Order Value Comparison")
            st.bar_chart({
                "Prime": prime_df["final_amount_inr"].mean(),
                "Non-Prime": non_prime_df["final_amount_inr"].mean()
            })

            # 📊 Retention Rate Comparison
            retention = df.groupby(["customer_id", "is_prime_member"])["order_date"].nunique().reset_index()
            st.subheader("📊 Retention Comparison")
            st.dataframe(retention)


        elif view_type == "retention":
            st.title("🔁 Customer Retention Dashboard")

            # ✅ Load transactions from SQL
            df = load_table("transactions")

            # --- Ensure order_date is datetime ---
            if "order_date" in df.columns:
                df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
                df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)

                # 📊 Unique Customers per Quarter
                retention_by_quarter = df.groupby(["order_quarter", "customer_id"]).size().reset_index(name="orders")
                retention_summary = retention_by_quarter.groupby("order_quarter")["customer_id"].nunique()
                st.subheader("📊 Unique Customers per Quarter")
                st.line_chart(retention_summary)

                # 🔄 New vs Repeat Customers
                first_order = df.groupby("customer_id")["order_date"].min().reset_index()
                df = df.merge(first_order, on="customer_id", suffixes=("", "_first"))
                df["is_new"] = df["order_date"].dt.to_period("Q") == df["order_date_first"].dt.to_period("Q")

                new_vs_repeat = df.groupby(["order_quarter", "is_new"])["customer_id"].nunique().reset_index()
                new_vs_repeat["type"] = new_vs_repeat["is_new"].map({True: "New", False: "Repeat"})

                st.subheader("🔄 New vs Repeat Customers by Quarter")
                fig = px.bar(
                    new_vs_repeat,
                    x="order_quarter",
                    y="customer_id",
                    color="type",
                    barmode="group",
                    labels={"customer_id": "Number of Customers", "order_quarter": "Quarter"},
                    title="Customer Retention Over Time",
                    color_discrete_sequence=["#FF9900", "#3366CC"]
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("The 'order_date' column is missing or not in datetime format.")




        elif view_type == "demographics":
            st.title("🌍 Demographics & Behavior Dashboard")

            # ✅ Load customer and transaction data from SQL
            df = load_table("customers")
            tx_df = load_table("transactions")

            # --- Age Group Distribution ---
            if "customer_age_group" in df.columns:
                st.subheader("🎂 Age Group Distribution")
                age_group_counts = df["customer_age_group"].value_counts().sort_index()
                st.bar_chart(age_group_counts)

            # --- Spending Patterns by Age Group ---
            if "customer_age_group" in df.columns and "final_amount_inr" in tx_df.columns:
                tx_df = tx_df.merge(df[["customer_id", "customer_age_group"]], on="customer_id", how="left")
                st.subheader("💸 Spending Patterns by Age Group")
                spend_by_age = tx_df.groupby("customer_age_group")["final_amount_inr"].mean()
                st.bar_chart(spend_by_age)

            # --- Age Group vs City Bar Chart ---
            if "customer_age_group" in df.columns and "customer_city" in df.columns:
                st.subheader("🏙️ Customer Distribution by Age Group and City")
                grouped = df.groupby(["customer_city", "customer_age_group"]).size().reset_index(name="count")

                fig = px.bar(
                    grouped,
                    x="customer_city",
                    y="count",
                    color="customer_age_group",
                    barmode="group",
                    title="Customer Age Group Distribution by City",
                    labels={
                        "count": "Number of Customers",
                        "customer_city": "City",
                        "customer_age_group": "Age Group"
                    },
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required columns 'customer_age_group' and 'customer_city' not found in the dataset.")





        elif view_type == "product_performance":
            st.title("📦 Product Performance Dashboard")

            # ✅ Load transactions and products data from SQL
            tx_df = load_table("transactions")
            products_df = load_table("products")

            # ✅ Merge product details into transaction data
            tx_df = tx_df.merge(products_df[["product_id", "product_rating", "brand", "subcategory"]], on="product_id", how="left")

            # ✅ Check for required columns
            required_cols = ["product_id", "final_amount_inr", "quantity", "product_rating", "return_status"]
            missing_cols = [col for col in required_cols if col not in tx_df.columns]

            if missing_cols:
                st.error(f"Missing columns after merge: {', '.join(missing_cols)}")
            else:
                # 📊 Aggregate product performance metrics
                perf_df = tx_df.groupby("product_id").agg({
                    "final_amount_inr": "sum",
                    "quantity": "sum",
                    "product_rating": "mean",
                    "return_status": lambda x: (x == "Returned").mean()
                }).sort_values(by="final_amount_inr", ascending=False)

                # 💰 Revenue by product
                st.subheader("💰 Top Products by Revenue")
                st.bar_chart(perf_df["final_amount_inr"].head(10))

                # 📦 Units sold
                st.subheader("📊 Units Sold")
                st.bar_chart(perf_df["quantity"].head(10))

                # ⭐ Average ratings
                st.subheader("⭐ Average Product Ratings")
                st.bar_chart(perf_df["product_rating"].head(10))

                # 🔁 Return rates
                st.subheader("🔁 Product Return Rates")
                st.bar_chart(perf_df["return_status"].head(10))





        elif view_type == "brand_analytics":
            st.title("📊 Brand Analytics Dashboard")

            # ✅ Load product and transaction data from SQL
            prod_df = load_table("products")
            tx_df = load_table("transactions")

            # ✅ Merge category and brand info into transactions
            tx_df = tx_df.merge(prod_df[["product_id", "brand", "category", "subcategory"]], on="product_id", how="left")

            # ✅ Filter valid entries
            tx_df = tx_df.dropna(subset=["brand", "category", "subcategory"])

            # 🧍 Customer Preferences: Most purchased brands
            brand_pref = tx_df["brand"].value_counts().sort_values(ascending=False)
            st.subheader("🧍 Customer Preference: Most Purchased Brands")
            st.bar_chart(brand_pref.head(10))

            # 📦 Category-wise Brand Share
            brand_cat_share = tx_df.groupby(["category", "brand"])["product_id"].count().unstack().fillna(0)
            brand_cat_share_pct = brand_cat_share.div(brand_cat_share.sum(axis=1), axis=0) * 100
            st.subheader("📦 Brand Share by Category (%)")
            st.dataframe(brand_cat_share_pct.round(2))

            # 📈 Market Share Evolution (Monthly)
            if "order_date" in tx_df.columns:
                tx_df["order_date"] = pd.to_datetime(tx_df["order_date"], errors="coerce")
                tx_df["order_year"] = tx_df["order_date"].dt.year
                tx_df["order_month"] = tx_df["order_date"].dt.month
                tx_df["period"] = tx_df["order_year"].astype(str) + "-" + tx_df["order_month"].astype(str)
                brand_monthly = tx_df.groupby(["period", "brand"])["product_id"].count().unstack().fillna(0)
                brand_monthly_pct = brand_monthly.div(brand_monthly.sum(axis=1), axis=0) * 100
                st.subheader("📈 Market Share Evolution Over Time")
                st.line_chart(brand_monthly_pct)

            # 🧠 Competitive Positioning: Brand Rating vs Volume
            if "product_rating" in prod_df.columns:
                rating_df = prod_df.dropna(subset=["brand", "product_rating"])
                brand_rating = rating_df.groupby("brand")["product_rating"].mean()
                brand_volume = tx_df["brand"].value_counts()
                brand_comp = pd.DataFrame({
                    "Avg Rating": brand_rating,
                    "Purchase Volume": brand_volume
                }).dropna()
                st.subheader("🧠 Competitive Positioning: Rating vs Purchase Volume")
                st.scatter_chart(brand_comp.reset_index()[["Avg Rating", "Purchase Volume"]])




        elif view_type == "inventory_optimization":
            st.title("📦 Inventory Optimization Dashboard")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' exists and is datetime
            if "order_date" not in time_df.columns:
                if "date" in time_df.columns:
                    time_df = time_df.rename(columns={"date": "order_date"})
                else:
                    st.warning("⚠️ 'order_date' column missing in time dimension data.")
                    st.stop()

            transactions_df = transactions_df.dropna(subset=["order_date"])
            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")

            # ✅ Enrich transactions with time dimensions
            transactions_df["order_year"] = transactions_df["order_date"].dt.year
            transactions_df["order_month"] = transactions_df["order_date"].dt.month
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            # ✅ Merge product info (excluding stock_qty)
            merged_df = transactions_df.merge(products_df[["product_id", "category", "subcategory", "brand"]], on="product_id", how="left")

            # 📊 Product Demand Patterns
            demand_df = merged_df.groupby("product_id")["transaction_id"].count().reset_index()
            demand_df = demand_df.merge(products_df[["product_id", "category", "brand"]], on="product_id", how="left")
            top_demand = demand_df.sort_values(by="transaction_id", ascending=False).head(10)

            st.subheader("📊 Top 10 Products by Demand")
            st.dataframe(top_demand)

            # 🌤️ Seasonal Trends
            seasonal_df = merged_df.groupby("order_month")["transaction_id"].count()
            st.subheader("🌤️ Monthly Demand Seasonality")
            if not seasonal_df.empty:
                st.bar_chart(seasonal_df)
            else:
                st.warning("No seasonal demand data available.")

            # 🔮 Demand Forecasting (Simple Linear Regression)
            from sklearn.linear_model import LinearRegression
            import numpy as np

            forecast_df = merged_df.groupby(["order_year", "order_month"])["transaction_id"].count().reset_index()
            forecast_df["date"] = pd.to_datetime(forecast_df["order_year"].astype(str) + "-" + forecast_df["order_month"].astype(str) + "-01")
            forecast_df["timestamp"] = forecast_df["date"].view("int64") // 10**9

            X = forecast_df[["timestamp"]]
            y = forecast_df["transaction_id"]

            model = LinearRegression()
            model.fit(X, y)

            future_dates = pd.date_range(start=forecast_df["date"].max(), periods=6, freq="M")
            future_ts = (future_dates.view("int64") // 10**9).reshape(-1, 1)
            future_preds = model.predict(future_ts)

            forecast_plot = pd.Series(future_preds, index=future_dates)
            st.subheader("🔮 Forecasted Product Demand")
            if not forecast_plot.empty:
                st.line_chart(forecast_plot)
            else:
                st.warning("Forecast data is empty.")




        elif view_type == "rating_review":
            st.title("⭐ Product Rating & Review Dashboard")

            # ✅ Load product and transaction data from SQL
            products_df = load_table("products")
            tx_df = load_table("transactions")

            # ✅ Check for required column
            if "product_rating" not in products_df.columns:
                st.error("Missing 'product_rating' column in products data.")
            else:
                # ✅ Merge average discounted price and sales volume from transactions
                price_df = tx_df.groupby("product_id")[["discounted_price_inr", "quantity", "final_amount_inr", "customer_rating"]].mean().reset_index()
                products_df = products_df.merge(price_df, on="product_id", how="left")

                # 📊 Rating Distribution
                rating_dist = products_df["product_rating"].value_counts().sort_index()
                st.subheader("📊 Rating Distribution")
                st.bar_chart(rating_dist)

                # 📈 Rating vs Avg Discounted Price
                if "discounted_price_inr" in products_df.columns:
                    rating_price_corr = products_df.groupby("product_rating")["discounted_price_inr"].mean()
                    st.subheader("📈 Rating vs Avg Discounted Price")
                    st.line_chart(rating_price_corr)

                # 💸 Rating vs Sales Volume
                if "quantity" in products_df.columns:
                    rating_sales_corr = products_df.groupby("product_rating")["quantity"].mean()
                    st.subheader("💸 Rating vs Avg Sales Volume")
                    st.line_chart(rating_sales_corr)

                # 🧪 Product Quality Insights: Low-rated products with high returns
                if "return_status" in tx_df.columns:
                    return_df = tx_df[tx_df["return_status"] == "Returned"]
                    return_df = return_df.merge(products_df[["product_id", "product_rating"]], on="product_id", how="left")
                    quality_flags = return_df[return_df["product_rating"] < 3].groupby("product_id")["product_rating"].count().sort_values(ascending=False)
                    st.subheader("🧪 Low-Rated Returned Products")
                    st.dataframe(quality_flags.head(10).reset_index().rename(columns={"product_rating": "low_rating_return_count"}))

                # 🧠 Review Sentiment (if available)
                if "review_sentiment" in products_df.columns:
                    sentiment_dist = products_df["review_sentiment"].value_counts(normalize=True).sort_values(ascending=False) * 100
                    st.subheader("🧠 Review Sentiment Distribution (%)")
                    st.bar_chart(sentiment_dist)

                # 🧠 Rating vs Customer Rating (from transactions)
                if "customer_rating" in products_df.columns:
                    rating_corr = products_df.groupby("product_rating")["customer_rating"].mean()
                    st.subheader("🧠 Product Rating vs Customer Rating")
                    st.line_chart(rating_corr)




        elif view_type == "new_launch":
            st.title("🚀 New Product Launch Dashboard")

            # ✅ Load transactions and products data from SQL
            tx_df = load_table("transactions")
            products_df = load_table("products")

            # ✅ Ensure order_year exists
            tx_df["order_date"] = pd.to_datetime(tx_df["order_date"], errors="coerce")
            tx_df["order_year"] = tx_df["order_date"].dt.year

            # ✅ Filter for latest year launches
            latest_year = tx_df["order_year"].max()
            launch_df = tx_df[tx_df["order_year"] == latest_year]

            # ✅ Merge product details
            launch_df = launch_df.merge(products_df[["product_id", "product_rating"]], on="product_id", how="left")

            # ✅ Check for required columns
            required_cols = ["product_id", "final_amount_inr", "quantity", "product_rating", "return_status"]
            missing_cols = [col for col in required_cols if col not in launch_df.columns]

            if missing_cols:
                st.error(f"Missing columns in merged launch data: {', '.join(missing_cols)}")
            else:
                # 📊 Aggregate launch performance
                launch_perf = launch_df.groupby("product_id").agg({
                    "final_amount_inr": "sum",
                    "quantity": "sum",
                    "product_rating": "mean",
                    "return_status": lambda x: (x == "Returned").mean()
                }).sort_values(by="final_amount_inr", ascending=False)

                # 📈 Revenue
                st.subheader("📈 Launch Revenue")
                st.bar_chart(launch_perf["final_amount_inr"].head(10))

                # 📦 Units Sold
                st.subheader("📦 Launch Units Sold")
                st.bar_chart(launch_perf["quantity"].head(10))

                # ⭐ Ratings
                st.subheader("⭐ Launch Ratings")
                st.bar_chart(launch_perf["product_rating"].head(10))

                # 🔁 Return Rates
                st.subheader("🔁 Launch Return Rates")
                st.bar_chart(launch_perf["return_status"].head(10))



        elif view_type == "delivery_performance":
            st.title("🚚 Delivery Performance Dashboard")

            # ✅ Load transactions and customers data from SQL
            tx_df = load_table("transactions")
            cust_df = load_table("customers")

            # ✅ Merge customer location info
            tx_df = tx_df.merge(cust_df[["customer_id", "customer_state"]], on="customer_id", how="left")

            # ✅ Check for required columns
            required_cols = ["delivery_days", "customer_state"]
            missing_cols = [col for col in required_cols if col not in tx_df.columns]

            if missing_cols:
                st.error(f"Missing columns in merged data: {', '.join(missing_cols)}")
            else:
                st.subheader("⏱️ Delivery Time Distribution")
                st.bar_chart(tx_df["delivery_days"].value_counts().sort_index())

                geo_perf = tx_df.groupby("customer_state")["delivery_days"].mean().sort_values()
                st.subheader("📍 Avg Delivery Days by State")
                st.bar_chart(geo_perf)

                on_time_rate = (tx_df["delivery_days"] <= 3).mean()
                st.metric("✅ On-Time Delivery Rate", f"{on_time_rate:.2%}")

                if "delivery_type" in tx_df.columns:
                    delivery_type_perf = tx_df.groupby("delivery_type")["delivery_days"].mean().sort_values()
                    st.subheader("⚙️ Avg Delivery Days by Delivery Type")
                    st.bar_chart(delivery_type_perf)

                delayed_states = geo_perf[geo_perf > 7]
                if not delayed_states.empty:
                    st.warning("🚩 States with Avg Delivery > 7 Days")
                    st.dataframe(delayed_states)

                if "order_date" in tx_df.columns:
                    tx_df["order_date"] = pd.to_datetime(tx_df["order_date"], errors="coerce")
                    tx_df["order_year"] = tx_df["order_date"].dt.year
                    tx_df["order_month"] = tx_df["order_date"].dt.month
                    monthly_delivery = tx_df.groupby(["order_year", "order_month"])["delivery_days"].mean().reset_index()
                    monthly_delivery["period"] = monthly_delivery["order_year"].astype(str) + "-" + monthly_delivery["order_month"].astype(str)
                    monthly_delivery = monthly_delivery.set_index("period")
                    st.subheader("📈 Avg Delivery Days Over Time")
                    st.line_chart(monthly_delivery["delivery_days"])



        elif view_type == "payment_analytics":
            st.title("💳 Payment Analytics Dashboard")

            # ✅ Load data from SQL
            customers_df = load_table("customers")
            products_df = load_table("products")
            transactions_df = load_table("transactions")
            time_df = load_table("time_dimension")

            # ✅ Ensure 'order_date' is datetime
            if "order_date" not in time_df.columns and "date" in time_df.columns:
                time_df = time_df.rename(columns={"date": "order_date"})

            transactions_df["order_date"] = pd.to_datetime(transactions_df["order_date"], errors="coerce")
            time_df["order_date"] = pd.to_datetime(time_df["order_date"], errors="coerce")

            transactions_df["order_year"] = transactions_df["order_date"].dt.year
            transactions_df["order_month"] = transactions_df["order_date"].dt.month
            transactions_df = transactions_df.merge(time_df, on="order_date", how="left")

            method_pref = transactions_df["payment_method"].value_counts()
            st.subheader("💰 Payment Method Preferences")
            st.bar_chart(method_pref)

            if "payment_status" in transactions_df.columns:
                success_rate = transactions_df["payment_status"].value_counts(normalize=True) * 100
                st.subheader("✅ Transaction Success Rate")
                st.metric("Success Rate", f"{success_rate.get('Success', 0):.2f}%")
                st.metric("Failure Rate", f"{success_rate.get('Failed', 0):.2f}%")

            trend_df = transactions_df.groupby(["order_year", "order_month", "payment_method"])["final_amount_inr"].sum().reset_index()
            trend_df["date"] = pd.to_datetime(trend_df["order_year"].astype(str) + "-" + trend_df["order_month"].astype(str) + "-01")
            pivot_trend = trend_df.pivot(index="date", columns="payment_method", values="final_amount_inr").fillna(0)
            st.subheader("📈 Payment Method Revenue Over Time")
            st.line_chart(pivot_trend)

            if "payment_gateway" in transactions_df.columns:
                gateway_perf = transactions_df.groupby("payment_gateway")["final_amount_inr"].sum().sort_values(ascending=False)
                gateway_success = transactions_df[transactions_df["payment_status"] == "Success"].groupby("payment_gateway")["order_id"].count()
                gateway_total = transactions_df.groupby("payment_gateway")["order_id"].count()
                gateway_success_rate = (gateway_success / gateway_total * 100).fillna(0)

                st.subheader("🤝 Financial Partner Performance")
                partner_df = pd.DataFrame({
                    "Total Revenue": gateway_perf,
                    "Success Rate (%)": gateway_success_rate
                }).sort_values(by="Total Revenue", ascending=False)
                st.dataframe(partner_df)

            if "payment_method" in transactions_df.columns:
                selected_method = st.selectbox("🔍 Select Payment Method", sorted(transactions_df["payment_method"].dropna().unique()))
                method_df = transactions_df[transactions_df["payment_method"] == selected_method]
                method_trend = method_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
                method_trend["date"] = pd.to_datetime(method_trend["order_year"].astype(str) + "-" + method_trend["order_month"].astype(str) + "-01")
                method_trend.set_index("date", inplace=True)
                st.subheader(f"📦 Revenue Trend for {selected_method}")
                st.line_chart(method_trend["final_amount_inr"])


        elif view_type == "return_cancellation":
            st.title("🔁 Return & Cancellation Dashboard")

            # ✅ Load transactions and products data from SQL
            tx_df = load_table("transactions")
            products_df = load_table("products")

            # ✅ Merge product category info
            tx_df = tx_df.merge(products_df[["product_id", "category"]], on="product_id", how="left")

            required_cols = ["return_status", "final_amount_inr", "category"]
            missing_cols = [col for col in required_cols if col not in tx_df.columns]

            if missing_cols:
                st.error(f"Missing columns in merged data: {', '.join(missing_cols)}")
            else:
                return_rate = (tx_df["return_status"] == "Returned").mean()
                st.metric("🔁 Overall Return Rate", f"{return_rate:.2%}")

                category_returns = tx_df.groupby("category")["return_status"].apply(lambda x: (x == "Returned").mean())
                st.subheader("📦 Return Rate by Category")
                st.bar_chart(category_returns)

                return_cost = tx_df[tx_df["return_status"] == "Returned"]["final_amount_inr"].sum()
                st.metric("💸 Cost Impact of Returns", f"₹{return_cost:,.0f}")

                category_cost = tx_df[tx_df["return_status"] == "Returned"].groupby("category")["final_amount_inr"].sum().sort_values(ascending=False)
                st.subheader("📊 Return Cost by Category")
                st.bar_chart(category_cost.head(10))

                high_risk = category_returns[category_returns > 0.15]
                if not high_risk.empty:
                    st.warning("🚩 Categories with Return Rate > 15%")
                    st.dataframe(high_risk)



        elif view_type == "customer_service":
            st.title("📞 Customer Service Dashboard")

            # ✅ Load transactions and customers data from SQL
            tx_df = load_table("transactions")
            cust_df = load_table("customers")

            # ✅ Merge customer tier info into transactions
            tx_df = tx_df.merge(cust_df[["customer_id", "customer_tier"]], on="customer_id", how="left")

            # ✅ Check for required columns
            required_cols = ["customer_rating", "flag_for_review", "customer_tier"]
            missing_cols = [col for col in required_cols if col not in tx_df.columns]

            if missing_cols:
                st.error(f"Missing columns in merged data: {', '.join(missing_cols)}")
            else:
                # 😊 Customer Satisfaction Ratings
                st.subheader("😊 Customer Satisfaction (Ratings)")
                st.bar_chart(tx_df["customer_rating"].value_counts().sort_index())

                # ⚠️ Flagged Orders
                flagged = tx_df[tx_df["flag_for_review"] == True]
                st.metric("⚠️ Flagged Orders for Review", len(flagged))

                # 🧾 Complaints by Tier
                st.subheader("🧾 Complaints by Tier")
                complaints_by_tier = flagged["customer_tier"].value_counts()
                st.bar_chart(complaints_by_tier)



        elif view_type == "supply_chain":
            st.title("🔗 Supply Chain Dashboard")

            # ✅ Load transactions, customers, and products data from SQL
            tx_df = load_table("transactions")
            cust_df = load_table("customers")
            prod_df = load_table("products")

            # ✅ Merge customer location and product category into transactions
            tx_df = tx_df.merge(cust_df[["customer_id", "customer_state"]], on="customer_id", how="left")
            tx_df = tx_df.merge(prod_df[["product_id", "category"]], on="product_id", how="left")

            # ✅ Check for required columns
            required_cols = ["delivery_days", "customer_state", "category", "final_amount_inr"]
            missing_cols = [col for col in required_cols if col not in tx_df.columns]

            if missing_cols:
                st.error(f"Missing columns in merged data: {', '.join(missing_cols)}")
            else:
                # 📍 Delivery Reliability by State
                delivery_reliability = tx_df.groupby("customer_state")["delivery_days"].apply(lambda x: (x <= 3).mean())
                st.subheader("📍 Delivery Reliability by State")
                st.bar_chart(delivery_reliability)

                # 💰 Average Cost by Category
                cost_by_category = tx_df.groupby("category")["final_amount_inr"].mean().sort_values(ascending=False)
                st.subheader("💰 Avg Cost by Category")
                st.bar_chart(cost_by_category)

                # 🚚 Delivery Performance by Category
                delivery_by_category = tx_df.groupby("category")["delivery_days"].mean().sort_values()
                st.subheader("🚚 Avg Delivery Days by Category")
                st.bar_chart(delivery_by_category)

                # 🚩 Flag Categories with High Delivery Time
                slow_categories = delivery_by_category[delivery_by_category > 7]
                if not slow_categories.empty:
                    st.warning("🚩 Categories with Avg Delivery > 7 Days")
                    st.dataframe(slow_categories)

                # 📦 Volume by Category
                if "quantity" in tx_df.columns:
                    volume_by_category = tx_df.groupby("category")["quantity"].sum().sort_values(ascending=False)
                    st.subheader("📦 Total Units Sold by Category")
                    st.bar_chart(volume_by_category.head(10))




        elif view_type == "predictive_analytics":
            st.title("📈 Predictive Analytics Dashboard")

            tx_df = load_table("transactions")
            cust_df = load_table("customers")
            prod_df = load_table("products")

            tx_df["order_date"] = pd.to_datetime(tx_df["order_date"], errors="coerce")
            tx_df["order_year"] = tx_df["order_date"].dt.year
            tx_df["order_month"] = tx_df["order_date"].dt.month

            # 📊 Sales Forecasting (Monthly Aggregation)
            monthly_sales = tx_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
            monthly_sales["period"] = monthly_sales["order_year"].astype(str) + "-" + monthly_sales["order_month"].astype(str)
            monthly_sales = monthly_sales.set_index("period")

            st.subheader("📊 Historical Sales Trend")
            st.line_chart(monthly_sales["final_amount_inr"])

            # 🔮 Placeholder for forecasting
            st.info("🔮 Forecasting model placeholder — integrate ARIMA, Prophet, or ML here.")

            # ⚠️ Churn Prediction
            last_order = tx_df.groupby("customer_id")["order_date"].max()
            churn_days = (pd.Timestamp.now() - pd.to_datetime(last_order)).dt.days
            churn_rate = (churn_days > 90).mean()
            st.subheader("⚠️ Customer Churn Prediction")
            st.metric("Estimated Churn Rate", f"{churn_rate:.2%}")

            # 📦 Demand Planning by Category
            tx_df = tx_df.merge(prod_df[["product_id", "category"]], on="product_id", how="left")
            demand_by_cat = tx_df.groupby(["order_month", "category"])["quantity"].sum().unstack().fillna(0)
            st.subheader("📦 Monthly Demand by Category")
            st.line_chart(demand_by_cat)

            # 🧮 Business Scenario Simulation
            st.subheader("🧮 Scenario Analysis: Revenue Impact")
            uplift = st.slider("Expected % Increase in Orders", min_value=0, max_value=100, value=10)
            projected_revenue = tx_df["final_amount_inr"].sum() * (1 + uplift / 100)
            st.metric("Projected Revenue", f"₹{projected_revenue:,.0f}")


        elif view_type == "market_intelligence":
            st.title("🧠 Market Intelligence Dashboard")

            prod_df = load_table("products")
            tx_df = load_table("transactions")

            pricing_df = tx_df.groupby("product_id")[["original_price_inr", "discounted_price_inr"]].mean().reset_index()
            prod_df = prod_df.merge(pricing_df, on="product_id", how="left")

            rating_cols = ["brand", "category", "subcategory", "product_rating"]
            missing_rating = [col for col in rating_cols if col not in prod_df.columns]
            if missing_rating:
                st.warning(f"⚠️ Missing columns in product data: {', '.join(missing_rating)}")

            if "original_price_inr" not in prod_df.columns or "discounted_price_inr" not in prod_df.columns:
                st.warning("⚠️ Pricing columns missing after merge: 'original_price_inr' or 'discounted_price_inr'.")
            else:
                if "brand" in prod_df.columns:
                    brand_price = prod_df.groupby("brand")[["original_price_inr", "discounted_price_inr"]].mean().sort_values(by="discounted_price_inr", ascending=False)
                    st.subheader("🏷️ Avg Pricing by Brand")
                    st.bar_chart(brand_price["discounted_price_inr"].head(10))

                prod_df["discount_pct"] = ((prod_df["original_price_inr"] - prod_df["discounted_price_inr"]) / prod_df["original_price_inr"]) * 100
                if "subcategory" in prod_df.columns:
                    subcat_discount = prod_df.groupby("subcategory")["discount_pct"].mean().sort_values(ascending=False)
                    st.subheader("💸 Avg Discount % by Subcategory")
                    st.bar_chart(subcat_discount.head(10))

                if "product_rating" in prod_df.columns:
                    st.subheader("🧭 Strategic Positioning: Price vs Rating")
                    st.scatter_chart(prod_df[["discounted_price_inr", "product_rating"]].dropna())

            if "category" in prod_df.columns and "product_rating" in prod_df.columns:
                cat_rating = prod_df.groupby("category")["product_rating"].mean().sort_values(ascending=False)
                st.subheader("📈 Avg Product Rating by Category")
                st.bar_chart(cat_rating)



        elif view_type == "cross_upsell":
            st.title("🔗 Cross-sell & Upsell Dashboard")

            tx_df = load_table("transactions")
            prod_df = load_table("products")

            tx_df = tx_df.merge(prod_df[["product_id", "category", "brand"]], on="product_id", how="left")

            # 🔗 Product Associations
            st.subheader("🔗 Product Associations")
            group_cols = ["customer_id"]
            if "order_id" in tx_df.columns:
                group_cols.append("order_id")

            basket_df = tx_df.groupby(group_cols)["product_id"].apply(list)
            pair_counts = {}

            for items in basket_df:
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        pair = tuple(sorted([items[i], items[j]]))
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1

            assoc_df = pd.DataFrame([
                {"Product Pair": f"{p[0]} & {p[1]}", "Count": c}
                for p, c in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
            ])

            st.write("📦 Top Co-purchased Product Pairs")
            st.dataframe(assoc_df.head(10))

            # ✅ Recommendation Effectiveness
            st.subheader("✅ Recommendation Effectiveness")
            repeat_df = tx_df.groupby("product_id")["customer_id"].nunique()
            st.bar_chart(repeat_df.sort_values(ascending=False).head(10))

            # 🧺 Bundle Opportunities
            st.subheader("🧺 Bundle Opportunities")
            bundle_df = tx_df.groupby("customer_id")["category"].nunique()
            st.bar_chart(bundle_df.head(20))

            # 💰 Revenue Optimization
            st.subheader("💰 Revenue by Bundle Category")
            bundle_rev = tx_df.groupby("category")["final_amount_inr"].sum().sort_values(ascending=False)
            st.bar_chart(bundle_rev.head(10))



        elif view_type == "seasonal_planning":
            st.title("📅 Seasonal Planning Dashboard")

            # ✅ Load transactions and products data from SQL
            tx_df = load_table("transactions")
            prod_df = load_table("products")

            tx_df = tx_df.merge(prod_df[["product_id", "category"]], on="product_id", how="left")
            tx_df["order_month"] = pd.to_datetime(tx_df["order_date"], errors="coerce").dt.month

            # 📦 Inventory Planning: Monthly demand by category
            monthly_demand = tx_df.groupby(["order_month", "category"])["quantity"].sum().unstack().fillna(0)
            st.subheader("📦 Monthly Demand by Category")
            st.line_chart(monthly_demand)

            # 🎉 Promotional Calendar: Festival sales performance
            if "is_festival_sale" in tx_df.columns and "festival_name" in tx_df.columns:
                festival_perf = tx_df[tx_df["is_festival_sale"] == True].groupby("festival_name")["final_amount_inr"].sum().sort_values(ascending=False)
                st.subheader("🎉 Festival Sale Performance")
                st.bar_chart(festival_perf)
            else:
                st.warning("Festival sale columns not found in dataset.")

            # 🧑‍🤝‍🧑 Resource Allocation: Peak demand months
            peak_months = tx_df.groupby("order_month")["quantity"].sum().sort_values(ascending=False)
            st.subheader("🧑‍🤝‍🧑 Resource Allocation: Peak Demand Months")
            st.bar_chart(peak_months)

            # 📈 Seasonal Optimization: Revenue vs Quantity
            seasonal_rev = tx_df.groupby("order_month")[["final_amount_inr", "quantity"]].sum()
            st.subheader("📈 Revenue vs Quantity by Month")
            st.line_chart(seasonal_rev)



        elif view_type == "bi_command_center":
            st.title("🧠 Business Intelligence Command Center")

            # ✅ Load datasets from SQL
            tx_df = load_table("transactions")
            cust_df = load_table("customers")
            prod_df = load_table("products")

            tx_df = tx_df.merge(prod_df[["product_id", "category", "brand"]], on="product_id", how="left")
            tx_df = tx_df.merge(cust_df[["customer_id", "customer_tier", "customer_state"]], on="customer_id", how="left")

            # ✅ Check for required columns
            required_cols = ["final_amount_inr", "order_date", "return_status", "delivery_days"]
            missing_cols = [col for col in required_cols if col not in tx_df.columns]

            if missing_cols:
                st.error(f"Missing columns in transactions data: {', '.join(missing_cols)}")
            else:
                tx_df["order_date"] = pd.to_datetime(tx_df["order_date"], errors="coerce")
                tx_df["order_year"] = tx_df["order_date"].dt.year
                tx_df["order_month"] = tx_df["order_date"].dt.month

                # 📊 Key Metrics
                st.subheader("📊 Key Business Metrics")
                total_revenue = tx_df["final_amount_inr"].sum()
                total_orders = len(tx_df)
                avg_order_value = total_revenue / total_orders if total_orders else 0

                last_order = tx_df.groupby("customer_id")["order_date"].max()
                churn_days = (pd.Timestamp.now() - pd.to_datetime(last_order)).dt.days
                churn_rate = (churn_days > 90).mean()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
                col2.metric("📦 Total Orders", total_orders)
                col3.metric("🧾 Avg Order Value", f"₹{avg_order_value:,.0f}")
                col4.metric("⚠️ Churn Rate", f"{churn_rate:.2%}")

                # 🚨 Automated Alerts
                st.subheader("🚨 Automated Alerts")
                high_return_rate = (tx_df["return_status"] == "Returned").mean()
                delayed_deliveries = (tx_df["delivery_days"] > 7).mean()

                if high_return_rate > 0.1:
                    st.error(f"⚠️ High Return Rate: {high_return_rate:.2%}")
                if delayed_deliveries > 0.15:
                    st.warning(f"⏱️ Delivery Delays Above Threshold: {delayed_deliveries:.2%}")

                # 📈 Performance Monitoring
                if "order_year" in tx_df.columns and "order_month" in tx_df.columns:
                    st.subheader("📈 Monthly Revenue Trend")
                    monthly_rev = tx_df.groupby(["order_year", "order_month"])["final_amount_inr"].sum().reset_index()
                    monthly_rev["period"] = monthly_rev["order_year"].astype(str) + "-" + monthly_rev["order_month"].astype(str)
                    monthly_rev = monthly_rev.set_index("period")
                    st.line_chart(monthly_rev["final_amount_inr"])
                else:
                    st.warning("Missing 'order_year' or 'order_month' columns for trend analysis.")

                # 🧠 Strategic Decision Support
                st.subheader("🧠 Strategic Scenario Simulation")
                uplift = st.slider("Expected % Increase in Orders", min_value=0, max_value=100, value=10)
                projected_revenue = total_revenue * (1 + uplift / 100)
                st.metric("📈 Projected Revenue", f"₹{projected_revenue:,.0f}")