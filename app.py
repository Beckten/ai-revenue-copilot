import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from openai import OpenAI

st.set_page_config(page_title="AI Revenue Copilot", layout="wide")

# -------------------------------------------------
# Styling
# -------------------------------------------------

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.section-card {
    background-color: #111827;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid #374151;
}

.insight-card {
    background-color: #0f172a;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #334155;
    margin-bottom: 14px;
}

.insight-card h4 {
    color: white;
}

.insight-card p {
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

segment_color_map = {
    "VIP": "#00E5A8",
    "High Value": "#FF6B6B",
    "Frequent Buyer": "#4D8BFF",
    "Low Value": "#7A8599"
}

risk_color_map = {
    "High Risk": "#FF4B4B",
    "Medium Risk": "#F59E0B",
    "Low Risk": "#22C55E"
}

# -------------------------------------------------
# Demo Dataset
# -------------------------------------------------

np.random.seed(42)

customers = [f"CUST-{i}" for i in range(1, 501)]
dates = pd.date_range(start="2024-01-01", periods=365)

df = pd.DataFrame({
    "CustomerID": np.random.choice(customers, 5000),
    "InvoiceDate": np.random.choice(dates, 5000),
    "Quantity": np.random.randint(1, 10, 5000),
    "UnitPrice": np.random.uniform(10, 250, 5000).round(2),
    "InvoiceNo": np.random.randint(10000, 99999, 5000)
})

st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["InvoiceDate"].min(), df["InvoiceDate"].max()]
)

df = df[
    (df["InvoiceDate"] >= pd.to_datetime(date_range[0])) &
    (df["InvoiceDate"] <= pd.to_datetime(date_range[1]))
]

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# -------------------------------------------------
# Customer Aggregation
# -------------------------------------------------

customer_df = df.groupby("CustomerID").agg(
    total_revenue=("Revenue", "sum"),
    order_count=("InvoiceNo", "nunique"),
    last_purchase=("InvoiceDate", "max"),
    avg_order_value=("Revenue", "mean")
).reset_index()

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
customer_df["recency_days"] = (snapshot_date - customer_df["last_purchase"]).dt.days

customer_df["churn_risk"] = np.where(
    customer_df["recency_days"] > 120,
    "High Risk",
    np.where(customer_df["recency_days"] > 60, "Medium Risk", "Low Risk")
)

# -------------------------------------------------
# Customer Segmentation
# -------------------------------------------------

revenue_75 = customer_df["total_revenue"].quantile(0.75)
revenue_90 = customer_df["total_revenue"].quantile(0.90)
orders_75 = customer_df["order_count"].quantile(0.75)

def segment_customer(row):
    if row["total_revenue"] >= revenue_90 and row["order_count"] >= orders_75:
        return "VIP"
    elif row["total_revenue"] >= revenue_75:
        return "High Value"
    elif row["order_count"] >= orders_75:
        return "Frequent Buyer"
    else:
        return "Low Value"

customer_df["segment"] = customer_df.apply(segment_customer, axis=1)

customer_df["priority_score"] = (
    (customer_df["total_revenue"] / customer_df["total_revenue"].max() * 50) +
    (customer_df["order_count"] / customer_df["order_count"].max() * 30) +
    ((1 - customer_df["recency_days"] / customer_df["recency_days"].max()) * 20)
).round(1)

# -------------------------------------------------
# ML Churn Probability Model
# -------------------------------------------------

features = ["total_revenue", "order_count", "recency_days"]
X = customer_df[features]
y = (customer_df["recency_days"] > 90).astype(int)

churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
churn_model.fit(X, y)

customer_df["churn_probability"] = churn_model.predict_proba(X)[:, 1]

customer_df["ai_priority_score"] = (
    customer_df["churn_probability"] * 50 +
    (customer_df["total_revenue"] / customer_df["total_revenue"].max()) * 30 +
    (customer_df["order_count"] / customer_df["order_count"].max()) * 20
).round(1)

# -------------------------------------------------
# Sidebar Account Filters
# -------------------------------------------------

st.sidebar.header("Account Filters")

selected_segments = st.sidebar.multiselect(
    "Customer Segment",
    options=sorted(customer_df["segment"].unique()),
    default=sorted(customer_df["segment"].unique())
)

selected_risks = st.sidebar.multiselect(
    "Churn Risk",
    options=sorted(customer_df["churn_risk"].unique()),
    default=sorted(customer_df["churn_risk"].unique())
)

filtered_customer_df = customer_df[
    customer_df["segment"].isin(selected_segments) &
    customer_df["churn_risk"].isin(selected_risks)
]

priority_accounts = filtered_customer_df.sort_values(
    "ai_priority_score", ascending=False
).head(15)

# -------------------------------------------------
# KPI Calculations
# -------------------------------------------------

total_revenue = df["Revenue"].sum()
total_customers = df["CustomerID"].nunique()
total_orders = df["InvoiceNo"].nunique()
avg_order_value = total_revenue / total_orders if total_orders != 0 else 0

# -------------------------------------------------
# Hero Section
# -------------------------------------------------

st.markdown("""
<div style="padding: 20px 0 10px 0;">
    <h1 style="font-size:56px; font-weight:700; margin-bottom:10px;">
        AI Revenue Copilot
    </h1>
    <p style="font-size:20px; color:#9CA3AF; max-width:850px;">
        Identify at-risk revenue, prioritize high-impact accounts, and recommend next-best sales actions.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Executive Overview
# -------------------------------------------------

st.header("Executive Revenue Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Customers", f"{total_customers:,}")
col3.metric("Orders", f"{total_orders:,}")
col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")

st.markdown("""
### Key Takeaways

- 📉 **Churn Risk Exists:** A portion of customers have not purchased recently, creating immediate revenue risk.
- 💰 **Revenue Concentration:** A smaller group of customers drives a meaningful share of total revenue.
- 🎯 **Action Opportunity:** Prioritized accounts highlight where sales teams should focus today.
""")

# -------------------------------------------------
# AI-Style Revenue Recommendations
# -------------------------------------------------

st.header("AI-Style Revenue Recommendations")

high_risk_count = filtered_customer_df[filtered_customer_df["churn_risk"] == "High Risk"].shape[0]
high_value_count = filtered_customer_df[filtered_customer_df["segment"].isin(["VIP", "High Value"])].shape[0]
top_priority = priority_accounts.iloc[0]

st.markdown(f"""
<div class="insight-card">
<h4>1. Protect at-risk revenue</h4>
<p>{high_risk_count} customers show high churn risk. Prioritize win-back outreach for accounts with long inactivity periods.</p>
</div>

<div class="insight-card">
<h4>2. Expand high-value accounts</h4>
<p>{high_value_count} customers are classified as VIP or High Value. These accounts should be targeted for upsell and expansion conversations.</p>
</div>

<div class="insight-card">
<h4>3. Start with the highest-priority account</h4>
<p>{top_priority["CustomerID"]} has the strongest combined revenue, frequency, recency, and churn probability profile with an AI priority score of {top_priority["ai_priority_score"]}.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Account Prioritization Engine
# -------------------------------------------------

st.header("Account Prioritization Engine")

st.subheader("Highest Priority Accounts")

st.dataframe(
    priority_accounts[
        [
            "CustomerID",
            "ai_priority_score",
            "churn_probability",
            "total_revenue",
            "order_count",
            "recency_days",
            "segment",
            "churn_risk"
        ]
    ],
    use_container_width=True
)

csv = priority_accounts.to_csv(index=False)

st.download_button(
    label="Download Priority Accounts",
    data=csv,
    file_name="priority_accounts.csv",
    mime="text/csv"
)

# -------------------------------------------------
# Highest Priority Account
# -------------------------------------------------

top_customer = priority_accounts.iloc[0]

st.subheader("🚨 Highest Priority Account")

col1, col2, col3 = st.columns(3)

col1.metric("Customer", top_customer["CustomerID"])
col2.metric("Priority Score", round(top_customer["ai_priority_score"], 1))
col3.metric("Churn Risk", top_customer["churn_risk"])

st.markdown(f"""
**Why this matters:**
- High revenue: ${top_customer['total_revenue']:,.0f}
- High inactivity: {int(top_customer['recency_days'])} days
- Elevated churn probability: {top_customer['churn_probability']:.2f}

👉 **Recommended Action:** Immediate outreach with retention or expansion incentive.
""")

# -------------------------------------------------
# AI Copilot Chat
# -------------------------------------------------

# -------------------------------------------------
# AI Copilot (SAFE DEMO VERSION)
# -------------------------------------------------

st.header("AI Revenue Copilot Chat")

USE_REAL_AI = False  # 🔒 

copilot_question = st.text_input(
    "Ask the copilot a revenue question",
    placeholder="Example: Who should I call today and why?"
)

if copilot_question:

    # Get top accounts context
    top_accounts = priority_accounts.head(5)

    if USE_REAL_AI and os.getenv("OPENAI_API_KEY"):
        # --- REAL AI (ONLY IF YOU TURNED ON) ---
        from openai import OpenAI
        client = OpenAI()

        top_accounts_context = top_accounts.to_string(index=False)

        prompt = f"""
You are an AI revenue strategist.

User question:
{copilot_question}

Top accounts:
{top_accounts_context}

Be concise and business-focused.
"""

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        st.markdown(response.output_text)

    else:
        # --- DEMO MODE (RECRUITER SAFE) ---

        top_customer = top_accounts.iloc[0]

        st.markdown(f"""
### 💡 AI Recommendation

Based on current revenue and churn signals, you should prioritize:

**{top_customer["CustomerID"]}**

**Why:**
- Revenue: ${top_customer["total_revenue"]:,.0f}
- Orders: {int(top_customer["order_count"])}
- Recency: {int(top_customer["recency_days"])} days
- Churn Risk: {top_customer["churn_risk"]}

👉 **Recommended Action:** Immediate outreach with a tailored retention or expansion offer.

---

### 🧠 Copilot Insight
This recommendation is driven by a combination of:
- High revenue contribution
- Elevated inactivity
- Strong likelihood of churn

Focusing on this account maximizes short-term revenue protection and long-term retention.
""")

        st.caption("Demo mode, AI responses are simulated for presentation.")

# -------------------------------------------------
# AI-Style Outreach Draft
# -------------------------------------------------

st.subheader("AI-Style Outreach Draft")

selected_customer = top_customer["CustomerID"]

# High Risk (Retention)
if top_customer["churn_risk"] == "High Risk":
    email_subject = f"{selected_customer}, quick check-in on your account"
    email_body = f"""
Hi {selected_customer},

I noticed it’s been about {int(top_customer["recency_days"])} days since your last activity.

Given the level of revenue your account has historically generated, I wanted to reach out directly to understand if anything has changed on your end or if there’s an opportunity for us to better support your priorities.

If it makes sense, I’d be happy to connect for 10–15 minutes this week to align and make sure you’re getting the most value.

Best,  
Beckten
"""

# High Value / VIP (Expansion)
elif top_customer["segment"] in ["VIP", "High Value"]:
    email_subject = f"{selected_customer}, potential opportunity to expand"
    email_body = f"""
Hi {selected_customer},

Your account has been one of the strongest in terms of engagement and revenue, so I wanted to reach out proactively.

There may be an opportunity to expand on what’s already working well, whether that’s increasing usage, optimizing current workflows, or exploring additional solutions.

If you're open to it, I’d value a quick conversation to walk through a few ideas and get your perspective.

Best,  
Beckten
"""

# Default (General Check-in)
else:
    email_subject = f"{selected_customer}, quick check-in"
    email_body = f"""
Hi {selected_customer},

I wanted to reach out and check in to see how things are going on your end.

If there’s anything we can help with or if priorities have shifted, I’m happy to connect and make sure you’re getting the most value from us.

Let me know if it would be helpful to sync briefly.

Best,  
Beckten
"""

# Display
st.markdown(f"**Subject:** {email_subject}")
st.text_area("Email Draft", email_body, height=250)

# Demo label (for recruiters)
st.caption("Demo mode, outreach drafts are AI-simulated for presentation.")

# -------------------------------------------------
# Revenue Risk Analysis
# -------------------------------------------------

st.header("Revenue Risk Analysis")

risk_summary = filtered_customer_df.groupby(
    ["segment", "churn_risk"]
)["total_revenue"].sum().reset_index()

high_value_mask = filtered_customer_df["segment"].isin(["VIP", "High Value"])

medium_risk_revenue = filtered_customer_df[
    high_value_mask & (filtered_customer_df["recency_days"] > 45)
]["total_revenue"].sum()

high_risk_revenue = filtered_customer_df[
    high_value_mask & (filtered_customer_df["recency_days"] > 90)
]["total_revenue"].sum()

col1, col2 = st.columns(2)

col1.metric("Revenue at Medium Risk", f"${medium_risk_revenue:,.0f}")
col2.metric("Revenue at High Risk", f"${high_risk_revenue:,.0f}")

st.subheader("Key Insight")

if high_risk_revenue > 0:
    st.write(f"${high_risk_revenue:,.0f} in high-value revenue is at immediate churn risk.")
elif medium_risk_revenue > 0:
    st.write(f"${medium_risk_revenue:,.0f} in high-value revenue shows early churn signals.")
else:
    st.write("No immediate churn risk detected among high-value customers.")

fig_risk = px.bar(
    risk_summary,
    x="segment",
    y="total_revenue",
    color="churn_risk",
    title="Revenue by Segment and Risk Level",
    barmode="group",
    color_discrete_map=risk_color_map
)

fig_risk.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="white"),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig_risk, use_container_width=True, key="risk_chart")

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1e3a8a, #0f172a);
    padding:20px;
    border-radius:16px;
    margin-bottom:20px;
    border:1px solid #1f2937;
">
    <h3 style="margin-bottom:5px;">🚨 Today’s Priority Action</h3>
    <p style="margin:0;">
        Contact <b>{top_customer["CustomerID"]}</b> immediately.
        This account shows high revenue (${top_customer["total_revenue"]:,.0f}) 
        and elevated churn risk.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Customer Intelligence
# -------------------------------------------------

st.header("Customer Intelligence")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Revenue Customers")
    st.dataframe(
        filtered_customer_df.sort_values("total_revenue", ascending=False)
        .head(10)[["CustomerID", "total_revenue", "order_count", "segment", "churn_risk"]],
        use_container_width=True
    )

with col2:
    st.subheader("At-Risk Customers")
    st.dataframe(
        filtered_customer_df[filtered_customer_df["churn_risk"] == "High Risk"]
        .sort_values("total_revenue", ascending=False)
        .head(10)[["CustomerID", "total_revenue", "order_count", "recency_days", "segment"]],
        use_container_width=True
    )

# -------------------------------------------------
# Revenue Trend
# -------------------------------------------------

st.header("Revenue Over Time")

daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["Revenue"].sum().reset_index()
daily_revenue.columns = ["Date", "Revenue"]

fig = px.line(
    daily_revenue,
    x="Date",
    y="Revenue",
    title="Daily Revenue Trend"
)

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="white"),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True, key="revenue_trend_chart")

# -------------------------------------------------
# Customer Segmentation
# -------------------------------------------------

st.header("Customer Segmentation & Revenue Strategy")

st.markdown("""
### How to interpret this section

- **Segment** explains customer value and buying behavior.
- **Churn Risk** explains how recently a customer has purchased.
- A customer can be **High Value** and still be **High Risk** if they have not purchased recently.
""")

segment_summary = filtered_customer_df.groupby("segment").agg(
    customers=("CustomerID", "count"),
    revenue=("total_revenue", "sum"),
    avg_revenue=("total_revenue", "mean"),
    avg_recency=("recency_days", "mean")
).reset_index()

segment_summary["revenue_share"] = (
    segment_summary["revenue"] / segment_summary["revenue"].sum() * 100
).round(1)

segment_summary_by_customers = segment_summary.sort_values("customers", ascending=False)
segment_summary_by_revenue = segment_summary.sort_values("revenue", ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig2 = px.bar(
        segment_summary_by_customers,
        x="segment",
        y="customers",
        title="Customer Count by Segment, Highest to Lowest",
        color="segment",
        color_discrete_map=segment_color_map
    )

    fig2.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )

    fig2.update_xaxes(
        categoryorder="array",
        categoryarray=segment_summary_by_customers["segment"]
    )

    st.plotly_chart(fig2, use_container_width=True, key="segment_count_chart")

with col2:
    fig4 = px.pie(
        segment_summary_by_revenue,
        names="segment",
        values="revenue",
        title="Revenue Concentration by Segment",
        hole=0.45,
        color="segment",
        color_discrete_map=segment_color_map
    )

    fig4.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig4, use_container_width=True, key="segment_revenue_pie")

st.subheader("Segment Strategy Table")

st.dataframe(
    segment_summary_by_revenue,
    use_container_width=True
)

st.subheader("Recommended Segment Plays")

card_cols = st.columns(2)

for i, row in segment_summary_by_revenue.reset_index(drop=True).iterrows():
    segment = row["segment"]

    if segment == "VIP":
        color = "#00E5A8"
        play = "Protect and expand with executive outreach."
    elif segment == "High Value":
        color = "#FF6B6B"
        play = "Upsell with personalized offers."
    elif segment == "Frequent Buyer":
        color = "#4D8BFF"
        play = "Grow wallet share and increase order value."
    else:
        color = "#7A8599"
        play = "Nurture through automated campaigns."

    with card_cols[i % 2]:
        st.markdown(f"""
        <div style="
            background-color:#111827;
            padding:16px;
            border-radius:14px;
            margin-bottom:14px;
            border-left:5px solid {color};
            min-height:145px;
        ">
            <h4 style="color:{color}; margin-bottom:6px;">{segment}</h4>
            <p style="margin:0; color:#d1d5db; font-size:15px;">
                <b>Customers:</b> {int(row['customers'])} <br>
                <b>Revenue Share:</b> {row['revenue_share']:.1f}% <br>
                <b>Play:</b> {play}
            </p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------
# Top Customers Chart
# -------------------------------------------------

st.header("Top Customers by Revenue")

top_customers_chart = (
    filtered_customer_df.groupby("CustomerID")["total_revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig3 = px.bar(
    top_customers_chart,
    x="total_revenue",
    y="CustomerID",
    orientation="h",
    title="Top 10 Customers",
    color="total_revenue",
    color_continuous_scale="Blues"
)

fig3.update_layout(
    template="plotly_dark",
    plot_bgcolor="#0f172a",
    paper_bgcolor="#0f172a",
    font=dict(color="white"),
    margin=dict(l=20, r=20, t=50, b=20),
    yaxis=dict(autorange="reversed")
)

st.plotly_chart(fig3, use_container_width=True, key="top_customers_chart")

# -------------------------------------------------
# Sales Action Plan
# -------------------------------------------------

st.header("Sales Action Plan")

st.markdown("""
**1. Expansion play:** Target VIP and high-value customers with strong purchase history for upsell conversations.

**2. Win-back play:** Re-engage high-revenue customers with high recency days before they fully churn.

**3. Technical demo play:** Use the prioritization engine to show sales leadership which accounts deserve attention first.

**4. Forecasting play:** Use revenue trend volatility to help sales leaders plan outreach and pipeline coverage.
""")

with st.expander("View sample transaction data"):
    st.dataframe(df.head(20), use_container_width=True)
