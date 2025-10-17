import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt

# ====================== Page Setup ======================
st.set_page_config(
    page_title="HR Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 HR Employee Attrition Dashboard")
st.markdown("An interactive and professional dashboard analyzing employee attrition patterns and HR metrics.")

# ====================== File Upload ======================
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Data File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")
    st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # ====================== Sidebar Navigation ======================
    section = st.sidebar.radio(
        "📊 Select Analysis Section:",
        [
            "Overview",
            "Demographics",
            "Department & Job Roles",
            "Salary Analysis",
            "Age Distribution",
            "Attrition vs Income Relations",
            "Numerical Columns Boxplots",
            "Attrition Insights",
            "Correlation"
        ]
    )

    colors = {"Yes": "#E74C3C", "No": "#2E86C1"}
    sns.set_theme(style="whitegrid")

    # ====================== 1. Overview ======================
    if section == "Overview":
        st.header("📋 Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplicate Values", df.duplicated().sum())

        st.markdown("### 🔹 Sample of the Data")
        st.dataframe(df.sample(5))

        st.markdown("### 🔹 Statistical Summary")
        st.dataframe(df.describe())

    # ====================== 2. Demographics ======================
    elif section == "Demographics":
        st.header("👥 Demographic Analysis")

        col1, col2 = st.columns(2)

        with col1:
            att_props = df["Attrition"].value_counts(normalize=True) * 100
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=att_props.index,
                values=att_props.values,
                textinfo='label+percent',
                marker=dict(colors=['#E74C3C', '#2E86C1']),
                hole=0.3
            ))
            fig.update_layout(title="Attrition Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig1 = px.bar(
                df.groupby(['Gender', 'Attrition'], as_index=False).size(),
                x='Gender', y='size', color='Attrition',
                color_discrete_map=colors, text='size',
                barmode='group', title='Attrition by Gender'
            )
            st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Age", nbins=20, color="Attrition",
                            color_discrete_map=colors, title="Age Distribution by Attrition")
        st.plotly_chart(fig2, use_container_width=True)

    # ====================== 3. Department & Job Roles ======================
    elif section == "Department & Job Roles":
        st.header("🏢 Department and Job Role Analysis")

        col1, col2 = st.columns(2)

        with col1:
            dept_attrition = df[df['Attrition'] == 'Yes']['Department'].value_counts()
            fig3 = go.Figure(go.Pie(
                labels=dept_attrition.index,
                values=dept_attrition.values,
                hole=0.4,
                marker=dict(colors=['#2E86C1', '#E74C3C', '#1E9D88']),
                textinfo='label+percent'
            ))
            fig3.update_layout(title="Attrition by Department")
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            job_attrition = df.groupby(['JobRole', 'Attrition'], as_index=False).size()
            job_attrition['Percentage'] = (
                job_attrition['size'] / job_attrition.groupby('JobRole')['size'].transform('sum') * 100
            )
            fig4 = px.bar(
                job_attrition, x='JobRole', y='Percentage',
                color='Attrition', text=job_attrition['Percentage'].apply(lambda x: f"{x:.1f}%"),
                color_discrete_map=colors, barmode='stack',
                title='Attrition by Job Role (%)'
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ====================== 4. Salary Analysis ======================
    elif section == "Salary Analysis":
        st.header("💰 Salary Analysis")

        col1, col2 = st.columns(2)

        with col1:
            avg_salary = (
                df.groupby("Department")["MonthlyIncome"]
                .mean().reset_index()
                .sort_values(by="MonthlyIncome", ascending=False)
            )
            fig5 = px.bar(
                avg_salary, x="Department", y="MonthlyIncome",
                text=avg_salary["MonthlyIncome"].apply(lambda x: f"${x:,.0f}"),
                color="MonthlyIncome", color_continuous_scale="Viridis",
                title="Average Monthly Income by Department"
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            fig6 = px.box(df, x="Attrition", y="MonthlyIncome",
                          color="Attrition", color_discrete_map=colors,
                          title="Monthly Income Distribution by Attrition")
            st.plotly_chart(fig6, use_container_width=True)

    # ====================== 5. Age Distribution ======================
    elif section == "Age Distribution":
        st.header("👤 Age Distribution of Employees")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df["Age"], bins=20, kde=True, color="#2E86C1", ax=ax)
        avg_age = df["Age"].mean()
        plt.axvline(avg_age, color="#E74C3C", linestyle="--", linewidth=2, label=f"Average Age: {avg_age:.1f}")
        plt.title("Age Distribution of Employees", fontsize=16)
        plt.legend()
        st.pyplot(fig)

        st.subheader("🏢 Attrition by Department")
        dept_attrition = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        fig = px.bar(
            dept_attrition, x='Department', y='Count', color='Attrition',
            text='Count', barmode='stack', color_discrete_map=colors,
            title='Attrition by Department'
        )
        st.plotly_chart(fig, use_container_width=True)

    # ====================== 6. Attrition vs Income Relations ======================
    elif section == "Attrition vs Income Relations":
        st.header("📈 Attrition vs Monthly Income Relations")

        attrition_palette = {"Yes": "#E74C3C", "No": "#2E86C1"}
        selected_cols = ['Age', 'DistanceFromHome', 'TotalWorkingYears', 'YearsAtCompany']

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        for idx, col in enumerate(selected_cols):
            sns.lineplot(
                data=df,
                x=col,
                y='MonthlyIncome',
                hue='Attrition',
                palette=attrition_palette,
                ax=axes[idx],
                linewidth=2.2
            )
            axes[idx].set_title(f"{col} vs Monthly Income by Attrition", fontsize=14, weight='bold')
            axes[idx].legend(title="Attrition", frameon=False)
        plt.tight_layout(pad=4)
        st.pyplot(fig)

    # ====================== 7. Numerical Columns Boxplots ======================
    elif section == "Numerical Columns Boxplots":
        st.header("📦 Boxplots for Numerical Columns")

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 0:
            rows = len(numerical_cols) // 3 + 1
            fig, axes = plt.subplots(rows, 3, figsize=(15, 12))
            axes = axes.flatten()
            for idx, col in enumerate(numerical_cols):
                sns.boxplot(data=df, x=col, ax=axes[idx], color="#76B7B2")
                axes[idx].set_title(col, fontsize=12, fontweight="bold")
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No numerical columns found in the dataset.")

    # ====================== 8. Attrition Insights ======================
    elif section == "Attrition Insights":
        st.header("📊 Additional Attrition Insights")

        fig7 = px.bar(
            df.groupby(['BusinessTravel', 'Attrition'], as_index=False).size(),
            x='BusinessTravel', y='size', color='Attrition',
            text='size', color_discrete_map=colors,
            barmode='group', title='Attrition by Business Travel'
        )
        st.plotly_chart(fig7, use_container_width=True)

        fig8 = px.histogram(df, x="YearsAtCompany", nbins=20, color="Attrition",
                            color_discrete_map=colors, title="Attrition by Years at Company")
        st.plotly_chart(fig8, use_container_width=True)

    # ====================== 9. Correlation ======================
    elif section == "Correlation":
        st.header("🔗 Correlation Heatmap")

        cat_cols = [i for i in df.columns if df[i].nunique() <= 5 or df[i].dtype == object]
        corr = df.drop(columns=cat_cols + ['EmployeeNumber'], errors='ignore').corr().round(2)
        x, y, z = corr.columns.tolist(), corr.index.tolist(), corr.to_numpy()
        fig9 = ff.create_annotated_heatmap(
            z=z, x=x, y=y, annotation_text=z,
            colorscale='Blues',
            hovertemplate="Correlation between %{x} and %{y} = %{z}<extra></extra>"
        )
        fig9.update_yaxes(autorange="reversed")
        fig9.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig9, use_container_width=True)

else:
    st.warning("📁 Please upload a CSV file to view the analysis.")
