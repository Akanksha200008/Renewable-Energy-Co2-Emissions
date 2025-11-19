import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pycountry
import numpy as np
from scipy import stats
from pathlib import Path

st.set_page_config(
    page_title="Renewable Energy & CO₂ Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def country_to_iso3(country_name):
    """Convert country name to ISO alpha-3 code"""
    try:
        return pycountry.countries.lookup(country_name).alpha_3
    except Exception:
        fixes = {
            "USA": "USA", "United States": "USA", "United States of America": "USA",
            "UK": "GBR", "Russia": "RUS", "Viet Nam": "VNM", "South Korea": "KOR",
            "North Korea": "PRK", "Czech Republic": "CZE", "Iran": "IRN",
        }
        return fixes.get(country_name, None)

@st.cache_data
def load_data():
    """Load and prepare the dataset"""
    df = pd.read_csv("Data/cleaned_energy_data.csv")
    df.columns = [c.strip() for c in df.columns]
    
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)
    
    if 'Country' in df.columns:
        df['iso_alpha3'] = df['Country'].apply(country_to_iso3)
    
    return df

def get_country_color_map(countries):
    """Create consistent color mapping for countries using distinct colors"""
    color_palette = px.colors.qualitative.Bold + px.colors.qualitative.Vivid + px.colors.qualitative.Safe
    color_map = {}
    for idx, country in enumerate(sorted(countries)):
        color_map[country] = color_palette[idx % len(color_palette)]
    return color_map

def calculate_growth_rate(df, country, metric):
    """Calculate year-over-year growth rate for a metric"""
    country_data = df[df['Country'] == country].sort_values('Year')
    if len(country_data) < 2:
        return None
    
    country_data = country_data.copy()
    country_data['Growth_Rate'] = country_data[metric].pct_change() * 100
    return country_data[['Year', 'Growth_Rate']].dropna()

def get_correlation_insights(corr_matrix, threshold=0.7):
    """Generate insights from correlation matrix"""
    insights = []
    
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                strong_correlations.append((var1, var2, corr_value))
    
    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for var1, var2, corr in strong_correlations[:5]:
        if corr > 0:
            insights.append(f"**Strong positive correlation ({corr:.2f})**: {var1} and {var2}")
        else:
            insights.append(f"**Strong negative correlation ({corr:.2f})**: {var1} and {var2}")
    
    return insights

df = load_data()

st.title("Renewable Energy & CO₂ Emissions Analysis Dashboard")
st.markdown("### Data Science Project: Global Renewable Energy Trends (2000-2023)")
st.markdown("---")

with st.sidebar:
    st.header(" Filters & Controls")
    
    st.subheader("Year Selection")
    years = sorted(df['Year'].unique())
    min_year, max_year = int(min(years)), int(max(years))
    
    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    single_year = st.select_slider(
        "Select Single Year (for maps & scatter)",
        options=years,
        value=max_year
    )
    
    st.subheader(" Country Selection")
    all_countries = sorted(df['Country'].unique().tolist())
    
    select_all = st.checkbox("Select All Countries", value=True)
    
    if select_all:
        selected_countries = all_countries
    else:
        selected_countries = st.multiselect(
            "Choose Countries",
            options=all_countries,
            default=all_countries[:5]
        )
    
    st.subheader("⚡ Energy Type Filter")
    energy_types = ['All'] + sorted(df['Energy_Type'].unique().tolist())
    selected_energy = st.selectbox("Energy Type", energy_types)
    
    st.markdown("---")
    st.info(f"**Data Range:** {year_range[0]} - {year_range[1]}\n\n**Countries:** {len(selected_countries)}")

df_filtered = df[
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1])
]

if not select_all:
    if selected_countries:
        df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
    else:
        df_filtered = df_filtered[df_filtered['Country'].isin([])]

if selected_energy != 'All':
    df_filtered = df_filtered[df_filtered['Energy_Type'] == selected_energy]

st.header("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_production = df_filtered['Production_GWh'].sum()
    st.metric("Total Production", f"{total_production/1e6:.2f}M GWh")

with col2:
    avg_emissions = df_filtered['CO2_Emissions'].mean()
    st.metric("Avg CO₂ Emissions", f"{avg_emissions/1e3:.1f}K")

with col3:
    total_investment = df_filtered['Investments'].sum()
    st.metric("Total Investment", f"${total_investment/1e12:.2f}T")

with col4:
    avg_renewable_pct = df_filtered['Proportion_of_Energy_from_Renewables'].mean()
    st.metric("Avg Renewable %", f"{avg_renewable_pct:.1f}%")

with col5:
    countries_count = df_filtered['Country'].nunique()
    st.metric("Countries Analyzed", countries_count)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Geographic Analysis", 
    "Trends & Growth", 
    "Correlations", 
    "Country Comparison",
    "Key Insights"
])

with tab1:
    st.header("Geographic Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"World Map - {single_year}")
        map_metric = st.selectbox(
            "Select Metric",
            ['Proportion_of_Energy_from_Renewables', 'Production_GWh', 'CO2_Emissions', 
             'Investments', 'Installed_Capacity_MW'],
            key="map_metric"
        )
        
        map_df = df_filtered[df_filtered['Year'] == single_year].dropna(subset=['iso_alpha3'])
        
        if not map_df.empty:
            fig_map = px.choropleth(
                map_df,
                locations='iso_alpha3',
                color=map_metric,
                hover_name='Country',
                hover_data={
                    'iso_alpha3': False,
                    map_metric: ':.2f',
                    'Energy_Type': True
                },
                title=f"{map_metric.replace('_', ' ')} by Country ({single_year})",
                color_continuous_scale='RdYlGn',
                projection='natural earth'
            )
            fig_map.update_layout(height=500, geo=dict(showframe=False, showcoastlines=True))
            st.plotly_chart(fig_map, width='stretch')
        else:
            st.warning("No data available for the selected filters")
    
    with col2:
        st.subheader("Top 10 Countries")
        top_metric = st.selectbox(
            "Rank by",
            ['Production_GWh', 'CO2_Emissions', 'Proportion_of_Energy_from_Renewables'],
            key="top_metric"
        )
        
        top_df = df_filtered[df_filtered['Year'] == single_year]
        
        if not top_df.empty:
            top_countries = top_df.nlargest(min(10, len(top_df)), top_metric)[['Country', top_metric]]
            
            fig_bar = px.bar(
                top_countries,
                x=top_metric,
                y='Country',
                orientation='h',
                title=f"Top {len(top_countries)} by {top_metric.replace('_', ' ')}",
                color=top_metric,
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_bar, width='stretch')
        else:
            st.warning("No data available for the selected filters")

with tab2:
    st.header("Trends & Growth Rate Analysis")
    
    st.subheader("Production & Emissions Over Time")
    
    if selected_countries:
        country_color_map = get_country_color_map(selected_countries)
        
        time_df = df_filtered.groupby(['Year', 'Country']).agg({
            'Production_GWh': 'sum',
            'CO2_Emissions': 'sum',
            'Proportion_of_Energy_from_Renewables': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_prod = px.line(
                time_df,
                x='Year',
                y='Production_GWh',
                color='Country',
                color_discrete_map=country_color_map,
                title="Renewable Energy Production Over Time",
                markers=True
            )
            fig_prod.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_prod, width='stretch')
            
            st.markdown("**What we see:** This chart shows the evolution of renewable energy production across selected countries. Rising trends indicate increased renewable energy adoption.")
        
        with col2:
            fig_co2 = px.line(
                time_df,
                x='Year',
                y='CO2_Emissions',
                color='Country',
                color_discrete_map=country_color_map,
                title="CO₂ Emissions Over Time",
                markers=True
            )
            fig_co2.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig_co2, width='stretch')
            
            st.markdown("** What we see:** CO₂ emissions trends. Ideally, we want to see declining trends as renewable energy increases.")
        
        st.subheader("Year-over-Year Growth Rates")
        
        growth_metric = st.selectbox(
            "Select metric for growth analysis",
            ['Production_GWh', 'CO2_Emissions', 'Investments', 'Proportion_of_Energy_from_Renewables']
        )
        
        if len(selected_countries) <= 10:
            growth_data = []
            for country in selected_countries:
                gr = calculate_growth_rate(df_filtered, country, growth_metric)
                if gr is not None and not gr.empty:
                    gr['Country'] = country
                    growth_data.append(gr)
            
            if growth_data:
                growth_df = pd.concat(growth_data, ignore_index=True)
                
                fig_growth = px.line(
                    growth_df,
                    x='Year',
                    y='Growth_Rate',
                    color='Country',
                    color_discrete_map=country_color_map,
                    title=f"Year-over-Year Growth Rate: {growth_metric.replace('_', ' ')}",
                    markers=True
                )
                fig_growth.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Growth")
                fig_growth.update_layout(height=400, yaxis_title="Growth Rate (%)")
                st.plotly_chart(fig_growth, width='stretch')
                
                avg_growth = growth_df.groupby('Country')['Growth_Rate'].mean().sort_values(ascending=False)
                st.markdown(f"** Conclusion:** Average growth rates show which countries are expanding renewable energy fastest. Positive values indicate growth, negative indicate decline.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**Highest Growth:** {avg_growth.index[0]} ({avg_growth.iloc[0]:.1f}% avg)")
                with col2:
                    if avg_growth.iloc[-1] < 0:
                        st.error(f"**Declining:** {avg_growth.index[-1]} ({avg_growth.iloc[-1]:.1f}% avg)")
                    else:
                        st.info(f"**Lowest Growth:** {avg_growth.index[-1]} ({avg_growth.iloc[-1]:.1f}% avg)")
        else:
            st.info("Select 10 or fewer countries to view growth rate analysis")
        
        st.subheader("Energy Type Distribution")
        
        energy_dist = df_filtered.groupby(['Year', 'Energy_Type'])['Production_GWh'].sum().reset_index()
        
        fig_area = px.area(
            energy_dist,
            x='Year',
            y='Production_GWh',
            color='Energy_Type',
            title="Energy Production by Type (Stacked Area Chart)",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_area.update_layout(height=400)
        st.plotly_chart(fig_area, width='stretch')
        
        st.markdown("** What we see:** The composition of renewable energy sources over time. This shows which renewable technologies dominate in the selected regions.")

with tab3:
    st.header("Correlation Analysis")
    
    st.subheader("Correlation Heatmap")
    
    key_vars = [
        'Production_GWh', 'CO2_Emissions', 'GDP', 'Population', 
        'Investments', 'Proportion_of_Energy_from_Renewables',
        'R&D_Expenditure', 'Installed_Capacity_MW', 'Energy_Consumption'
    ]
    
    available_vars = [v for v in key_vars if v in df_filtered.columns]
    
    if len(available_vars) >= 3:
        corr_df = df_filtered[available_vars].dropna()
        
        if not corr_df.empty:
            corr = corr_df.corr()
            
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                aspect='auto',
                title="Correlation Matrix: Key Energy & Economic Indicators",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(
                height=700,
                width=900,
                xaxis_title="",
                yaxis_title=""
            )
            fig_corr.update_xaxes(tickangle=45)
            st.plotly_chart(fig_corr, width='stretch')
            
            st.markdown("###  How to Read This Heatmap:")
            st.markdown("""
            - **Values range from -1 to +1**
            - **+1 (Dark Red)**: Perfect positive correlation - variables increase together
            - **0 (White)**: No correlation
            - **-1 (Dark Blue)**: Perfect negative correlation - one increases as other decreases
            - **|0.7-1.0|**: Strong correlation
            - **|0.4-0.7|**: Moderate correlation
            - **|0.0-0.4|**: Weak correlation
            """)
            
            st.subheader("Key Insights from Correlations")
            insights = get_correlation_insights(corr, threshold=0.6)
            
            if insights:
                for insight in insights:
                    st.markdown(insight)
            else:
                st.info("No strong correlations (>0.6) found in the selected data")
            
            st.subheader("Detailed Correlation Values")
            corr_styled = corr.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1).format("{:.3f}")
            st.dataframe(corr_styled, width='stretch')
    else:
        st.warning("Insufficient data for correlation analysis")

with tab4:
    st.header("Country-by-Country Comparison")
    
    st.subheader(f"GDP vs Production Scatter Plot ({single_year})")
    
    scatter_df = df_filtered[df_filtered['Year'] == single_year].copy()
    
    if not scatter_df.empty and 'GDP' in scatter_df.columns and 'Production_GWh' in scatter_df.columns:
        country_color_map = get_country_color_map(scatter_df['Country'].unique())
        
        fig_scatter = px.scatter(
            scatter_df,
            x='GDP',
            y='Production_GWh',
            size='Population',
            color='Country',
            color_discrete_map=country_color_map,
            hover_name='Country',
            hover_data={
                'GDP': ':,.0f',
                'Production_GWh': ':,.2f',
                'Population': ':,',
                'Energy_Type': True,
                'Country': False
            },
            title=f"GDP vs Renewable Production - Each point is a country with distinct color",
            labels={
                'GDP': 'GDP (USD)',
                'Production_GWh': 'Renewable Energy Production (GWh)'
            }
        )
        
        if len(scatter_df) > 2:
            z = np.polyfit(scatter_df['GDP'].dropna(), scatter_df['Production_GWh'].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(scatter_df['GDP'].min(), scatter_df['GDP'].max(), 100)
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash', width=2)
                )
            )
        
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, width='stretch')
        
        st.markdown("### What This Shows:")
        st.markdown("""
        - **Each bubble** represents a country (with distinct color for easy identification)
        - **Bubble size** represents population
        - **X-axis** shows economic size (GDP)
        - **Y-axis** shows renewable energy production
        - **Red dashed line** shows the general trend
        
        **Conclusion:** Countries above the trend line produce more renewable energy than expected for their economic size,
        indicating stronger commitment to renewables. Countries below may have growth opportunities.
        """)
        
        correlation = scatter_df[['GDP', 'Production_GWh']].corr().iloc[0, 1]
        st.info(f"**Correlation coefficient:** {correlation:.3f} - {'Positive' if correlation > 0 else 'Negative'} relationship between GDP and renewable production")
    else:
        st.warning("No data available for scatter plot")
    
    st.subheader(" Multi-Metric Comparison")
    
    if selected_countries and len(selected_countries) <= 15:
        comparison_df = df_filtered[
            df_filtered['Year'] == single_year
        ].groupby('Country').agg({
            'Production_GWh': 'sum',
            'CO2_Emissions': 'sum',
            'Proportion_of_Energy_from_Renewables': 'mean',
            'Investments': 'sum'
        }).reset_index()
        
        metric_choice = st.selectbox(
            "Select metric to compare",
            ['Production_GWh', 'CO2_Emissions', 'Proportion_of_Energy_from_Renewables', 'Investments']
        )
        
        comparison_df = comparison_df.sort_values(metric_choice, ascending=False)
        
        fig_comparison = px.bar(
            comparison_df,
            x='Country',
            y=metric_choice,
            color='Country',
            color_discrete_map=get_country_color_map(comparison_df['Country'].unique()),
            title=f"{metric_choice.replace('_', ' ')} by Country ({single_year})"
        )
        fig_comparison.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_comparison, width='stretch')

with tab5:
    st.header(" Key Insights & Conclusions")
    
    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your filter selections.")
    else:
        st.subheader(" Executive Summary")
        
        production_by_country = df_filtered.groupby('Country')['Production_GWh'].sum()
        emissions_by_country = df_filtered.groupby('Country')['CO2_Emissions'].mean()
        renewable_by_country = df_filtered.groupby('Country')['Proportion_of_Energy_from_Renewables'].mean()
        
        if not production_by_country.empty and not emissions_by_country.empty and not renewable_by_country.empty:
            top_producer = production_by_country.idxmax()
            top_production = production_by_country.max()
            
            lowest_emissions = emissions_by_country.idxmin()
            lowest_emission_val = emissions_by_country.min()
            
            highest_renewable_pct = renewable_by_country.idxmax()
            highest_renewable_val = renewable_by_country.max()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"""
                ** Top Producer**
                
                {top_producer}
                
                {top_production/1e3:.1f}K GWh
                """)
            
            with col2:
                st.success(f"""
                ** Cleanest Energy**
                
                {lowest_emissions}
                
                {lowest_emission_val/1e3:.1f}K CO₂
                """)
            
            with col3:
                st.success(f"""
                **Highest Renewable %**
                
                {highest_renewable_pct}
                
                {highest_renewable_val:.1f}%
                """)
        else:
            st.info("Insufficient data for executive summary with current filters.")
    
    st.markdown("---")
    
    st.subheader("Data Analysis Conclusions")
    
    st.markdown("### 1️.Renewable Energy Production Trends")
    yearly_production = df_filtered.groupby('Year')['Production_GWh'].sum()
    if len(yearly_production) > 1:
        total_growth = ((yearly_production.iloc[-1] - yearly_production.iloc[0]) / yearly_production.iloc[0]) * 100
        st.markdown(f"""
        - Overall renewable energy production has **{'increased' if total_growth > 0 else 'decreased'}** by **{abs(total_growth):.1f}%** 
          from {year_range[0]} to {year_range[1]}
        - This indicates **{'positive momentum' if total_growth > 0 else 'concerning decline'}** in renewable energy adoption
        """)
    
    st.markdown("### 2. CO₂ Emissions vs Renewable Energy")
    if not df_filtered.empty and 'CO2_Emissions' in df_filtered.columns and 'Proportion_of_Energy_from_Renewables' in df_filtered.columns:
        corr_data = df_filtered[['CO2_Emissions', 'Proportion_of_Energy_from_Renewables']].dropna()
        if len(corr_data) > 1:
            corr_co2_renewable = corr_data.corr().iloc[0, 1]
            st.markdown(f"""
            - Correlation between CO₂ emissions and renewable energy proportion: **{corr_co2_renewable:.3f}**
            - This **{'negative' if corr_co2_renewable < 0 else 'positive'}** correlation suggests that higher renewable energy adoption 
              is **{'associated with lower' if corr_co2_renewable < 0 else 'not clearly associated with lower'}** emissions
            """)
        else:
            st.info("Insufficient data to calculate CO₂ vs renewable energy correlation")
    
    st.markdown("### 3️. Economic Factors")
    if not df_filtered.empty and 'GDP' in df_filtered.columns and 'Production_GWh' in df_filtered.columns:
        gdp_data = df_filtered[['GDP', 'Production_GWh']].dropna()
        if len(gdp_data) > 1:
            gdp_prod_corr = gdp_data.corr().iloc[0, 1]
            st.markdown(f"""
            - Correlation between GDP and renewable production: **{gdp_prod_corr:.3f}**
            - **{'Wealthier nations tend to invest more in renewable energy' if gdp_prod_corr > 0.5 else 'Economic size is not the only factor in renewable energy adoption'}**
            """)
        else:
            st.info("Insufficient data to calculate GDP vs production correlation")
    
    st.markdown("### 4. Investment Impact")
    if not df_filtered.empty and 'Investments' in df_filtered.columns and 'Production_GWh' in df_filtered.columns:
        inv_data = df_filtered[['Investments', 'Production_GWh']].dropna()
        if len(inv_data) > 1:
            inv_prod_corr = inv_data.corr().iloc[0, 1]
            st.markdown(f"""
            - Correlation between investment and production: **{inv_prod_corr:.3f}**
            - **{'Strong evidence that investment drives renewable energy capacity' if inv_prod_corr > 0.5 else 'Investment shows moderate relationship with production capacity'}**
            """)
        else:
            st.info("Insufficient data to calculate investment vs production correlation")
    
    st.markdown("---")
    
    st.subheader(" Recommendations for Further Analysis")
    st.markdown("""
    1. **Statistical Significance Testing**: Conduct hypothesis tests to validate correlations
    2. **Time Series Forecasting**: Build predictive models for future renewable energy trends
    3. **Policy Impact Analysis**: Investigate the effect of government policies on renewable adoption
    4. **Technology Breakdown**: Deep dive into which renewable technologies (solar, wind, hydro) are most effective
    5. **Regional Patterns**: Analyze geographical and climate factors influencing renewable energy potential
    """)
    
    st.markdown("---")
    st.subheader(" Export Data")
    
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"renewable_energy_filtered_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )
    
    with col2:
        summary_stats = df_filtered[available_vars].describe()
        summary_csv = summary_stats.to_csv()
        st.download_button(
            label="Download Summary Statistics (CSV)",
            data=summary_csv,
            file_name=f"summary_statistics_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv"
        )
