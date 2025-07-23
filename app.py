import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import re
from typing import Tuple, Optional

# Set page config
st.set_page_config(
    page_title="Restaurant Peak Hours Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .highlight {
        background-color: #ffd700;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def find_header_row(df: pd.DataFrame) -> int:
    """Find the row index containing the header in Petpooja Order Master report."""
    header_keywords = ['invoice no.', 'date', 'net sales', 'order type', 'status']
    
    for idx, row in df.iterrows():
        row_str = ' '.join(str(cell).lower() for cell in row.values if pd.notna(cell))
        if all(keyword in row_str for keyword in header_keywords[:3]):
            return idx
    return 0

def clean_column_name(column_name: str) -> str:
    """Clean and standardize column names."""
    return str(column_name).strip().lower() if pd.notna(column_name) else 'unknown'

def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load and preprocess the uploaded Petpooja Order Master Excel file."""
    try:
        # Read all data to find the header row
        df_raw = pd.read_excel(uploaded_file, header=None)
        header_row = find_header_row(df_raw)
        
        # Read again with the correct header row
        df = pd.read_excel(uploaded_file, header=header_row)
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Required columns mapping
        column_mapping = {
            'date': next((col for col in df.columns if 'date' in col), None),
            'net_sales': next((col for col in df.columns if 'net sales' in col and '₹' in col), None),
            'order_type': next((col for col in df.columns if 'order type' in col), None),
            'status': next((col for col in df.columns if 'status' in col), None)
        }
        
        # Check required columns
        if not all([column_mapping['date'], column_mapping['net_sales']]):
            missing = [k for k, v in column_mapping.items() if not v and k in ['date', 'net_sales']]
            st.error(f"Missing required columns: {', '.join(missing).replace('_', ' ').title()}")
            return None, None
        
        # Select and rename columns
        columns_to_keep = [v for v in column_mapping.values() if v is not None]
        df = df[columns_to_keep].rename(columns={
            column_mapping['date']: 'datetime',
            column_mapping['net_sales']: 'revenue'
        })
        
        # Convert and clean data
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df = df[df['datetime'].notna() & df['revenue'].notna()]
        
        # Add time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['order_count'] = 1
        
        # Filter by status if available
        if 'status' in df.columns:
            df = df[df['status'].str.lower() == 'success']
        
        return df, column_mapping
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def plot_hourly_analysis(df, metric='order_count'):
    """
    Create hourly analysis plot with peak hours highlighted.
    
    Args:
        df: DataFrame containing the data
        metric: 'order_count' for number of orders, 'revenue' for total sales
    """
    # Aggregate by hour
    hourly_data = df.groupby('hour').agg({
        'order_count': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Find peak hour(s)
    max_value = hourly_data[metric].max()
    peak_hours = hourly_data[hourly_data[metric] == max_value]['hour'].tolist()
    
    # Determine y-axis label and title based on metric
    y_label = 'Number of Orders' if metric == 'order_count' else 'Total Revenue (₹)'
    title = 'Hourly Order Count' if metric == 'order_count' else 'Hourly Revenue (₹)'
    
    # Create bar chart
    fig = px.bar(
        hourly_data, 
        x='hour', 
        y=metric,
        title=title,
        labels={'hour': 'Hour of Day', metric: y_label},
        color_discrete_sequence=['#4CAF50']
    )
    
    # Highlight peak hours
    for hour in peak_hours:
        fig.add_vrect(
            x0=hour-0.4, x1=hour+0.4,
            fillcolor="gold",
            opacity=0.3,
            line_width=0,
            layer="below"
        )
    
    # Customize hover template
    if metric == 'order_count':
        hover_template = '<b>%{x}:00 - %{x}:59</b><br>Orders: %{y:,.0f}<extra></extra>'
    else:
        hover_template = '<b>%{x}:00 - %{x}:59</b><br>Revenue: ₹%{y:,.2f}<extra></extra>'
    
    # Update layout
    fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}" for h in range(24)],
            title='Hour of Day'
        ),
        yaxis=dict(
            tickformat=",.0f" if metric == 'order_count' else ",.2f",
            title=y_label
        ),
        hovermode='x',
        showlegend=False,
        margin=dict(t=50, b=100, l=50, r=50)
    )
    
    return fig, peak_hours, max_value

def plot_heatmap(df, metric='order_count'):
    """
    Create heatmap of day vs hour.
    
    Args:
        df: DataFrame containing the data
        metric: 'order_count' for number of orders, 'revenue' for total sales
    """
    # Create pivot table for heatmap
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Aggregate data
    if metric == 'order_count':
        heatmap_data = df.pivot_table(
            index='day_of_week',
            columns='hour',
            values='order_count',
            aggfunc='count',
            fill_value=0
        )
    else:  # revenue
        heatmap_data = df.pivot_table(
            index='day_of_week',
            columns='hour',
            values='revenue',
            aggfunc='sum',
            fill_value=0
        )
    
    # Reindex to ensure correct day order
    heatmap_data = heatmap_data.reindex(days_order)
    
    # Format labels
    y_label = 'Day of Week'
    x_label = 'Hour of Day'
    title = 'Weekly Order Count Heatmap' if metric == 'order_count' else 'Weekly Revenue Heatmap (₹)'
    
    # Create heatmap with better colors
    fig = px.imshow(
        heatmap_data,
        labels=dict(x=x_label, y=y_label, color='Orders' if metric == 'order_count' else 'Revenue (₹)'),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        title=title,
        aspect="auto",
        color_continuous_scale='Greens'
    )
    
    # Customize hover template
    if metric == 'order_count':
        hover_template = '<b>%{y}, %{x}:00-%{x}:59</b><br>Orders: %{z:,.0f}<extra></extra>'
    else:
        hover_template = '<b>%{y}, %{x}:00-%{x}:59</b><br>Revenue: ₹%{z:,.2f}<extra></extra>'
    
    # Update layout
    fig.update_traces(hovertemplate=hover_template)
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}" for h in range(0, 24, 2)]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=days_order,
            ticktext=[day[:3] for day in days_order]
        ),
        coloraxis_colorbar=dict(
            title='Orders' if metric == 'order_count' else 'Revenue (₹)',
            tickformat=",.0f" if metric == 'order_count' else ",.2f"
        ),
        margin=dict(t=50, b=100, l=100, r=50)
    )
    
    return fig

def generate_insights(peak_hours, max_value, metric, df):
    """
    Generate business insights based on peak hours and sales patterns.
    
    Args:
        peak_hours: List of peak hours (0-23)
        max_value: Maximum value (orders or revenue) at peak
        metric: 'order_count' or 'revenue'
        df: Full DataFrame for additional analysis
    """
    insights = []
    
    # Format peak hour(s) information
    if len(peak_hours) == 1:
        hour_str = f"{peak_hours[0]:02d}:00 - {peak_hours[0]+1:02d}"
        if metric == 'order_count':
            insights.append(f"• Peak order volume at {hour_str} with {int(max_value)} orders.")
        else:
            insights.append(f"• Peak revenue at {hour_str} with ₹{max_value:,.2f} in sales.")
    else:
        hours_str = ", ".join([f"{h:02d}:00-{h+1:02d}" for h in peak_hours])
        if metric == 'order_count':
            insights.append(f"• Multiple peak order hours: {hours_str} with {int(max_value)} orders each.")
        else:
            insights.append(f"• Multiple peak revenue hours: {hours_str} with ₹{max_value:,.2f} each.")
    
    # Time-based insights
    peak_hour = peak_hours[0]
    if peak_hour >= 21 or peak_hour <= 4:  # Late night/early morning
        insights.append("• Strong late-night business detected. Consider extending hours or adding late-night specials.")
    elif 17 <= peak_hour <= 20:  # Dinner
        insights.append("• Dinner rush confirmed. Ensure adequate staffing and consider a pre-set dinner menu.")
    elif 11 <= peak_hour <= 14:  # Lunch
        insights.append("• Clear lunch peak. Optimize your lunch menu and staff scheduling accordingly.")
    elif 7 <= peak_hour <= 10:  # Breakfast
        insights.append("• Morning business is strong. Consider expanding breakfast/brunch offerings.")
    
    # Day of week analysis
    if 'day_of_week' in df.columns:
        busiest_day = df['day_of_week'].value_counts().idxmax()
        slowest_day = df['day_of_week'].value_counts().idxmin()
        
        if busiest_day != slowest_day:
            insights.append(f"• {busiest_day}s are typically your busiest days, while {slowest_day}s are the quietest.")
    
    # Revenue per order insight (if both metrics available)
    if 'revenue' in df.columns and 'order_count' in df.columns:
        avg_order_value = df['revenue'].sum() / df['order_count'].sum()
        insights.append(f"• Average order value: ₹{avg_order_value:,.2f}")
    
    return insights

def main():
    # Title and description
    st.title("🍽️ Restaurant Peak Hours Analyzer")
    st.markdown("""
    Upload your Petpooja Order Master report to analyze peak business hours and optimize your operations.
    The app will automatically detect the data structure and provide insights.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Petpooja Order Master Report (Excel)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Add a loading spinner while processing the file
        with st.spinner('Analyzing your data...'):
            # Load and preprocess data
            df, column_mapping = load_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.success("✅ Data loaded successfully!")
            
            # Show data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Orders", f"{len(df):,}")
            with col2:
                st.metric("Total Revenue", f"₹{df['revenue'].sum():,.2f}")
            with col3:
                date_range = f"{df['datetime'].min().strftime('%b %d, %Y')} - {df['datetime'].max().strftime('%b %d, %Y')}"
                st.metric("Date Range", date_range)
            
            # Show data preview with expander
            with st.expander("View sample data", expanded=False):
                st.dataframe(df.head(10))
            
            # Add a divider
            st.markdown("---")
            
            # Metric selection
            metric = st.radio(
                "Select analysis view:",
                ["order_count", "revenue"],
                format_func=lambda x: "Order Volume" if x == "order_count" else "Revenue",
                horizontal=True
            )
            
            # Add a toggle for order type filtering if available
            filter_by_type = False
            if 'order_type' in column_mapping and column_mapping['order_type'] in df.columns:
                filter_by_type = st.checkbox("Filter by order type", value=False)
            
            # Apply order type filter if enabled
            if filter_by_type and 'order_type' in df.columns:
                order_types = sorted(df['order_type'].dropna().unique())
                if order_types:
                    selected_types = st.multiselect(
                        "Select order types to include:",
                        options=order_types,
                        default=order_types
                    )
                    if selected_types:
                        df = df[df['order_type'].isin(selected_types)]
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Hourly Analysis", "📅 Weekly Heatmap"])
            
            with tab1:
                # Hourly analysis
                fig_hourly, peak_hours, max_value = plot_hourly_analysis(df, metric)
                st.plotly_chart(fig_hourly, use_container_width=True)
                
                # Display peak hour information
                st.subheader("🔍 Business Insights")
                insights = generate_insights(peak_hours, max_value, metric, df)
                for insight in insights:
                    st.markdown(insight)
                
                # Add export button
                st.download_button(
                    label="📥 Export Hourly Data",
                    data=df.groupby('hour').agg({
                        'order_count': 'sum',
                        'revenue': 'sum'
                    }).reset_index().to_csv(index=False),
                    file_name="hourly_analysis.csv",
                    mime="text/csv"
                )
            
            with tab2:
                # Weekly heatmap
                fig_heatmap = plot_heatmap(df, metric)
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Heatmap insights
                st.markdown("""
                **How to read this heatmap:**
                - Each cell shows the total orders/revenue for that specific day and hour
                - Darker green indicates higher activity
                - Hover over cells to see exact values
                - Look for patterns to optimize staffing and promotions
                """)
        
        elif df is not None and df.empty:
            st.warning("No valid data found in the uploaded file. Please check the file format.")
    else:
        # Show sample data structure and instructions
        with st.expander("ℹ️ How to use this tool", expanded=True):
            st.markdown("""
            ### Expected Excel Format:
            This app works with Petpooja Order Master reports. Your file should include these columns:
            - **Date**: Timestamp of each order
            - **Net Sales (₹)(M.A - D)**: Total amount for each order
            - (Optional) Order Type: Dine-In, Delivery, Takeaway, etc.
            - (Optional) Status: Filter for 'Success' orders
            
            ### How it works:
            1. Export your order data from Petpooja as an Excel file
            2. Upload it using the button above
            3. View peak hours and business insights
            4. Toggle between order volume and revenue views
            5. Filter by order type if needed
            
            The app will automatically detect the data structure and process your orders.
            """)
        
        st.info("👆 Please upload an Excel file to begin analysis")

if __name__ == "__main__":
    main()
