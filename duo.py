import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="PetPooja Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

def clean_data(df):
    """
    Clean and extract required columns from the uploaded data.
    
    Args:
        df: Raw DataFrame from uploaded file
        
    Returns:
        Tuple of (cleaned DataFrame, error message if any)
    """
    try:
        # Store original row count
        original_rows = len(df)
        
        # Make a copy to avoid modifying the original DataFrame
        df_clean = df.copy()
        
        # 1. Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # 2. Convert column names to lowercase and strip whitespace
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        
        # 3. Map possible column names to standard names
        column_mapping = {
            'restaurant': ['restaurant', 'outlet', 'location', 'store', 'branch', 'restaurant name'],
            'hour': ['hour', 'time', 'hr', 'order hour', 'sale hour', 'transaction hour'],
            'item': ['item', 'product', 'menu item', 'dish', 'menu'],
            'price': ['price', 'unit price', 'rate', 'unitprice', 'cost'],
            'quantity': ['quantity', 'qty', 'count', 'number', 'amount'],
            'total': ['total', 'amount', 'revenue', 'sale amount', 'gross']
        }
        
        # Find actual column names in the DataFrame
        actual_columns = {}
        missing_columns = []
        
        for std_name, possible_names in column_mapping.items():
            found = False
            for name in possible_names:
                if name in df_clean.columns:
                    actual_columns[std_name] = name
                    found = True
                    break
            if not found:
                missing_columns.append(std_name)
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # 4. Keep only required columns and rename them
        required_columns = {v: k for k, v in actual_columns.items()}
        df_clean = df_clean[list(actual_columns.values())].rename(columns=required_columns)
        
        # 5. Convert hour to 24-hour format (handle both numeric and string formats)
        if df_clean['hour'].dtype == 'object':
            # Handle string hours in '01 PM', '11 AM' format
            def convert_12h_to_24h(time_str):
                try:
                    # Handle NaN or None values
                    if pd.isna(time_str):
                        return None
                        
                    # Convert to string and strip whitespace
                    time_str = str(time_str).strip().upper()
                    
                    # Extract hour and period (AM/PM)
                    if 'AM' in time_str or 'PM' in time_str:
                        time_part = time_str.split()[0]  # Get the hour part
                        period = 'AM' if 'AM' in time_str else 'PM'
                        
                        # Convert hour to int
                        hour = int(''.join(filter(str.isdigit, time_part)) or 0)
                        
                        # Handle 12 AM (midnight) and 12 PM (noon)
                        if period == 'AM':
                            return 0 if hour == 12 else hour
                        else:  # PM
                            return 12 if hour == 12 else hour + 12
                    
                    # If no AM/PM, assume 24-hour format
                    return int(''.join(filter(str.isdigit, time_str)) or 0)
                    
                except (ValueError, AttributeError):
                    return None
            
            # Apply the conversion to the hour column
            df_clean['hour'] = df_clean['hour'].apply(convert_12h_to_24h)
            
            # Drop rows where hour conversion failed
            df_clean = df_clean.dropna(subset=['hour'])
            df_clean['hour'] = df_clean['hour'].astype(int)
        
        # 6. Convert numeric columns to appropriate types
        numeric_cols = ['price', 'quantity', 'total']
        for col in numeric_cols:
            # Remove any non-numeric characters and convert to float
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace(r'[^\d.]', '', regex=True),
                errors='coerce'
            )
        
        # 7. Remove rows with null values in key columns
        key_columns = ['restaurant', 'hour', 'item', 'price', 'quantity', 'total']
        df_clean = df_clean.dropna(subset=key_columns)
        
        # 8. Ensure hour is between 0 and 23
        df_clean = df_clean[(df_clean['hour'] >= 0) & (df_clean['hour'] <= 23)]
        
        # 9. Ensure quantity and total are positive
        df_clean = df_clean[(df_clean['quantity'] > 0) & (df_clean['total'] > 0)]
        
        # 10. Reset index after filtering
        df_clean = df_clean.reset_index(drop=True)
        
        # Standardize column names to match the rest of the app
        df_clean = df_clean.rename(columns={'restaurant': 'restaurant_name'})
        
        # Log cleaning results
        cleaned_rows = len(df_clean)
        removed_rows = original_rows - cleaned_rows
        
        if removed_rows > 0:
            st.info(f"Cleaning complete: {removed_rows} rows removed, {cleaned_rows} rows remaining.")
        
        return df_clean, None
        
    except Exception as e:
        return None, f"Error during data cleaning: {str(e)}"

def validate_and_summarize_data(df):
    """
    Validate the cleaned data and display summary statistics.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check if data is empty after cleaning
        if df.empty:
            return False, "No valid data remains after cleaning."
        
        # Check for negative values in numeric columns
        numeric_cols = ['price', 'quantity', 'total']
        negative_values = {}
        for col in numeric_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    negative_values[col] = neg_count
        
        # Check hour range
        invalid_hours = df[~df['hour'].between(0, 23)]
        
        # Display summary statistics
        st.subheader("📊 Data Summary")
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Restaurants", df['restaurant_name'].nunique())
        with col3:
            st.metric("Unique Items", df['item'].nunique())
        
        # Hour range
        hour_min = int(df['hour'].min())
        hour_max = int(df['hour'].max())
        st.write("### Hour Range")
        st.write(f"{hour_min:02d}:00 - {hour_max:02d}:59")
        
        # Show validation warnings
        if negative_values:
            warning_msg = "⚠️ Warning: Negative values found in: "
            warning_msg += ", ".join([f"{k} ({v} rows)" for k, v in negative_values.items()])
            st.warning(warning_msg)
        
        if not invalid_hours.empty:
            st.warning(f"⚠️ Warning: Found {len(invalid_hours)} rows with invalid hour values (not 0-23)")
        
        # Show data preview
        st.subheader("🔍 Data Preview (First 10 Rows)")
        st.dataframe(df.head(10))
        
        # Show data types
        with st.expander("View Data Types"):
            st.write(df.dtypes.astype(str))
        
        # Show summary statistics
        with st.expander("View Summary Statistics"):
            st.write(df[['price', 'quantity', 'total']].describe())
        
        return True, None
        
    except Exception as e:
        return False, f"Error during data validation: {str(e)}"

def load_and_validate_data(uploaded_file):
    """
    Load and validate the uploaded file.
    
    Args:
        uploaded_file: The uploaded file object (CSV or Excel)
        
    Returns:
        Tuple of (DataFrame, list of missing columns, error message)
    """
    try:
        # Load the data
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if df.empty:
            return None, [], "Uploaded file is empty."
            
        # Clean the data using the clean_data function
        df_clean, error = clean_data(df)
        if error:
            return None, [], error
            
        # Validate and display summary
        is_valid, validation_error = validate_and_summarize_data(df_clean)
        if not is_valid:
            return None, [], validation_error
        
        return df_clean, [], None
        
    except Exception as e:
        return None, [], f"Error processing file: {str(e)}"

def generate_business_insights(hourly_revenue, period_revenue=None):
    """
    Generate business insights based on revenue data.
    
    Args:
        hourly_revenue: DataFrame with hourly revenue data
        period_revenue: Optional DataFrame with period revenue data
        
    Returns:
        List of insight strings
    """
    insights = []
    
    try:
        # Check if data is empty
        if hourly_revenue.empty:
            return ["Insufficient data to generate insights."]
            
        # 1. Identify peak hours
        if not hourly_revenue.empty and 'total' in hourly_revenue.columns:
            max_hour = hourly_revenue.loc[hourly_revenue['total'].idxmax()]
            insights.append(
                f"**Peak Hour:** The busiest hour is {int(max_hour['hour'])}:00 with "
                f"₹{max_hour['total']:,.2f} in revenue."
            )
            
            # 2. Identify slow hours (bottom 25% of revenue)
            revenue_25th_percentile = hourly_revenue['total'].quantile(0.25)
            slow_hours = hourly_revenue[hourly_revenue['total'] <= revenue_25th_percentile]
            
            if not slow_hours.empty and 'hour' in slow_hours.columns:
                slow_hour_list = ", ".join([f"{int(h)}:00" for h in slow_hours['hour']])
                insights.append(
                    f"**Slow Hours:** Consider promotions during these hours: {slow_hour_list}"
                )
        
        # 3. Compare time periods (if provided)
        if period_revenue is not None and not period_revenue.empty:
            try:
                if 'time_period' in period_revenue.columns and 'revenue' in period_revenue.columns:
                    lunch_data = period_revenue[period_revenue['time_period'].str.contains('Afternoon', na=False)]
                    dinner_data = period_revenue[period_revenue['time_period'].str.contains('Evening', na=False)]
                    
                    if not lunch_data.empty and not dinner_data.empty:
                        lunch_avg = lunch_data['revenue'].iloc[0]
                        dinner_avg = dinner_data['revenue'].iloc[0]
                        
                        if lunch_avg > dinner_avg * 1.2:  # 20% higher lunch revenue
                            insights.append(
                                "**Lunch Dominance:** Lunch service generates significantly more revenue than dinner. "
                                "Consider evening promotions to boost dinner business."
                            )
                        elif dinner_avg > lunch_avg * 1.2:  # 20% higher dinner revenue
                            insights.append(
                                "**Dinner Dominance:** Dinner service generates significantly more revenue than lunch. "
                                "Explore ways to boost lunchtime business."
                            )
            except (IndexError, KeyError):
                pass  # Skip period comparison if data format issues
        
        return insights if insights else ["No significant patterns detected in the data."]
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return ["Unable to generate insights due to data limitations."]

def process_hourly_revenue(df, date=None):
    """
    Process the cleaned data to analyze hourly revenue.
    
    Args:
        df: Cleaned DataFrame with 'hour' and 'total' columns
        date: Optional date to use for the analysis (default: today)
        
    Returns:
        DataFrame with hourly revenue data
    """
    try:
        # Group by hour and sum the total revenue
        hourly_revenue = df.groupby('hour')['total'].sum().reset_index()
        
        # Convert hour to datetime for better plotting
        if date is None:
            date = pd.Timestamp.today().date()
            
        hourly_revenue['datetime'] = pd.to_datetime(
            f"{date} " + hourly_revenue['hour'].astype(str).str.zfill(2) + ":00:00"
        )
        
        # Format time for display (12-hour format with AM/PM)
        hourly_revenue['time_display'] = hourly_revenue['datetime'].dt.strftime('%I %p').str.strip()
        
        return hourly_revenue.sort_values('hour')
    except Exception as e:
        st.error(f"Error processing hourly revenue data: {str(e)}")
        return pd.DataFrame()

def analyze_time_periods(df):
    """
    Analyze revenue by time periods.
    
    Args:
        df: DataFrame with 'hour' and 'total' columns
        
    Returns:
        DataFrame with time period analysis
    """
    try:
        # Define time periods
        conditions = [
            (df['hour'] >= 6) & (df['hour'] < 12),  # Morning: 6 AM - 11:59 AM
            (df['hour'] >= 12) & (df['hour'] < 18),  # Afternoon: 12 PM - 5:59 PM
            (df['hour'] >= 18) | (df['hour'] < 6)    # Evening: 6 PM - 5:59 AM
        ]
        period_names = ['Morning (6 AM - 12 PM)', 'Afternoon (12 PM - 6 PM)', 'Evening (6 PM - 6 AM)']
        
        # Add time period column
        df['time_period'] = np.select(conditions, period_names, default='Other')
        
        # Group by time period and calculate metrics
        period_revenue = df.groupby('time_period', observed=False).agg(
            revenue=('total', 'sum'),
            order_count=('total', 'count')
        ).reset_index()
        
        # Calculate percentage of total revenue
        total_rev = period_revenue['revenue'].sum()
        if total_rev > 0:
            period_revenue['percentage'] = (period_revenue['revenue'] / total_rev * 100).round(1)
        else:
            period_revenue['percentage'] = 0
        
        # Sort by time period order
        period_order = {period: i for i, period in enumerate(period_names)}
        period_revenue['sort_order'] = period_revenue['time_period'].map(period_order)
        period_revenue = period_revenue.sort_values('sort_order').drop('sort_order', axis=1)
        
        return period_revenue
    except Exception as e:
        st.error(f"Error analyzing time periods: {str(e)}")
        return pd.DataFrame()

# Initialize session state for data persistence
if 'df' not in st.session_state:
    st.session_state.df = None

# Initialize target session states
if 'daily_target' not in st.session_state:
    st.session_state.daily_target = 10000
if 'weekly_target' not in st.session_state:
    st.session_state.weekly_target = 70000
if 'monthly_target' not in st.session_state:
    st.session_state.monthly_target = 300000

# Initialize table configuration session states
if 'num_tables' not in st.session_state:
    st.session_state.num_tables = 20
if 'seats_per_table' not in st.session_state:
    st.session_state.seats_per_table = 4
if 'target_service_time' not in st.session_state:
    st.session_state.target_service_time = 45

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis",
    ["Data Overview", "Peak Hours Analysis", "Revenue Trends Analysis", "Sales Target Performance", "Table Turnover Rate", "Customer Lifetime Value"]
)

# Add clear data button
if st.sidebar.button("Clear Data"):
    st.session_state.df = None
    st.rerun()

# Main content
st.title("PetPooja Analytics Dashboard")

if page == "Data Overview":
    st.header("📊 Data Overview")
    
    # Instructions
    st.markdown("""
    ### Welcome to Data Overview
    
    Upload and validate your restaurant sales data to get started with the analysis.
    
    **Please upload the 'All Restaurants Sales: Hourly Item Wise' report from PetPooja dashboard.**  
    Required columns: Restaurant Name, Hour, Item, Price, Quantity, Total
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your sales data",
        type=['csv', 'xlsx'],
        key="data_overview_upload"
    )
    
    if uploaded_file is not None:
        # Load and validate data
        df, missing_columns, error = load_and_validate_data(uploaded_file)
        
        if error:
            st.error(f"Error: {error}")
        elif missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.warning("Please ensure your file contains all required columns.")
        elif df.empty:
            st.warning("No valid data found in the uploaded file after validation.")
        else:
            st.success("✅ File successfully loaded and validated!")
            
            # Store data in session state
            st.session_state.df = df
            
            # Display basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Unique Restaurants", df['restaurant_name'].nunique())
            with col3:
                st.metric("Total Revenue", f"₹{df['total'].sum():,.2f}")

elif page == "Peak Hours Analysis":
    st.header("📊 Peak Hours Analysis")
    
    # Instructions
    st.markdown("""
    ### Welcome to Peak Hours Analysis
    
    This section helps you analyze your restaurant's busiest hours.
    
    **Please upload the 'All Restaurants Sales: Hourly Item Wise' report from PetPooja dashboard.**  
    Required columns: Restaurant Name, Hour, Item, Price, Quantity, Total
    """)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.success("✅ Using previously loaded data!")
        
        # Display data summary
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Unique Restaurants", df['restaurant_name'].nunique())
        
        # --- Peak Hours Analysis ---
        st.markdown("---")
        st.header("📈 Peak Hours Analysis")
        
        try:
            # Group data by hour and calculate metrics
            hourly_data = df.groupby('hour').agg({
                'total': 'sum',
                'quantity': 'sum',
                'item': 'count'  # Count of orders
            }).reset_index()
            
            # Find peak hours
            peak_revenue_hour = hourly_data.loc[hourly_data['total'].idxmax()]
            peak_quantity_hour = hourly_data.loc[hourly_data['quantity'].idxmax()]
            
            # Display peak hour information
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Peak Revenue Hour", 
                         f"{int(peak_revenue_hour['hour'])}:00 - {int(peak_revenue_hour['hour'])+1}:00",
                         f"₹{peak_revenue_hour['total']:,.2f}")
            with col2:
                st.metric("Peak Quantity Hour", 
                         f"{int(peak_quantity_hour['hour'])}:00 - {int(peak_quantity_hour['hour'])+1}:00",
                         f"{int(peak_quantity_hour['quantity'])} items")
            
            # Create charts
            st.subheader("Revenue by Hour")
            st.bar_chart(
                data=hourly_data,
                x='hour',
                y='total',
                use_container_width=True
            )
            
            st.subheader("Quantity Sold by Hour")
            st.bar_chart(
                data=hourly_data,
                x='hour',
                y='quantity',
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error in peak hours analysis: {str(e)}")
    else:
        st.warning("Please upload data in the 'Data Overview' section first.")

elif page == "Revenue Trends Analysis":
    st.header("💰 Revenue Trends Analysis")
    
    # Instructions
    st.markdown("""
    ### Analyzing Hourly Revenue Patterns
    This section shows revenue trends by hour based on your sales data.
    """)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.success("✅ Using previously loaded data!")
        
        try:
            # Process the data for hourly revenue
            hourly_revenue = process_hourly_revenue(df)
            
            if not hourly_revenue.empty:
                # Display the processed data
                st.subheader("Hourly Revenue Summary")
                
                # Calculate key metrics
                total_revenue = hourly_revenue['total'].sum()
                avg_revenue = hourly_revenue['total'].mean()
                
                # Find peak and low hours
                peak_idx = hourly_revenue['total'].idxmax()
                low_idx = hourly_revenue['total'].idxmin()
                peak_hour = hourly_revenue.loc[peak_idx]
                low_hour = hourly_revenue.loc[low_idx]
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
                with col2:
                    st.metric("Peak Hour", 
                             f"{peak_hour['time_display']}", 
                             f"₹{peak_hour['total']:,.2f}")
                with col3:
                    st.metric("Avg. Hourly Revenue", f"₹{avg_revenue:,.2f}")
                
                # Create and display the line chart
                st.subheader("Hourly Revenue Distribution")
                
                # Calculate 7-hour rolling average
                hourly_revenue['rolling_avg'] = hourly_revenue['total'].rolling(
                    window=7,  # 7-hour window
                    min_periods=1,  # Return value even if fewer than 7 data points
                    center=True  # Center the window
                ).mean()
                
                # Create the line chart with both actual and rolling average
                fig = px.line(
                    hourly_revenue,
                    x='time_display',
                    y=['total', 'rolling_avg'],
                    labels={
                        'time_display': 'Hour of Day',
                        'value': 'Revenue (₹)',
                        'variable': 'Metric'
                    },
                    title='Hourly Revenue Trends with 7-Hour Rolling Average',
                    template='plotly_white'
                )
                
                # Update layout for better readability
                fig.update_layout(
                    yaxis_tickprefix='₹',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary metrics
                st.markdown(f"""
                ### Revenue Insights
                - **Peak Revenue Hour:** {peak_hour['time_display']} with **₹{peak_hour['total']:,.2f}**
                - **Lowest Revenue Hour:** {low_hour['time_display']} with **₹{low_hour['total']:,.2f}**
                - **Average Revenue per Hour:** **₹{avg_revenue:,.2f}**
                """)
                
                # --- Time Period Analysis ---
                st.subheader("Revenue by Time Period")
                
                # Analyze revenue by time periods
                period_revenue = analyze_time_periods(df)
                
                if not period_revenue.empty:
                    # Create bar chart
                    fig_period = px.bar(
                        period_revenue,
                        x='time_period',
                        y='revenue',
                        text='percentage',
                        labels={
                            'time_period': 'Time Period',
                            'revenue': 'Revenue (₹)',
                            'percentage': 'Percentage of Total'
                        },
                        title='Revenue Distribution by Time Period',
                        template='plotly_white'
                    )
                    
                    # Update layout
                    fig_period.update_layout(
                        xaxis_title=None,
                        yaxis_title='Revenue (₹)',
                        yaxis_tickprefix='₹',
                        showlegend=False,
                        height=500
                    )
                    
                    # Add value labels
                    fig_period.update_traces(
                        texttemplate='%{text}%',
                        textposition='outside'
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig_period, use_container_width=True)
                    
                    # Generate and display business insights
                    with st.expander("📈 Business Insights", expanded=True):
                        st.markdown("### Key Business Insights")
                        
                        # Generate insights
                        insights = generate_business_insights(hourly_revenue, period_revenue)
                        
                        # Display insights as bullet points
                        for insight in insights[:5]:  # Show max 5 most important insights
                            st.markdown(f"- {insight}")
                        
                        # Add timestamp
                        st.caption(f"*Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
                
                # Show the data table in an expander
                with st.expander("View Hourly Revenue Data"):
                    st.dataframe(hourly_revenue[['time_display', 'total']].rename(columns={
                        'time_display': 'Time',
                        'total': 'Revenue (₹)'
                    }), hide_index=True)
            else:
                st.error("Unable to process revenue data. Please check your data format.")
                
        except Exception as e:
            st.error(f"Error in revenue trends analysis: {str(e)}")
    else:
        st.warning("Please upload data in the 'Data Overview' section first.")

elif page == "Sales Target Performance":
    st.header("🎯 Sales Target Performance")
    
    # Instructions
    st.markdown("""
    ### Performance Against Sales Targets
    
    Set your sales targets and track your actual performance against them.
    This helps you understand if you're meeting your business goals and identify areas for improvement.
    """)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.success("✅ Using previously loaded data!")
        
        # Target Setting Section
        st.subheader("📊 Set Your Sales Targets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_target = st.number_input(
                "Daily Revenue Target (₹)", 
                min_value=0, 
                value=st.session_state.get('daily_target', 10000),
                step=1000
            )
        
        with col2:
            weekly_target = st.number_input(
                "Weekly Revenue Target (₹)", 
                min_value=0, 
                value=st.session_state.get('weekly_target', 70000),
                step=5000
            )
        
        with col3:
            monthly_target = st.number_input(
                "Monthly Revenue Target (₹)", 
                min_value=0, 
                value=st.session_state.get('monthly_target', 300000),
                step=10000
            )
        
        if st.button("💾 Save Targets"):
            st.session_state.daily_target = daily_target
            st.session_state.weekly_target = weekly_target
            st.session_state.monthly_target = monthly_target
            st.success("Targets saved successfully!")
        
        # Performance Analysis
        if all(key in st.session_state for key in ['daily_target', 'weekly_target', 'monthly_target']):
            st.markdown("---")
            st.subheader("📈 Performance Analysis")
            
            try:
                # Calculate actual performance
                total_revenue = df['total'].sum()
                unique_hours = df['hour'].nunique()
                
                # Calculate averages (simulating daily/weekly performance)
                daily_avg = total_revenue / max(1, unique_hours / 24) if unique_hours > 0 else 0
                weekly_avg = daily_avg * 7
                monthly_performance = total_revenue  # Treat current data as monthly sample
                
                # Calculate achievement percentages
                daily_achievement = (daily_avg / st.session_state.daily_target) * 100 if st.session_state.daily_target > 0 else 0
                weekly_achievement = (weekly_avg / st.session_state.weekly_target) * 100 if st.session_state.weekly_target > 0 else 0
                monthly_achievement = (monthly_performance / st.session_state.monthly_target) * 100 if st.session_state.monthly_target > 0 else 0
                
                # Display performance metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Daily Performance", 
                        f"{daily_achievement:.1f}%",
                        f"₹{daily_avg:,.0f} / ₹{st.session_state.daily_target:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Weekly Performance", 
                        f"{weekly_achievement:.1f}%",
                        f"₹{weekly_avg:,.0f} / ₹{st.session_state.weekly_target:,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "Monthly Performance", 
                        f"{monthly_achievement:.1f}%",
                        f"₹{monthly_performance:,.0f} / ₹{st.session_state.monthly_target:,.0f}"
                    )
                
                # Performance Chart
                st.subheader("🎯 Target Achievement Overview")
                
                performance_data = pd.DataFrame({
                    'Period': ['Daily', 'Weekly', 'Monthly'],
                    'Achievement': [daily_achievement, weekly_achievement, monthly_achievement],
                    'Target': [100, 100, 100]
                })
                
                fig = px.bar(
                    performance_data,
                    x='Period',
                    y='Achievement',
                    title='Target Achievement Percentage',
                    color='Achievement',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    range_color=[0, 150]
                )
                
                # Add target line at 100%
                fig.add_hline(
                    y=100,
                    line_dash='dash',
                    line_color='black',
                    annotation_text='Target (100%)',
                    annotation_position='right'
                )
                
                fig.update_layout(
                    yaxis_title='Achievement (%)',
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Business Insights
                with st.expander("📊 Performance Insights", expanded=True):
                    st.markdown("### Target Performance Analysis")
                    
                    insights = []
                    
                    # Overall performance insight
                    avg_performance = (daily_achievement + weekly_achievement + monthly_achievement) / 3
                    if avg_performance >= 100:
                        insights.append("🎉 **Excellent Performance!** You're consistently exceeding your sales targets across all periods.")
                    elif avg_performance >= 80:
                        insights.append("👍 **Good Performance!** You're close to your targets. Small improvements can help you exceed them.")
                    else:
                        insights.append("⚠️ **Below Target Performance.** Consider analyzing your peak hours and revenue trends to identify improvement opportunities.")
                    
                    # Specific period insights
                    if daily_achievement < 80:
                        insights.append("📅 **Daily Target:** Focus on increasing daily sales through promotions or extended hours during peak times.")
                    
                    if weekly_achievement >= 100:
                        insights.append("📊 **Weekly Strength:** Your weekly performance is strong. Consider replicating successful strategies.")
                    
                    if monthly_achievement < 90:
                        insights.append("📈 **Monthly Focus:** Your monthly performance needs attention. Consider seasonal marketing or menu optimization.")
                    
                    # Best performing period
                    best_period = performance_data.loc[performance_data['Achievement'].idxmax(), 'Period']
                    insights.append(f"🏆 **Best Performance:** Your {best_period.lower()} performance is strongest at {performance_data['Achievement'].max():.1f}%.")
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
                    
                    # Recommendations
                    st.markdown("### 💡 Recommendations")
                    if avg_performance < 90:
                        st.markdown("- Review your peak hours analysis to identify underutilized time slots")
                        st.markdown("- Consider promotional activities during slow periods")
                        st.markdown("- Analyze your most profitable menu items and promote them")
                    else:
                        st.markdown("- Maintain current successful strategies")
                        st.markdown("- Consider increasing targets for continued growth")
                        st.markdown("- Explore expansion opportunities")
                    
                    st.caption(f"*Analysis generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
            
            except Exception as e:
                st.error(f"Error in target performance analysis: {str(e)}")
        
        else:            
            st.info("👆 Please set your sales targets above to see performance analysis.")
    
    else:
        st.warning("Please upload data in the 'Data Overview' section first.")

elif page == "Table Turnover Rate":
    st.header("🔄 Table Turnover Rate Analysis")
    
    # Instructions
    st.markdown("""
    ### Real Table Turnover Rate Analysis with Dual Reports
    
    This analysis combines your hourly sales data with actual table usage data to provide precise table turnover insights.
    Upload both the hourly sales report (already loaded) and the table report for comprehensive analysis.
    """)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.success("✅ Hourly sales data loaded!")
        
        # Table Report Upload Section
        st.subheader("📋 Upload Table Report")
        st.info("Upload the table.csv file you received from PetPooja dashboard for real table turnover analysis")
        
        table_file = st.file_uploader(
            "Upload Table Report (CSV)",
            type=['csv'],
            key="table_report_upload",
            help="Upload the table-specific report from PetPooja dashboard with columns: Table Number, Order End Date, Order Start Time, Customer Count, Order Type Name"
        )
        
        if table_file is not None:
            try:
                # Load and clean table data
                table_df = pd.read_csv(table_file)
                
                st.info("🔧 Processing table data...")
                
                # Clean table data
                table_df_clean = table_df.copy()
                
                # Remove invalid table numbers (like "Shivam")
                valid_table_mask = pd.to_numeric(table_df_clean['Table Number'], errors='coerce').notna()
                table_df_clean = table_df_clean[valid_table_mask]
                table_df_clean['Table Number'] = pd.to_numeric(table_df_clean['Table Number'])
                
                # Remove rows with invalid Order Start Time
                invalid_time_mask = table_df_clean['Order Start Time'].str.contains('Shivam|undefined|NaN', na=True, regex=True)
                table_df_clean = table_df_clean[~invalid_time_mask]
                
                # Convert datetime columns
                table_df_clean['Order End Date'] = pd.to_datetime(table_df_clean['Order End Date'], errors='coerce')
                table_df_clean['Order Start Time'] = pd.to_datetime(table_df_clean['Order Start Time'], errors='coerce')
                
                # Remove rows with failed datetime conversion
                table_df_clean = table_df_clean.dropna(subset=['Order End Date', 'Order Start Time'])
                
                # Handle Customer Count (set realistic default party size if 0)
                table_df_clean['Customer Count'] = table_df_clean['Customer Count'].replace(0, 2.5)
                
                # Filter for Dine In orders only
                table_df_clean = table_df_clean[table_df_clean['Order Type Name'] == 'Dine In']
                
                # Extract hour from Order Start Time
                table_df_clean['hour'] = table_df_clean['Order Start Time'].dt.hour
                
                # Calculate service duration (if both start and end times are available)
                table_df_clean['service_duration'] = (
                    table_df_clean['Order End Date'] - table_df_clean['Order Start Time']
                ).dt.total_seconds() / 60  # Convert to minutes
                
                # Set realistic service duration bounds (15 minutes to 3 hours)
                table_df_clean['service_duration'] = table_df_clean['service_duration'].clip(15, 180)
                
                st.success("✅ Table data processed successfully!")
                
                # Display combined data summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Valid Table Records", len(table_df_clean))
                with col2:
                    active_tables = table_df_clean['Table Number'].nunique()
                    st.metric("Active Tables", active_tables)
                with col3:
                    avg_service_time = table_df_clean['service_duration'].mean()
                    st.metric("Avg Service Time", f"{avg_service_time:.0f} min")
                with col4:
                    total_customers = table_df_clean['Customer Count'].sum()
                    st.metric("Total Customers Served", f"{total_customers:.0f}")
                
                # Combined Analysis Section
                st.markdown("---")
                st.subheader("📊 Combined Table Turnover Analysis")
                
                # Calculate real hourly table usage
                hourly_table_usage = table_df_clean.groupby('hour').agg({
                    'Table Number': 'count',           # Number of table uses per hour
                    'Customer Count': 'sum',           # Total customers served per hour
                    'service_duration': 'mean'         # Average service time per hour
                }).reset_index()
                hourly_table_usage.columns = ['hour', 'table_uses', 'customers_served', 'avg_service_time']
                
                # Merge with hourly sales data
                hourly_sales = df.groupby('hour')['total'].sum().reset_index()
                
                combined_data = pd.merge(
                    hourly_sales,
                    hourly_table_usage,
                    on='hour',
                    how='outer'
                ).fillna(0)
                
                # Calculate real metrics
                combined_data['revenue_per_table_use'] = combined_data['total'] / combined_data['table_uses'].replace(0, 1)
                combined_data['revenue_per_customer'] = combined_data['total'] / combined_data['customers_served'].replace(0, 1)
                
                # Key performance metrics
                avg_table_uses_hour = combined_data['table_uses'].mean()
                peak_usage_hour = combined_data.loc[combined_data['table_uses'].idxmax()]
                total_table_uses = combined_data['table_uses'].sum()
                avg_revenue_per_use = combined_data['revenue_per_table_use'].mean()
                
                # Display real metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Avg Table Uses/Hour", 
                        f"{avg_table_uses_hour:.1f}",
                        "actual table seatings"
                    )
                
                with col2:
                    st.metric(
                        "Peak Usage Hour", 
                        f"{int(peak_usage_hour['hour'])}:00",
                        f"{int(peak_usage_hour['table_uses'])} uses"
                    )
                
                with col3:
                    st.metric(
                        "Total Table Uses", 
                        f"{int(total_table_uses)}",
                        "in analyzed period"
                    )
                
                with col4:
                    st.metric(
                        "Revenue/Table Use", 
                        f"₹{avg_revenue_per_use:,.0f}",
                        "per table seating"
                    )
                
                # Visualization: Real Table Usage vs Revenue
                st.subheader("📈 Hourly Table Usage vs Revenue")
                
                fig_combined = px.scatter(
                    combined_data,
                    x='table_uses',
                    y='total',
                    size='customers_served',
                    hover_data=['hour', 'avg_service_time'],
                    title='Table Usage vs Revenue by Hour',
                    labels={
                        'table_uses': 'Number of Table Uses',
                        'total': 'Revenue (₹)',
                        'customers_served': 'Customers Served'
                    },
                    template='plotly_white'
                )
                
                fig_combined.update_layout(height=500)
                st.plotly_chart(fig_combined, use_container_width=True)
                
                # Individual Table Performance
                st.subheader("🏪 Individual Table Performance")
                
                # Table-wise analysis
                table_performance = table_df_clean.groupby('Table Number').agg({
                    'Order Start Time': 'count',      # Number of uses
                    'Customer Count': 'sum',          # Total customers
                    'service_duration': 'mean'        # Average service time
                }).reset_index()
                table_performance.columns = ['Table Number', 'Total Uses', 'Total Customers', 'Avg Service Time']
                
                # Merge with revenue data by table (approximate based on proportions)
                table_revenue = []
                for table_num in table_performance['Table Number']:
                    table_orders = table_df_clean[table_df_clean['Table Number'] == table_num]
                    table_hours = table_orders['hour'].value_counts()
                    est_revenue = 0
                    for hour, count in table_hours.items():
                        hour_revenue = combined_data[combined_data['hour'] == hour]['total'].values
                        if len(hour_revenue) > 0:
                            # Proportional revenue allocation
                            total_uses_that_hour = combined_data[combined_data['hour'] == hour]['table_uses'].values[0]
                            if total_uses_that_hour > 0:
                                est_revenue += (hour_revenue[0] * count / total_uses_that_hour)
                    table_revenue.append(est_revenue)
                
                table_performance['Est Revenue'] = table_revenue
                table_performance['Revenue per Use'] = table_performance['Est Revenue'] / table_performance['Total Uses']
                
                # Create table performance chart
                fig_table_perf = px.bar(
                    table_performance,
                    x='Table Number',
                    y='Total Uses',
                    color='Revenue per Use',
                    title='Table Performance: Usage and Revenue per Use',
                    labels={
                        'Table Number': 'Table Number',
                        'Total Uses': 'Number of Uses',
                        'Revenue per Use': 'Revenue per Use (₹)'
                    },
                    template='plotly_white'
                )
                
                fig_table_perf.update_layout(height=400)
                st.plotly_chart(fig_table_perf, use_container_width=True)
                
                # Service Time Analysis
                st.subheader("⏱️ Service Time Patterns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Service time distribution
                    fig_service = px.histogram(
                        table_df_clean,
                        x='service_duration',
                        nbins=20,
                        title='Service Time Distribution',
                        labels={'service_duration': 'Service Duration (minutes)'}
                    )
                    st.plotly_chart(fig_service, use_container_width=True)
                
                with col2:
                    # Service time by hour
                    service_by_hour = table_df_clean.groupby('hour')['service_duration'].mean().reset_index()
                    fig_service_hour = px.line(
                        service_by_hour,
                        x='hour',
                        y='service_duration',
                        title='Average Service Time by Hour',
                        labels={'service_duration': 'Service Duration (minutes)'}
                    )
                    st.plotly_chart(fig_service_hour, use_container_width=True)
                
                # Real Business Insights
                with st.expander("🔄 Real Table Turnover Insights", expanded=True):
                    st.markdown("### Data-Driven Table Performance Analysis")
                    
                    insights = []
                    
                    # Most/least popular tables
                    most_used_table = table_performance.loc[table_performance['Total Uses'].idxmax()]
                    least_used_table = table_performance.loc[table_performance['Total Uses'].idxmin()]
                    
                    insights.append(f"🏆 **Most Popular Table:** Table {int(most_used_table['Table Number'])} with {int(most_used_table['Total Uses'])} uses")
                    insights.append(f"📉 **Least Used Table:** Table {int(least_used_table['Table Number'])} with {int(least_used_table['Total Uses'])} uses")
                    
                    # Revenue efficiency
                    highest_revenue_table = table_performance.loc[table_performance['Revenue per Use'].idxmax()]
                    insights.append(f"💰 **Highest Revenue/Use:** Table {int(highest_revenue_table['Table Number'])} generating ₹{highest_revenue_table['Revenue per Use']:,.0f} per use")
                    
                    # Service time insights
                    avg_service = table_df_clean['service_duration'].mean()
                    if avg_service > 60:
                        insights.append(f"⏱️ **Service Time:** Average of {avg_service:.0f} minutes suggests opportunity to speed up service")
                    else:
                        insights.append(f"⚡ **Efficient Service:** Average {avg_service:.0f} minutes service time is excellent")
                    
                    # Peak efficiency
                    peak_hour_int = int(peak_usage_hour['hour'])
                    insights.append(f"🕐 **Peak Efficiency:** {peak_hour_int}:00 shows highest table utilization with {int(peak_usage_hour['table_uses'])} uses")
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
                    
                    # Actionable recommendations
                    st.markdown("### 💡 Data-Driven Recommendations")
                    
                    recommendations = []
                    
                    # Table optimization
                    usage_variance = table_performance['Total Uses'].std()
                    if usage_variance > 2:
                        recommendations.append("📐 **Table Layout:** High usage variance suggests repositioning popular table configurations")
                    
                    # Service optimization
                    if avg_service > 45:
                        recommendations.append("⚡ **Service Speed:** Consider process improvements to reduce average service time")
                    
                    # Revenue optimization
                    revenue_variance = table_performance['Revenue per Use'].std()
                    if revenue_variance > 200:  # High variance in revenue per table
                        recommendations.append("💰 **Revenue Strategy:** Focus on increasing revenue at lower-performing tables")
                    
                    # Capacity utilization
                    if avg_table_uses_hour < active_tables * 0.5:  # Less than 50% utilization
                        recommendations.append("📈 **Capacity Utilization:** Significant opportunity to increase table turnover")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    st.caption(f"*Real data analysis generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
                
                # Detailed data tables
                with st.expander("📊 View Detailed Performance Data"):
                    st.markdown("#### Combined Hourly Data")
                    display_combined = combined_data[['hour', 'table_uses', 'total', 'customers_served', 'avg_service_time']].copy()
                    display_combined.columns = ['Hour', 'Table Uses', 'Revenue (₹)', 'Customers', 'Avg Service (min)']
                    display_combined['Hour'] = display_combined['Hour'].apply(lambda x: f"{int(x)}:00")
                    st.dataframe(display_combined, hide_index=True)
                    
                    st.markdown("#### Individual Table Performance")
                    display_tables = table_performance[['Table Number', 'Total Uses', 'Total Customers', 'Avg Service Time', 'Est Revenue', 'Revenue per Use']].copy()
                    display_tables['Est Revenue'] = display_tables['Est Revenue'].round(0)
                    display_tables['Revenue per Use'] = display_tables['Revenue per Use'].round(0)
                    display_tables['Avg Service Time'] = display_tables['Avg Service Time'].round(1)
                    st.dataframe(display_tables, hide_index=True)
            
            except Exception as e:
                st.error(f"Error processing table data: {str(e)}")
                st.info("Please ensure the table report CSV has columns: Table Number, Order End Date, Order Start Time, Customer Count, Order Type Name")
        
        else:
            st.info("👆 Upload the table report to see real table turnover analysis using both reports.")
            
            # Show estimated analysis while waiting for table report
            st.markdown("---")
            st.subheader("📊 Estimated Table Analysis (Sales Data Only)")
            st.info("This is estimated analysis based on sales data. Upload table report above for accurate analysis.")
            
            # Table Configuration for estimation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_tables = st.number_input("Est. Number of Tables", min_value=1, value=8, step=1)
            with col2:
                seats_per_table = st.number_input("Est. Seats per Table", min_value=1, value=4, step=1)
            with col3:
                target_service_time = st.number_input("Target Service Time (min)", min_value=15, value=45, step=5)
            
            # Basic estimated metrics
            total_capacity = num_tables * seats_per_table
            hourly_data = df.groupby('hour').agg({'total': 'sum', 'item': 'count'}).reset_index()
            hourly_data['estimated_customers'] = hourly_data['item'] * 2.5
            hourly_data['estimated_turns'] = hourly_data['estimated_customers'] / total_capacity
            
            avg_estimated_turns = hourly_data['estimated_turns'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Avg Turns/Hour", f"{avg_estimated_turns:.2f}")
            with col2:
                st.metric("Estimated Total Capacity", f"{total_capacity} seats")
            
            st.info("⬆️ Upload table report above for precise analysis with real table usage data!")
    
    else:
        st.warning("Please upload sales data in the 'Data Overview' section first.")

elif page == "Customer Lifetime Value":
    st.header("👥 Customer Lifetime Value Analysis")
    
    # Instructions
    st.markdown("""
    ### Customer Lifetime Value (CLV) Analysis
    
    Analyze customer behavior patterns, lifetime value, and segmentation to optimize retention strategies.
    Upload the customer transaction report from PetPooja dashboard for comprehensive customer insights.
    """)
    
    if st.session_state.df is not None:
        st.success("✅ Sales data loaded!")
        
        # Customer Report Upload Section
        st.subheader("📋 Upload Customer Report")
        st.info("Upload the customer.csv file from PetPooja dashboard for CLV analysis")
        
        customer_file = st.file_uploader(
            "Upload Customer Report (CSV)",
            type=['csv'],
            key="customer_report_upload",
            help="Upload customer transaction data with columns: Customer_Name, Net_Sales, Customer_ID, Transaction_Timestamp, etc."
        )
        
        if customer_file is not None:
            try:
                # Load customer data
                customer_df = pd.read_csv(customer_file)
                
                st.info("🔧 Processing customer data...")
                
                # Clean customer data
                customer_df_clean = customer_df.copy()
                
                # Convert data types
                customer_df_clean['Net_Sales'] = pd.to_numeric(customer_df_clean['Net_Sales'], errors='coerce')
                customer_df_clean['Customer_ID'] = pd.to_numeric(customer_df_clean['Customer_ID'], errors='coerce')
                customer_df_clean['Order_Frequency'] = pd.to_numeric(customer_df_clean['Order_Frequency'], errors='coerce')
                
                # Convert date columns
                customer_df_clean['Transaction_Date'] = pd.to_datetime(customer_df_clean['Transaction_Date'])
                customer_df_clean['First_Visit_Date'] = pd.to_datetime(customer_df_clean['First_Visit_Date'])
                customer_df_clean['Last_Visit_Date'] = pd.to_datetime(customer_df_clean['Last_Visit_Date'])
                customer_df_clean['Transaction_Timestamp'] = pd.to_datetime(customer_df_clean['Transaction_Timestamp'])
                
                # Handle customer identification (anonymous vs registered)
                customer_df_clean['Customer_Type'] = customer_df_clean.apply(
                    lambda x: 'Registered' if x['Customer_ID'] != 0 and str(x['Customer_Name']).lower() != 'null' 
                    else 'Anonymous', axis=1
                )
                
                # Create unique customer identifier
                customer_df_clean['Unique_Customer_ID'] = customer_df_clean.apply(
                    lambda x: f"REG_{x['Customer_ID']}" if x['Customer_Type'] == 'Registered' 
                    else f"ANON_{x.name}", axis=1
                )
                
                st.success("✅ Customer data processed successfully!")
                
                # Display customer data summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(customer_df_clean))
                with col2:
                    registered_customers = (customer_df_clean['Customer_Type'] == 'Registered').sum()
                    st.metric("Registered Customers", registered_customers)
                with col3:
                    anonymous_customers = (customer_df_clean['Customer_Type'] == 'Anonymous').sum()
                    st.metric("Anonymous Customers", anonymous_customers)
                with col4:
                    total_revenue = customer_df_clean['Net_Sales'].sum()
                    st.metric("Total Customer Revenue", f"₹{total_revenue:,.2f}")
                
                # Customer Lifetime Value Analysis
                st.markdown("---")
                st.subheader("📊 Customer Lifetime Value Analysis")
                
                # Calculate CLV metrics per customer
                clv_metrics = customer_df_clean.groupby('Unique_Customer_ID').agg({
                    'Net_Sales': ['sum', 'mean', 'count'],
                    'Transaction_Date': ['min', 'max'],
                    'Customer_Type': 'first',
                    'Customer_Name': 'first',
                    'Phone_Number': 'first'
                }).reset_index()
                
                # Flatten column names
                clv_metrics.columns = [
                    'Unique_Customer_ID', 'Total_Revenue', 'Avg_Order_Value', 'Total_Orders',
                    'First_Order_Date', 'Last_Order_Date', 'Customer_Type', 'Customer_Name', 'Phone_Number'
                ]
                
                # Calculate customer lifespan in days
                clv_metrics['Customer_Lifespan_Days'] = (
                    clv_metrics['Last_Order_Date'] - clv_metrics['First_Order_Date']
                ).dt.days + 1  # Add 1 to include same-day customers
                
                # Calculate order frequency (orders per day)
                clv_metrics['Order_Frequency'] = clv_metrics['Total_Orders'] / clv_metrics['Customer_Lifespan_Days']
                clv_metrics['Order_Frequency'] = clv_metrics['Order_Frequency'].fillna(1)  # For single-day customers
                
                # Calculate predicted CLV (simplified model)
                # CLV = AOV × Order Frequency × Customer Lifespan (estimated)
                avg_lifespan = clv_metrics['Customer_Lifespan_Days'].mean()
                clv_metrics['Predicted_CLV'] = (
                    clv_metrics['Avg_Order_Value'] * 
                    clv_metrics['Order_Frequency'] * 
                    avg_lifespan
                )
                
                # Customer Segmentation
                # Define segments based on Total Revenue and Order Frequency
                revenue_75th = clv_metrics['Total_Revenue'].quantile(0.75)
                revenue_25th = clv_metrics['Total_Revenue'].quantile(0.25)
                freq_median = clv_metrics['Total_Orders'].median()
                
                def segment_customer(row):
                    if row['Total_Revenue'] >= revenue_75th and row['Total_Orders'] >= freq_median:
                        return 'VIP Customers'
                    elif row['Total_Revenue'] >= revenue_75th:
                        return 'High Value'
                    elif row['Total_Orders'] >= freq_median:
                        return 'Frequent Customers'
                    elif row['Total_Revenue'] <= revenue_25th:
                        return 'Low Value'
                    else:
                        return 'Regular Customers'
                
                clv_metrics['Customer_Segment'] = clv_metrics.apply(segment_customer, axis=1)
                
                # Display key CLV metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_clv = clv_metrics['Total_Revenue'].mean()
                    st.metric("Avg CLV", f"₹{avg_clv:,.2f}")
                
                with col2:
                    avg_aov = clv_metrics['Avg_Order_Value'].mean()
                    st.metric("Avg Order Value", f"₹{avg_aov:,.2f}")
                
                with col3:
                    avg_orders = clv_metrics['Total_Orders'].mean()
                    st.metric("Avg Orders per Customer", f"{avg_orders:.1f}")
                
                with col4:
                    avg_lifespan_display = clv_metrics['Customer_Lifespan_Days'].mean()
                    st.metric("Avg Customer Lifespan", f"{avg_lifespan_display:.0f} days")
                
                # Customer Segmentation Visualization
                st.subheader("👥 Customer Segmentation")
                
                segment_summary = clv_metrics.groupby('Customer_Segment').agg({
                    'Unique_Customer_ID': 'count',
                    'Total_Revenue': 'sum',
                    'Avg_Order_Value': 'mean'
                }).reset_index()
                segment_summary.columns = ['Segment', 'Customer_Count', 'Total_Revenue', 'Avg_AOV']
                segment_summary['Revenue_Percentage'] = (
                    segment_summary['Total_Revenue'] / segment_summary['Total_Revenue'].sum() * 100
                ).round(1)
                
                # Create segment visualization
                fig_segments = px.scatter(
                    clv_metrics,
                    x='Total_Orders',
                    y='Total_Revenue',
                    color='Customer_Segment',
                    size='Avg_Order_Value',
                    hover_data=['Customer_Name', 'Customer_Type'],
                    title='Customer Segmentation: Orders vs Revenue',
                    labels={
                        'Total_Orders': 'Total Orders',
                        'Total_Revenue': 'Total Revenue (₹)',
                        'Customer_Segment': 'Segment'
                    },
                    template='plotly_white'
                )
                
                fig_segments.update_layout(height=500)
                st.plotly_chart(fig_segments, use_container_width=True)
                
                # Segment Summary
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_segment_count = px.pie(
                        segment_summary,
                        values='Customer_Count',
                        names='Segment',
                        title='Customer Distribution by Segment',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_segment_count, use_container_width=True)
                
                with col2:
                    fig_segment_revenue = px.bar(
                        segment_summary,
                        x='Segment',
                        y='Revenue_Percentage',
                        title='Revenue Contribution by Segment (%)',
                        template='plotly_white'
                    )
                    fig_segment_revenue.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_segment_revenue, use_container_width=True)
                
                # Top Customers Analysis
                st.subheader("🏆 Top Customers Analysis")
                
                top_customers = clv_metrics.nlargest(10, 'Total_Revenue')[
                    ['Customer_Name', 'Customer_Type', 'Total_Revenue', 'Total_Orders', 'Avg_Order_Value', 'Customer_Segment']
                ]
                
                # Replace null with "Anonymous Customer"
                top_customers['Customer_Name'] = top_customers['Customer_Name'].apply(
                    lambda x: 'Anonymous Customer' if str(x).lower() == 'null' else x
                )
                
                fig_top_customers = px.bar(
                    top_customers,
                    x='Customer_Name',
                    y='Total_Revenue',
                    color='Customer_Segment',
                    title='Top 10 Customers by Revenue',
                    labels={'Total_Revenue': 'Total Revenue (₹)'},
                    template='plotly_white'
                )
                fig_top_customers.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_top_customers, use_container_width=True)
                
                # Customer Behavior Patterns
                st.subheader("📈 Customer Behavior Patterns")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Order frequency distribution
                    fig_freq = px.histogram(
                        clv_metrics,
                        x='Total_Orders',
                        nbins=20,
                        title='Order Frequency Distribution',
                        labels={'Total_Orders': 'Number of Orders'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_freq, use_container_width=True)
                
                with col2:
                    # AOV distribution
                    fig_aov = px.histogram(
                        clv_metrics,
                        x='Avg_Order_Value',
                        nbins=20,
                        title='Average Order Value Distribution',
                        labels={'Avg_Order_Value': 'Average Order Value (₹)'},
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_aov, use_container_width=True)
                
                # Customer Type Analysis
                st.subheader("📱 Registered vs Anonymous Customers")
                
                customer_type_analysis = clv_metrics.groupby('Customer_Type').agg({
                    'Unique_Customer_ID': 'count',
                    'Total_Revenue': ['sum', 'mean'],
                    'Avg_Order_Value': 'mean',
                    'Total_Orders': 'mean'
                }).round(2)
                
                customer_type_analysis.columns = [
                    'Count', 'Total_Revenue', 'Avg_Revenue', 'Avg_AOV', 'Avg_Orders'
                ]
                customer_type_analysis = customer_type_analysis.reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Customer Type Comparison")
                    st.dataframe(customer_type_analysis, hide_index=True)
                
                with col2:
                    if len(customer_type_analysis) > 1:
                        reg_avg_revenue = customer_type_analysis[
                            customer_type_analysis['Customer_Type'] == 'Registered'
                        ]['Avg_Revenue'].iloc[0] if not customer_type_analysis[
                            customer_type_analysis['Customer_Type'] == 'Registered'
                        ].empty else 0
                        
                        anon_avg_revenue = customer_type_analysis[
                            customer_type_analysis['Customer_Type'] == 'Anonymous'
                        ]['Avg_Revenue'].iloc[0] if not customer_type_analysis[
                            customer_type_analysis['Customer_Type'] == 'Anonymous'
                        ].empty else 0
                        
                        if reg_avg_revenue > 0 and anon_avg_revenue > 0:
                            value_difference = ((reg_avg_revenue - anon_avg_revenue) / anon_avg_revenue * 100)
                            st.metric(
                                "Registered vs Anonymous",
                                f"{value_difference:+.1f}%",
                                "Revenue difference"
                            )
                
                # Business Insights
                with st.expander("👥 Customer Lifetime Value Insights", expanded=True):
                    st.markdown("### CLV Analysis Insights")
                    
                    insights = []
                    
                    # Customer base analysis
                    total_customers = len(clv_metrics)
                    registered_pct = (clv_metrics['Customer_Type'] == 'Registered').mean() * 100
                    insights.append(f"📊 **Customer Base:** {total_customers} customers with {registered_pct:.1f}% registered")
                    
                    # VIP customers
                    vip_customers = len(clv_metrics[clv_metrics['Customer_Segment'] == 'VIP Customers'])
                    vip_revenue_pct = segment_summary[segment_summary['Segment'] == 'VIP Customers']['Revenue_Percentage'].iloc[0] if not segment_summary[segment_summary['Segment'] == 'VIP Customers'].empty else 0
                    if vip_customers > 0:
                        insights.append(f"👑 **VIP Customers:** {vip_customers} customers generate {vip_revenue_pct:.1f}% of total revenue")
                    
                    # High value insights
                    high_value_customers = len(clv_metrics[clv_metrics['Customer_Segment'] == 'High Value'])
                    if high_value_customers > 0:
                        insights.append(f"💎 **High Value Segment:** {high_value_customers} customers show high revenue potential")
                    
                    # Customer retention opportunity
                    single_order_customers = len(clv_metrics[clv_metrics['Total_Orders'] == 1])
                    single_order_pct = (single_order_customers / total_customers) * 100
                    insights.append(f"🔄 **Retention Opportunity:** {single_order_pct:.1f}% are one-time customers")
                    
                    # Revenue concentration
                    top_20_pct_customers = int(total_customers * 0.2)
                    if top_20_pct_customers > 0:
                        top_customers_revenue = clv_metrics.nlargest(top_20_pct_customers, 'Total_Revenue')['Total_Revenue'].sum()
                        top_customers_pct = (top_customers_revenue / clv_metrics['Total_Revenue'].sum()) * 100
                        insights.append(f"📈 **Revenue Concentration:** Top 20% customers contribute {top_customers_pct:.1f}% of revenue")
                    
                    for insight in insights:
                        st.markdown(f"- {insight}")
                    
                    # Actionable recommendations
                    st.markdown("### 💡 CLV Optimization Recommendations")
                    
                    recommendations = []
                    
                    # Registration strategy
                    if registered_pct < 30:
                        recommendations.append("📱 **Customer Registration:** Implement incentives to convert anonymous customers to registered users")
                    
                    # Retention strategy
                    if single_order_pct > 60:
                        recommendations.append("🔄 **Retention Focus:** Develop follow-up campaigns for one-time customers")
                    
                    # VIP program
                    if vip_customers > 0:
                        recommendations.append("👑 **VIP Program:** Create exclusive offers and experiences for high-value customers")
                    
                    # Frequency improvement
                    avg_frequency = clv_metrics['Order_Frequency'].mean()
                    if avg_frequency < 0.1:  # Less than once per 10 days
                        recommendations.append("📅 **Frequency Building:** Launch loyalty programs to increase visit frequency")
                    
                    # AOV improvement
                    if avg_aov < 500:
                        recommendations.append("💰 **AOV Enhancement:** Implement upselling strategies and combo offers")
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    st.caption(f"*CLV analysis generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
                
                # Detailed customer data
                with st.expander("📊 View Detailed Customer Data"):
                    st.markdown("#### Customer Segments Summary")
                    st.dataframe(segment_summary, hide_index=True)
                    
                    st.markdown("#### Top 20 Customers by CLV")
                    top_20_display = clv_metrics.nlargest(20, 'Total_Revenue')[
                        ['Customer_Name', 'Customer_Type', 'Total_Revenue', 'Total_Orders', 
                         'Avg_Order_Value', 'Customer_Segment', 'Customer_Lifespan_Days']
                    ].copy()
                    top_20_display['Customer_Name'] = top_20_display['Customer_Name'].apply(
                        lambda x: 'Anonymous' if str(x).lower() == 'null' else x
                    )
                    top_20_display.columns = [
                        'Customer', 'Type', 'Total Revenue (₹)', 'Orders', 
                        'AOV (₹)', 'Segment', 'Lifespan (Days)'
                    ]
                    st.dataframe(top_20_display, hide_index=True)
            
            except Exception as e:
                st.error(f"Error processing customer data: {str(e)}")
                st.info("Please ensure the customer CSV has the required columns: Customer_Name, Net_Sales, Customer_ID, Transaction_Timestamp, etc.")
        
        else:
            st.info("👆 Upload the customer report to analyze Customer Lifetime Value and customer segments.")
    
    else:
        st.warning("Please upload sales data in the 'Data Overview' section first.")
