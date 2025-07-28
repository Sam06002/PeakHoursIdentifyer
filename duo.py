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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis",
    ["Data Overview", "Peak Hours Analysis", "Revenue Trends Analysis", "Sales Target Performance"]
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
