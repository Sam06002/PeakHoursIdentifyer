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
            
        # Standardize column names to match the rest of the app
        df_clean = df_clean.rename(columns={
            'restaurant': 'restaurant_name'
        })
        
        # Validate and display summary
        is_valid, validation_error = validate_and_summarize_data(df_clean)
        if not is_valid:
            return None, [], validation_error
        
        return df_clean, [], None
        
    except Exception as e:
        return None, [], f"Error processing file: {str(e)}"
    
    try:
        # Load the data
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if df.empty:
            return None, [], "Uploaded file is empty."
        
        # Clean column names (remove extra spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map actual columns to standard names
        column_mapping = {}
        missing_columns = []
        
        for std_name, possible_names in required_columns.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    column_mapping[std_name] = name
                    found = True
                    break
            
            if not found:
                missing_columns.append(std_name)
        
        if missing_columns:
            return df, missing_columns, None
        
        # Standardize column names
        df_clean = df.rename(columns={
            column_mapping['restaurant']: 'restaurant_name',
            column_mapping['hour']: 'hour',
            column_mapping['item']: 'item',
            column_mapping['price']: 'price',
            column_mapping['quantity']: 'quantity',
            column_mapping['total']: 'total'
        })
        
        # Convert data types
        numeric_cols = ['hour', 'price', 'quantity', 'total']
        for col in numeric_cols:
            # Try to convert to numeric, forcing errors to NaN
            df_clean[col] = pd.to_numeric(
                df_clean[col].astype(str).str.replace(r'[^\d.]', '', regex=True), 
                errors='coerce'
            )
        
        # Remove rows with null values in key columns
        key_columns = ['restaurant_name', 'hour', 'item', 'price', 'quantity', 'total']
        df_clean = df_clean.dropna(subset=key_columns)
        
        # Ensure hour is between 0 and 23
        df_clean = df_clean[(df_clean['hour'] >= 0) & (df_clean['hour'] <= 23)]
        
        # Ensure quantity and total are positive
        df_clean = df_clean[(df_clean['quantity'] > 0) & (df_clean['total'] > 0)]
        
        return df_clean, [], None
        
    except Exception as e:
        return None, [], f"Error processing file: {str(e)}"

def generate_business_insights(hourly_revenue, period_revenue):
    """
    Generate business insights based on revenue data.
    
    Args:
        hourly_revenue: DataFrame with hourly revenue data
        period_revenue: DataFrame with period revenue data
        
    Returns:
        List of insight strings
    """
    insights = []
    
    try:
        # 1. Peak sales period
        peak_hour = hourly_revenue.loc[hourly_revenue['total'].idxmax()]
        insights.append(
            f"**Peak Sales Period:** {peak_hour['time_display']} with **₹{peak_hour['total']:,.2f}** in revenue "
            f"(highest of the day)"
        )
        
        # 2. Revenue growth between consecutive hours
        hourly_revenue['revenue_growth'] = hourly_revenue['total'].pct_change() * 100
        max_growth = hourly_revenue.nlargest(1, 'revenue_growth')
        if not max_growth.empty and max_growth['revenue_growth'].iloc[0] > 0:
            growth_time = max_growth['time_display'].iloc[0]
            growth_pct = max_growth['revenue_growth'].iloc[0]
            insights.append(
                f"**Biggest Revenue Jump:** {growth_time} with a {growth_pct:.1f}% increase from the previous hour"
            )
        
        # 3. Busiest vs slowest periods
        if not period_revenue.empty:
            busiest = period_revenue.loc[period_revenue['revenue'].idxmax()]
            slowest = period_revenue.loc[period_revenue['revenue'].idxmin()]
            
            insights.append(
                f"**Busiest Period:** {busiest['time_period']} generates {busiest['percentage']:.1f}% of daily revenue "
                f"(₹{busiest['revenue']:,.2f})"
            )
            
            if slowest['revenue'] > 0:
                ratio = busiest['revenue'] / slowest['revenue']
                if ratio > 2:  # Only suggest optimization if there's significant difference
                    insights.append(
                        f"**Optimization Opportunity:** Consider increasing staff or promotions during {slowest['time_period']} "
                        f"(only {slowest['percentage']:.1f}% of daily revenue)"
                    )
        
        # 4. Revenue consistency
        revenue_std = hourly_revenue['total'].std()
        revenue_mean = hourly_revenue['total'].mean()
        if revenue_mean > 0:
            cv = (revenue_std / revenue_mean) * 100
            if cv > 50:
                insights.append(
                    "**Revenue Volatility:** High variation in hourly revenue. Consider analyzing factors "
                    "causing these fluctuations."
                )
            else:
                insights.append(
                    "**Revenue Stability:** Consistent revenue flow throughout the day "
                    f"(coefficient of variation: {cv:.1f}%)"
                )
        
        # 5. Identify potential lunch/dinner rush
        lunch_hours = hourly_revenue[hourly_revenue['hour'].between(11, 14)]
        dinner_hours = hourly_revenue[hourly_revenue['hour'].between(19, 22)]
        
        if not lunch_hours.empty and not dinner_hours.empty:
            lunch_avg = lunch_hours['total'].mean()
            dinner_avg = dinner_hours['total'].mean()
            
            if lunch_avg > dinner_avg * 1.2:  # 20% higher lunch revenue
                insights.append(
                    "**Strong Lunch Business:** Lunch revenue is significantly higher than dinner. "
                    "Consider capitalizing on this trend with targeted lunch promotions."
                )
            elif dinner_avg > lunch_avg * 1.2:  # 20% higher dinner revenue
                insights.append(
                    "**Dinner Dominance:** Dinner service generates significantly more revenue than lunch. "
                    "Explore ways to boost lunchtime business."
                )
        
        return insights
    
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        return ["Unable to generate insights due to data limitations."]

def analyze_time_periods(df):
    """
    Analyze revenue by time periods (Morning/Afternoon/Evening).
    
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
        period_revenue['percentage'] = (period_revenue['revenue'] / total_rev * 100).round(1)
        
        # Sort by time period order
        period_order = {period: i for i, period in enumerate(period_names)}
        period_revenue['sort_order'] = period_revenue['time_period'].map(period_order)
        period_revenue = period_revenue.sort_values('sort_order').drop('sort_order', axis=1)
        
        return period_revenue
    except Exception as e:
        st.error(f"Error analyzing time periods: {str(e)}")
        return pd.DataFrame()

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
        hourly_revenue['time_display'] = hourly_revenue['datetime'].dt.strftime('%I %p')
        
        return hourly_revenue.sort_values('hour')
    except Exception as e:
        st.error(f"Error processing hourly revenue data: {str(e)}")
        return pd.DataFrame()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Analysis",
    ["Data Overview", "Peak Hours Analysis", "Revenue Trends Analysis"]
)

# Main content
st.title("PetPooja Analytics Dashboard")

if page == "Peak Hours Analysis":
    st.header("📊 Peak Hours Analysis")
    
    # Instructions
    st.markdown("""
    ### Welcome to Peak Hours Analysis
    
    This section helps you analyze your restaurant's busiest hours.
    
    **Please upload the 'All Restaurants Sales: Hourly Item Wise' report from PetPooja dashboard.**  
    Required columns: Restaurant Name, Hour, Item, Price, Quantity, Total
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your sales data",
        type=['csv', 'xlsx'],
        key="peak_hours_upload"
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
            
            # Display data summary
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Unique Restaurants", df['restaurant_name'].nunique())
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Show data statistics
            with st.expander("View Data Statistics"):
                st.write("### Data Types")
                st.write(df.dtypes.astype(str))
                
                st.write("### Summary Statistics")
                st.dataframe(df.describe())
            
            # --- Peak Hours Analysis ---
            st.markdown("---")
            st.header("📈 Peak Hours Analysis")
            
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
    
elif page == "Revenue Trends Analysis":
    st.header("💰 Revenue Trends Analysis")
    
    # Instructions
    st.markdown("""
    ### Analyzing Hourly Revenue Patterns
    This section shows revenue trends by hour based on your sales data.
    """)
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please upload and process your data in the 'Data Overview' section first.")
    else:
        # Process the data for hourly revenue
        hourly_revenue = process_hourly_revenue(st.session_state.df)
        
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
            line_shape='spline',
            template='plotly_white',
            color_discrete_map={
                'total': '#1f77b4',  # Blue for actual values
                'rolling_avg': '#ff7f0e'  # Orange for rolling average
            }
        )
        
        # Add peak and low point markers
        fig.add_scatter(
            x=[peak_hour['time_display']],
            y=[peak_hour['total']],
            mode='markers+text',
            marker=dict(color='red', size=12, symbol='star'),
            name='Peak Hour',
            text=['Peak'],
            textposition='top center',
            showlegend=False
        )
        
        fig.add_scatter(
            x=[low_hour['time_display']],
            y=[low_hour['total']],
            mode='markers+text',
            marker=dict(color='blue', size=10, symbol='x'),
            name='Lowest Hour',
            text=['Low'],
            textposition='bottom center',
            showlegend=False
        )
        
        # Add average line
        fig.add_hline(
            y=avg_revenue,
            line_dash='dash',
            line_color='green',
            annotation_text=f'Average: ₹{avg_revenue:,.2f}',
            annotation_position='bottom right'
        )
        
        # Update layout for better readability
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=hourly_revenue['time_display'],
                ticktext=hourly_revenue['time_display']
            ),
            yaxis_tickprefix='₹',
            hovermode='x unified',
            height=500,
            showlegend=False
        )
        
        # Update line styles and hover template
        fig.for_each_trace(lambda t: t.update(
            line=dict(width=3 if t.name == 'total' else 2.5),  # Slightly thinner line for rolling average
            mode='lines+markers' if t.name == 'total' else 'lines',  # Markers only for actual values
            name='Hourly Revenue' if t.name == 'total' else '7-Hour Rolling Avg',
            hovertemplate='<b>%{x}</b><br>' +
                        ('Revenue: ₹%{y:,.2f}<extra></extra>' if t.name == 'total' 
                         else '7-Hour Avg: ₹%{y:,.2f}<extra></extra>')
        ))
        
        # Make rolling average line dashed
        fig.update_traces(
            line=dict(dash='dash'),
            selector={"name": "7-Hour Rolling Avg"}
        )
        
        # Display summary metrics
        st.markdown("""
        ### Revenue Insights
        - **Peak Revenue Hour:** {} with **₹{:,.2f}**
        - **Lowest Revenue Hour:** {} with **₹{:,.2f}**
        - **Average Revenue per Hour:** **₹{:,.2f}**
        """.format(
            peak_hour['time_display'], peak_hour['total'],
            low_hour['time_display'], low_hour['total'],
            avg_revenue
        ))
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Time Period Analysis ---
        st.subheader("Revenue by Time Period")
        
        # Analyze revenue by time periods
        period_revenue = analyze_time_periods(hourly_revenue)
        
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
                color='time_period',
                color_discrete_sequence=px.colors.qualitative.Pastel,
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
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<br>%{customdata[0]:.1f}% of Total<extra></extra>',
                customdata=period_revenue[['percentage']]
            )
            
            # Display the chart
            st.plotly_chart(fig_period, use_container_width=True)
            
            # Display time period insights
            max_period = period_revenue.loc[period_revenue['revenue'].idxmax()]
            st.markdown("""
            ### Time Period Insights
            - **Highest Revenue Period:** {} with **₹{:,.2f}** ({}% of total)
            - **Revenue Distribution:** Morning: {}% | Afternoon: {}% | Evening: {}%
            """.format(
                max_period['time_period'],
                max_period['revenue'],
                max_period['percentage'],
                period_revenue[period_revenue['time_period'].str.startswith('Morning')]['percentage'].iloc[0],
                period_revenue[period_revenue['time_period'].str.startswith('Afternoon')]['percentage'].iloc[0],
                period_revenue[period_revenue['time_period'].str.startswith('Evening')]['percentage'].iloc[0]
            ))
            
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

else:  # Data Overview
    st.header("📊 Data Overview")
    
    # Instructions
    ### Welcome to Data Overview
    
    Upload and validate your restaurant sales data to get started with the analysis.
    
    **Please upload the 'All Restaurants Sales: Hourly Item Wise' report from PetPooja dashboard.**  
    Required columns: Restaurant Name, Hour, Item, Price, Quantity, Total
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your sales data",
        type=['csv', 'xlsx'],
        key="revenue_trends_upload"
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
            
            # Display data summary
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Unique Restaurants", df['restaurant_name'].nunique())
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Show data statistics
            with st.expander("View Data Statistics"):
                st.write("### Data Types")
                st.write(df.dtypes.astype(str))
                
                st.write("### Summary Statistics")
                st.dataframe(df.describe())
            
            # --- Revenue Trends Analysis ---
            st.markdown("---")
            st.header("📊 Revenue Trends")
            
            # Add restaurant filter if multiple restaurants exist
            unique_restaurants = df['restaurant_name'].unique()
            if len(unique_restaurants) > 1:
                selected_restaurant = st.selectbox(
                    "Select Restaurant",
                    ["All Restaurants"] + sorted(unique_restaurants.tolist())
                )
                
                # Filter data based on selection
                if selected_restaurant != "All Restaurants":
                    df_filtered = df[df['restaurant_name'] == selected_restaurant].copy()
                else:
                    df_filtered = df.copy()
            else:
                selected_restaurant = unique_restaurants[0]
                df_filtered = df.copy()
            
            # Group data by hour for the selected restaurant(s)
            hourly_revenue = df_filtered.groupby('hour')['total'].sum().reset_index()
            
            # Calculate metrics
            total_revenue = hourly_revenue['total'].sum()
            avg_revenue_per_hour = hourly_revenue['total'].mean()
            best_hour_row = hourly_revenue.loc[hourly_revenue['total'].idxmax()]
            best_hour = int(best_hour_row['hour'])
            best_hour_revenue = best_hour_row['total']
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
            with col2:
                st.metric("Avg. Revenue/Hour", f"₹{avg_revenue_per_hour:,.2f}")
            with col3:
                st.metric("Best Hour", 
                         f"{best_hour}:00 - {best_hour+1}:00",
                         f"₹{best_hour_revenue:,.2f}")
            
            # Create line chart
            st.subheader("Hourly Revenue Trend")
            st.line_chart(
                data=hourly_revenue,
                x='hour',
                y='total',
                use_container_width=True
            )
