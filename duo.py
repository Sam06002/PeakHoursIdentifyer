import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

st.set_page_config(page_title="PetPooja Analytics Dashboard", page_icon="📊", layout="wide")

# --- Data Cleaning Function ---
def clean_data(df):
    try:
        original_rows = len(df)
        df_clean = df.copy()
        df_clean = df_clean.dropna(how='all')
        df_clean.columns = df_clean.columns.str.strip().str.lower()
        
        column_mapping = {
            'restaurant': ['restaurant', 'outlet', 'location', 'store', 'branch', 'restaurant name'],
            'hour': ['hour', 'time', 'hr', 'order hour', 'sale hour', 'transaction hour'],
            'item': ['item', 'product', 'menu item', 'dish', 'menu'],
            'price': ['price', 'unit price', 'rate', 'unitprice', 'cost'],
            'quantity': ['quantity', 'qty', 'count', 'number', 'amount'],
            'total': ['total', 'amount', 'revenue', 'sale amount', 'gross']
        }
        
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
        
        required_columns = {v: k for k, v in actual_columns.items()}
        df_clean = df_clean[list(actual_columns.values())].rename(columns=required_columns)

        # Convert hour to 24-hour format
        if df_clean['hour'].dtype == 'object':
            def convert_12h_to_24h(time_str):
                try:
                    if pd.isna(time_str): 
                        return None
                    time_str = str(time_str).strip().upper()
                    if 'AM' in time_str or 'PM' in time_str:
                        time_part = time_str.split()[0]
                        period = 'AM' if 'AM' in time_str else 'PM'
                        hour = int(''.join(filter(str.isdigit, time_part)) or 0)
                        if period == 'AM': 
                            return 0 if hour == 12 else hour
                        else: 
                            return 12 if hour == 12 else hour + 12
                    return int(''.join(filter(str.isdigit, time_str)) or 0)
                except:
                    return None
            
            df_clean['hour'] = df_clean['hour'].apply(convert_12h_to_24h)
            df_clean = df_clean.dropna(subset=['hour'])
            df_clean['hour'] = df_clean['hour'].astype(int)

        # Convert numeric columns
        for col in ['price', 'quantity', 'total']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove invalid rows
        key_cols = ['restaurant', 'hour', 'item', 'price', 'quantity', 'total']
        df_clean = df_clean.dropna(subset=key_cols)
        df_clean = df_clean[(df_clean['hour'] >= 0) & (df_clean['hour'] <= 23)]
        df_clean = df_clean[(df_clean['quantity'] > 0) & (df_clean['total'] > 0)]
        df_clean = df_clean.rename(columns={'restaurant': 'restaurant_name'}).reset_index(drop=True)
        
        cleaned_rows = len(df_clean)
        removed_rows = original_rows - cleaned_rows
        if removed_rows > 0:
            st.sidebar.info(f"Data cleaned: {removed_rows} rows removed, {cleaned_rows} remain.")
        
        return df_clean, None
    except Exception as e:
        return None, f"Error cleaning sales data: {str(e)}"

# --- Altair Chart Functions ---
def create_bar_chart(data, x_col, y_col, title, x_title, y_title, color='steelblue'):
    chart = alt.Chart(data).mark_bar(color=color).encode(
        x=alt.X(f'{x_col}:O', title=x_title, sort='ascending'),
        y=alt.Y(f'{y_col}:Q', title=y_title),
        tooltip=[f'{x_col}:O', f'{y_col}:Q']
    ).properties(
        title=title,
        height=400,
        width='container'
    )
    return chart

def create_line_chart(data, x_col, y_col, title, x_title, y_title, color='orange'):
    chart = alt.Chart(data).mark_line(point=True, color=color).encode(
        x=alt.X(f'{x_col}:O', title=x_title, sort='ascending'),
        y=alt.Y(f'{y_col}:Q', title=y_title),
        tooltip=[f'{x_col}:O', f'{y_col}:Q']
    ).properties(
        title=title,
        height=400,
        width='container'
    )
    return chart

# --- Session State Initialization ---
session_keys = ['sales_df', 'table_df', 'customer_df', 'aov_df', 'daily_target', 'weekly_target', 'monthly_target']
for key in session_keys:
    if key not in st.session_state: 
        if 'target' in key:
            st.session_state[key] = 10000 if 'daily' in key else (70000 if 'weekly' in key else 300000)
        else:
            st.session_state[key] = None

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🗂️ Upload Your Data")
    
    sales_file = st.file_uploader("1. Hourly Sales Report", type=['csv', 'xlsx'])
    table_file = st.file_uploader("2. Table Report", type=['csv'])
    customer_file = st.file_uploader("3. Customer Report", type=['csv'])
    aov_file = st.file_uploader("4. Order Details Report (AOV)", type=['csv'])

    if st.button("🗑️ Clear All Data"):
        for k in session_keys:
            if 'target' in k:
                st.session_state[k] = 10000 if 'daily' in k else (70000 if 'weekly' in k else 300000)
            else:
                st.session_state[k] = None
        st.rerun()

    # Process uploaded files
    if sales_file and st.session_state['sales_df'] is None:
        try:
            df_raw = pd.read_csv(sales_file) if sales_file.name.endswith('.csv') else pd.read_excel(sales_file)
            df_clean, error = clean_data(df_raw)
            if error:
                st.error(error)
            else:
                st.session_state['sales_df'] = df_clean
                st.success("✅ Sales data loaded!")
        except Exception as e:
            st.error(f"Error loading sales file: {str(e)}")

    if table_file and st.session_state['table_df'] is None:
        try:
            st.session_state['table_df'] = pd.read_csv(table_file)
            st.success("✅ Table report loaded!")
        except Exception as e:
            st.error(f"Error loading table file: {str(e)}")

    if customer_file and st.session_state['customer_df'] is None:
        try:
            st.session_state['customer_df'] = pd.read_csv(customer_file)
            st.success("✅ Customer report loaded!")
        except Exception as e:
            st.error(f"Error loading customer file: {str(e)}")

    if aov_file and st.session_state['aov_df'] is None:
        try:
            st.session_state['aov_df'] = pd.read_csv(aov_file)
            st.success("✅ Order Details loaded!")
        except Exception as e:
            st.error(f"Error loading AOV file: {str(e)}")

# --- Main Dashboard ---
st.title("📊 PetPooja Advanced Analytics Dashboard")

if st.session_state['sales_df'] is None:
    st.info("👋 Welcome! Please upload your 'Hourly Sales Report' in the sidebar to get started.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://petpooja.com/assets/images/logo/logo-icon.svg", width=200)
else:
    df_sales = st.session_state['sales_df']
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📈 Performance", 
        "🎯 Sales Targets",
        "🔄 Table Turnover",
        "👥 Customer Value",
        "🛒 Average Order Value"
    ])

    # ===== TAB 1: OVERVIEW =====
    with tab1:
        st.header("Sales Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df_sales):,}")
        with col2:
            st.metric("Unique Restaurants", df_sales['restaurant_name'].nunique())
        with col3:
            st.metric("Total Items Sold", f"{df_sales['quantity'].sum():,.0f}")
        with col4:
            st.metric("Total Revenue", f"₹{df_sales['total'].sum():,.2f}")
        
        st.subheader("📋 Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Range:**")
            st.write(f"• Hours: {df_sales['hour'].min()}:00 to {df_sales['hour'].max()}:00")
            st.write(f"• Unique Items: {df_sales['item'].nunique()}")
            st.write(f"• Restaurants: {df_sales['restaurant_name'].nunique()}")
        
        with col2:
            st.write("**Key Metrics:**")
            st.write(f"• Average per order: ₹{df_sales['total'].mean():.2f}")
            st.write(f"• Highest order: ₹{df_sales['total'].max():.2f}")
            st.write(f"• Average items per order: {df_sales['quantity'].mean():.1f}")
        
        with st.expander("📊 View Full Dataset"):
            st.dataframe(df_sales, use_container_width=True)

    # ===== TAB 2: PERFORMANCE =====
    with tab2:
        st.header("Peak Hours & Revenue Trends")
        
        try:
            # Calculate hourly metrics
            hourly_data = df_sales.groupby('hour').agg({
                'total': 'sum',
                'quantity': 'sum',
                'item': 'count'
            }).reset_index()
            hourly_data.columns = ['hour', 'total_revenue', 'total_quantity', 'order_count']
            
            if len(hourly_data) > 0:
                # Peak hour metrics
                peak_revenue_hour = hourly_data.loc[hourly_data['total_revenue'].idxmax()]
                peak_orders_hour = hourly_data.loc[hourly_data['order_count'].idxmax()]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Peak Revenue Hour", 
                        f"{int(peak_revenue_hour['hour'])}:00",
                        f"₹{peak_revenue_hour['total_revenue']:,.0f}"
                    )
                with col2:
                    st.metric(
                        "Peak Orders Hour",
                        f"{int(peak_orders_hour['hour'])}:00", 
                        f"{int(peak_orders_hour['order_count'])} orders"
                    )
                with col3:
                    st.metric(
                        "Avg Hourly Revenue",
                        f"₹{hourly_data['total_revenue'].mean():,.0f}"
                    )
                
                # Charts side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    revenue_chart = create_bar_chart(
                        hourly_data, 'hour', 'total_revenue',
                        'Revenue by Hour', 'Hour of Day', 'Revenue (₹)',
                        color='lightblue'
                    )
                    st.altair_chart(revenue_chart, use_container_width=True)
                
                with col2:
                    orders_chart = create_bar_chart(
                        hourly_data, 'hour', 'order_count',
                        'Orders by Hour', 'Hour of Day', 'Number of Orders',
                        color='lightgreen'
                    )
                    st.altair_chart(orders_chart, use_container_width=True)
                
                # Revenue trend line
                st.subheader("📈 Revenue Trend Analysis")
                trend_chart = create_line_chart(
                    hourly_data, 'hour', 'total_revenue',
                    'Hourly Revenue Trend', 'Hour of Day', 'Revenue (₹)',
                    color='orange'
                )
                st.altair_chart(trend_chart, use_container_width=True)
                
                # Alternative: Streamlit native charts (guaranteed to work)
                st.subheader("📊 Alternative View (Streamlit Native)")
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(hourly_data.set_index('hour')['total_revenue'], height=300)
                with col2:
                    st.line_chart(hourly_data.set_index('hour')['order_count'], height=300)
                
                with st.expander("📊 View Hourly Data"):
                    st.dataframe(hourly_data, hide_index=True)
            else:
                st.error("No data available for charts.")
                
        except Exception as e:
            st.error(f"Error creating performance charts: {str(e)}")

    # ===== TAB 3: SALES TARGETS =====
    with tab3:
        st.header("Sales Target Performance")
        
        # Target input section
        st.subheader("🎯 Set Your Sales Targets")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            daily_target = st.number_input(
                "Daily Revenue Target (₹)",
                min_value=0,
                value=st.session_state.daily_target,
                step=1000
            )
        
        with col2:
            weekly_target = st.number_input(
                "Weekly Revenue Target (₹)",
                min_value=0,
                value=st.session_state.weekly_target,
                step=5000
            )
        
        with col3:
            monthly_target = st.number_input(
                "Monthly Revenue Target (₹)",
                min_value=0,
                value=st.session_state.monthly_target,
                step=10000
            )
        
        if st.button("💾 Save Targets"):
            st.session_state.daily_target = daily_target
            st.session_state.weekly_target = weekly_target  
            st.session_state.monthly_target = monthly_target
            st.success("Targets saved successfully!")
        
        # Performance calculation
        st.markdown("---")
        st.subheader("📊 Performance Analysis")
        
        try:
            total_revenue = df_sales['total'].sum()
            unique_hours = df_sales['hour'].nunique()
            
            # Estimate daily performance
            daily_avg = total_revenue / max(1, unique_hours / 24) if unique_hours > 0 else 0
            weekly_avg = daily_avg * 7
            monthly_performance = total_revenue
            
            # Achievement percentages
            daily_achievement = (daily_avg / st.session_state.daily_target) * 100 if st.session_state.daily_target > 0 else 0
            weekly_achievement = (weekly_avg / st.session_state.weekly_target) * 100 if st.session_state.weekly_target > 0 else 0
            monthly_achievement = (monthly_performance / st.session_state.monthly_target) * 100 if st.session_state.monthly_target > 0 else 0
            
            # Display metrics
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
            
            # Performance chart using Streamlit native (guaranteed to work)
            st.subheader("🎯 Target Achievement Overview")
            performance_data = pd.DataFrame({
                'Period': ['Daily', 'Weekly', 'Monthly'],
                'Achievement': [daily_achievement, weekly_achievement, monthly_achievement]
            })
            st.bar_chart(performance_data.set_index('Period')['Achievement'], height=400)
            
            # Progress bars as additional visualization
            st.subheader("📊 Progress Indicators")
            st.progress(min(daily_achievement / 100, 1.0), text=f"Daily: {daily_achievement:.1f}%")
            st.progress(min(weekly_achievement / 100, 1.0), text=f"Weekly: {weekly_achievement:.1f}%")
            st.progress(min(monthly_achievement / 100, 1.0), text=f"Monthly: {monthly_achievement:.1f}%")
            
        except Exception as e:
            st.error(f"Error in performance calculation: {str(e)}")

    # ===== TAB 4: TABLE TURNOVER =====
    with tab4:
        st.header("Table Turnover Analysis")
        
        if st.session_state['table_df'] is None:
            st.info("📋 Upload your 'Table Report' in the sidebar to enable detailed table turnover analysis.")
            
            # Show estimated analysis
            st.subheader("📊 Estimated Table Analysis (Based on Sales Data)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_tables = st.number_input("Estimated Number of Tables", min_value=1, value=8, step=1)
            with col2:
                seats_per_table = st.number_input("Average Seats per Table", min_value=1, value=4, step=1)
            with col3:
                target_service_time = st.number_input("Target Service Time (minutes)", min_value=15, value=45, step=5)
            
            # Basic calculations
            total_capacity = num_tables * seats_per_table
            hourly_data = df_sales.groupby('hour').agg({
                'total': 'sum',
                'item': 'count'
            }).reset_index()
            
            # Estimate customers (assuming 2.5 people per order on average)
            hourly_data['estimated_customers'] = hourly_data['item'] * 2.5
            hourly_data['estimated_turns'] = hourly_data['estimated_customers'] / total_capacity
            
            avg_turns = hourly_data['estimated_turns'].mean()
            max_turns = hourly_data['estimated_turns'].max()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Avg Turns/Hour", f"{avg_turns:.2f}")
            with col2:
                st.metric("Estimated Peak Turns/Hour", f"{max_turns:.2f}")
            with col3:
                st.metric("Total Seating Capacity", f"{total_capacity} seats")
            
            # Chart using Altair
            turns_chart = create_bar_chart(
                hourly_data, 'hour', 'estimated_turns',
                'Estimated Table Turns by Hour', 'Hour of Day', 'Estimated Turns',
                color='purple'
            )
            st.altair_chart(turns_chart, use_container_width=True)
            
            # Alternative native chart
            st.bar_chart(hourly_data.set_index('hour')['estimated_turns'], height=300)
            
        else:
            table_df = st.session_state['table_df']
            st.success(f"✅ Table data loaded: {len(table_df)} records")
            
            st.subheader("📊 Table Usage Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Table Records", len(table_df))
            with col2:
                if 'Table Number' in table_df.columns:
                    st.metric("Unique Tables", table_df['Table Number'].nunique())
            
            with st.expander("📊 View Table Data"):
                st.dataframe(table_df, use_container_width=True)

    # ===== TAB 5: CUSTOMER LIFETIME VALUE (FIXED VERSION) =====
    with tab5:
        st.header("Customer Lifetime Value (CLV)")
        
        if st.session_state['customer_df'] is None:
            st.info("📋 Upload your 'Customer Report' in the sidebar to enable CLV analysis.")
        else:
            customer_df = st.session_state['customer_df']
            st.success(f"✅ Customer data loaded: {len(customer_df)} records")
            
            try:
                # Advanced data cleaning and processing
                customer_df_clean = customer_df.copy()
                
                # Clean and convert Net_Sales column
                if 'Net_Sales' in customer_df_clean.columns:
                    # Handle various data formats and clean the Net_Sales column
                    customer_df_clean['Net_Sales_Original'] = customer_df_clean['Net_Sales'].copy()
                    
                    # Convert to string first, then clean
                    customer_df_clean['Net_Sales'] = customer_df_clean['Net_Sales'].astype(str)
                    
                    # Remove currency symbols, commas, and other non-numeric characters
                    customer_df_clean['Net_Sales'] = customer_df_clean['Net_Sales'].str.replace('₹', '', regex=False)
                    customer_df_clean['Net_Sales'] = customer_df_clean['Net_Sales'].str.replace(',', '', regex=False)
                    customer_df_clean['Net_Sales'] = customer_df_clean['Net_Sales'].str.replace('[^0-9.-]', '', regex=True)
                    
                    # Convert to numeric, replacing invalid values with 0
                    customer_df_clean['Net_Sales'] = pd.to_numeric(customer_df_clean['Net_Sales'], errors='coerce').fillna(0)
                    
                    # Remove rows with zero or negative sales
                    customer_df_clean = customer_df_clean[customer_df_clean['Net_Sales'] > 0]
                    
                    if len(customer_df_clean) > 0:
                        # Calculate key metrics
                        total_customers = len(customer_df_clean)
                        total_revenue = customer_df_clean['Net_Sales'].sum()
                        avg_clv = customer_df_clean['Net_Sales'].mean()
                        median_clv = customer_df_clean['Net_Sales'].median()
                        max_clv = customer_df_clean['Net_Sales'].max()
                        min_clv = customer_df_clean['Net_Sales'].min()
                        
                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Customers", f"{total_customers:,}")
                        with col2:
                            st.metric("Total Customer Revenue", f"₹{total_revenue:,.2f}")
                        with col3:
                            st.metric("Average CLV", f"₹{avg_clv:,.2f}")
                        with col4:
                            st.metric("Median CLV", f"₹{median_clv:,.2f}")
                        
                        # Customer distribution using native chart
                        st.subheader("📈 Customer Value Distribution")
                        try:
                            # Create bins for visualization
                            customer_bins = pd.cut(customer_df_clean['Net_Sales'], bins=10)
                            distribution = customer_bins.value_counts().sort_index()
                            st.bar_chart(distribution, height=400)
                        except Exception as viz_error:
                            st.warning(f"Chart error: {str(viz_error)}")
                            # Alternative: Simple histogram
                            st.bar_chart(customer_df_clean['Net_Sales'].value_counts().head(20), height=300)
                        
                        # Business insights
                        st.subheader("💡 Business Insights")
                        insights = [
                            f"💎 **Top Customer Value**: ₹{max_clv:,.2f}",
                            f"📊 **Value Range**: ₹{min_clv:,.2f} to ₹{max_clv:,.2f}",
                            f"📈 **Revenue Concentration**: Top 20% of customers likely contribute 60-80% of revenue"
                        ]
                        
                        for insight in insights:
                            st.write(f"- {insight}")
                    
                    else:
                        st.warning("No valid customer data found after cleaning.")
                
                else:
                    st.error("Net_Sales column not found in the uploaded customer data.")
                
                # Show raw data
                with st.expander("📊 View Customer Data"):
                    st.dataframe(customer_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing customer data: {str(e)}")

    # ===== TAB 6: AVERAGE ORDER VALUE =====
    with tab6:
        st.header("Average Order Value (AOV)")
        
        if st.session_state['aov_df'] is None:
            st.info("📋 Upload the 'Order Details Report' in the sidebar to enable AOV analysis.")
        else:
            aov_df = st.session_state['aov_df'].copy()
            
            try:
                # Clean AOV data
                aov_df['total'] = pd.to_numeric(aov_df['total'], errors='coerce')
                aov_df['created_date'] = pd.to_datetime(aov_df['created_date'], errors='coerce')
                aov_df = aov_df.dropna(subset=['total', 'created_date'])
                
                # Filter valid orders
                valid_orders = aov_df[aov_df['status'] == 1].copy()
                # Remove cancelled orders if column exists
                if 'order_status' in valid_orders.columns:
                    valid_orders = valid_orders[~valid_orders['order_status'].str.lower().str.contains('cancel', na=False)]
                
                if len(valid_orders) > 0:
                    aov = valid_orders['total'].mean()
                    total_orders = len(valid_orders)
                    total_revenue = valid_orders['total'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Order Value", f"₹{aov:,.2f}")
                    with col2:
                        st.metric("Total Valid Orders", f"{total_orders:,}")
                    with col3:
                        st.metric("Total Revenue", f"₹{total_revenue:,.2f}")
                    
                    # AOV trend over time using Altair
                    daily_aov = valid_orders.groupby(valid_orders['created_date'].dt.date)['total'].mean().reset_index()
                    daily_aov.columns = ['Date', 'AOV']
                    daily_aov = daily_aov.dropna()
                    
                    if len(daily_aov) > 1:
                        st.subheader("📈 Daily AOV Trend")
                        
                        # Convert date to string for Altair compatibility
                        daily_aov['Date_str'] = daily_aov['Date'].astype(str)
                        
                        aov_trend_chart = alt.Chart(daily_aov).mark_line(point=True, color='green').encode(
                            x=alt.X('Date_str:T', title='Date'),
                            y=alt.Y('AOV:Q', title='Average Order Value (₹)'),
                            tooltip=['Date_str:T', 'AOV:Q']
                        ).properties(
                            title='Daily AOV Trend',
                            height=400
                        )
                        st.altair_chart(aov_trend_chart, use_container_width=True)
                        
                        # Alternative native chart
                        st.line_chart(daily_aov.set_index('Date')['AOV'], height=300)
                    
                    # Order value distribution
                    st.subheader("📊 Order Value Distribution")
                    
                    # Using native chart for guaranteed display
                    order_bins = pd.cut(valid_orders['total'], bins=20)
                    distribution = order_bins.value_counts().sort_index()
                    st.bar_chart(distribution, height=400)
                    
                    with st.expander("📊 View Order Details Data"):
                        st.dataframe(valid_orders, use_container_width=True)
                else:
                    st.warning("No valid orders found in the data.")
                    
            except Exception as e:
                st.error(f"Error processing AOV data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**PetPooja Analytics Dashboard** - Built with Streamlit & Altair")
