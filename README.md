# Restaurant Peak Hours Analyzer 🍽️

A Streamlit web application that helps restaurant owners and managers analyze their order data to identify peak business hours, optimize staffing, and improve operational efficiency.

![Peak Hours Analyzer](https://img.shields.io/badge/Status-Active-success)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Upload Excel/CSV** files containing restaurant order data
- **Visualize** peak hours with interactive charts
- **Heatmap** analysis of order patterns by day and hour
- **Business insights** and recommendations
- **Responsive design** works on desktop and mobile

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Sam06002/PeakHoursIdentifyer.git
   cd PeakHoursIdentifyer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

1. **Prepare your data**:
   - Export your restaurant order data in Excel (`.xlsx`, `.xls`) or CSV format
   - Ensure your data includes at least these columns:
     - Order timestamp or separate date and time columns
     - Order amount or total
     - Order status (to filter out cancelled orders)

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```

3. **Using the application**:
   - Click "Browse files" to upload your order data
   - The app will automatically detect the header row
   - Select the appropriate columns for date, time, and amount
   - View the interactive visualizations and insights

## Data Format

The application works best with data in the following format:

| Order ID | Order Date | Order Time | Amount | Status  |
|----------|------------|------------|--------|---------|
| 1001     | 2023-01-01 | 12:30:00   | 45.50  | COMPLETE|
| 1002     | 2023-01-01 | 12:45:00   | 32.75  | COMPLETE|
| ...      | ...        | ...        | ...    | ...     |

## Features in Detail

### 1. Hourly Analysis
- View order count and revenue by hour
- Identify peak business hours
- Compare different days of the week

### 2. Day-Hour Heatmap
- Visualize order patterns across days and hours
- Quickly spot busy periods
- Understand weekly patterns

### 3. Business Insights
- Get automated insights about your peak hours
- Data-driven recommendations for staffing
- Identify potential opportunities for promotions

## Troubleshooting

- **File not loading**: Ensure your file is not open in another program
- **Column detection issues**: Check that your file has a header row with clear column names
- **Visualization errors**: Make sure date/time columns are properly formatted

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data visualization powered by [Plotly](https://plotly.com/)
- Data processing with [Pandas](https://pandas.pydata.org/)
