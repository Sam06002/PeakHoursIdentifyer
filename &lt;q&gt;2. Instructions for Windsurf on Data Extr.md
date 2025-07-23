<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# <q>2. Instructions for Windsurf on Data Extraction

When modifying the app, inform Windsurf to:

Skip initial metadata rows to reach the main data (look for the row headers that include Invoice No., Date, etc.).

Extract the following columns for processing:

Date (ensure it’s parsed as a datetime)

Net Sales (₹)(M.A - D) (use for revenue aggregation)

For peak hour analysis, group data by hour extracted from the Date column, then:

Sum revenue by hour (for revenue-based peaks).

Count orders by hour (for order-count-based peaks).

Optionally, for deeper analysis, allow users to:

Filter by Order Type (e.g., Dine-In, Delivery).

Exclude orders where Status is not ‘Success’.

3. Example: Relevant Data Structure After Extraction
DateNet Sales (₹)(M.A - D)2025-05-31 23:24:0118202025-05-31 23:09:212370......
4. Data Extraction Pipeline (For Windsurf)

Read the Excel file, locate the actual headers row (Invoice No., Date, etc.).

Extract and clean only the rows containing real order data.

Convert the Date column to Python datetime.

Use the hour part for peak analysis.

Aggregate by hour:

Order Count: number of rows per hour.

Revenue: sum of Net Sales (₹)(M.A - D) per hour.

5. Additional Recommendations

Ensure time values are not affected by time zone differences.

Exclude any rows that have missing/invalid timestamps or sales values.

Optionally, add an alert in the UI if these columns cannot be found.

6. Information to Provide to Developers
Required columns from ‘Order Master Report’ for peak hour analysis:

Date (parsed as datetime; used for time grouping)

Net Sales (₹)(M.A - D) (used for revenue aggregation)

(Optional): Invoice No. (for order counting if format changes)

(Optional but recommended): Order Type, Status

State that the column headers may not be on the first row due to initial metadata, so look for the header row and proceed from there.
Providing this focused extraction logic will help ensure that the peak hours dashboard only uses accurate, clean data for clear, actionable business insights.</q> provide me a proper prompt which i can directly copy paste in windsurf ai

Certainly! Here is a **copy-paste-ready prompt** for Windsurf AI to modify your Streamlit peak hours app, based on the specifics of the Petpooja Order Master Report:

**WINDSURF DEVELOPMENT PROMPT:**
---
**Task:**
Modify the Streamlit Peak Hours Identification app to work specifically with the Petpooja "Order Master" report Excel file.

**Required Data Extraction \& Cleaning Instructions:**

1. **Header Row Detection:**
    - Locate the row containing the actual headers (look for "Invoice No.", "Date", etc.) since the report may have initial metadata rows.
    - Extract and use only the actual order data rows under these headers.
2. **Column Selection:**
    - Select at minimum these columns:
        - `Date` (ensure this is parsed as a Python datetime object)
        - `Net Sales (₹)(M.A - D)`
    - (Optional/Future) Also support extracting `Order Type` and `Status` if present.
3. **Data Preparation Steps:**
    - Convert the `Date` column to datetime.
    - Drop/ignore rows with missing or invalid `Date` or `Net Sales (₹)(M.A - D)` values.
    - Optionally, filter data to only include orders where `Status` is "Success".
    - Optionally, allow user filtering by `Order Type` (e.g., Dine-In, Delivery).
4. **Peak Hour Aggregation:**
    - Extract the hour from the `Date` column.
    - Aggregate by hour:
        - a) **Order Count:** Number of orders per hour (count rows).
        - b) **Revenue:** Sum `Net Sales (₹)(M.A - D)` per hour.
5. **Visualization \& UI:**
    - Display bar charts of both order counts per hour and revenue per hour.
    - Clearly highlight the peak hour(s) in the chart and display callouts/text for the busiest time slot(s).
    - If required columns are missing or not found, show a clear alert in the UI.
6. **Robustness:**
    - Ensure code works even if the header row is NOT the first row in the sheet.
    - Handle case where multiple sheets or extra blank rows are present.
    - Make error handling user-friendly for failed uploads or missing columns.

**File:**

- Single file, `app.py` (Streamlit).

**Sample Data Structure after Extraction:**


| Date | Net Sales (₹)(M.A - D) |
| :-- | :-- |
| 2025-05-31 23:24:01 | 1820 |
| 2025-05-31 23:09:21 | 2370 |

**End Output:**
A robust Streamlit dashboard that takes an Order Master report as input and accurately visualizes peak hours by both order count and revenue, extracting and cleaning data as per the specifications above.

***Copy and paste the above prompt into Windsurf. If you need additional filtering (Order Type, Status), mention it in your next update!***

