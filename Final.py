import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from prophet import Prophet

# --- Data Cleaning Functions ---
def missing_values(df):
    df['Category'] = df['Category'].str.strip().str.capitalize()
    print("Missing values column wise\n")
    missing = df.isnull().sum()
    print(missing)
    return missing

def drop_duplicate(df):
    print("Dropping duplicate rows...")
    df_cleaned = df.drop_duplicates()
    return df_cleaned

def changetype(df):
    print("Changing data types and handling missing values for 'Unit_Sold'...")
    df = df.copy()
    df['Unit_Sold'] = df['Unit_Sold'].fillna(0).astype(int) 
    return df

# --- Data Analysis & Visualization Functions ---
def analyze_data(df):
    print("\n--- Starting Data Analysis ---")
    print("Average Selling price:", df['Selling_Price'].mean())
    print("Maximum selling price:", df['Selling_Price'].max())
    print("Minimum Unit Sold:", df['Unit_Sold'].min())
    print("\nTotal product by category:\n", df.groupby('Category')['Product_Name'].count())
    return df.sort_values(by='Total_Revenue', ascending=False)

def plot_sales_by_state(df):
    print("Generating plot for sales by state...")
    data = df.groupby('State')['Unit_Sold'].sum()
    data.plot(kind='bar', color='#007FFF', figsize=(12, 6))
    plt.title("Total Units Sold by State")
    plt.xlabel("States")
    plt.ylabel("Total Units Sold")
    plt.tight_layout()
    plt.show()

def plot_revenue_by_category(df):
    print("Generating plot for revenue by category...")
    revenue_by_product = df.groupby('Category')['Total_Revenue'].sum()
    revenue_by_product.plot(kind='pie', autopct='%1.1f%%', title='Product Share by Total Revenue', startangle=90, figsize=(8, 8))
    plt.ylabel('')  
    plt.tight_layout()
    plt.show()

# --- Forecasting Function ---

def run_sales_forecast(df):
    """
    Runs a sales forecast using Prophet on the cleaned data.
    """
    print("\n--- Starting Sales Forecast ---")
    # 1. Prepare the data for Prophet
    df_forecast = df[['Date', 'Unit_Sold']].copy()
    df_forecast.rename(columns={'Date': 'ds', 'Unit_Sold': 'y'}, inplace=True)

    # 1: Drop rows where the date is missing BEFORE converting.
    df_forecast.dropna(subset=['ds'], inplace=True)
    
    # 2: Add 'dayfirst=True' to correctly interpret dates like 18-09-2025.
    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'], dayfirst=True)
    
    # 2. Initialize and fit the model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_forecast)
    
    # 3. Create future dates and predict
    future = model.make_future_dataframe(periods=90) # Forecast 90 days ahead
    forecast = model.predict(future)
    
    print("Forecast generated successfully.")
    
    # 4. Plot the forecast
    model.plot(forecast, xlabel='Date', ylabel='Units Sold')
    plt.title('Sales Forecast (Next 90 Days)')
    plt.show()
    
    # 5. Export the forecast results
    forecast_output_path = 'sales_forecast_output.xlsx'
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(forecast_output_path, index=False)
    print(f"Forecast data saved to '{forecast_output_path}'")
    print("--- Forecast Complete ---")

# --- Save Function ---

def save_data(df, output_path):
    print(f"Saving cleaned data to {output_path}...")
    df.to_excel(output_path, index=False)
    print("Save complete.")

# --- Main Pipeline ---

def main():
    filepath = "Sales.xlsx"
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return

    # --- Run Cleaning and Analysis ---
    missing_values(df)
    df = drop_duplicate(df)
    df = changetype(df)    
    df = analyze_data(df)
    
    # --- Run Visualizations ---
    plot_sales_by_state(df)
    plot_revenue_by_category(df)

    # --- Save and Forecast ---
    save_data(df, 'Sales_cleaned.xlsx')
    
    run_sales_forecast(df)

# --- Script Execution ---
if __name__ == "__main__":
    main()