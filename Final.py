import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

#Read Excel file 
filepath="Sales.xlsx"
df=pd.read_excel(filepath)
#Print the Excel data
print(df)


#Data Cleaning

#find the Missing Values
def missing_values(df):
    print("Missing values column wise\n")
    missing= df.isnull().sum()
    print(missing)
    return missing

#Drop the Duplicate Values
def drop_duplicate(df):
    print("Droping duplicate rows:\n")
    df_cleaned=df.drop_duplicates()
    return df_cleaned

#Change the Data type
def changetype(df):
    print("change datatype:\n")
    df=df.copy()
    df['Unit_Sold']=df['Unit_Sold'].fillna(0).astype(float)
    return df

#Fill The null values
def fillnull(df):
    return df.fillna(1)


#Data Analysis
def analyze_data(df):
    #Find The average of the selling Price
    print("Average Selling price",df['Selling_Price'].mean())
    
    #Count the number of product
    print("Count the no of product",df['Product_Name'].value_counts())
    
    #Find The Maximum of the selling Price
    print("maximum selling price",df['Selling_Price'].max())

    #Find The Minimum of the unit sold
    print("Minimum Unit Sold",df['Unit_Sold'].min())
    
    #Display the product by category
    grouped=df.groupby('Category')['Product_Name'].count()
    print("total product by category:\n",grouped)
    
    #sort the data by Total_revenue 
    return df.sort_values(by='Total_Revenue',ascending=False)

#save the Modified file
def save_data(output_path):
    df.to_excel(output_path,index=False)

#Bar Chart
data = df.groupby('State')['Product_Name'].count()
data.plot(kind='bar', color='#007FFF')
plt.title("Product Sales by State")
plt.xlabel("States")
plt.ylabel("Number of Products Sold")
plt.tight_layout()
plt.show()

#pie Chart
def product_total_Revenue(df):
    revenue_by_product = df.groupby('Category')['Total_Revenue'].sum()
    revenue_by_product.plot(kind='pie',autopct='%0.1f%%',title='Product Share by Total Revenue',startangle=90)
    plt.ylabel('')  
    plt.tight_layout()
    plt.show()
   
    
    

#Main Function
def main(df):
    missing_values(df)
    df=drop_duplicate(df)
    df=changetype(df)
    df=fillnull(df)
    df=analyze_data(df)
    product_total_Revenue(df)

    save_data('Sales_cleaned.xlsx')

if __name__=="__main__":
    main(df)

