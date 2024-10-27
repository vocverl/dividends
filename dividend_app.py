import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

def process_dividend_data(df):
    # Filter and prepare data
    df = df[df['Introduction'] == 'Dividends']
    df = df[df['Header'] == 'Data']
    df['Amount'] = pd.to_numeric(df['Amount'])
    df['PayDate'] = pd.to_datetime(df['PayDate'], format='%Y%m%d')
    return df

def create_monthly_heatmap(df):
    # Create year and month columns for the heatmap
    df['Year'] = df['PayDate'].dt.year
    df['Month'] = df['PayDate'].dt.month
    
    # Create pivot table for the heatmap
    pivot_table = df.pivot_table(
        values='Amount',
        index='Month',
        columns='Year',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # First create heatmap for non-zero values (orange to green)
    colors = ['#FFA500', '#FFFF00', '#90EE90', '#00FF00']  # orange -> yellow -> light green -> green
    custom_cmap = sns.color_palette(colors, as_cmap=True)
    
    # Create mask for zero values
    zero_mask = pivot_table == 0
    
    # Plot non-zero values
    sns.heatmap(pivot_table,
                annot=True,
                fmt='.0f',
                cmap=custom_cmap,
                mask=zero_mask,
                cbar_kws={'label': 'Dividend Amount ($)'})
    
    # Plot zero values in gray
    sns.heatmap(pivot_table,
                annot=True,
                fmt='.0f',
                cmap=['#D3D3D3'],  # Light gray
                mask=~zero_mask,
                cbar=False)
    
    plt.title('Dividend Heatmap: Month vs Year')
    plt.xlabel('Year')
    plt.ylabel('Month')
    
    # Adjust month labels to show month names
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.yticks(np.arange(0.5, 12.5), month_labels)
    
    return fig

def create_symbol_heatmap(df):
    # Create year and symbol pivot table
    df['Year'] = df['PayDate'].dt.year
    pivot_table = df.pivot_table(
        values='Amount',
        index='Symbol',
        columns='Year',
        aggfunc='sum',
        fill_value=0
    )
    
    # Sort by total dividend
    pivot_table['Total'] = pivot_table.sum(axis=1)
    pivot_table = pivot_table.sort_values('Total', ascending=False).drop('Total', axis=1)
    
    # Take top 15 symbols
    pivot_table = pivot_table.head(15)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for zero values
    zero_mask = pivot_table == 0
    
    # Create heatmap with same colors as monthly heatmap
    colors = ['#FFA500', '#FFFF00', '#90EE90', '#00FF00']
    custom_cmap = sns.color_palette(colors, as_cmap=True)
    
    # Plot non-zero values
    sns.heatmap(pivot_table,
                annot=True,
                fmt='.0f',
                cmap=custom_cmap,
                mask=zero_mask,
                cbar_kws={'label': 'Dividend Amount ($)'})
    
    # Plot zero values in gray
    sns.heatmap(pivot_table,
                annot=True,
                fmt='.0f',
                cmap=['#D3D3D3'],  # Light gray
                mask=~zero_mask,
                cbar=False)
    
    plt.title('Dividend Heatmap by Symbol')
    plt.xlabel('Year')
    plt.ylabel('Symbol')
    return fig

def analyze_dividend_growth(df):
    # Calculate dividend growth per symbol
    df['Year'] = df['PayDate'].dt.year
    yearly_div_per_symbol = df.pivot_table(
        values='Amount',
        index='Symbol',
        columns='Year',
        aggfunc='sum',
        fill_value=0
    )
    
    # Calculate growth percentage between first and last year
    first_year = yearly_div_per_symbol.columns[0]
    last_year = yearly_div_per_symbol.columns[-1]
    
    growth_df = pd.DataFrame({
        'Symbol': yearly_div_per_symbol.index,
        'First_Year_Amount': yearly_div_per_symbol[first_year],
        'Last_Year_Amount': yearly_div_per_symbol[last_year]
    })
    
    # Calculate growth percentage
    growth_df['Growth_Pct'] = ((growth_df['Last_Year_Amount'] - growth_df['First_Year_Amount']) / 
                              growth_df['First_Year_Amount'] * 100)
    
    # Filter where both years have dividend > 0
    growth_df = growth_df[
        (growth_df['First_Year_Amount'] > 0) & 
        (growth_df['Last_Year_Amount'] > 0)
    ].sort_values('Growth_Pct', ascending=False)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#228B22' if x >= 0 else '#FF4136' for x in growth_df['Growth_Pct']]
    
    growth_df['Growth_Pct'].plot(kind='bar', color=colors)
    plt.title(f'Dividend Growth {first_year}-{last_year} (%)')
    plt.xlabel('Symbol')
    plt.ylabel('Growth Percentage')
    plt.xticks(range(len(growth_df)), growth_df['Symbol'], rotation=45, ha='right')
    
    # Add values above/below bars
    for i, v in enumerate(growth_df['Growth_Pct']):
        plt.text(i, v + (5 if v >= 0 else -5), 
                f'{v:,.1f}%', 
                ha='center', 
                va='bottom' if v >= 0 else 'top')
    
    plt.tight_layout()
    return fig, growth_df
def calculate_growth_metrics(df):
    yearly_div = df.groupby(df['PayDate'].dt.year)['Amount'].sum()
    
    if len(yearly_div) > 1:
        total_growth = ((yearly_div.iloc[-1] / yearly_div.iloc[0]) - 1) * 100
        avg_annual_growth = (pow(yearly_div.iloc[-1] / yearly_div.iloc[0], 1/(len(yearly_div)-1)) - 1) * 100
    else:
        total_growth = 0
        avg_annual_growth = 0
        
    return yearly_div, total_growth, avg_annual_growth

def main():
    st.set_page_config(page_title="Dividend Portfolio Analyzer", layout="wide")
    st.title("Dividend Portfolio Analyzer")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dividend CSV file", type=['csv'])

    if uploaded_file is not None:
        # Read data
        df = pd.read_csv(uploaded_file, names=['Introduction', 'Header', 'PayDate', 
                        'Ex-Date', 'Symbol', 'Note', 'Quantity', 'DividendPerShare', 'Amount'])
        df = process_dividend_data(df)
        
        # Date filter in sidebar
        min_date = df['PayDate'].min().date()
        max_date = df['PayDate'].max().date()
        date_range = st.sidebar.date_input(
            "Select period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data based on selected dates
        mask = (df['PayDate'].dt.date >= date_range[0]) & (df['PayDate'].dt.date <= date_range[1])
        df_filtered = df[mask]
        
        # Main statistics
        st.subheader("Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_div = df_filtered['Amount'].sum()
            st.metric("Total Dividend", f"${total_div:,.2f}")
        
        with col2:
            monthly_avg = df_filtered.groupby(df_filtered['PayDate'].dt.strftime('%Y-%m'))['Amount'].sum().mean()
            st.metric("Monthly Average", f"${monthly_avg:,.2f}")
        
        with col3:
            unique_symbols = df_filtered['Symbol'].nunique()
            st.metric("Number of Stocks", unique_symbols)
            
        with col4:
            avg_per_payment = df_filtered['Amount'].mean()
            st.metric("Average per Payment", f"${avg_per_payment:,.2f}")
        
        # Graphs in two columns
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Total Dividend per Stock
            st.subheader("Total Dividend per Stock")
            dividend_per_symbool = df_filtered.groupby('Symbol')['Amount'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            dividend_per_symbool.plot(kind='bar')
            plt.xticks(rotation=45, ha='right')
            plt.title("Total Dividend per Stock")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Dividend Growth per Year
            st.subheader("Dividend Growth per Year")
            yearly_div, total_growth, avg_annual_growth = calculate_growth_metrics(df_filtered)
            fig, ax = plt.subplots(figsize=(10, 6))
            yearly_div.plot(kind='bar')
            plt.title(f"Total Dividend per Year (Growth: {total_growth:.1f}%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_right:
            # Dividend per Month
            st.subheader("Dividend per Month")
            df_filtered['Month'] = df_filtered['PayDate'].dt.strftime('%Y-%m')
            monthly_div = df_filtered.groupby('Month')['Amount'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            monthly_div.plot(kind='bar')
            plt.xticks(rotation=45, ha='right')
            plt.title("Dividend per Month")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Monthly Heatmap
            st.subheader("Monthly Dividend Heatmap")
            heatmap_fig = create_monthly_heatmap(df_filtered)
            st.pyplot(heatmap_fig)
            plt.close()

        # Symbol Heatmap
        st.subheader("Symbol Dividend Heatmap")
        symbol_heatmap_fig = create_symbol_heatmap(df_filtered)
        st.pyplot(symbol_heatmap_fig)
        plt.close()

        # Dividend Growth Analysis
        st.subheader("Dividend Growth Analysis")
        growth_fig, growth_df = analyze_dividend_growth(df_filtered)
        st.pyplot(growth_fig)
        plt.close()

        # Detailed statistics
        st.subheader("Detailed Statistics")
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.subheader("Top 10 Dividend Payers")
            top_10 = dividend_per_symbool.head(10)
            st.table(top_10.round(2).to_frame().style.format("${:,.2f}"))
            
        with col_stats2:
            st.subheader("Dividend Frequency")
            dividend_frequency = df_filtered.groupby('Symbol').size().sort_values(ascending=False)
            st.table(dividend_frequency.head(10).to_frame().rename(columns={0: 'Number of Payments'}))

        # Show growth data in table
        st.subheader("Detailed Dividend Growth")
        growth_table = growth_df.copy()
        growth_table = growth_table.round(2)
        growth_table.columns = ['Symbol', f'Amount {growth_table.columns[1]}', 
                              f'Amount {growth_table.columns[2]}', 'Growth %']
        st.table(growth_table.head(10).style.format({
            f'Amount {growth_table.columns[1]}': '${:,.2f}',
            f'Amount {growth_table.columns[2]}': '${:,.2f}',
            'Growth %': '{:,.1f}%'
        }))
        
        # Download functionality
        st.subheader("Download Data")
        st.download_button(
            label="Download analyzed data as CSV",
            data=df_filtered.to_csv().encode('utf-8'),
            file_name='dividend_analysis.csv',
            mime='text/csv',
        )

        # Extra information
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset Information")
        st.sidebar.write(f"Period: {min_date.strftime('%d-%m-%Y')} to {max_date.strftime('%d-%m-%Y')}")
        st.sidebar.write(f"Number of transactions: {len(df_filtered)}")
        st.sidebar.write(f"Average annual growth: {avg_annual_growth:.1f}%")

if __name__ == "__main__":
    main()