import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Ultimate Dividend Portfolio Dashboard", layout="wide")

def load_portfolio_data(file):
    """Load and process portfolio data."""
    content = file.read().decode('utf-8').splitlines()
    portfolio_rows = []
    headers = None
    
    for line in content:
        parts = line.split(',')
        if len(parts) > 0:
            if parts[0] == 'Open Position Summary' and parts[1] == 'Header':
                headers = parts[2:]  # Skip the first two columns
            elif parts[0] == 'Open Position Summary' and parts[1] == 'Data' and 'Total' not in parts:
                portfolio_rows.append(parts[2:])  # Skip the first two columns
    
    if headers and portfolio_rows:
        df = pd.DataFrame(portfolio_rows, columns=headers)
        # Convert numeric columns
        numeric_cols = ['Quantity', 'Value', 'Cost Basis', 'UnrealizedP&L']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    return None

def load_dividend_data(file):
    """Load and process dividend data."""
    try:
        content = file.read().decode('utf-8').splitlines()
        dividend_rows = []
        
        for line in content:
            parts = line.split(',')
            if len(parts) >= 9 and parts[0] == 'Dividends' and parts[1] == 'Data':
                dividend_rows.append({
                    'PayDate': parts[2],
                    'Symbol': parts[4],
                    'Amount': parts[8]
                })
        
        if dividend_rows:
            df = pd.DataFrame(dividend_rows)
            df['PayDate'] = pd.to_datetime(df['PayDate'], format='%Y%m%d')
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
            return df
        return pd.DataFrame(columns=['PayDate', 'Symbol', 'Amount'])
        
    except Exception as e:
        st.error(f"Error processing dividend data: {str(e)}")
        return pd.DataFrame(columns=['PayDate', 'Symbol', 'Amount'])

def filter_dividend_data(dividend_df, period='All', year=None, symbol=None):
    """Filter dividend data based on selected filters."""
    filtered_df = dividend_df.copy()
    
    # Period filter
    if period != 'All':
        years = int(period.replace('Y', ''))
        latest_date = filtered_df['PayDate'].max()
        filtered_df = filtered_df[filtered_df['PayDate'] > latest_date - pd.DateOffset(years=years)]
    
    # Year filter
    if year is not None and year != 'All':
        filtered_df = filtered_df[filtered_df['PayDate'].dt.year == year]
    
    # Symbol filter
    if symbol is not None and symbol != 'All':
        filtered_df = filtered_df[filtered_df['Symbol'] == symbol]
    
    return filtered_df

def create_dividend_trend_chart(dividend_df):
    """Create monthly dividend trend chart."""
    monthly_data = dividend_df.groupby(pd.Grouper(key='PayDate', freq='M'))['Amount'].sum().reset_index()
    
    fig = px.line(monthly_data, x='PayDate', y='Amount',
                  title='Monthly Dividend Income Trend',
                  labels={'Amount': 'Dividend Amount ($)', 'PayDate': 'Date'})
    
    fig.update_layout(
        height=300,
        showlegend=False,
        hovermode='x unified',
        yaxis_tickprefix='$'
    )
    
    return fig

def create_sector_allocation(portfolio_df):
    """Create sector allocation pie chart."""
    sector_data = portfolio_df.groupby('Sector')['Value'].sum().reset_index()
    sector_data['Percentage'] = (sector_data['Value'] / sector_data['Value'].sum() * 100).round(2)
    
    fig = px.pie(sector_data, 
                 values='Value', 
                 names='Sector',
                 title='Sector Allocation',
                 hole=0.4)
    
    fig.update_traces(textposition='outside', textinfo='label+percent')
    fig.update_layout(showlegend=True, height=350)
    
    return fig

def create_monthly_income_heatmap(dividend_df):
    """Create monthly income heatmap."""
    pivot_data = dividend_df.pivot_table(
        values='Amount',
        index=dividend_df['PayDate'].dt.month,
        columns=dividend_df['PayDate'].dt.year,
        aggfunc='sum',
        fill_value=0
    )
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    colorscale = [
        [0, 'rgb(200,200,200)'],      # Grey for zero
        [0.0001, 'rgb(255,0,0)'],     # Red for low values
        [0.5, 'rgb(255,255,0)'],      # Yellow for middle values
        [1, 'rgb(0,255,0)']           # Green for high values
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=month_names,
        colorscale=colorscale,
        text=np.round(pivot_data.values, 2),
        texttemplate='$%{text:,.0f}',
        textfont={"size": 10, "color": "black"},
        hoverongaps=False,
        zmin=0,
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Income: $%{z:,.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title='Monthly Dividend Income Distribution',
        height=400,
        margin=dict(t=50)
    )
    
    fig.update_traces(colorbar_title="Income ($)", showscale=True)
    
    return fig

def create_income_bubble_chart(portfolio_df, dividend_df):
    """Create bubble chart showing position size vs income generation."""
    latest_date = dividend_df['PayDate'].max()
    one_year_ago = latest_date - pd.DateOffset(years=1)
    annual_income = dividend_df[dividend_df['PayDate'] > one_year_ago].groupby('Symbol')['Amount'].sum()
    
    bubble_data = portfolio_df.copy()
    bubble_data['Annual_Income'] = bubble_data['Symbol'].map(annual_income).fillna(0)
    bubble_data['Current_Yield'] = (bubble_data['Annual_Income'] / bubble_data['Value'] * 100)
    
    fig = px.scatter(bubble_data,
                    x='Current_Yield',
                    y='Annual_Income',
                    size='Value',
                    color='Sector',
                    hover_name='Symbol',
                    title='Position Size vs Income Generation',
                    labels={
                        'Current_Yield': 'Current Yield (%)',
                        'Annual_Income': 'Annual Income ($)',
                        'Value': 'Position Size ($)'
                    })
    
    fig.update_layout(height=350, yaxis_tickprefix='$')
    
    return fig
def create_dividend_per_stock_chart(dividend_df):
    """Create dividend per stock chart."""
    stock_data = dividend_df.groupby('Symbol')['Amount'].sum().sort_values(ascending=False)
    
    fig = px.bar(stock_data.head(10), 
                 title='Top 10 Dividend Contributing Stocks',
                 labels={'value': 'Annual Dividend Income ($)', 'Symbol': 'Stock'})
    
    fig.update_layout(height=300, showlegend=False, yaxis_tickprefix='$')
    
    return fig

def create_dividend_growth_chart(dividend_df):
    """Create yearly dividend growth chart."""
    yearly_data = dividend_df.groupby(dividend_df['PayDate'].dt.year)['Amount'].sum().reset_index()
    yearly_data['YoY_Growth'] = yearly_data['Amount'].pct_change() * 100
    
    fig = px.bar(yearly_data, 
                 x='PayDate', 
                 y='Amount',
                 title='Yearly Dividend Income',
                 labels={'Amount': 'Total Dividends ($)', 'PayDate': 'Year'})
    
    fig.update_layout(height=300, showlegend=False, yaxis_tickprefix='$')
    
    return fig

def calculate_metrics(portfolio_df, dividend_df):
    """Calculate portfolio and dividend metrics."""
    # Portfolio metrics
    portfolio_metrics = {
        'Total Value': f"${portfolio_df['Value'].sum():,.2f}",
        'Annual Income': f"${dividend_df[dividend_df['PayDate'] > dividend_df['PayDate'].max() - pd.DateOffset(years=1)]['Amount'].sum():,.2f}",
        'Current Yield': f"{(dividend_df[dividend_df['PayDate'] > dividend_df['PayDate'].max() - pd.DateOffset(years=1)]['Amount'].sum() / portfolio_df['Value'].sum() * 100):,.2f}%",
        '# Positions': len(portfolio_df)
    }
    
    # Dividend metrics
    monthly_income = dividend_df.groupby(dividend_df['PayDate'].dt.to_period('M'))['Amount'].sum()
    yoy_growth = ((monthly_income[-12:].mean() / monthly_income[-24:-12].mean() - 1) * 100) if len(monthly_income) >= 24 else 0
    
    dividend_metrics = {
        'Total Dividend': f"${dividend_df['Amount'].sum():,.2f}",
        'Monthly Average': f"${monthly_income.mean():,.2f}",
        'YoY Growth': f"{yoy_growth:+.1f}%"
    }
    
    return portfolio_metrics, dividend_metrics

def main():
    st.title('Ultimate Dividend Portfolio Dashboard')
    
    # File upload section
    col1, col2 = st.columns(2)
    with col1:
        portfolio_file = st.file_uploader("Upload Portfolio.csv", type=['csv'])
    with col2:
        dividend_file = st.file_uploader("Upload Dividends.csv", type=['csv'])
    
    if not portfolio_file or not dividend_file:
        st.warning('Please upload both CSV files to view the dashboard.')
        return
    
    try:
        # Load data
        portfolio_df = load_portfolio_data(portfolio_file)
        dividend_df = load_dividend_data(dividend_file)
        
        if portfolio_df is None or dividend_df.empty:
            st.error("Error processing data files.")
            return
            
        # Filters section
        st.header("Filters")
        filter_cols = st.columns(3)
        with filter_cols[0]:
            selected_period = st.selectbox("Select Period", ["All", "1Y", "2Y", "3Y"])
        with filter_cols[1]:
            years = sorted(dividend_df['PayDate'].dt.year.unique())
            selected_year = st.selectbox("Select Year", ["All"] + list(years))
        with filter_cols[2]:
            symbols = sorted(dividend_df['Symbol'].unique())
            selected_symbol = st.selectbox("Select Symbol", ["All"] + list(symbols))

        # Apply filters
        filtered_dividend_df = filter_dividend_data(
            dividend_df, 
            period=selected_period,
            year=selected_year,
            symbol=selected_symbol
        )
        
        # Calculate metrics with filtered data
        portfolio_metrics, dividend_metrics = calculate_metrics(portfolio_df, filtered_dividend_df)
        
        # Display Portfolio Metrics
        st.header("Portfolio Metrics")
        cols = st.columns(len(portfolio_metrics))
        for col, (metric, value) in zip(cols, portfolio_metrics.items()):
            col.metric(metric, value)
            
        # Display Dividend Metrics
        st.header("Dividend Metrics")
        cols = st.columns(len(dividend_metrics))
        for col, (metric, value) in zip(cols, dividend_metrics.items()):
            col.metric(metric, value)
        
        # Portfolio Analysis
        st.header("Portfolio Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_sector_allocation(portfolio_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_income_bubble_chart(portfolio_df, filtered_dividend_df), use_container_width=True)
        
        # Dividend Analysis
        st.header("Dividend Analysis")
        st.plotly_chart(create_dividend_trend_chart(filtered_dividend_df), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_dividend_per_stock_chart(filtered_dividend_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_dividend_growth_chart(filtered_dividend_df), use_container_width=True)
        
        # Monthly Income Heatmap
        st.header("Monthly Income Analysis")
        st.plotly_chart(create_monthly_income_heatmap(filtered_dividend_df), use_container_width=True)
        
        # Top Holdings
        st.header("Top Holdings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 by Value")
            top_10_value = portfolio_df.nlargest(10, 'Value')[['Symbol', 'Sector', 'Value']]
            top_10_value['% of Portfolio'] = (top_10_value['Value'] / portfolio_df['Value'].sum() * 100).round(2)
            top_10_value['Value'] = top_10_value['Value'].apply(lambda x: f"${x:,.2f}")
            top_10_value['% of Portfolio'] = top_10_value['% of Portfolio'].apply(lambda x: f"{x}%")
            st.dataframe(top_10_value)
        
        with col2:
            st.subheader("Top 10 by Income")
            latest_date = filtered_dividend_df['PayDate'].max()
            one_year_ago = latest_date - pd.DateOffset(years=1)
            annual_income = filtered_dividend_df[filtered_dividend_df['PayDate'] > one_year_ago].groupby('Symbol')['Amount'].sum()
            
            top_10_income = pd.DataFrame(annual_income).join(
                portfolio_df.set_index('Symbol')[['Sector', 'Value']], 
                on='Symbol'
            ).nlargest(10, 'Amount')
            
            top_10_income['Yield'] = (top_10_income['Amount'] / top_10_income['Value'] * 100).round(2)
            top_10_income['Amount'] = top_10_income['Amount'].apply(lambda x: f"${x:,.2f}")
            top_10_income['Yield'] = top_10_income['Yield'].apply(lambda x: f"{x}%")
            st.dataframe(top_10_income[['Sector', 'Amount', 'Yield']])

    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        st.write('Please ensure your CSV files are in the correct format.')

if __name__ == "__main__":
    main()