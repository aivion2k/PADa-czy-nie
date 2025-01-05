import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Analiza Danych Meteorologicznych", layout="wide")

df = pd.read_csv('dane/dane_stacja_600.csv')
df['data_pomiaru'] = pd.to_datetime(df['data_pomiaru'])

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Rozkład Danych", 
    "Temperatura", 
    "Korelacje",
    "Statystyki Miesięczne",
    "Róża Wiatrów",
    "Wartości Ekstremalne",
    "Dekompozycja"
])

with st.sidebar:
    st.header("Informacje o danych")
    st.write("Podstawowe statystyki:")
    st.write(df.describe())

# Tab 1: Rozkład danych
with tab1:
    st.header("Rozkład parametrów meteorologicznych")
    
    fig = plt.figure(figsize=(15, 10))
    for i, column in enumerate(['temperatura', 'predkosc_wiatru', 'wilgotnosc_wzgledna', 'cisnienie'], 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Rozkład: {column}')
    plt.tight_layout()
    st.pyplot(fig)

# Tab 2: Temperatura
with tab2:
    st.header("Analiza temperatury")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Średnia dzienna temperatura")
        daily_avg = df.groupby('data_pomiaru')['temperatura'].mean()
        fig = plt.figure(figsize=(10, 6))
        plt.plot(daily_avg.index, daily_avg.values)
        plt.title('Średnia dzienna temperatura')
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Średni dobowy przebieg temperatury")
        df['godzina'] = pd.to_numeric(df['godzina_pomiaru'])
        hourly_avg = df.groupby('godzina').agg({
            'temperatura': 'mean',
            'wilgotnosc_wzgledna': 'mean',
            'predkosc_wiatru': 'mean'
        }).round(2)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(hourly_avg.index, hourly_avg['temperatura'], marker='o')
        plt.title('Średni dobowy przebieg temperatury')
        plt.xlabel('Godzina')
        plt.ylabel('Temperatura (°C)')
        plt.grid(True)
        st.pyplot(fig)

# Tab 3: Korelacje
with tab3:
    st.header("Korelacje między parametrami")
    
    df_corr = df.copy()
    df_corr['suma_opadu'] = df_corr['suma_opadu'] / 4  # Korekta opadów
    
    df_corr['pora_roku'] = df_corr['data_pomiaru'].dt.month.map({
        12: 1, 1: 1, 2: 1,  # Zima: 1
        3: 2, 4: 2, 5: 2,  # Wiosna: 2
        6: 3, 7: 3, 8: 3,  # Lato: 3
        9: 4, 10: 4, 11: 4  # Jesień: 4
    })
    
    df_corr['godzina'] = pd.to_numeric(df_corr['godzina_pomiaru'])
    
    correlation_matrix = df_corr[[
        'temperatura', 'predkosc_wiatru', 'wilgotnosc_wzgledna',
        'cisnienie', 'suma_opadu', 'kierunek_wiatru', 'godzina', 'pora_roku'
    ]].corr()
    
    labels = {
        'temperatura': 'Temperatura',
        'predkosc_wiatru': 'Prędkość wiatru',
        'wilgotnosc_wzgledna': 'Wilgotność',
        'cisnienie': 'Ciśnienie',
        'suma_opadu': 'Suma opadów',
        'kierunek_wiatru': 'Kierunek wiatru',
        'godzina': 'Godzina',
        'pora_roku': 'Pora roku'
    }
    
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        vmin=-1, vmax=1,
        xticklabels=[labels[col] for col in correlation_matrix.columns],
        yticklabels=[labels[col] for col in correlation_matrix.columns]
    )
    plt.title('Macierz korelacji parametrów meteorologicznych', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Najsilniejsze korelacje:")
    correlations = correlation_matrix.unstack()
    sorted_correlations = correlations.sort_values(key=abs, ascending=False)
    sorted_correlations = sorted_correlations[sorted_correlations != 1.0]
    
    for idx, val in sorted_correlations[:10].items():
        var1, var2 = idx
        st.write(f"{labels[var1]} - {labels[var2]}: {val:.3f}")

# Tab 4: Statystyki miesięczne
with tab4:
    st.header("Statystyki miesięczne")
    
    df['miesiac'] = df['data_pomiaru'].dt.month
    monthly_stats = df.groupby('miesiac').agg({
        'temperatura': ['mean', 'std'],
        'suma_opadu': 'sum',
        'predkosc_wiatru': 'mean',
        'wilgotnosc_wzgledna': 'mean'
    }).round(2)
    
    st.dataframe(monthly_stats)
    
    st.subheader("Suma opadów miesięczna")
    monthly_precipitation = (df.groupby(df['data_pomiaru'].dt.strftime('%Y-%m'))['suma_opadu']
                           .sum()
                           .div(4)
                           .reset_index())
    
    fig = plt.figure(figsize=(20, 6))
    monthly_precipitation.plot(kind='bar', x='data_pomiaru', y='suma_opadu')
    plt.title('Suma opadów miesięczna [mm]')
    plt.xlabel('Miesiąc')
    plt.ylabel('Suma opadów [mm]')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(monthly_precipitation['suma_opadu']):
        plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Średnia suma miesięczna", f"{monthly_precipitation['suma_opadu'].mean():.1f} mm")
    col2.metric("Maksymalna suma miesięczna", f"{monthly_precipitation['suma_opadu'].max():.1f} mm")
    col3.metric("Minimalna suma miesięczna", f"{monthly_precipitation['suma_opadu'].min():.1f} mm")

# Tab 5: Róża wiatrów
with tab5:
    st.header("Róża wiatrów")
    
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(111, projection='polar')
    kierunki = df['kierunek_wiatru']
    predkosci = df['predkosc_wiatru']
    plt.hist2d(np.radians(kierunki), predkosci, bins=[16, 16])
    plt.title('Róża wiatrów')
    st.pyplot(fig)

# Tab 6: Wartości ekstremalne
with tab6:
    st.header("Wartości ekstremalne")
    
    for column in ['temperatura', 'predkosc_wiatru', 'suma_opadu']:
        st.subheader(column)
        col1, col2 = st.columns(2)
        
        max_val = df[column].max()
        max_date = df.loc[df[column].idxmax(), 'data_pomiaru']
        min_val = df[column].min()
        min_date = df.loc[df[column].idxmin(), 'data_pomiaru']
        
        col1.metric("Maksimum", f"{max_val:.2f}", f"Data: {max_date}")
        col2.metric("Minimum", f"{min_val:.2f}", f"Data: {min_date}")

# Tab 7: Dekompozycja
with tab7:
    st.header("Dekompozycja szeregów czasowych")
    
    parameter = st.selectbox(
        "Wybierz parametr do dekompozycji:",
        ['temperatura', 'predkosc_wiatru', 'wilgotnosc_wzgledna', 'cisnienie', 'suma_opadu']
    )
    
    df_2023 = df[df['data_pomiaru'].dt.year == 2023]
    decomposition = seasonal_decompose(
        df_2023[parameter],
        period=24*7,
        extrapolate_trend='freq'
    )
    
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(15,12))
    fig.suptitle(f'Dekompozycja szeregu czasowego - {parameter}')
    decomposition.observed.plot(ax=ax1, title='Obserwacje')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Sezonowość')
    decomposition.resid.plot(ax=ax4, title='Reszty')
    plt.tight_layout()
    st.pyplot(fig)