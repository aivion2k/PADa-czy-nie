# Analiza Danych Meteorologicznych

Projekt obejmuje kompleksową analizę danych meteorologicznych stacji Bielsko-Biała, od ich pobrania przez API, przez analizę eksploracyjną, po zaawansowane modelowanie i wizualizację w aplikacji interaktywnej. Dane można pobrać dla dowolnej stacji meteorologicznej z listy dostępnej pod adresem: https://danepubliczne.imgw.pl/api/data/synop/

## Struktura Projektu

Projekt składa się z czterech głównych komponentów:

### 1. Pobieranie Danych (`download_data.ipynb`)
- Pobieranie danych historycznych z API IMGW
- ID stacji jest pobierane z ostatnich trzech cyfr identyfikatora stacji (np. dla Bielsko-Biała używamy '600')
- Automatyczne tworzenie katalogu 'dane' i zapisywanie plików w formacie ZIP
- Wstępne przetwarzanie i czyszczenie danych

Lista wszystkich dostępnych stacji: https://danepubliczne.imgw.pl/api/data/synop/

### 2. Analiza Danych (`data_analysis.ipynb`)
- Eksploracyjna analiza danych (EDA)
- Wizualizacje rozkładów parametrów meteorologicznych
- Analiza korelacji między zmiennymi
- Analiza trendów czasowych
- Dekompozycja szeregów czasowych

### 3. Modelowanie (`modeling.ipynb`)
- Implementacja modeli regresji do przewidywania temperatury
  - Regresja liniowa (R² = 0.799)
  - Las losowy (R² = 0.942)
- Implementacja modeli klasyfikacji do przewidywania pór roku
  - Drzewo decyzyjne (Accuracy = 0.665)
  - Las losowy (Accuracy = 0.732)
  - Regresja logistyczna (Accuracy = 0.617)
- Ocena i porównanie modeli

### 4. Aplikacja Streamlit (`streamlit.py`)
- Interaktywna aplikacja do wizualizacji danych
- Wykresy i statystyki w czasie rzeczywistym
- Możliwość eksploracji różnych aspektów danych
- Wizualizacja wyników modelowania
- Uruchomienie: `streamlit run streamlit.py`

## Wymagania

```python
pip install -r requirements.txt
```

Główne zależności:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit
- statsmodels

## Struktura Danych

Dane zawierają następujące parametry meteorologiczne:
- Temperatura
- Prędkość wiatru
- Wilgotność względna
- Ciśnienie
- Kierunek wiatru
- Suma opadów

## Główne Wnioski

1. **Analiza danych**:
   - Silne zależności między parametrami meteorologicznymi
   - Wyraźne wzorce sezonowe i dobowe
   - Znaczący wpływ pory dnia na parametry pogodowe

2. **Modelowanie**:
   - Las losowy najlepiej radzi sobie zarówno w regresji jak i klasyfikacji
   - Temperatura jest kluczowym parametrem w określaniu pory roku
   - Modele nieliniowe znacząco przewyższają modele liniowe

## Autor
Michał Król
