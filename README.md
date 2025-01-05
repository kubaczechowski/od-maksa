# Proponowanie kawy na podstawie rozpoznawania wieku oraz płci klienta kawiarni

## Opis aplikacji

Aplikacja analizująca zdjęcie użytkownika, wykonane w czasie rzeczywistym lub wczytane z urządzenia, aby dokonać predykcji. Na podstawie uzyskanych wyników proponowane są kawy najczęściej wybierane przez osoby o określonym wieku i płci. Projekt opracowano z myślą o wdrożeniu w sieci Starbucks, aby wzbogacić doświadczenie klientów o spersonalizowane propozycje produktów. Do trenowania modelu wykorzystano zbiór danych z platformy Kaggle, zawierający około 30 tysięcy zdjęć twarzy.

Aplikacja posiada interfejs graficzny przedstawiony poniżej:
![image](https://github.com/user-attachments/assets/2a495c4d-8610-44c7-b682-2f05370c25e0)

Na podstawie dostarczonego zdjęcia użytkownik otrzymuje rekomendację. Przykład rekomendacji jest przedstawiony poniżej:
![image](https://github.com/user-attachments/assets/54ae2cbd-879f-49da-a084-422162cf552e)


## Jak uruchomić
Należy wejść do folderu projektu a następnie wykonać komendę

```
streamlit run client_app.py 
```

## Technologie
Do stworzenia prototypu zostały wykorzystane poniższe technologie:
 - Streamlit
 - Python
 - Keras
 - TensorFlow
