import numpy as np
import matplotlib.pyplot as plt

# Parametry symulacji
N_values = [100, 1000, 10000]  # Liczby kroków
num_samples = 5000  # Liczba realizacji procesu błędzenia losowego
num_bins = 20  # Liczba przedziałów histogramu

# Definicja funkcji gęstości prawdopodobieństwa dla P_N
def probability_density_PN(x):
    """Funkcja gęstości prawdopodobieństwa f_N(x) = 1 / (pi * sqrt(x * (1-x)))"""
    return np.where((x > 0) & (x < 1), 1 / (np.pi * np.sqrt(x * (1 - x))), 0)

# Symulacja dla różnych N
for N in N_values:
    PN_samples = []  # Lista wartości frakcji czasu P_N

    for _ in range(num_samples):
        # Symulacja błędzenia losowego
        X = np.random.choice([-1, 1], size=N)  # N losowych kroków ±1
        S_N = np.cumsum(X)  # Pozycje w kolejnych krokach
        
        # Obliczanie L_N (ile razy S_N było nad osią OX)
        L_N = np.sum(S_N > 0)  
        P_N = L_N / N  # Frakcja czasu nad osią OX
        
        PN_samples.append(P_N)

    # Konwersja listy na numpy array
    PN_samples = np.array(PN_samples)

    # Tworzenie histogramu
    plt.figure(figsize=(8, 5))
    plt.hist(PN_samples, bins=num_bins, density=True, alpha=0.6, color='b', label="Empiryczna gęstość")

    # Rysowanie funkcji gęstości f_N(x)
    x_values = np.linspace(0.01, 0.99, 100)  # Zakres (0,1) bez krańców, aby uniknąć dzielenia przez 0
    y_values = probability_density_PN(x_values)
    plt.plot(x_values, y_values, 'r-', linewidth=2, label=r"$\frac{1}{\pi \sqrt{x(1-x)}}$")

    # Opis wykresu
    plt.xlabel(r"$P_N$")
    plt.ylabel("Gęstość prawdopodobieństwa")
    plt.title(f"Histogram P_N dla N={N}")
    plt.legend()
    plt.grid()

    # Zapis wykresu do pliku
    filename = f"histogram_PN_N{N}.png"
    plt.savefig(filename)
    print(f"Wykres histogramu dla N={N} zapisano jako {filename}")

    plt.close()  # Zamknięcie wykresu, aby zwolnić pamięć
