import csv
import random

def gerar_csv(nome_arquivo='dados.csv', n_linhas=50):
    cores = ['Vermelho', 'Azul', 'Verde', 'Amarelo', 'Preto', 'Branco']
    avaliacoes = ['Ruim', 'Médio', 'Bom', 'Excelente']  # Categórico ordinal
    temperaturas = [round(random.uniform(-10, 40), 1) for _ in range(n_linhas)]  # Intervalo
    idades = [random.randint(1, 100) for _ in range(n_linhas)]  # Ratio

    with open(nome_arquivo, mode='w', newline='', encoding='utf-8') as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(['Cor', 'Avaliação', 'Temperatura (°C)', 'Idade'])
        for _ in range(n_linhas):
            linha = [
                random.choice(cores),       # Categórico nominal
                random.choice(avaliacoes),  # Categórico ordinal
                random.choice(temperaturas),# Numérico intervalo
                random.choice(idades)       # Numérico ratio
            ]
            escritor.writerow(linha)

if __name__ == "__main__":
    gerar_csv('dummydata.csv', 100)
