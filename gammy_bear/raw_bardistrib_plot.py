import math
from collections import Counter
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import skew, kurtosis

def detectar_distribuicao(df, col):
  intvalues = []
  for value in df[col]:
    intvalues.append(value)
  valores = np.array(intvalues)
  skewness = skew(valores)
  kurt = kurtosis(valores)
  if abs(skewness) < 0.5:
      if abs(kurt) < 1:
          return 'Aprox. Normal'
      elif kurt > 1:
          return 'Leptocúrtica'
      else:
          return 'Platicúrtica'
  elif skewness > 0.5:
      return 'Assimetria à direita'
  elif skewness < -0.5:
      return 'Assimetria à esquerda'
  return 'Indefinida'

def class_distributions_grid(data_dict, cols=3, figsize=(5, 5)):
    total = len(data_dict)
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows))
    fig.patch.set_facecolor('black')  # fundo da figura

    axes = axes.flatten()

    for idx, (class_name, values) in enumerate(data_dict.items()):
        intvalues = []
        for value in values:
            intvalues.append(values[value])

        counts = Counter(values)
        keys = list(counts.keys())
        freq = list(counts.values())

        media = np.mean(intvalues)
        desvio = np.std(intvalues)
        skewness = skew(np.array(intvalues))
        kurt = kurtosis(np.array(intvalues))
        tipo_dist = detectar_distribuicao(intvalues)

        ax = axes[idx]
        ax.bar(keys, freq, color='blue')  # barras em azul claro

        # Fundo preto pro gráfico
        ax.set_facecolor('black')

        # Linhas de média e std
        ax.axhline(media, color='red', linestyle='--', label='Média')
        ax.axhline(media + desvio, color='orange', linestyle=':', label='+1σ')
        ax.axhline(media - desvio, color='orange', linestyle=':', label='-1σ')

        # Título com fonte branca
        ax.set_title(
            f'{class_name}\n'
            f'Média: {media:.2f}, Std: {desvio:.2f}\n'
            f'Skew: {skewness:.2f}, Kurtosis: {kurt:.2f}\n'
            f'Distribuição: {tipo_dist}',
            fontsize=10,
            color='white'
        )

        # Labels brancos
        ax.set_xlabel('Valor', color='white')
        ax.set_ylabel('Frequência', color='white')

        # Eixos e ticks brancos
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')

        # Grid branco
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='white')
        ax.legend(fontsize=8, facecolor='black', edgecolor='white', labelcolor='white')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
