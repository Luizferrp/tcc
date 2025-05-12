from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
label_encoder = LabelEncoder()

class analyzer():
  def __init__(self, df):
    self.df = df
    self.cacheddata = {}
    for i in df.columns:
      self.cacheddata[i] = {}
      self.cacheddata[i]['dataType'] = pd.api.types.is_numeric_dtype(df[i])
      self.cacheddata[i]['total_of_classes'] = self.total_of_classes(i)
      self.cacheddata[i]['class_distribution'] = self.class_distribution(i)
      if self.df[i].dtype == "object":
        self.df[i] = label_encoder.fit_transform(self.df[i])
      if not self.cacheddata[i]['dataType']:
         print("A coluna ", i, " não é numérica, então não pode ser analisada.")
         self.cacheddata['indices'] = self.codificar_coluna(i)
         self.df[i] = label_encoder.fit_transform(self.df[i])
      else:
        self.cacheddata[i]['distribution'] = self.detectar_distribuicao(i)
        self.cacheddata[i]['skewness'] = self.calcular_skewness(df[i])
        self.cacheddata[i]['kurtosis'] = self.calcular_kurtosis(df[i])
        self.cacheddata[i]['resumo_estatistico'] = self.resumo_estatistico(i)
    self.df = pd.DataFrame(MaxAbsScaler().fit_transform(self.df))

  def codificar_coluna(self, coluna):
    valores_unicos = self.df[coluna].unique()
    mapa = {valor: idx for idx, valor in enumerate(valores_unicos)}
    coluna_transformada = self.df[coluna].map(mapa)
    return coluna_transformada, mapa
    
  def total_of_classes(self, col):
    unique = len(self.df[col].unique())
    return unique
  
  def class_distribution(self, col):
      buffer = self.df[col].value_counts()
      return buffer
  
  def distribution_grid_plot(self, data_dict, cols=3, figsize=(15, 30)):
    total = len(data_dict)
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # transforma matriz 2D em lista pra facilitar o loop

    for idx, (class_name, values) in enumerate(data_dict.items()):
        counts = Counter(values)
        keys = list(counts.keys())
        freq = list(counts.values())

        ax = axes[idx]
        ax.bar(keys, freq, color='mediumseagreen')
        ax.set_title(class_name)
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Remove eixos vazios (caso o número de gráficos não preencha toda a grade)
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
  
  def resumo_estatistico(self, col):
    intvalues = []
    for value in self.df[col]:
        intvalues.append(value)
    valores = np.array(intvalues)
    media = np.mean(valores)
    desvio = np.std(valores)
    return media, desvio

  def detectar_distribuicao(self, col):
    intvalues = []
    for value in self.df[col]:
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


  def calcular_skewness(self, valores):
      return skew(np.array(valores))

  def calcular_kurtosis(self, valores):
      return kurtosis(np.array(valores))

  def plot_class_distributions_grid(self, data_dict, cols=3, figsize=(5, 5)):
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

        media, desvio = self.resumo_estatistico(intvalues)
        skewness = self.calcular_skewness(intvalues)
        kurt = self.calcular_kurtosis(intvalues)
        tipo_dist = self.detectar_distribuicao(intvalues)

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

    def boxplot_class_distributions_grid(self, data_dict, cols=3, figsize=(18, 45)):
      total = len(data_dict)
      rows = math.ceil(total / cols)

      fig, axes = plt.subplots(rows, cols, figsize=figsize)
      fig.patch.set_facecolor('black')  # fundo da figura

      axes = axes.flatten()

      for idx, (class_name, values) in enumerate(data_dict.items()):
          intvalues = []
          for value in values:
              intvalues.append(values[value])

          counts = Counter(values)
          keys = list(counts.keys())
          freq = list(counts.values())

          media, desvio = self.resumo_estatistico(intvalues)
          skewness = self.calcular_skewness(intvalues)
          kurt = self.calcular_kurtosis(intvalues)
          tipo_dist = self.detectar_distribuicao(intvalues)

          sns.boxplot(
                  data=df, y=idx,
                  boxprops={'color': 'blue'},    # Quartis
                  medianprops={'color': 'red'},  # Mediana
                  whiskerprops={'color': 'green'}, capprops={'color': 'green'}  # Extremos
          )

      fig.tight_layout()
      plt.show()

  def plot_numericas(df, numericas):
    for col in numericas:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Boxplot
        sns.boxplot(data=df, y=col, ax=ax[0])
        ax[0].set_title(f'Boxplot - {col}')

        # Histograma com KDE
        sns.histplot(df[col], kde=True, ax=ax[1])
        ax[1].set_title(f'Distribuição - {col}')

        # Estatísticas
        media = df[col].mean()
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        stats_text = f'Média: {media:.2f}\nSkewness: {skew:.2f}\nKurtosis: {kurt:.2f}'

        # Inserir estatísticas como texto no gráfico de distribuição
        ax[1].text(
            0.95, 0.95, stats_text,
            transform=ax[1].transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
        )

        plt.tight_layout()
        plt.show()
  def top_correlacoes(df, top_n=5, metodo='pearson'):
    # Filtra apenas colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])

    # Calcula matriz de correlação
    corr = df_numerico.corr(method=metodo)

    # Ignora autocorrelações (diagonal) e pega somente pares únicos
    corr_pairs = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Var1', 'level_1': 'Var2', 0: 'Correlation'})
    )

    # Remove duplicatas (A,B) e (B,A)
    corr_pairs['Ordered'] = corr_pairs.apply(lambda row: tuple(sorted((row['Var1'], row['Var2']))), axis=1)
    corr_pairs = corr_pairs.drop_duplicates('Ordered').drop(columns='Ordered')

    # Ordena pelas maiores correlações absolutas
    top_corr = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)

    return top_corr.head(top_n)

  def top_correlacoes(df, top_n=5, metodo='pearson'):
    # Filtra apenas colunas numéricas
    df_numerico = df.select_dtypes(include=[np.number])

    # Calcula matriz de correlação
    corr = df_numerico.corr(method=metodo)

    # Ignora autocorrelações (diagonal) e pega somente pares únicos
    corr_pairs = (
        corr.where(~np.eye(corr.shape[0], dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'Var1', 'level_1': 'Var2', 0: 'Correlation'})
    )

    # Remove duplicatas (A,B) e (B,A)
    corr_pairs['Ordered'] = corr_pairs.apply(lambda row: tuple(sorted((row['Var1'], row['Var2']))), axis=1)
    corr_pairs = corr_pairs.drop_duplicates('Ordered').drop(columns='Ordered')

    # Ordena pelas maiores correlações absolutas
    top_corr = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)

    return top_corr.head(top_n)

  def separar_colunas(df):
    colunas_categoricas = []
    colunas_numericas = []

    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            colunas_numericas.append(col)
        else:
            colunas_categoricas.append(col)
    
    return colunas_categoricas, colunas_numericas
  
  def gen_plot(self, target=None):
    for coluna in self.df.columns:  
      if self.cacheddata[coluna]['dataType']:
        coluna = self.df[coluna]
        valores = coluna.dropna()
        media = valores.mean()
        desvio = valores.std()
        curtose = kurtosis(valores)
        assimetria = skew(valores)

        # Classificar distribuição
        if abs(assimetria) < 0.5:
            tipo = 'Aprox. Normal' if abs(curtose) < 1 else ('Leptocúrtica' if curtose > 1 else 'Platicúrtica')
        elif assimetria > 0.5:
            tipo = 'Assimetria à Direita'
        else:
            tipo = 'Assimetria à Esquerda'

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Boxplot
        sns.boxplot(data=valores, ax=ax[0])
        ax[0].set_title(f'Boxplot\nKurtosis: {curtose:.2f} | Distribuição: {tipo}')
        ax[0].set_ylabel(coluna.name)

        # "Gráfico de velas" com quartis
        q1 = valores.quantile(0.25)
        q2 = valores.quantile(0.50)
        q3 = valores.quantile(0.75)

        ax[1].hlines(y=1, xmin=q1, xmax=q3, linewidth=10, color='skyblue')
        ax[1].vlines(x=[q1, q2, q3], ymin=0.9, ymax=1.1, color='blue')
        ax[1].vlines(x=valores.min(), ymin=1, ymax=1, color='gray', linestyle='--')
        ax[1].vlines(x=valores.max(), ymin=1, ymax=1, color='gray', linestyle='--')
        ax[1].set_ylim(0.8, 1.2)
        ax[1].set_yticks([])
        ax[1].set_title('Gráfico de Velas (Baseado em Quartis)')
        ax[1].set_xlabel(coluna.name)

        plt.tight_layout()
        plt.show()

    else:
        if target is None:
            # Gráfico de barras simples (frequência)
            freq = self.df[coluna].value_counts()
            plt.figure(figsize=(10, 5))
            sns.barplot(x=freq.index, y=freq.values)
            plt.title(f'Distribuição de {self.df[coluna].name}')
            plt.ylabel('Frequência')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            # Gráfico de barras empilhadas com base em outra variável (ex: rótulo)
            df_temp = pd.DataFrame({coluna.name: coluna, 'target': target})
            crosstab = pd.crosstab(df_temp[coluna.name], df_temp['target'], normalize='index')

            crosstab.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab20')
            plt.title(f'{coluna.name} vs {target.name} (Barras Empilhadas Normalizadas)')
            plt.ylabel('Proporção')
            plt.xticks(rotation=45)
            plt.legend(title='Classe')
            plt.tight_layout()
            plt.show()