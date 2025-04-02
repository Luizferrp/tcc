[Referencia] 
---
  - Autores: Springer, Berlin, Heidelberg
  - Data: 28 March 2008
  - Titulo: Early Stopping - But When?
  - Disponibilização: [Part of the book series: Lecture Notes in Computer Science ((LNCS,volume 1524))](https://www.springer.com/series/558)

[Tabela 1]
---
  - TipoDaPublicação: Livro
  - Qualis: None
  - ConferênciaOuPeriódico: Livro

[Tabela 2]
---
  - AnoDePublicação: 2002

[Tabela 3]
---
  - Autores: Springer, Berlin, Heidelberg
  - TemaPrincipal: Earlie Exists
  - Método: NN
  - BaseAmostral: None
  - Resultados: 
  - Ferramentas: Python
  - LimitaçõesOuCriticas: a experimentação teórica do earlie exits
  
[outros]
  - citacao: Prechelt, L. (1998). Early Stopping - But When?. In: Orr, G.B., Müller, KR. (eds) Neural Networks: Tricks of the Trade. Lecture Notes in Computer Science, vol 1524. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-49430-8_3

[Resumo]


Backpropagation can be very slow particularly for multilayered networks where the cost surface is typically non-quadratic, non-convex, and high dimensional with many local minima and/or flat regions. There is no formula to guarantee that (1) the network will converge to a good solution, (2) convergence is swift, or (3) convergence even occurs at all

Adaptive learning rates Many authors, including Sompolinsky et al. [37],
Darken & Moody [9], Sutton [38], Murata et al. [28] have proposed rules for
automatically adapting the learning rates (see also [16]). These rules control the
speed of convergence by increasing or decreasing the learning rate based on the
error.

A técnica básica de early stopping interrompe o treinamento de uma rede neural assim que o erro na validação começa a aumentar, evitando overfitting. O processo segue estes passos:

Dividir os dados: Separar o conjunto de treinamento em treino e validação (exemplo: 2:1).

Treinar e monitorar: O modelo é treinado apenas no conjunto de treino, enquanto a validação é verificada periodicamente (exemplo: a cada 5 épocas).

Parar no ponto ideal: Se o erro de validação aumentar em relação à última verificação, o treinamento é interrompido.

Usar os melhores pesos: Os pesos da rede do último momento antes do aumento do erro de validação são usados como o resultado final do treinamento.

Essa técnica assume que a erro de validação reflete a capacidade de generalização do modelo, ajudando a evitar ajustes excessivos aos dados de treinamento.




techinics:
  Equalize the Learning Speeds
– give each weight its own learning rate
– learning rates should be proportional to the square root of the
number of inputs to the unit
– weights in lower layers should typically be larger than in the
higher layers

citations:
 - Why Early Stopping?
    - When training a neural network, one is usually interested in obtaining a network with optimal generalization performance. However, all standard neural network architectures such as the fully connected multi-layer perceptron are prone to overfitting

