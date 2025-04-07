Based on the search results, I'll help you fill out the template with the information available from the document:

[Referencia]
---
- Autores: Hanan Hindy, Ethan Bayne, Robert Atkinson, Christos Tachtatzis, Jean-Noël Colin, Xavier Bellekens [1]
- Data: 14 October 2020 [1]
- Titulo: Utilising Deep Learning Techniques for Effective Zero-Day Attack Detection [1]
- Disponibilização: Open access article under CC BY license [2]

[Tabela 1]
---
- TipoDaPublicação: Article
- Qualis: Not specified in the document
- ConferênciaOuPeriódico: Electronics (MDPI)

[Tabela 2]
---
- AnoDePublicação: 2020 [1]

[Tabela 3]
---
- Autores: From multiple institutions (Abertay University, University of Strathclyde, University of Namur) [1]
- TemaPrincipal: Zero-day cyber-attack detection using deep learning techniques [3]
- Método: Autoencoder implementation compared against One-Class Support Vector Machine (SVM) [3]
- BaseAmostral: Two datasets: CICIDS2017 and NSL-KDD [4]
- Resultados: Detection accuracy of 89-99% for NSL-KDD dataset and 75-98% for CICIDS2017 dataset [3]
- Ferramentas: Deep Learning, specifically autoencoders and One-Class SVM [5]
- LimitaçõesOuCriticas: The NSL-KDD dataset shows similar detection trends due to limited number and variance of attacks covered [6]

[Outros]
- DOI: Not explicitly mentioned in the provided chunks

[Resumo]
The paper proposes an autoencoder implementation for detecting zero-day attacks in cybersecurity. The research aims to build an Intrusion Detection System (IDS) model with high recall while maintaining low false-negative rates. The study compares the proposed autoencoder model against a One-Class Support Vector Machine using two well-known datasets (CICIDS2017 and NSL-KDD). The results demonstrate that autoencoders are particularly effective at detecting complex zero-day attacks, achieving high detection accuracy rates across both datasets. [3]

Citations:
[1]: DLTechniquesForEffectiveZero-DayAttackDetection Page 1
[2]: DLTechniquesForEffectiveZero-DayAttackDetection Page 16
[3]: DLTechniquesForEffectiveZero-DayAttackDetection Page 1
[4]: DLTechniquesForEffectiveZero-DayAttackDetection Page 6
[5]: DLTechniquesForEffectiveZero-DayAttackDetection Page 13
[6]: DLTechniquesForEffectiveZero-DayAttackDetection Page 12
---
this uses a autoenconder with a SVM
offers the capabilities of Deep Learning (DL) to serve as outlier detection for zero-day attacks with high recall
autoencoder acts as a light-weight outlier detector, which could then be used for zero-day attacks detection,




Citations:
 - The increase in both the number and sheer variety of new cyber-attacks poses a tremendous challenge for IDS solutions that rely on a database of historical attack signatures
 - Current outlier-based zero-day detection research suffers from high false-negative rates
 - Their findings showed that a zero-day attack can exist for a substantial period of time (average of 10 months) before they are detected and can compromise systems during that period
 - statistical study that shows that 62% of the attacks are identified after compromising systems.
 - Moreover, the number of zero-day attacks in 2019 exceeds the previous three years
 - urgent need for more effective attack detection models