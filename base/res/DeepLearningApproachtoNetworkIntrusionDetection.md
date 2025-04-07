[Referencia](#referencia)

---
  - Autores: Nathan Shone; Tran Nguyen Ngoc; Vu Dinh Phai; Qi Shi
  - Data: (February 2018)

  - Titulo: A Deep Learning Approach to Network Intrusion Detection
  - Disponibilização: [IEEE Transactions on Emerging Topics in Computational Intelligence ( Volume: 2, Issue: 1, February 2018)
](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7433297)


[Tabela 1](#tabela-1)

---
  - TipoDaPublicação: IEEE TRANSACTIONS ON EMERGING TOPICS IN COMPUTATIONAL INTELLIGENC
  - Qualis: A1
  - ConferênciaOuPeriódico: periódico

[Tabela 2](#tabela-2)

---
  - AnoDePublicação: 2018

[Tabela 3](#tabela-3)

---
  - Autores: Nathan Shone; Tran Nguyen Ngoc; Vu Dinh Phai; Qi Shi
  - TemaPrincipal: Deep Learning to Network Intrusion Detection
  - Método:  nonsymmetric deep autoencoder
  - BaseAmostral: NSL-KDD KDD'99
  - Resultados: ~40-98% training boost speed
  - Ferramentas: TensorFlow
  - LimitaçõesOuCriticas:
  
[Resumo](#resumo)


Desafios de sistemas mapeados:
---
  - mostly models opposes to anomaly detection techniques
  - bc high false error rate
  - difficulty in obtaining reliable training data
  - longevity of training data
  - behavioural dynamics
The specifics of this challenge are to create a widely-accepted anomaly detection technique capable of overcoming limitations

about classical mathods as swallow classifiers:
---
  - there are limitations with these techniques, such as the comparatively high level of human expert interaction required; expert knowledge is needed to process data e.g. identi-fying useful data and patterns

argument 4 me:
--- 
  - the drastic growth in the volume of network data, which is set to continue. This growth can be predominantly attributed to increasing levels
  - these volumes requires techniques that can analyse data in an increas-ingly rapid,
  - analysis needs to be more de-tailed and contextually-aware, which means shifting away from abstract and high-level observations. For example, behavioural changes need to be easily attributable to specific elements of a network, e.g. individual users, operating system versions or protocols

Base
---
NIDS Challenges
Network monitoring has been used extensively for the pur-poses of security, forensics and anomaly detection. However,recent advances have created many new obstacles for NIDSs.Some of the most pertinent issues include:
1) Volume - The volume of data both stored and passingthrough networks continues to increase. It is forecast thatby 2020, the amount of data in existence will top 44 ZB[4]. As such, the traffic capacity of modern networks hasdrastically increased to facilitate the volume of traffic ob-served. Many modern backbone links are now operating atwirespeeds of 100 Gbps or more. To contextualise this, a100 Gbps link is capable of handling 148,809,524 packetsper second [5]. Hence, to operate at wirespeed, a NIDSwould need to be capable of completing the analysis of apacket within 6.72 ns. Providing NIDS at such a speed isdifficult and ensuring satisfactory levels of accuracy, ef-fectiveness and efficiency also presents a significant chal-lenge.
2) Accuracy - To maintain the aforementioned levels of ac-curacy, existing techniques cannot be relied upon. There-fore, greater levels of granularity, depth and contextualunderstanding are required to provide a more holistic andaccurate view. Unfortunately, this comes with various fi-nancial, computational and time costs.
3) Diversity - Recent years have seen an increase in thenumber of new or customised protocols being utilised inmodern networks. This can be partially attributed to thenumber of devices with network and/or Internet connec-tivity. As a result, it is becoming increasingly difficult todifferentiate between normal and abnormal traffic and/orbehaviours.
4) Dynamics - Given the diversity and flexibility of modernnetworks, the behaviour is dynamic and difficult to predict.In turn, this leads to difficulty in establishing a reliablebehavioural norm. It also raises concerns as to the lifespanof learning models.
5) Low-frequency attacks - These types of attacks haveoften thwarted previous anomaly detection techniques,including artificial intelligence approaches. The problemstems from imbalances in the training dataset, meaningthat NIDS offer weaker detection precision when facedwith these types of low frequency attacks.
6) Adaptability - Modern networks have adopted many newtechnologies to reduce their reliance on static technolo-gies and management styles. Therefore, there is morewidespread usage of dynamic technologies such as con-tainerisation, virtualisation and Software Defined Net-works. NIDSs will need to be able to adapt to the usage ofsuch technologies and the side effects they bring about.B. Deep LearningDeep learning is an advanced sub-field of machine learning,which advances Machine Learning closer to Artificial Intelli-gence. It facilitates the modelling of complex relationships andconcepts [6] using multiple levels of representation. Supervisedand unsupervised learning algorithms are used to construct suc-cessively higher levels of abstraction, defined using the outputfeatures from lower levels [7].
1) Auto-Encoder: A popular technique currently utilisedwithin deep learning research is auto-encoders, which is utilisedby our proposed solution (detailed in Section IV) An auto-encoder is an unsupervised neural network-based feature ex-traction algorithm, which learns the best parameters required toreconstruct its output as close to its input as possible. One ofit desirable characteristics is the capability to provide more apowerful and non-linear generalisation than Principle Compo-nent Analysis (PCA) This is achieved by applying backpropagation and setting thetarget values to be equal to the inputs. In other words, it istrying to learn an approximation to the identity function. Anauto-encoder typically has an input layer, output layer (with thesame dimension as the input layer)
and a hidden layer. Thishidden layer normally has a smaller dimension than that of theinput (known as an undercomplete or sparse auto-encoder)
Anexample of an auto-encoder is shown in Fig. 1.Most researchers [8]—[10] use auto-encoders as a non-lineartransformation to discover interesting data structures, by im-posing other constraints on the network, and compare the re-sults with those of PCA (linear transformation)
These methodsare based on the encoder-decoder paradigm. The input is first

It proposes:
  a mixing between swallow and deep classifiers using a randon forcast up in they autoencoder
  this will let them have the dynamics from a deep learning model, with low or no human interaction and a gerneric interpretation
  but the random forecast will provide it's well know high accuracy standards
  * the result from the DL model will passes into the random forecast withou a flatten layout 