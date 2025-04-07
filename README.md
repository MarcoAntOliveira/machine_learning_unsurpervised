## unsupervisioned learning
this repository is destined for unsupervisioned learning

### Missing Regression
Ao invés de substituir os dados por uma media para todos os ddados faltantes pela média ou mediana, é usado um moddelo de regressão.



### Aprendizado Supervisionado x Aprendizado não supervisionado
A divisão entre algoritmos dde machine learning supervisonado e não supervisionado
- machine learning não supervisionado, usa algoritmos de machine learning para analisar e agrupar conjuntos de dados não rotulados
e agrupa os dados de acorddo com os criterios selecionados pelo prorio algoritmo.

  Armazenamento em cluster é uma técnica de mineração de dados que agrupa dados não rotulados com base em suas semelhanças ou diferenças.
  Algoritmos de armazenamento em cluster são usados para processar objetos de dados não classificados e brutos em grupos representados por
  estruturas ou padrões nas informações. Os algoritmos de armazenamento em cluster podem ser categorizados em alguns tipos, especificamente
  exclusivos, sobrepostos, hierárquicos e probabilísticos.

- Aprendizagem supervisionada, também conhecida como aprendizado de máquina supervisionado, é uma subcategoria de aprendizado de máquina e inteligência artificial.
 É definido pelo uso de conjuntos de dados rotulados para treinar algoritmos que classificam dados ou preveem resultados com precisão.


### K - Means
K-means é um algoritmo de agrupamento de dados que divide um conjunto de dados em clusters similares. É uma técnica de aprendizado não supervisionado que é muito utilizada na análise de dados e no aprendizado de máquina.
Como funciona
1. O algoritmo K-means divide os dados em clusters com base na distância entre os centroides.
2. O centroide é a média ou mediana de todos os pontos dentro do cluster.
3. O algoritmo iterativamente divide os dados em clusters, minimizando a variância em cada cluster.
#### iteração
o algoritmo itera sobre as distancia das amostras aos centroides, modificando a qual cluster cada amostra pertence. se houverr troca de cluster, o algoritmo itera novamente até que não haja mais trocas .

#### quantidade de clusters
De acordo com resultaddos empiricos foi temos a seguinte formula:
$$k =\sqrt\frac{n}{2}$$
##### Elbow
Uma seguinte sugestão é criar um grafico de performance por quantidade de cluster
#### algoritmo em python
```python
normalizador = MinMaxScaler(feature_range=(0,1))
x_norm = normalizador.fit_transform(x)

modelo = KMeans(n_clusters=2, random_state=16)
modelo.fit(x_norm)
print(modelo.cluster_centers_)

print(modelo.predict((x_norm)))  //arrayy com os valores das predições
```

### PCA - principal component analisis
A tecnica busca  diminuir a dimensionalidade  dos datasets em  que se está trabalhando.É um jeito de reduzir, em poucos fatores comuns, variáveis que guardam interrelações, criando assim, “novas variáveis” capazes de representar as variáveis originais em conjunto.


#### correlação de pearson
Correlação, em estatística, é a mensuração do quanto uma variável se relaciona a outra, tanto em termos de intensidade, quanto em direção. Dentre as diversas métricas utilizadas, a mais famosa é o Coeficiente de Correlação de Pearson, que mede o relacionamento linear entre duas variáveis. A medida varia entre -1 (máxima correlação negativa, ou seja, variações em uma variável X estão inversamente relacionadas às variações em Y) e 1 (máxima correlação positiva, isto é, variações na variável X estão diretamente relacionadas às variações na variável Y). O valor 0 significa a ausência de correlação.

O Coeficiente de Correlação de Pearson é uma métrica adimensional que consiste na razão entre a covariância de duas variáveis e a raiz quadrada do produto de suas variâncias, o que significa, na prática, o produtos dos desvios-padrão das variáveis. A fórmula parece cabeluda, mas nem é.
$$r_{xy} = \frac{\text{Cov}(x,y)}{s_x s_y} = \frac{\sum_{i=1}^{n} (x_i - \bar{x}) (y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$


### Metodos Ensemble

O metodo consiste que ao invés de usar um unico algoritmo , usar varios e construi a resposta como uma combinação de várias soluções


### Random forest
Random forest is an ensemble method, meaning it combines the predictions of multiple models (in this case, decision trees) to make a final prediction.
#### Decision Trees:
Each tree in the forest is a decision tree, a model that uses a tree-like structure to make decisions based on features.
Randomness:
The "random" in "random forest" refers to two key aspects:
Bootstrap Sampling: Each decision tree is trained on a random subset of the training data (with replacement), called a bootstrap sample.
Random Feature Selection: At each node in a decision tree, a random subset of features is considered for splitting, rather than using all features.

