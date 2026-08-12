<div align="center">

# Detecção Híbrida de Discurso de Ódio

### Baseline Histórico CNN + Bi-LSTM no Civil Comments

[![Read in English](https://img.shields.io/badge/Read%20in-English-0077B5?style=for-the-badge&logo=google-translate&logoColor=white)](README.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange" alt="TensorFlow 2.x">
  <img src="https://img.shields.io/badge/Status-Baseline_Histórico-yellow" alt="Status: Baseline Histórico">
  <a href="https://colab.research.google.com/github/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete/blob/main/notebooks/Hatespeech_Detection_LSTM_CNN.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab">
  </a>
</p>

<img src="assets/resultado_neos_v4.png" alt="Resultados históricos do experimento Civil Comments" width="100%">

*Saída histórica preservada para rastreabilidade. Ela não representa o benchmark validado de um modelo corrigido.*

</div>

---

## Sobre o Projeto

Este repositório preserva um experimento de pesquisa para classificação multiclasse de toxicidade / categorias relacionadas a discurso de ódio utilizando o dataset **Civil Comments** e uma arquitetura neural paralela **CNN + Bi-LSTM**.

O experimento passou por diferentes abordagens até chegar à arquitetura híbrida paralela atualmente preservada no notebook histórico. Os embeddings são treinados do zero, em vez de inicializados com vetores pré-treinados como GloVe ou Word2Vec.

O repositório está sendo preparado para um ciclo controlado de correção e reavaliação. O notebook atual é mantido intencionalmente como baseline histórico para que mudanças futuras possam ser comparadas com o experimento original.

---

## Estado Atual do Projeto

O experimento salvo **não é considerado um modelo final nem pronto para produção**.

| Item | Estado atual |
| :--- | :--- |
| Arquitetura | Baseline histórico CNN + Bi-LSTM paralelo |
| Dataset | Civil Comments |
| Macro F1 salvo | ~0,90 (execução histórica) |
| Overfitting | **Confirmado** |
| Pipeline de avaliação | Precisa ser revisado antes de uso como benchmark |
| Estratégia de labels | Precisa de revisão explícita |
| Correção do modelo | Planejada |
| Benchmark final validado | **Ainda não disponível** |

O notebook salvo reporta macro F1 geral em torno de **0,90**, mas esse valor deve ser interpretado somente como um **resultado experimental histórico**. O overfitting confirmado e as preocupações ainda não resolvidas no protocolo de avaliação impedem que essa métrica seja usada como evidência da qualidade final do modelo.

Veja [Problemas Conhecidos](docs/KNOWN_ISSUES.md) e [Histórico do Experimento](docs/EXPERIMENT_HISTORY.md) para a interpretação técnica atual.

---

## Evolução do Projeto

### 1. Fase Inicial — Protótipo One-vs-Rest

- **Abordagem:** classificação em dois estágios One-vs-Rest inspirada em Pitsilis et al. (2022).
- **Arquitetura:** múltiplos classificadores Bi-LSTM independentes.
- **Resultado histórico:** aproximadamente 71% de acurácia.
- **Limitação observada:** complexidade no gerenciamento dos modelos e baixa qualidade na integração das probabilidades.

### 2. Fase Intermediária — Híbrido Sequencial

- **Abordagem:** extração de características com CNN seguida de processamento recorrente (CNN → RNN).
- **Resultado histórico:** aproximadamente 86% de acurácia.
- **Limitação observada:** o desenho sequencial aparentava descartar informações contextuais necessárias à etapa recorrente.

### 3. Baseline Histórico — Híbrido Paralelo

- **Abordagem:** a mesma entrada de embedding é processada por ramos independentes de contexto e padrões locais, sendo fundida antes da classificação.
- **Arquitetura:** ramo Bi-LSTM + ramo CNN multi-kernel.
- **Experimento salvo:** macro F1 em torno de 0,90.
- **Interpretação atual:** útil como baseline experimental, mas ainda não é um benchmark validado devido ao overfitting confirmado e às preocupações ainda não resolvidas na avaliação.

<div align="center">
  <img src="assets/resultado_neos_v3.png" alt="Resultado histórico da evolução da arquitetura" width="80%">
</div>

---

## Arquitetura

O modelo histórico processa a entrada tokenizada em dois ramos paralelos:

1. **Embedding** — embedding treinável inicializado do zero.
2. **Ramo contextual (Bi-LSTM)** — modela contexto sequencial e ordem dos tokens.
3. **Ramo de padrões (CNN Multi-Kernel)** — kernels Conv1D de diferentes tamanhos capturam padrões locais.
4. **Fusão e classificação** — as saídas dos ramos são concatenadas, seguidas por camadas densas e um classificador softmax.

Uma variante com GRU também foi explorada historicamente, mas o experimento preservado utiliza Bi-LSTM.

---

## Dados e Procedimento Histórico de Treinamento

- **Dataset:** [Jigsaw / Civil Comments](https://huggingface.co/datasets/civil_comments)
- **Balanceamento histórico:** undersampling das classes maiores e oversampling das menores.
- **Validação histórica:** validação cruzada estratificada com 5 folds.
- **Controle de treinamento:** Early Stopping.
- **Embedding:** treinado do zero.

Esses itens descrevem o que o notebook preservado executa; eles **não** significam que o protocolo atual foi aprovado como metodologia final. Em particular, a ordem de pré-processamento/balanceamento e a construção das labels serão revisadas antes do novo treinamento.

---

## Resultados Históricos

O notebook preservado contém as saídas da execução anterior e reporta aproximadamente:

| Métrica | Resultado histórico salvo |
| :--- | :--- |
| Acurácia | ~0,90 |
| Macro F1 | ~0,90 |
| Macro Recall | ~0,90 |

Essas métricas são mantidas para rastreabilidade. Elas só serão substituídas como benchmark do projeto após a correção do pipeline experimental e uma nova avaliação independente.

---

## Limitações Conhecidas

O baseline atual possui as seguintes limitações documentadas:

- overfitting do modelo confirmado;
- pré-processamento e balanceamento ocorrem antes da divisão da validação cruzada e, portanto, precisam ser corrigidos/revisados;
- a construção das labels precisa de revisão explícita, incluindo o tratamento de `severe_toxicity`;
- as métricas históricas não são consideradas o benchmark final;
- atualmente não há evidência suficiente para afirmar prontidão para produção ou generalização entre idiomas.

Detalhes: [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

---

## Estrutura do Repositório

```text
.
├── AGENTS.md
├── README.md
├── README_PT.md
├── requirements.txt
├── notebooks/
│   └── Hatespeech_Detection_LSTM_CNN.ipynb
├── src/
│   └── hate_speech_detection/
├── tests/
├── docs/
│   ├── KNOWN_ISSUES.md
│   └── EXPERIMENT_HISTORY.md
└── assets/
```

O notebook contém atualmente o experimento histórico. As áreas `src/` e `tests/` foram preparadas intencionalmente para a implementação corrigida e seu trabalho de validação.

---

## Como Executar o Notebook Histórico

### Requisitos

- Python 3.8+
- TensorFlow 2.x
- dependências listadas em `requirements.txt`

### Configuração local

```bash
git clone https://github.com/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete.git
cd Hatespeech_Detection_Civil_Comments_NLP_Obsolete
pip install -r requirements.txt
```

Depois, abra `notebooks/Hatespeech_Detection_LSTM_CNN.ipynb` em um ambiente compatível com Jupyter ou utilize o botão do Colab no topo deste README.

> O notebook está preservado como baseline histórico e contém limitações metodológicas conhecidas. Executá-lo reproduz o caminho experimental antigo; ele não representa o futuro pipeline corrigido.

---

## Referências Bibliográficas

1. **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv:2206.04162, 2022.
2. **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3. **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4. **JIGSAW / GOOGLE.** *Jigsaw Unintended Bias in Toxicity Classification*. Kaggle, 2019.

---

## Licença

Distribuído sob a licença Apache 2.0. Veja [LICENSE](LICENSE) para mais detalhes.
