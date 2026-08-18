<div align="center">

# Detecção Híbrida de Discurso de Ódio

### Baseline Histórico CNN + Bi-LSTM no Civil Comments

[![Read in English](https://img.shields.io/badge/Read%20in-English-0077B5?style=for-the-badge&logo=google-translate&logoColor=white)](README.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange" alt="TensorFlow 2.21">
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

O repositório está agora em um ciclo controlado de correção e reavaliação. O notebook histórico é mantido intencionalmente para que o novo caminho experimental possa ser comparado com ele sem reescrever as evidências anteriores.

---

## Estado Atual do Projeto

O experimento salvo **não é considerado um modelo final nem pronto para produção**.

| Item | Estado atual |
| :--- | :--- |
| Arquitetura | Baseline histórico CNN + Bi-LSTM paralelo |
| Dataset | Civil Comments |
| Macro F1 salvo | ~0,90 (execução histórica) |
| Overfitting | **Confirmado** |
| Pipeline de avaliação | Runners leakage-safe e hierárquico implementados; novo benchmark pendente |
| Estratégia de labels | Estratégia hierárquica em dois estágios definida; gate inicial selecionado em `0.4` |
| Correção do modelo | Em andamento |
| Benchmark final validado | **Ainda não disponível** |

O notebook salvo reporta macro F1 geral em torno de **0,90**, mas esse valor deve ser interpretado somente como um **resultado experimental histórico**. O overfitting confirmado e as questões ainda não resolvidas sobre a qualidade do modelo impedem que essa métrica seja usada como evidência da qualidade final.

Veja [Problemas Conhecidos](docs/KNOWN_ISSUES.md), [Histórico do Experimento](docs/EXPERIMENT_HISTORY.md), [Estratégia de Targets](docs/TARGET_STRATEGY.md) e [Reprodutibilidade](docs/REPRODUCIBILITY.md) para a interpretação técnica atual.

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
- **Interpretação atual:** útil como baseline experimental, mas ainda não é um benchmark validado porque o overfitting foi confirmado e o pipeline corrigido ainda não produziu métricas substitutas.

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

- **Dataset:** [Google / Civil Comments](https://huggingface.co/datasets/google/civil_comments)
- **Balanceamento histórico:** undersampling das classes maiores e oversampling das menores.
- **Validação histórica:** validação cruzada estratificada com 5 folds.
- **Controle de treinamento:** Early Stopping.
- **Embedding:** treinado do zero.

Esses itens descrevem o que o notebook preservado executa; eles **não** significam que o protocolo histórico seja a metodologia final. O caminho corrigido agora isola o pré-processamento por fold e usa a estratégia hierárquica documentada em `docs/TARGET_STRATEGY.md`.

---

## Resultados Históricos

O notebook preservado contém as saídas da execução anterior e reporta aproximadamente:

| Métrica | Resultado histórico salvo |
| :--- | :--- |
| Acurácia | ~0,90 |
| Macro F1 | ~0,90 |
| Macro Recall | ~0,90 |

Essas métricas são mantidas para rastreabilidade. Elas só serão substituídas como benchmark do projeto após o novo treinamento e uma nova avaliação independente do pipeline corrigido.

---

## Limitações Conhecidas

O projeto ainda possui as seguintes limitações documentadas:

- overfitting do modelo confirmado;
- o runner hierárquico ainda não produziu métricas substitutas de benchmark;
- erros de roteamento no Stage 1 podem se propagar para o Stage 2 e precisam ser medidos end-to-end;
- o gate `0.4` ainda perde uma pequena parcela dos exemplos positivos de subtipo, principalmente `sexual_explicit`;
- calibração de threshold por label, fairness, robustez e prontidão para produção ainda não foram validadas;
- as métricas históricas não são consideradas o benchmark final.

Detalhes: [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) e [docs/TARGET_STRATEGY.md](docs/TARGET_STRATEGY.md).

---

## Estrutura do Repositório

```text
.
├── .python-version
├── AGENTS.md
├── README.md
├── README_PT.md
├── requirements.txt
├── notebooks/
│   └── Hatespeech_Detection_LSTM_CNN.ipynb
├── scripts/
│   ├── analyze_gate_coverage.py
│   ├── run_hierarchical_cv.py
│   └── run_leakage_safe_cv.py
├── src/
│   └── hate_speech_detection/
├── tests/
├── docs/
│   ├── KNOWN_ISSUES.md
│   ├── EXPERIMENT_HISTORY.md
│   ├── REPRODUCIBILITY.md
│   └── TARGET_STRATEGY.md
└── assets/
```

O notebook permanece como o experimento histórico. As áreas `src/`, `scripts/` e `tests/` contêm o caminho de implementação e validação corrigido.

---

## Configuração do Runtime Corrigido

O runtime corrigido está fixado em **Python 3.12** e nas versões exatas registradas em `requirements.txt`.

Usando `uv`:

```bash
git clone https://github.com/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete.git
cd Hatespeech_Detection_Civil_Comments_NLP_Obsolete
uv venv --python 3.12
uv pip install -r requirements.txt
```

Execute a validação do repositório:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

Reproduza a análise de gate já medida:

```bash
python scripts/analyze_gate_coverage.py
```

Inicie o experimento hierárquico apenas quando uma execução completa de treinamento for desejada:

```bash
python scripts/run_hierarchical_cv.py
```

O notebook histórico é anterior ao runtime atualmente fixado. As versões exatas do ambiente original não foram registradas, portanto as dependências atuais não devem ser descritas como o ambiente histórico original.

---

## Notebook Histórico

Abra `notebooks/Hatespeech_Detection_LSTM_CNN.ipynb` em um ambiente compatível com Jupyter ou utilize o botão do Colab no topo deste README.

> O notebook é preservado como evidência histórica e contém limitações metodológicas conhecidas. Ele não é alterado para representar o pipeline corrigido.

---

## Referências Bibliográficas

1. **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv:2206.04162, 2022.
2. **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3. **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4. **JIGSAW / GOOGLE.** *Jigsaw Unintended Bias in Toxicity Classification*. Kaggle, 2019.

---

## Licença

Distribuído sob a licença Apache 2.0. Veja [LICENSE](LICENSE) para mais detalhes.
