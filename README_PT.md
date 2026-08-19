<div align="center">

# Detecção Híbrida de Discurso de Ódio

### Benchmark acadêmico final CNN + Bi-LSTM hierárquico no Civil Comments

[![Read in English](https://img.shields.io/badge/Read%20in-English-0077B5?style=for-the-badge&logo=google-translate&logoColor=white)](README.md)

<p>
  <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-2.20-orange" alt="TensorFlow 2.20">
  <img src="https://img.shields.io/badge/Status-Pesquisa_Concluída-brightgreen" alt="Status: Pesquisa Concluída">
  <a href="https://colab.research.google.com/github/Umbura/Hatespeech_Detection_Civil_Comments_NLP_Obsolete/blob/main/notebooks/final/HateSpeech_Final_Hierarchical.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir Notebook Final no Colab">
  </a>
</p>

</div>

---

## Sobre o Projeto

Este repositório contém um experimento acadêmico de NLP para classificação de toxicidade e categorias relacionadas a discurso de ódio utilizando o dataset **Civil Comments** e um encoder paralelo **CNN + Bi-LSTM**.

O notebook original é preservado como evidência histórica, mas o resultado científico final vem de um pipeline **hierárquico em dois estágios**, corrigido metodologicamente, com pré-processamento isolado por fold, EarlyStopping na validação interna e seleção nested dos thresholds de decisão.

O projeto é considerado concluído dentro do escopo acadêmico atual. Ele **não** reivindica estado da arte nem prontidão para produção.

---

## Benchmark Final da Pesquisa

O benchmark oficial do repositório é o experimento completo da PR #10 com seleção nested de thresholds:

| Avaliação | Resultado |
|---|---:|
| Stage 1 F1 tunado | 0.6319 |
| Stage 1 recall tunado | 0.5653 |
| Stage 1 PR-AUC / AP | 0.7095 |
| Stage 1 ROC-AUC | 0.9195 |
| Stage 2 oracle Macro F1 tunado | 0.5959 |
| **Macro F1 end-to-end tunado** | **0.4412** |

Uma segunda execução completa do mesmo protocolo produziu **0.4427 de Macro F1 end-to-end**. Essa repetição é mantida como evidência de estabilidade; `0.4412` continua sendo o benchmark principal para evitar selecionar retroativamente apenas a execução ligeiramente maior.

Métricas completas: [results/FINAL_RESULTS.md](results/FINAL_RESULTS.md) e [results/final_metrics.json](results/final_metrics.json).

---

## Principal Resultado Científico

Os mesmos modelos treinados foram avaliados com thresholds fixos e thresholds selecionados na validação interna:

| Avaliação end-to-end | Macro F1 |
|---|---:|
| Roteamento fixo `0.40` / labels `0.50` | 0.3496 |
| **Thresholds selecionados de forma nested** | **0.4412** |

O ganho foi de `+0.0916` absoluto, aproximadamente **+26,2% relativo**.

A principal conclusão científica não é apenas o valor final de F1. O experimento mostra que thresholds fixos estavam desalinhados com o problema hierárquico e desbalanceado. A seleção dos thresholds apenas na validação interna recuperou uma parcela importante do desempenho sem substituir a arquitetura CNN + Bi-LSTM.

---

## Formulação Hierárquica

### Stage 1 — roteamento

O Stage 1 aprende as saídas fracionárias do Civil Comments:

- `toxicity`;
- `severe_toxicity` como saída auxiliar.

A definição de verdade de referência para roteamento é:

```text
toxicity >= 0.4
```

No split de treino completo com `1.804.874` comentários, esse gate encaminha 201.476 exemplos e cobre **99,578%** das amostras que possuem ao menos um subtipo positivo do Stage 2.

### Stage 2 — classificação multilabel

Os comentários roteados são classificados independentemente para:

- `obscene`;
- `threat`;
- `insult`;
- `identity_attack`;
- `sexual_explicit`.

Os labels sobrepostos são preservados, em vez de reduzidos a uma única classe dependente da ordem.

---

## Protocolo de Validação

O experimento final utiliza dois folds externos. Em cada fold:

1. o fold externo é reservado somente para avaliação;
2. o restante dos dados é dividido em treino e validação interna comum;
3. pré-processamento aprendido é ajustado somente no treino;
4. EarlyStopping do Stage 1 e Stage 2 utiliza somente validação interna;
5. thresholds dos labels do Stage 2 são escolhidos somente na validação interna;
6. o threshold de roteamento do Stage 1 é escolhido pela Macro F1 end-to-end da validação interna;
7. os thresholds são congelados antes da avaliação externa.

O fold externo não influencia treinamento, EarlyStopping ou escolha dos thresholds.

---

## Resultados End-to-End por Label

| Label | F1 |
|---|---:|
| obscene | 0.5115 |
| threat | 0.2951 |
| insult | 0.6343 |
| identity_attack | 0.3761 |
| sexual_explicit | 0.3892 |
| **Macro F1** | **0.4412** |

O Stage 2 oracle alcança `0.5959` de Macro F1, mas esse valor não representa o sistema completo porque pressupõe roteamento perfeito.

---

## Experimento Histórico

`notebooks/Hatespeech_Detection_LSTM_CNN.ipynb` preserva o experimento original e as métricas salvas em torno de Macro F1 `~0.90`.

Essas métricas históricas **não** são o benchmark atual porque a construção dos targets e o protocolo de validação original possuem limitações metodológicas documentadas. O notebook permanece inalterado para rastreabilidade.

O notebook acadêmico canônico agora é:

`notebooks/final/HateSpeech_Final_Hierarchical.ipynb`

---

## Estrutura do Repositório

```text
.
├── README.md
├── README_PT.md
├── requirements.txt
├── notebooks/
│   ├── README.md
│   ├── Hatespeech_Detection_LSTM_CNN.ipynb     # histórico
│   └── final/
│       └── HateSpeech_Final_Hierarchical.ipynb # notebook canônico
├── results/
│   ├── FINAL_RESULTS.md
│   └── final_metrics.json
├── scripts/
│   ├── analyze_gate_coverage.py
│   ├── run_hierarchical_cv.py
│   ├── run_leakage_safe_cv.py
│   └── run_route_head_cv.py
├── src/
│   └── hate_speech_detection/
├── tests/
├── docs/
│   ├── EXPERIMENT_HISTORY.md
│   ├── KNOWN_ISSUES.md
│   ├── REPRODUCIBILITY.md
│   └── TARGET_STRATEGY.md
└── assets/
```

`run_hierarchical_cv.py` é o runner canônico do experimento final. `run_route_head_cv.py` permanece como experimento exploratório posterior e não é a fonte do benchmark final.

---

## Reproduzir o Experimento Final

Crie o ambiente local a partir da raiz do repositório:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

Execute a validação estrutural:

```bash
python -m compileall -q src scripts
python -m unittest discover -s tests -p "test_*.py" -v
```

Verifique a cobertura do gate no dataset completo:

```bash
python scripts/analyze_gate_coverage.py
```

Smoke diagnóstico:

```bash
python scripts/run_hierarchical_cv.py --max-samples 50000 --n-splits 2 --epochs 1
```

Experimento completo:

```bash
python scripts/run_hierarchical_cv.py --n-splits 2 --epochs 5
```

Métricas de execuções com `--max-samples` são apenas diagnósticas e não devem ser apresentadas como benchmark do dataset completo.

---

## Limitações

- foram utilizados dois folds externos devido ao custo computacional do treinamento completo;
- ainda existe pressão de overfitting em épocas iniciais;
- erros de roteamento se propagam permanentemente para o Stage 2;
- `sexual_explicit` é a categoria mais afetada pelo gate de verdade `toxicity >= 0.4`;
- fairness e robustez por subgrupos não foram validadas;
- o benchmark final do repositório é uma estimativa por validação cruzada, não uma avaliação congelada no split oficial de teste;
- o projeto não reivindica estado da arte nem prontidão para produção.

Veja [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) para a interpretação completa.

---

## Referências Bibliográficas

1. **PITSILIS, G. K.** *Improved two-stage hate speech classification for twitter based on Deep Neural Networks*. arXiv:2206.04162, 2022.
2. **ZHOU, C. et al.** *A C-LSTM Neural Network for Text Classification*. COLING 2016.
3. **SCHUSTER, M.; PALIWAL, K. K.** *Bidirectional recurrent neural networks*. IEEE Transactions on Signal Processing, 1997.
4. **JIGSAW / GOOGLE.** *Civil Comments / Jigsaw Unintended Bias in Toxicity Classification*.

---

## Licença

Distribuído sob a licença Apache 2.0. Veja [LICENSE](LICENSE).
