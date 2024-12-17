# LLM-zero-to-hero

### Capítulo 1: Fundamentos de NLP  
**Conceitos:**  
- O que é NLP?  
- Tokenização básica (por espaços, pontuação).  
- Remoção de stopwords, normalização de texto.  
**Prática:**  
- Exemplo de limpeza e pré-processamento de texto.  
- Geração de n-grams.  
**Saída no GitHub:**  
- Notebook com exemplo de limpeza e tokenização de um texto.  
- Glossário inicial de termos em um README.

### Capítulo 2: Tokenização Moderna e Introdução a Embeddings  
**Conceitos:**  
- Tokenizers subword (BPE, WordPiece).  
- Diferença entre tokenização tradicional vs. tokenização subword.  
- Conceito de embeddings (não necessariamente treinar, mas entender).  
**Prática:**  
- Uso de tokenizers da Hugging Face.  
- Análise de como modelos pré-treinados tokenizam textos.  
**Saída no GitHub:**  
- Notebook comparando tokenização com NLTK e Hugging Face Tokenizers.  
- Anotações sobre embeddings e links para leituras adicionais.

### Capítulo 3: Introdução a Transformers e Modelos Pré-Treinados  
**Conceitos:**  
- Arquitetura Transformer (Atenção, Encoder-Decoder, etc.).  
- Modelos pré-treinados (BERT, GPT-2, DistilGPT-2) e suas diferenças.  
**Prática:**  
- Carregar um modelo leve (DistilGPT-2) localmente.  
- Realizar inferência em CPU.  
**Saída no GitHub:**  
- Notebook que carrega um modelo pré-treinado e gera texto.  
- Esquema visual (diagrama) da arquitetura Transformer no README.

### Capítulo 4: Inferência e Otimizações Básicas  
**Conceitos:**  
- CPU vs GPU vs TPU.  
- Quantização (8-bit, 4-bit) e sua utilidade.  
- Trade-offs: velocidade vs. qualidade.  
**Prática:**  
- Aplicar quantização simples usando bitsandbytes ou métodos do transformers.  
- Medir tempo de inferência antes/depois da quantização.  
**Saída no GitHub:**  
- Notebook mostrando inferência quantizada.  
- Tabela de comparação de desempenho no README.

### Capítulo 5: Fine-Tuning Convencional  
**Conceitos:**  
- Por que fine-tuning?  
- Fluxo de trabalho: dataset, split (treino/validação/teste), hiperparâmetros básicos.  
- Métricas de avaliação (acurácia, F1-score).  
**Prática:**  
- Fine-tuning simples de um modelo BERT-like em um dataset de classificação de sentimento no Colab com GPU.  
**Saída no GitHub:**  
- Notebook contendo o processo de fine-tuning passo a passo.  
- Resultados (métricas e gráficos) no README.

### Capítulo 6: Parameter-Efficient Fine-Tuning (PEFT) e LoRA  
**Conceitos:**  
- Motivação para PEFT.  
- Conceito do LoRA (Low-Rank Adaptation).  
**Prática:**  
- Aplicar LoRA em um modelo pequeno e comparar com o fine-tuning completo.  
- Ver redução de custo computacional.  
**Saída no GitHub:**  
- Notebook de exemplo com PEFT usando a library `peft`.  
- Documento comparando custos e resultados vs. fine-tuning tradicional.

### Capítulo 7: Explorando Modelos LLaMA-like e Grandes LLMs  
**Conceitos:**  
- Modelos grandes (LLaMA, LLaMA2, Falcon, BLOOM) e seus requisitos de hardware.  
- Limitações de rodar localmente.  
**Prática:**  
- Rodar inferência de um LLaMA-like model quantizado (4-bit) localmente com `llama.cpp`.  
- Fazer um fine-tuning leve usando LoRA em Colab.  
**Saída no GitHub:**  
- Notebook com inferência local (CPU) e scripts de quantização.  
- Notebook de fine-tuning LoRA no Colab.  
- README explicando desafios de escalar para modelos maiores.

### Capítulo 8: Prompting e Ajuste Fino da Qualidade  
**Conceitos:**  
- Prompt engineering: como formatar instruções para obter melhor resultado.  
- Ajuste fino qualitativo via prompts.  
**Prática:**  
- Comparar respostas de um modelo antes e depois de modificar o prompt.  
- Criar um mini-conjunto de prompts e avaliar a qualidade das respostas.  
**Saída no GitHub:**  
- Notebook com exemplos de prompting.  
- Guia de boas práticas em prompt engineering no README.

### Capítulo 9: Projeto Final - Construindo uma Aplicação de NLP/LLM  
**Conceitos:**  
- Pipeline completo: coleta de dados, limpeza, tokenização, fine-tuning, inferência.  
- Documentação e versionamento.  
**Prática:**  
- Escolher um caso de uso simples (ex: gerar respostas num estilo específico, classificação customizada, FAQs).  
- Criar dataset, aplicar LoRA, testar resultado final.  
- Documentar todo o processo no GitHub.  
**Saída no GitHub:**  
- Repositório com toda a pipeline, do dado cru ao modelo ajustado.  
- README detalhado, explicando cada passo, resultados e limitações.

### Capítulo 10: Explorar Técnicas Mais Recentes  
**Conceitos:**  
- RLHF (Reinforcement Learning from Human Feedback).  
- Instrução Fine-Tuning (Instruction Tuning).  
- Multimodalidade (texto + imagens, etc.).  
**Prática:**  
- Apenas leitura e experimentos se viável.  
**Saída no GitHub:**  
- Notas conceituais e links para papers.  
- Casos de teste simples, se possível.

