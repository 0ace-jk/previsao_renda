# 🔨 Projeto de Previsão de Renda

[![Project Demo GIF](./assets/preview_streamlit_projeto_2.gif)](https://previsao-renda-acejk.streamlit.app/)

## 📝 Descrição

Este projeto utiliza a metodologia *CRISP-DM* para desenvolver um modelo de previsão de renda de clientes. A análise e a modelagem são realizadas em um [Jupyter Notebook](.projeto_2.ipynb), e a solução final é apresentada em uma aplicação interativa com [Streamlit](https://previsao-renda-acejk.streamlit.app/).

A capacidade de prever a renda de clientes é um recurso valioso para instituições financeiras, auxiliando na tomada de decisões estratégicas e na gestão de riscos. Um modelo preciso pode ser um diferencial competitivo, otimizando a concessão de crédito e prevenindo a inadimplência.

## 🚀 Como Executar Localmente

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

**Pré-requisitos:**
- Python 3.12.10

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Windows, use: venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação Streamlit:**
   ```bash
   streamlit run projeto_2.py
   ```

## 📊 Dados

O modelo foi treinado com o conjunto de dados `previsao_de_renda.csv`, que contém informações socioeconômicas sobre os clientes. A variável alvo do modelo é a `renda`.

**Principais variáveis utilizadas:**
- `sexo`: Gênero do cliente.
- `posse_de_veiculo`: Indica se o cliente possui veículo.
- `posse_de_imovel`: Indica se o cliente possui imóvel.
- `qtd_filhos`: Número de filhos do cliente.
- `tipo_renda`: Fonte de renda (ex: Assalariado, Empresário).
- `educacao`: Nível de escolaridade.
- `estado_civil`: Estado civil do cliente.
- `idade`: Idade do cliente.
- `tempo_emprego`: Duração do emprego atual, em anos.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.12
- **Bibliotecas de Análise de Dados:** Pandas, Scikit-learn, ydata-profiling
- **Visualização e Interface:** Streamlit

**Nota importante:** Este projeto foi desenvolvido com a versão 3.12 do Python, pois a biblioteca `ydata-profiling` possui restrições de compatibilidade com versões mais recentes.

Agradeço por seu interesse no projeto!
