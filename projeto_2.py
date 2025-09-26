# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from ydata_profiling import ProfileReport
import time

st.set_page_config(
    page_title='Análise de previsão de Renda',
    layout='centered',
    page_icon='💰',
    initial_sidebar_state='expanded',
    menu_items={'About': 'Projeto de Análise de Previsão de Renda desenvolvido por Antonio Moura Jr durante o curso de Data Science da EBAC.'}
)


@st.cache_data # Cache para não recarregar os datos a cada interação
def carregar_dados():
    df = pd.read_csv('./input/previsao_de_renda.csv')
    return df

@st.cache_data
def preprocessar_dados(_df):
    # Pré-processamento dos dados
    df_encoded = _df.drop(columns=['Unnamed: 0', 'data_ref', 'id_cliente']).copy()
    df_encoded.loc[df_encoded['tipo_renda'] == 'Pensionista', 'tempo_emprego'] = \
        df_encoded.loc[df_encoded['tipo_renda'] == 'Pensionista', 'tempo_emprego'].fillna(-1)
    df_encoded.loc[df_encoded['tipo_renda'] == 'Pensionista', 'tempo_emprego'].value_counts()
    df_encoded['renda_log'] = np.log1p(df_encoded['renda'])
    ordem_escolaridade = ['Primário', 'Secundário', 'Superior incompleto', 'Superior completo', 'Pós graduação']
    mapa_escolaridade = {nivel: i for i, nivel in enumerate(ordem_escolaridade)}
    df_encoded['educacao_ordinal'] = df_encoded['educacao'].map(mapa_escolaridade)
    df_encoded.drop(columns=['educacao', 'renda'], inplace=True)
    df_encoded = pd.get_dummies(df_encoded, drop_first=True, dtype=int)
    df_encoded.posse_de_imovel = df_encoded.posse_de_imovel.astype(int)
    df_encoded.posse_de_veiculo = df_encoded.posse_de_veiculo.astype(int)

    return df_encoded

@st.cache_data
def gerar_relatorio(_df):
    # Gerar relatório
    profile = ProfileReport(_df, title='Análise de Previsão de Renda')

    # Converter o relatório para HTML
    report_html_str = profile.to_html()
    soup = BeautifulSoup(report_html_str, 'html.parser')

    # Remover a barra de navegação (A barra tem gerado problemas de renderização no Streamlit)
    nav_bar = soup.find('nav')
    if nav_bar:
        nav_bar.decompose()

    for a_tag in soup.find_all('a'):
        if a_tag.has_attr('href') and a_tag['href'].startswith('#'):
            del a_tag['href']
        if a_tag.has_attr('target'):
            del a_tag['target']

    
    return str(soup)

def otimizar_random_forest(profundidades, folhas, X_train, X_test, y_train, y_test):
    r2s = []
    indicador_profundidade = []
    indicador_folha = []
    
    total_iterations = len(profundidades) * len(folhas)
    st.markdown('---')
    my_bar = st.progress(0, text="Otimização em andamento... Por favor, aguarde.")
    iteration_count = 0

    for profundidade in profundidades:
        for folha in folhas:
            iteration_count += 1

            time.sleep(0.1)
            modelo = RandomForestRegressor(max_depth=profundidade, min_samples_leaf=folha, random_state=42)
            modelo.fit(X_train, y_train)
            r2 = modelo.score(X_test, y_test)
            
            r2s.append(r2)
            indicador_profundidade.append(profundidade)
            indicador_folha.append(folha)
            
            percent_complete = iteration_count / total_iterations
            my_bar.progress(percent_complete, text=f"Progresso: {iteration_count}/{total_iterations} combinações testadas.")

    renda_r2 = pd.DataFrame({'r2': r2s, 'profundidade': indicador_profundidade, 'n_minimo': indicador_folha})
    sns.heatmap(renda_r2.pivot(index='profundidade',
                columns='n_minimo', values='r2'), vmin=.4)
    renda_r2.pivot(index='profundidade', columns='n_minimo', values='r2')

    melhor_r2 = 0
    pivot_df = renda_r2.pivot(index='profundidade', columns='n_minimo', values='r2')
    for folha in pivot_df.columns:
        for profundidade in pivot_df.index:
            if pivot_df.loc[profundidade, folha] > melhor_r2:
                melhor_folha, melhor_profundidade, melhor_r2 = folha, profundidade, pivot_df.loc[profundidade, folha]
                print(melhor_folha, melhor_profundidade, melhor_r2)

    return melhor_folha, melhor_profundidade, melhor_r2

def side_bar():
    # Barra lateral de navegação
    st.markdown('# Previsão de renda')
    st.markdown('----')

    st.sidebar.title('Navegação')

    sidebar = st.sidebar

    if sidebar.button('Página Inicial 🏠'):
        st.session_state.pagina_atual = 'view_home'
        st.rerun() # Opcional, mas garante uma transição imediata e limpa

    if sidebar.button('Etapa 1 - CRISP-DM: Entendimento do Negócio 🔍'):
        st.session_state.pagina_atual = 'view_entendimento'
        st.rerun() # Opcional, mas garante uma transição imediata e limpa

    if sidebar.button('Etapa 2 - CRISP-DM: Entendimento dos Dados 🔍'):
        st.session_state.pagina_atual = 'view_eda'
        st.rerun() # Opcional, mas garante uma transição imediata e limpa

    if sidebar.button('Etapa 3 Crisp-DM: Preparação dos dados 🔍'):
        st.session_state.pagina_atual = 'view_preprocessamento'
        st.rerun() # Opcional, mas garante uma transição imediata e limpa

    if sidebar.button('Etapa 4 Crisp-DM: Modelagem 🔍'):
        st.session_state.pagina_atual = 'view_modelagem'
        st.rerun() # Opcional, mas garante uma transição imediata e limpa

    if sidebar.button('Etapa 5 Crisp-DM: Avaliação dos resultados 🔍'):
        st.session_state.pagina_atual = 'view_conclusao'
        st.rerun() # Opcional, mas garante uma transição imediata e limpa
def rodape():
    # Rodapé
    st.markdown('----')
    st.markdown('Projeto desenvolvido por Antonio Moura Jr. | [LinkedIn](https://www.linkedin.com/in/antoniomourajr/) | [GitHub](https://github.com/0ace-jk/) | [Kaggle](https://www.kaggle.com/antoniojunior1998)')


def view_home():
    # Página inicial
    side_bar()
    st.title('Meu Projeto de Análise de Renda: Uma Jornada com Dados 🚀')

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, 1])
    col_texto_b.markdown(
        """
        Olá! Seja bem-vindo(a) à apresentação do meu projeto de análise de dados. Esta aplicação foi criada para demonstrar, de forma interativa, todas as etapas de uma investigação sobre a previsão de renda utilizando uma base de dados real.

        Neste projeto, meu objetivo foi mergulhar em um conjunto de dados para descobrir padrões, testar hipóteses e construir um modelo de Machine Learning funcional. Todo o processo foi guiado pela metodologia CRISP-DM, um framework que ajuda a organizar o raciocínio e as tarefas em um projeto de dados.

        Esta aplicação é dividida nas fases do CRISP-DM. Convido você a navegar pelas seções usando o menu lateral:

        - Comece pelo Entendimento do Negócio: Para entender o "porquê" deste projeto.

        - Mergulhe no Entendimento dos Dados: Onde você encontrará uma análise exploratória completa e interativa.

        - Acompanhe a Preparação dos Dados: Veja como os dados brutos foram transformados e preparados para a modelagem.

        - Descubra a Modelagem: Conheça os modelos que treinei e seus fundamentos.

        - Confira a Avaliação: Veja se os modelos performaram bem e se atingimos nossos objetivos.
        """
    )

    st.markdown('----')
    st.markdown('## Como me encontrar:')
    col_perfil_a, col_perfil_b, col_perfil_c = st.columns([10, 1, 1])
    col_perfil_a.link_button('LinkedIn', url=('https://www.linkedin.com/in/antoniomourajr/'))
    col_perfil_a.link_button('GitHub', url=('https://github.com/0ace-jk/'))
    col_perfil_a.link_button('Kaggle', url=('https://www.kaggle.com/antoniojunior1998'))


def view_entendimento():
    # Página de entendimento do negócio
    side_bar()
    st.title('Etapa 1 - CRISP-DM: Entendimento do Negócio')
    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])
    col_texto_b.write(
        """
        A primeira etapa do **CRISP-DM (Cross Industry Standard Process for Data Mining)** é dedicada ao **entendimento do negócio**.  
        Antes de qualquer análise ou modelagem, é fundamental compreender o contexto do problema, os objetivos estratégicos da organização e como a solução baseada em dados poderá gerar valor.

        No nosso caso, estamos trabalhando com uma **base de dados de clientes de uma instituição financeira**.  
        O objetivo principal é **desenvolver um modelo preditivo capaz de estimar, com a maior precisão e consistência possível, o valor da variável *"renda"* de cada cliente**.  

        Essa previsão é relevante para a instituição, pois auxilia em processos como:  
        - Avaliação de crédito e definição de limites;  
        - Análise de risco financeiro;  
        - Segmentação de clientes para produtos e serviços;  
        - Suporte a decisões estratégicas relacionadas à concessão de empréstimos e investimentos.  

        Com um bom entendimento do negócio, podemos alinhar expectativas, definir métricas de sucesso e garantir que os resultados obtidos no final do processo atendam às necessidades reais da instituição.
        """
    )

    rodape()


def view_eda():
    # Página de entendimento dos dados
    side_bar()
    st.title('Etapa 2 - CRISP-DM: Entendimento dos Dados')

    # Carrega os dados usando a função em cache
    df_dados = carregar_dados()

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])

    col_texto_b.markdown(
        """
        Após compreender o problema de negócio, o próximo passo do processo **CRISP-DM** é o **entendimento dos dados**.  
        Nessa etapa, buscamos conhecer a fundo a base disponível, explorando suas características, qualidade e limitações.  

        O objetivo é responder perguntas como:  
        - Quais informações estão disponíveis na base?  
        - Qual a qualidade desses dados (existem valores ausentes, inconsistências ou outliers)?  
        - Existem variáveis irrelevantes ou redundantes?  
        - Quais relações iniciais podem ser observadas entre as variáveis e a variável-alvo (*renda*)?  

        ### Contexto da Base de Dados
        A base utilizada corresponde a informações de **clientes de uma instituição financeira**, contendo variáveis demográficas, socioeconômicas e relacionadas ao perfil de consumo.  

        Essa etapa é essencial para:  
        - Garantir que os dados estejam em condições adequadas para análise;  
        - Identificar possíveis ajustes e transformações necessárias na etapa de **Preparação dos Dados**;  
        - Levantar hipóteses iniciais que poderão ser confirmadas ou rejeitadas nas análises posteriores.  

        Em resumo, o entendimento dos dados funciona como um diagnóstico inicial, que guiará a preparação e a modelagem, assegurando que o modelo preditivo seja construído sobre uma base confiável e representativa.
        """
    )

    col_analyse_a, col_analyse_b = st.columns([1, 1])

    st.markdown(
        '''
        | Variável                | Descrição                                                   | Tipo                                              |
        | ----------------------- |:-----------------------------------------------------------:| -------------------------------------------------:|
        | data_ref                |  Data de Referência                                         | <span style="color:red">Objeto (str)</span>       |
        | id_cliente              |  Id do cliente                                              | <span style="color:red">Inteiro</span>            |
        | sexo                    |  Sexo do cliente **M** = Masculino **F** = Feminino         | <span style="color:red">Objeto (str)</span>       |
        | posse_de_veiculo        |  Cliente tem veiculo? **True** = Sim, **False** = Não       | <span style="color:red">Booleano</span>           |
        | posse_de_imovel         |  Cliente tem imovel? **True** = Sim, **False** = Não        | <span style="color:red">Booleano</span>           |
        | qtd_filhos              |  Quantidade de filhos                                       | <span style="color:red">Inteiro</span>            |
        | tipo_renda              |  Tipo de renda. (Assalariado, Empresário, etc)              | <span style="color:red">Objeto (str)</span>       |
        | educacao                |  Tipo de educação que o cliente completou                   | <span style="color:red">Objeto (str)</span>       |
        | estado_civil            |  Estado civil. (Casado, Solteiro, etc)                      | <span style="color:red">Objeto (str)</span>       |
        | tipo_residencia         |  Tipo de residência. (Casa, Aluguel, etc)                   | <span style="color:red">Objeto (str)</span>       |
        | idade                   |  Idade do cliente                                           | <span style="color:red">Inteiro</span>            |
        | tempo_emprego           |  Tempo em que o cliente está empregado                      | <span style="color:red">Ponto Flutuante</span>    |
        | qt_pessoas_residencia   |  Quantidade de pessoas que vivem na residência do cliente   | <span style="color:red">Ponto Flutuante</span>    |
        | renda                   |  Valor da renda                                             | <span style="color:red">Ponto Flutuante</span>    |
        ''', unsafe_allow_html=True
    )

    st.markdown('---')
    st.markdown('### Entendimento dos dados - Univariada')
    # Gera o relatório HTML usando a nova função em cache
    html_relatorio = gerar_relatorio(df_dados.copy())
    # Exibe o HTML final no Streamlit
    st.components.v1.html(html_relatorio, height=800, scrolling=True)

    st.markdown('---')

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])
    col_texto_b.markdown(
        '''
        ### Observamos alguns pontos interessantes, como os valores faltantes da variável *'tempo_emprego'* ter uma quantidade grande de **missing values** porem, ao crusarmos esses dados com a variável *'tipo_renda'* observamos que esses valores faltantes são predominantemente de pensionistas.

        #### Mas por que pensionistas tem dados faltantes em tempo de emprego? Vamos entender.

        - Pensionistas apresentam tempo de emprego igual a zero nos dados porque, geralmente, o sistema de registos considera que um pensionista não está em atividade de emprego, sendo o tempo de trabalho já encerrado e focado no recebimento de uma pensão.
        - Em resumo, o pensionista recebe um benefício que substitui a remuneração, e não um salário, logo o registo de tempo de emprego é nulo, pois não há atividade laboral.

        '''
    )

    st.markdown('---')
    st.markdown('# Entendimento dos dados - Bivariada')

    corr = df_dados.corr(numeric_only=True)
    corr

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])

    col_texto_b.markdown(
        '''
        Com uma simples tabela de correlação conseguimos observar alguns detalhes importantes de relações que temos em nossos dados.

        Na tabela, comparamos cada variavel entre elas, o valor <span style="color:red">**[1]**</span> significa que as variaveis comparadas tem uma relação muito forte, se o valor for <span style="color:red">**[0]**</span>, significa que elas não tem nenhuma relação, e se o valor for <span style="color:red">**[-1]**</span> significa que elas tem uma relação inversa muito forte!
        - *Quantidade de pessoas na residência* se relaciona muito com a *quantidade de filhos*, o que faz total sentido.
        - A *Idade* tem uma grande correlação com o *Tempo de emprego*, mais uma correlação forte esperada.

        Mas estamos buscando entender a variável *Renda*, entao vamos analisar ela mais de perto.
        - O *Tempo de emprego* e *Posse de veículo* possuem uma maior correlação com a renda, pelo menos a primeira vista.

        Vamos observar como elas se comportam.
        ''', unsafe_allow_html=True
    )

    container = st.container()

    col_chart_a, col_chart_b = container.columns([.5, .5], border=True)

    col_chart_a.markdown('#### (Amostra de 250 indivíduos)')

    col_chart_b.markdown('#### (15000 indivíduos)')

    col_chart_a.scatter_chart(df_dados.sample(n=250), y='tempo_emprego', x='renda', color='posse_de_veiculo', x_label='Renda', y_label='Tempo de emprego')

    sns.set_theme(style='ticks')
    sns.jointplot(data=df_dados, x='renda', y='tempo_emprego', hue='posse_de_veiculo')
    col_chart_b.pyplot(plt)
    plt.close()

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])

    col_texto_b.markdown(
        '''
        ---

        Mesmo com um grafico de dispersão fica dificil observar essa correlação.

        Está nítido a necessidade de preparação desses dados para conseguirmos visualizar com clareza.
        '''
    )

    rodape()


def view_preprocessamento():
    # Página de entendimento dos dados
    side_bar()
    st.title('Etapa 3 Crisp-DM: Preparação dos dados')

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])

    col_texto_b.markdown(
        '''
        Nessa etapa realizamos tipicamente as seguintes operações com os dados:

        - **seleção**: Já temos os dados selecionados adequadamente?
        - **limpeza**: Precisaremos identificar e tratar dados faltantes
        - **construção**: construção de novas variáveis
        - **integração**: Temos apenas uma fonte de dados, não é necessário integração
        - **formatação**: Os dados já se encontram em formatos úteis?


        '''
    )

    st.markdown(
        '''
        Vamos rever os dados que temos.

        Coluna | Tipo de dado | Obs
        :------|:------------:|:-----
        Unnamed: 0 | int64 | Essa coluna é apenas o index dos dados, vamos retirar do nosso modelo pois ela não agregará em nada.
        data_ref | object | A data é algo importante ao analisar séries temporais, porém, esse não é nosso objetivo com esse modelo, vamos retirar ela tbm.
        id_cliente | int64 | Como a coluna de index, esse id do cliente não será necessario, vamos retira-lo também.
        sexo | object | Uma variável com F e M, como boa pratica, vamos atribuir um 1 para Feminino e 0 para Masculino para nosso modelo entender de maneira correta essa variável.
        posse_de_veiculo | bool | Um booleano, sem problemas, mas como uma boa pratica, vamos atribuir valores com 0 e 1.
        posse_de_imovel | bool | Mesma situação da posse_de_veiculo, vamos atribuir valores com 0 e 1.
        qtd_filhos | int64 | Tudo certo com essa variavel.
        tipo_renda | object | Uma variável qualitativa nominal, vamos usar o One-Hot Encoding(Dummies).
        educacao | object | Uma variável qualitativa ordinal, aqui vamos usar uma técnica diferente e mais eficiente para nosso modelo, Codificação ordinal.
        estado_civil | object | Uma variável qualitativa nominal, vamos usar o One-Hot Encoding(Dummies).
        tipo_residencia | object | Uma variável qualitativa nominal, vamos usar o One-Hot Encoding(Dummies).
        idade | int64 | Tudo certo com a idade, não possúi outliers, vamos manter ela como está.
        tempo_emprego | float64 | Aparentemente tudo correto por aqui, detalhe de alguns valores faltantes como discutido anteriormente.
        qt_pessoas_residencia | float64 | Tudo certo por aqui também.
        renda | float64 | Nossa variável alvo, vamos aplicar uma técnica nela e em outras variáveis para "Normalizar" essas curvas muitos acentuadas.
        '''
    )

    st.markdown('---')

    df = carregar_dados()

    st.markdown('### Dados brutos')
    st.markdown('Abaixo temos uma amostra dos dados brutos da renda, antes de qualquer pré-processamento.')

    sns.set_theme('notebook')
    sns.histplot(df, x='renda', kde=True)
    st.pyplot(plt)
    plt.close()

    st.markdown(
        '''
        Veja que a curva de renda é muito inclinada, com muitos dados concentrados em rendas mais baixas e uma cauda longa de rendas mais altas.
        '''
    )
    st.markdown('---')

    df_encoded = preprocessar_dados(df)

    st.markdown('### Dados pré-processados')
    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])
    col_texto_b.markdown('Abaixo temos uma amostra dos dados já pré-processados, prontos para serem usados em um modelo de Machine Learning.')
    st.dataframe(df_encoded.head(10))

    st.markdown('#### Aplicamos o log na variável renda para "normalizar" a curva, deixando-a mais próxima de uma distribuição normal.')

    sns.set_theme('notebook')
    sns.histplot(df_encoded, x='renda_log', kde=True)
    st.pyplot(plt)

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])
    col_texto_b.markdown(
        '''
        ### Para saber exatamente como tudo foi pré-processado, veja o notebook [projeto-2.ipynb](https://github.com/0ace-jk/previsao_renda/blob/main/projeto-2.ipynb)  no GitHub.
        '''
    )

    rodape()


def view_modelagem():
    # Página de entendimento dos dados
    side_bar()
    st.title('Etapa 4 Crisp-DM: Modelagem')

    col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1]) 

    col_texto_b.markdown(
        '''
        Nessa etapa que realizaremos a construção do modelo. Os passos típicos são:
        - Selecionar a técnica de modelagem
        - Desenho do teste
        - Avaliação do modelo
        '''
    )

    col_texto_b.markdown(
        '''
        Vamos separar nossa base de testes entre variaveis independentes, ***X***, e variavel alvo ***y***

        Nossa base possui 15.000 registros.
        Optamos por utilizar 80% dos dados para treino (12.000 registros) e 20% para teste (3.000 registros).

        Essa divisão garante que o modelo tenha dados suficientes para aprender e, ao mesmo tempo, que possamos avaliar seu desempenho em situações novas, simulando cenários reais.

        ---

        ### Execução (Treinamento) do Modelo

        Nesta etapa, ensinamos o modelo de Machine Learning a reconhecer padrões nos dados.

        - O algoritmo Random Forest cria diversas árvores de decisão, cada uma aprendendo padrões diferentes da base.

        - Essas árvores são combinadas para gerar um resultado final mais robusto e confiável (daí o nome floresta aleatória).

        Em resumo: executar o modelo significa treinar a Random Forest para entender o comportamento dos clientes a partir do histórico fornecido.
        '''
    )

    df_encoded = preprocessar_dados(carregar_dados())
    X = df_encoded.drop(columns=['renda_log'], axis=1).copy()
    y = df_encoded['renda_log'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=4)

    st.markdown(
        '''
        ---

        Com o botão abaixo, você pode iniciar o processo de otimização do modelo RandomForest, que ajusta os hiperparâmetros para melhorar a performance do modelo.
        '''
    )

    st.markdown('<p style="color:red;">A otimização pode levar vários minutos, tenha em mente que essa etapa pode ser demorada.</p>', unsafe_allow_html=True)
    

    if st.button("Iniciar Otimização do RandomForest"):

        melhor_folha_1, melhor_profundidade_1, melhor_r2_1 = otimizar_random_forest(range(1, 92, 10), range(1, 92, 10), X_train, X_test, y_train, y_test)

        melhor_folha_2, melhor_profundidade_2, melhor_r2_2 = otimizar_random_forest(range(max(melhor_profundidade_1-11, 0)+1, melhor_profundidade_1+11, 3), range (max(melhor_folha_1-11, 0)+1, melhor_folha_1+11, 3), X_train, X_test, y_train, y_test)

        melhor_folha_3, melhor_profundidade_3, melhor_r2_3 = otimizar_random_forest(range(max(melhor_profundidade_2-3, 0)+1, melhor_profundidade_2+4), range (max(melhor_folha_2-3, 0)+1, melhor_folha_2+4), X_train, X_test, y_train, y_test)

        col_texto_a, col_texto_b, col_texto_c = st.columns([.1, 3, .1])
        st.markdown('---')
        st.markdown(f'Com isso consigo o melhor resultado para o min_sample_leaf, e para o max_depth.')
        st.markdown(f'- min_sample_leaf = **{melhor_folha_3}**')
        st.markdown(f'- max_depth = **{melhor_profundidade_3}**')



    rodape()


def view_conclusao():
    side_bar()
    st.title('Etapa 5 Crisp-DM: Avaliação dos resultados')

    st.markdown(
        '''
        O cross_val_score do scikit-learn é usado para rodar a validação cruzada do modelo — ou seja, treinar e avaliar várias vezes em diferentes divisões dos dados, obtendo uma média mais confiável do desempenho.

        Esse será o nosso método de avaliação final do modelo.
        '''
    )

    st.markdown('---')
    st.markdown('Para não ser necessário rodar novamente a otimização do RandomForest, já que o melhor resultado foi obtido, vamos treinar o modelo com esses hiperparâmetros e avaliar ele com o cross_val_score.')
    st.markdown('Relembrando os melhores hiperparâmetros encontrados: min_samples_leaf = 1 e max_depth = 15.')

    df_encoded = preprocessar_dados(carregar_dados())
    X = df_encoded.drop(columns=['renda_log'], axis=1).copy()
    y = df_encoded['renda_log'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=4)

    modelo = RandomForestRegressor(max_depth=15, min_samples_leaf=1, random_state=4)
    modelo.fit(X_train, y_train)

    r2_modelo = modelo.score(X_test, y_test)
    st.write(f'O R² do modelo alcançado foi de: {r2_modelo:.4f}')



    scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")
    st.write(f'R² em cada fold: ')
    col_a, col_b = st.columns([.3, .7])
    col_a.write(scores)
    st.write(f'\nR² médio: ', scores.mean())
    st.write(f'Desvio padrão: ', scores.std())

    rodape()





if 'pagina_atual' not in st.session_state:
    st.session_state.pagina_atual = 'view_home'
if st.session_state.pagina_atual == 'view_home':
    view_home()
elif st.session_state.pagina_atual == 'view_entendimento':
    view_entendimento()
elif st.session_state.pagina_atual == 'view_eda':
    view_eda()
elif st.session_state.pagina_atual == 'view_preprocessamento':
    view_preprocessamento()
elif st.session_state.pagina_atual == 'view_modelagem':
    view_modelagem()
elif st.session_state.pagina_atual == 'view_conclusao':
    view_conclusao()
