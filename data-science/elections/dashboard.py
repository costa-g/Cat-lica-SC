# dashboard.py

import streamlit as st
import pandas as pd
import os

# Caminho completo para o mapa
mapa_path = os.path.abspath("resultado_eleicoes_mapa.html")

# Configuração do título e introdução do dashboard
st.title("Dashboard de Análise das Eleições Municipais 2024")
st.write("Este dashboard apresenta uma análise dos dados das eleições municipais de 2024, "
         "incluindo insights sobre poder econômico, coligações, redes sociais e propostas de governo.")

# Caminho para a pasta de saída com os gráficos e tabelas
output_path = "output/"

# Insight 1: Média de Bens Declarados - Prefeitos Eleitos vs Não Eleitos
st.subheader("Insight 1: Média de Bens Declarados - Prefeitos Eleitos vs Não Eleitos")
st.image(output_path + "media_bens_eleitos.png", caption="Média de Bens Declarados pelos Prefeitos Eleitos e Não Eleitos")

# Insight 2: Coligações com Maior Número de Eleitos
st.subheader("Insight 2: Coligações com Maior Número de Eleitos")
st.image(output_path + "coligacoes_eleitos.png", caption="Número de Eleitos por Coligação x Número de Partidos")

# Insight 3: Partido com Maior Quantidade de Candidatos por UF
st.subheader("Insight 3: Partido com Maior Quantidade de Candidatos por UF")
st.image(output_path + "partido_maior_por_uf.png", caption="Partido com Maior Quantidade de Candidatos por UF")
st.write("Dados:")
maior_partido_por_uf = pd.read_csv(output_path + "maior_partido_por_uf.csv")
st.dataframe(maior_partido_por_uf)

# Insight 4: Tendência Regional para Candidaturas por Partido
st.subheader("Insight 4: Tendência Regional para Candidaturas por Partido")
st.image(output_path + "distribuicao_partido_regiao.png", caption="Distribuição de Candidaturas por Partido e Região")
st.write("Dados:")
candidatos_por_regiao = pd.read_csv(output_path + "distribuicao_partido_regiao.csv")
st.dataframe(candidatos_por_regiao)

# Insight 5: Partido Dominante por UF (Prefeito, Vice e Vereadores)
st.subheader("Insight 5: Partido Dominante por UF (Prefeito, Vice e Vereadores)")
st.image(output_path + "partido_dominante_por_uf.png", caption="Partido Dominante por UF")
st.write("Dados:")
partido_dominante_uf = pd.read_csv(output_path + "partido_dominante_uf.csv")
st.dataframe(partido_dominante_uf)

# Insight 6: Distribuição de Candidatos Indígenas e Quilombolas por Região
st.subheader("Insight 6: Distribuição de Candidatos Indígenas e Quilombolas por Região")
st.image(output_path + "indigenas_quilombolas_regiao.png", caption="Número de Candidatos Indígenas e Quilombolas por Região")
st.write("Dados:")
indigenas_por_regiao = pd.read_csv(output_path + "indigenas_por_regiao.csv")
quilombolas_por_regiao = pd.read_csv(output_path + "quilombolas_por_regiao.csv")
st.write("Candidatos Indígenas por Região:")
st.dataframe(indigenas_por_regiao)
st.write("Candidatos Quilombolas por Região:")
st.dataframe(quilombolas_por_regiao)

# Insight 7: Rede Social Preferida dos Candidatos por Partido e UF
st.subheader("Insight 7: Rede Social Preferida dos Candidatos por Partido e UF")
st.image(output_path + "rede_social_uf.png", caption="Rede Social Preferida por UF")
st.write("Dados:")
redes_por_partido_uf = pd.read_csv(output_path + "redes_por_partido_uf.csv")
st.dataframe(redes_por_partido_uf)

# Insight 8: Principais Termos nas Propostas de Governo
st.subheader("Insight 8: Principais Termos nas Propostas de Governo")
st.image(output_path + "nuvem_termos_propostas.png", caption="Nuvem de Palavras das Propostas de Governo")
st.write("Dados:")
termos_propostas = pd.read_csv(output_path + "termos_propostas.csv")
st.dataframe(termos_propostas)

# Insight 9: Mapeamento do Resultado das Eleições com Folium
st.subheader("Insight 9: Mapeamento do Resultado das Eleições com Folium")
st.write("Clique no link abaixo para visualizar o mapa dos resultados das eleições:")
# Lê o conteúdo do arquivo HTML e incorpora no Streamlit
with open("resultado_eleicoes_mapa.html", "r", encoding="utf-8") as file:
    mapa_html = file.read()

st.components.v1.html(mapa_html, height=600, scrolling=True)

st.write("Este mapa interativo exibe o partido vencedor em cada município, colorido de acordo com a legenda.")

# Rodapé
st.write("Fonte: Dados das Eleições 2024")
