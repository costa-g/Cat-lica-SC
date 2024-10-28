# scripts/análise_eleitoral.py

import os
import pandas as pd
from glob import glob
import fitz
import nltk
from nltk.corpus import stopwords
from collections import Counter
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Verificar e baixar stopwords apenas se ainda não estiverem instaladas
def ensure_stopwords():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data", "corpora", "stopwords")
    if not os.path.exists(nltk_data_path):
        nltk.download('stopwords')
    return set(stopwords.words('portuguese'))

# Carregar as stopwords
stop_words = ensure_stopwords()

# Função para extrair texto dos PDFs
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            text += pdf[page_num].get_text()
    return text

# Função para garantir que a pasta output exista
def ensure_output_directory():
    if not os.path.exists("output"):
        os.makedirs("output")

ensure_output_directory()

# Função para carregar todos os arquivos em uma pasta específica
def load_data_from_folder(folder_path, file_pattern="*.csv"):
    all_files = glob(os.path.join(folder_path, file_pattern))
    df_list = [pd.read_csv(file, sep=';', encoding='latin1') for file in all_files]
    return pd.concat(df_list, ignore_index=True)

# Caminhos para as pastas de dados
data_paths = {
    "candidatos": "./data/candidatos/",
    "candidatos_bens": "./data/candidatos_bens/",
    "candidatos_info_complementar": "./data/candidatos_info_complementar/",
    "candidatos_redes_sociais": "./data/candidatos_redes_sociais/",
    "coligacoes": "./data/coligacoes/",
    "motivo_cassacao": "./data/motivo_cassacao/",
    "vagas": "./data/vagas/"
}

# Carregando os dados
dados_candidatos = load_data_from_folder(data_paths["candidatos"])
dados_bens = load_data_from_folder(data_paths["candidatos_bens"])
dados_info_complementar = load_data_from_folder(data_paths["candidatos_info_complementar"])
dados_redes_sociais = load_data_from_folder(data_paths["candidatos_redes_sociais"])
dados_coligacoes = load_data_from_folder(data_paths["coligacoes"])
dados_motivo_cassacao = load_data_from_folder(data_paths["motivo_cassacao"])
dados_vagas = load_data_from_folder(data_paths["vagas"])

# Funções para cada insight

def insight_1_economia_influencia_eleicao(dados_candidatos, dados_bens):
    prefeitos_eleitos = dados_candidatos[(dados_candidatos['DS_CARGO'] == 'prefeito') & (dados_candidatos['DS_SIT_TOT_TURNO'] == 'eleito')]
    bens_eleitos = dados_bens[dados_bens['SQ_CANDIDATO'].isin(prefeitos_eleitos['SQ_CANDIDATO'])]
    soma_bens_eleitos = bens_eleitos['VR_BEM_CANDIDATO'].mean()
    bens_nao_eleitos = dados_bens[~dados_bens['SQ_CANDIDATO'].isin(prefeitos_eleitos['SQ_CANDIDATO'])]['VR_BEM_CANDIDATO'].mean()
    soma_bens_nao_eleitos = bens_nao_eleitos['VR_BEM_CANDIDATO'].mean()

    bens_media = pd.DataFrame({'Status': ['Eleitos', 'Não Eleitos'], 'Média de Bens Declarados': [soma_bens_eleitos, soma_bens_nao_eleitos]})
    bens_media.to_csv("output/media_bens_eleitos.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(['Eleitos', 'Não Eleitos'], [soma_bens_eleitos, soma_bens_nao_eleitos], color=['blue', 'red'])
    plt.title("Média de Bens Declarados por Prefeitos Eleitos vs Não Eleitos")
    plt.ylabel("Média de Bens Declarados (R$)")
    plt.savefig("output/media_bens_eleitos.png")
    plt.close()

def insight_2_coligacoes_disputas_vitoria(dados_candidatos, dados_coligacoes):
    dados_coligacoes['NUMERO_PARTIDOS'] = dados_coligacoes['DS_COMPOSICAO_FEDERACAO'].str.count(',') + 1
    coligacoes_eleitos = dados_candidatos[dados_candidatos['DS_SIT_TOT_TURNO'] == 'eleito']
    coligacoes_resultados = coligacoes_eleitos.groupby('SQ_COLIGACAO').size().reset_index(name='NUM_ELEITOS')
    coligacoes_detalhadas = dados_coligacoes.merge(coligacoes_resultados, on='SQ_COLIGACAO')
    coligacoes_detalhadas.to_csv("output/coligacoes_detalhadas.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=coligacoes_detalhadas, x='NUMERO_PARTIDOS', y='NUM_ELEITOS', hue='NM_COLIGACAO')
    plt.title("Número de Eleitos por Coligação x Número de Partidos")
    plt.xlabel("Número de Partidos")
    plt.ylabel("Número de Eleitos")
    plt.legend([], [], frameon=False)
    plt.savefig("output/coligacoes_eleitos.png")
    plt.close()

def insight_3_maior_partido_uf(dados_candidatos):
    partidos_por_uf = dados_candidatos.groupby(['SG_UF', 'SG_PARTIDO']).size().reset_index(name='NUM_CANDIDATOS')
    maior_partido_por_uf = partidos_por_uf.loc[partidos_por_uf.groupby('SG_UF')['NUM_CANDIDATOS'].idxmax()]
    maior_partido_por_uf.to_csv("output/maior_partido_por_uf.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=maior_partido_por_uf, y='SG_UF', x='NUM_CANDIDATOS', hue='SG_PARTIDO', dodge=False)
    plt.title("Partido com Maior Quantidade de Candidatos por UF")
    plt.xlabel("Número de Candidatos")
    plt.ylabel("UF")
    plt.legend(title="Partido")
    plt.savefig("output/partido_maior_por_uf.png")
    plt.close()

def insight_4_tendencia_regional_partido(dados_candidatos):
    ufs_para_regioes = {
        "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
        "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
        "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
        "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
        "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
        "PR": "Sul", "RS": "Sul", "SC": "Sul"
    }
    dados_candidatos['REGIAO'] = dados_candidatos['SG_UF'].map(ufs_para_regioes)
    candidatos_por_regiao = dados_candidatos.groupby(['REGIAO', 'SG_PARTIDO']).size().reset_index(name='NUM_CANDIDATOS')
    candidatos_por_regiao.to_csv("output/distribuicao_partido_regiao.csv", index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=candidatos_por_regiao, x='REGIAO', y='NUM_CANDIDATOS', hue='SG_PARTIDO')
    plt.title("Distribuição de Candidaturas por Partido e Região")
    plt.xlabel("Região")
    plt.ylabel("Número de Candidatos")
    plt.legend(title="Partido", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("output/distribuicao_partido_regiao.png")
    plt.close()

def insight_5_partido_dominante_cargo(dados_candidatos):
    cargos_importantes = ['prefeito', 'vice-prefeito', 'vereador']
    dados_cargos = dados_candidatos[dados_candidatos['DS_CARGO'].isin(cargos_importantes)]
    partido_dominante_uf = dados_cargos.groupby(['SG_UF', 'SG_PARTIDO']).size().reset_index(name='TOTAL_CANDIDATOS')
    partido_dominante_uf = partido_dominante_uf.loc[partido_dominante_uf.groupby('SG_UF')['TOTAL_CANDIDATOS'].idxmax()]
    partido_dominante_uf.to_csv("output/partido_dominante_uf.csv", index=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=partido_dominante_uf, y='SG_UF', x='TOTAL_CANDIDATOS', hue='SG_PARTIDO', dodge=False)
    plt.title("Partido Dominante por UF (Prefeito, Vice e Vereadores)")
    plt.xlabel("Total de Candidatos")
    plt.ylabel("UF")
    plt.legend(title="Partido")
    plt.savefig("output/partido_dominante_por_uf.png")
    plt.close()

def insight_6_candidatos_indigenas_quilombolas(dados_candidatos, dados_info_complementar):
    ufs_para_regioes = {
        "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
        "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
        "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
        "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
        "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
        "PR": "Sul", "RS": "Sul", "SC": "Sul"
    }
    candidatos_indigenas = dados_info_complementar[dados_info_complementar['CD_ETNIA_INDIGENA'] != 0]
    candidatos_quilombolas = dados_info_complementar[dados_info_complementar['ST_QUILOMBOLA'] == 'S']
    candidatos_indigenas['REGIAO'] = candidatos_indigenas['SG_UF'].map(ufs_para_regioes)
    candidatos_quilombolas['REGIAO'] = candidatos_quilombolas['SG_UF'].map(ufs_para_regioes)

    indigenas_por_regiao = candidatos_indigenas.groupby('REGIAO').size().reset_index(name='NUM_INDIGENAS')
    quilombolas_por_regiao = candidatos_quilombolas.groupby('REGIAO').size().reset_index(name='NUM_QUILOMBOLAS')

    indigenas_por_regiao.to_csv("output/indigenas_por_regiao.csv", index=False)
    quilombolas_por_regiao.to_csv("output/quilombolas_por_regiao.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=indigenas_por_regiao, x='REGIAO', y='NUM_INDIGENAS', ax=ax[0], palette="Blues")
    ax[0].set_title("Número de Candidatos Indígenas por Região")

    sns.barplot(data=quilombolas_por_regiao, x='REGIAO', y='NUM_QUILOMBOLAS', ax=ax[1], palette="Greens")
    ax[1].set_title("Número de Candidatos Quilombolas por Região")

    plt.tight_layout()
    plt.savefig("output/indigenas_quilombolas_regiao.png")
    plt.close()

def insight_7_rede_social_preferida(dados_redes_sociais):
    dados_redes_sociais['TIPO_REDE'] = dados_redes_sociais['DS_URL'].str.extract(r'(facebook|instagram|twitter|youtube|linkedin)', expand=False).fillna('outros')
    redes_por_partido_uf = dados_redes_sociais.groupby(['SG_UF', 'TIPO_REDE']).size().reset_index(name='NUM_CANDIDATOS')
    redes_por_partido_uf.to_csv("output/redes_por_partido_uf.csv", index=False)

    plt.figure(figsize=(12, 6))
    sns.countplot(data=dados_redes_sociais, x='SG_UF', hue='TIPO_REDE', order=sorted(dados_redes_sociais['SG_UF'].unique()))
    plt.title("Rede Social Preferida dos Candidatos por UF")
    plt.xlabel("UF")
    plt.ylabel("Número de Candidatos")
    plt.legend(title="Rede Social", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig("output/rede_social_uf.png")
    plt.close()

def insight_8_termos_propostas_governo(stop_words, path_propostas='./data/candidatos_propostas_governo/SC/'):
    all_terms = []
    for file_name in os.listdir(path_propostas):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(path_propostas, file_name)
            text = extract_text_from_pdf(file_path)
            tokens = [word for word in text.lower().split() if word.isalpha() and word not in stop_words]
            all_terms.extend(tokens)
    termos_frequentes = Counter(all_terms).most_common(10)
    pd.DataFrame(termos_frequentes, columns=['Termo', 'Frequência']).to_csv("output/termos_propostas.csv", index=False)

    text = ' '.join([term for term, _ in termos_frequentes])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Principais Termos nas Propostas de Governo")
    plt.savefig("output/nuvem_termos_propostas.png")
    plt.close()

def insight_9_mapa_resultados_eleicao():
    resultados_municipios = pd.DataFrame({
        'MUNICIPIO': ['Município A', 'Município B', 'Município C'],
        'LATITUDE': [-23.5505, -22.9068, -21.1775],
        'LONGITUDE': [-46.6333, -43.1729, -44.8709],
        'PARTIDO_VENCEDOR': ['partido_a', 'partido_b', 'partido_c']
    })
    mapa_brasil = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)
    cores_partidos = {'partido_a': 'blue', 'partido_b': 'red', 'partido_c': 'green'}
    for _, row in resultados_municipios.iterrows():
        folium.CircleMarker(
            location=(row['LATITUDE'], row['LONGITUDE']),
            radius=6,
            color=cores_partidos.get(row['PARTIDO_VENCEDOR'], 'gray'),
            fill=True,
            fill_opacity=0.7,
            popup=row['MUNICIPIO']
        ).add_to(mapa_brasil)
    mapa_brasil.save("output/resultado_eleicoes_mapa.html")
    print("Mapa gerado e salvo como resultado_eleicoes_mapa.html")

# Função principal para executar todos os insights em paralelo
def main():
    insights = [
        (insight_1_economia_influencia_eleicao, [dados_candidatos, dados_bens]),
        (insight_2_coligacoes_disputas_vitoria, [dados_candidatos, dados_coligacoes]),
        (insight_3_maior_partido_uf, [dados_candidatos]),
        (insight_4_tendencia_regional_partido, [dados_candidatos]),
        (insight_5_partido_dominante_cargo, [dados_candidatos]),
        (insight_6_candidatos_indigenas_quilombolas, [dados_candidatos, dados_info_complementar]),
        (insight_7_rede_social_preferida, [dados_redes_sociais]),
        (insight_8_termos_propostas_governo, [stop_words]),
        (insight_9_mapa_resultados_eleicao, []),
    ]

    # Barra de progresso com ProcessPoolExecutor para paralelizar
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(insight, *args) for insight, args in insights]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processando Insights"):
            pass  # Cada tarefa completada atualiza a barra de progresso

if __name__ == "__main__":
    main()
