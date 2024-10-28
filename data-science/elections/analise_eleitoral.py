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
import concurrent.futures

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
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page_num in range(pdf.page_count):
                text += pdf[page_num].get_text()
        return text
    except Exception as e:
        print(f"Erro ao extrair texto do PDF '{file_path}': {e}")
        return ""

# Função para processar e extrair texto dos PDFs em uma pasta de propostas
def process_pdf_file(file_name, path_propostas):
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(os.path.join(path_propostas, file_name))
    return ""

# Função para garantir que a pasta output exista
def ensure_output_directory():
    if not os.path.exists("output"):
        os.makedirs("output")

ensure_output_directory()

# Função auxiliar global para leitura de arquivos
def load_file(file):
    try:
        return pd.read_csv(file, sep=';', encoding='latin1')
    except Exception as e:
        print(f"Erro ao carregar o arquivo '{file}': {e}")
    return pd.DataFrame()  # Retorna um DataFrame vazio para manter a estrutura

# Função para carregar todos os arquivos em uma pasta específica com ProcessPoolExecutor
def load_data_from_folder(folder_path, file_pattern="*.csv"):
    all_files = glob(os.path.join(folder_path, file_pattern))
    
    # Usa ProcessPoolExecutor para carregar os arquivos em paralelo
    with ProcessPoolExecutor() as executor:
        df_list = list(executor.map(load_file, all_files))

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


# Funções para cada insight com tratamento de exceções

def insight_1_economia_influencia_eleicao(dados_candidatos, dados_bens):
    print("\nInsight 1: Economia e Influência na Eleição")
    try:
        prefeitos_eleitos = dados_candidatos[(dados_candidatos['DS_CARGO'] == 'prefeito') & (dados_candidatos['DS_SIT_TOT_TURNO'] == 'eleito')]
        bens_eleitos = dados_bens[dados_bens['SQ_CANDIDATO'].isin(prefeitos_eleitos['SQ_CANDIDATO'])]
        soma_bens_eleitos = bens_eleitos['VR_BEM_CANDIDATO'].mean()
        bens_nao_eleitos = dados_bens[~dados_bens['SQ_CANDIDATO'].isin(prefeitos_eleitos['SQ_CANDIDATO'])]['VR_BEM_CANDIDATO'].mean()

        print("Carregou os dados e fez a média.")
        
        bens_media = pd.DataFrame({'Status': ['Eleitos', 'Não Eleitos'], 'Média de Bens Declarados': [soma_bens_eleitos, bens_nao_eleitos]})
        bens_media.to_csv("output/media_bens_eleitos.csv", index=False)

        print("Salvou a média em um arquivo CSV.")

        plt.figure(figsize=(8, 5))
        plt.bar(['Eleitos', 'Não Eleitos'], [soma_bens_eleitos, bens_nao_eleitos], color=['blue', 'red'])
        plt.title("Média de Bens Declarados por Prefeitos Eleitos vs Não Eleitos")
        plt.ylabel("Média de Bens Declarados (R$)")
        plt.savefig("output/media_bens_eleitos.png")
        plt.close()

        print("Salvou o gráfico em uma imagem PNG.")
    except Exception as e:
        print(f"Erro no insight 1 - economia_influencia_eleicao: {e}")

def insight_2_coligacoes_disputas_vitoria(dados_candidatos, dados_coligacoes):
    print("Insight 2: Coligações e Disputas de Vitoria")
    try:
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
    except Exception as e:
        print(f"Erro no insight 2 - coligacoes_disputas_vitoria: {e}")

def insight_3_maior_partido_uf(dados_candidatos):
    print("Insight 3: Maior Partido por UF")
    try:
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
    except Exception as e:
        print(f"Erro no insight 3 - maior_partido_uf: {e}")

def insight_4_tendencia_regional_partido(dados_candidatos):
    print("Insight 4: Tendência Regional por Partido")
    try:
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
    except Exception as e:
        print(f"Erro no insight 4 - tendencia_regional_partido: {e}")

def insight_5_partido_dominante_cargo(dados_candidatos):
    print("Insight 5: Partido Dominante por Cargo")
    try:
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
    except Exception as e:
        print(f"Erro no insight 5 - partido_dominante_cargo: {e}")

def insight_6_candidatos_indigenas_quilombolas(dados_candidatos, dados_info_complementar):
    print("Insight 6: Candidatos Indígenas e Quilombolas")
    try:
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
    except Exception as e:
        print(f"Erro no insight 6 - candidatos_indigenas_quilombolas: {e}")

def insight_7_rede_social_preferida(dados_redes_sociais):
    print("Insight 7: Rede Social Preferida")
    try:
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
    except Exception as e:
        print(f"Erro no insight 7 - rede_social_preferida: {e}")

def insight_8_termos_propostas_governo(stop_words, path_propostas='./data/candidatos_propostas_governo/SC/'):
    try:
        all_terms = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(
                process_pdf_file, os.listdir(path_propostas), [path_propostas] * len(os.listdir(path_propostas))
            ))
        for text in results:
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
    except Exception as e:
        print(f"Erro no insight 8 - termos_propostas: {e}")


def insight_9_mapa_resultados_eleicao():
    try:
        resultados_municipios = pd.DataFrame({
            'MUNICIPIO': ['Município A', 'Município B', 'Município C'],
            'LATITUDE': [-23.5505, -22.9068, -21.1775],
            'LONGITUDE': [-46.6333, -43.1729, -44.8709],
            'PARTIDO_VENCEDOR': ['partido_a', 'partido_b', 'partido_c']
        })
        
        # Mapa inicial
        mapa_brasil = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)
        cores_partidos = {'partido_a': 'blue', 'partido_b': 'red', 'partido_c': 'green'}
        
        # Função auxiliar para adicionar marcadores
        def add_marker(row):
            folium.CircleMarker(
                location=(row['LATITUDE'], row['LONGITUDE']),
                radius=6,
                color=cores_partidos.get(row['PARTIDO_VENCEDOR'], 'gray'),
                fill=True,
                fill_opacity=0.7,
                popup=row['MUNICIPIO']
            ).add_to(mapa_brasil)

        # Adiciona os marcadores em paralelo
        with ProcessPoolExecutor() as executor:
            executor.map(add_marker, [row for _, row in resultados_municipios.iterrows()])

        mapa_brasil.save("output/resultado_eleicoes_mapa.html")
        print("Mapa gerado e salvo como resultado_eleicoes_mapa.html")
        
    except Exception as e:
        print(f"Erro no insight 9 - mapa_resultados_eleicao: {e}")

# Função principal para executar todos os insights
def main():

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

    # Verifique se os dados estão carregados corretamente
    try:
        assert not dados_candidatos.empty, "Erro: dados de candidatos estão vazios!"
        assert not dados_bens.empty, "Erro: dados de bens estão vazios!"
        assert not dados_info_complementar.empty, "Erro: dados de informações complementares são vazios!"
        assert not dados_redes_sociais.empty, "Erro: dados de redes sociais estão vazios!"
        assert not dados_coligacoes.empty, "Erro: dados de coligações estão vazios!"
        assert not dados_motivo_cassacao.empty, "Erro: dados de motivos de cassação estão vazios!"
        assert not dados_vagas.empty, "Erro: dados de vagas estão vazios!"
    except AssertionError as e:
        print(e)

    insights = [
        (insight_1_economia_influencia_eleicao, [dados_candidatos, dados_bens]),
        # (insight_2_coligacoes_disputas_vitoria, [dados_candidatos, dados_coligacoes]),
        # (insight_3_maior_partido_uf, [dados_candidatos]),
        # (insight_4_tendencia_regional_partido, [dados_candidatos]),
        # (insight_5_partido_dominante_cargo, [dados_candidatos]),
        # (insight_6_candidatos_indigenas_quilombolas, [dados_candidatos, dados_info_complementar]),
        # (insight_7_rede_social_preferida, [dados_redes_sociais]),
        # (insight_8_termos_propostas_governo, [stop_words]),
        # (insight_9_mapa_resultados_eleicao, []),
    ]

    with tqdm(total=len(insights), desc="Processando Insights", unit="insight") as pbar:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(insight, *args) for insight, args in insights]
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

if __name__ == "__main__":
    main()