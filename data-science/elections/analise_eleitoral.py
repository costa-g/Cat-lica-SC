import os
import pandas as pd
import fitz
import nltk
import folium
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from glob import glob
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from concurrent.futures import ProcessPoolExecutor

def ensure_output_directory():
    if not os.path.exists("output"):
        os.makedirs("output")

ensure_output_directory()

def ensure_stopwords():
    nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data", "corpora", "stopwords")
    if not os.path.exists(nltk_data_path):
        nltk.download('stopwords')
    return set(stopwords.words('portuguese'))

stop_words = ensure_stopwords()

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

def process_pdf_file(file_name, path_propostas):
    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(os.path.join(path_propostas, file_name))
    return ""

def load_file(file):
    try:
        return pd.read_csv(file, sep=';', encoding='latin1')
    except Exception as e:
        print(f"Erro ao carregar o arquivo '{file}': {e}")
    return pd.DataFrame()

def load_data_from_folder(folder_path, file_pattern="*.csv"):
    all_files = glob(os.path.join(folder_path, file_pattern))
    
    with ProcessPoolExecutor() as executor:
        df_list = list(executor.map(load_file, all_files))

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


# Funções para cada insight

def insight_1_economia_influencia_eleicao(dados_candidatos, dados_bens):
    print("\nInsight 1: Economia e Influência na Eleição")
    
    try:
        prefeitos_eleitos = dados_candidatos[
            (dados_candidatos['DS_CARGO'].str.lower() == 'prefeito') &
            (dados_candidatos['DS_SIT_TOT_TURNO'].str.lower() == 'eleito')
        ][['SQ_CANDIDATO', 'NM_CANDIDATO']]

        if prefeitos_eleitos.empty:
            print("Nenhum prefeito eleito encontrado nos dados de candidatos.")
            return
        
        dados_bens['VR_BEM_CANDIDATO'] = pd.to_numeric(
            dados_bens['VR_BEM_CANDIDATO'].replace(',', '.', regex=True), errors='coerce'
        )

        bens_eleitos = dados_bens[dados_bens['SQ_CANDIDATO'].isin(prefeitos_eleitos['SQ_CANDIDATO'])]
        total_bens_por_candidato = bens_eleitos.groupby('SQ_CANDIDATO')['VR_BEM_CANDIDATO'].sum().reset_index()

        if total_bens_por_candidato.empty:
            print("Nenhum bem declarado encontrado para os prefeitos eleitos.")
            return

        top_10_bens = total_bens_por_candidato.nlargest(10, 'VR_BEM_CANDIDATO')
        top_10_bens = top_10_bens.merge(prefeitos_eleitos, on='SQ_CANDIDATO')
        top_10_bens.to_csv("output/total_bens_prefeitos_eleitos.csv", index=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=top_10_bens, 
            x='NM_CANDIDATO', 
            y='VR_BEM_CANDIDATO', 
            color='blue'
        )
        plt.title("Top 10 Prefeitos Eleitos com Maior Total de Bens Declarados")
        plt.xlabel("Nome do Candidato")
        plt.ylabel("Total de Bens Declarados (R$)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig("output/total_bens_prefeitos_eleitos.png")
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Erro no insight 1 - economia_influencia_eleicao: {e}")


def insight_2_coligacoes_disputas_vitoria(dados_candidatos, dados_coligacoes):
    print("Insight 2: Coligações e Disputas de Vitória")
    try:
        dados_coligacoes['NUMERO_PARTIDOS'] = dados_coligacoes['DS_COMPOSICAO_FEDERACAO'].str.count(',') + 1
        
        coligacoes_eleitos = dados_candidatos[dados_candidatos['DS_SIT_TOT_TURNO'].str.lower() == 'eleito']
        coligacoes_resultados = coligacoes_eleitos.groupby(['SQ_COLIGACAO', 'SG_UF']).size().reset_index(name='NUM_ELEITOS')

        coligacoes_detalhadas = dados_coligacoes.merge(coligacoes_resultados, on='SQ_COLIGACAO', how='left')
        coligacoes_detalhadas['NUM_ELEITOS'] = coligacoes_detalhadas['NUM_ELEITOS'].fillna(0)
        coligacoes_detalhadas = coligacoes_detalhadas.sort_values(by=['NUMERO_PARTIDOS', 'NUM_ELEITOS'], ascending=[False, False])
        coligacoes_detalhadas.to_csv("output/coligacoes_detalhadas.csv", index=False)
        
        plt.figure(figsize=(14, 8))
        
        scatter = sns.scatterplot(
            data=coligacoes_detalhadas, 
            x='NUMERO_PARTIDOS', 
            y='NUM_ELEITOS', 
            size='NUM_ELEITOS', 
            hue='SG_UF_x', 
            sizes=(40, 400),
            alpha=0.7, 
            legend='full'
        )

        plt.title("Coligações: Número de Eleitos por Número de Partidos e UF")
        plt.xlabel("Número de Partidos na Coligação")
        plt.ylabel("Número de Eleitos")
        scatter.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        plt.savefig("output/coligacoes_eleitos.png")
        plt.show()
        
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

def insight_9_mapa_resultados_eleicao(dados_candidatos, dados_municipios):
    try:
        candidatos_eleitos = dados_candidatos[
            (dados_candidatos['DS_SIT_TOT_TURNO'].str.lower() == 'eleito') &
            (dados_candidatos['DS_CARGO'].str.lower() == 'prefeito')
        ].copy()
        
        candidatos_eleitos.loc[:, 'NM_UE'] = candidatos_eleitos['NM_UE'].str.lower()
        candidatos_eleitos.loc[:, 'SG_UF'] = candidatos_eleitos['SG_UF'].str.lower()
        dados_municipios.loc[:, 'MUNICIPIO'] = dados_municipios['MUNICIPIO'].str.lower()
        dados_municipios.loc[:, 'SG_UF'] = dados_municipios['SG_UF'].str.lower()
        
        resultados_municipios = pd.merge(
            candidatos_eleitos[['NM_UE', 'SG_PARTIDO', 'SG_UF']],
            dados_municipios[['MUNICIPIO', 'LATITUDE', 'LONGITUDE', 'SG_UF']],
            left_on=['NM_UE', 'SG_UF'],
            right_on=['MUNICIPIO', 'SG_UF'],
            how='inner'
        )

        resultados_municipios = resultados_municipios.rename(columns={
            'NM_UE': 'MUNICIPIO',
            'SG_PARTIDO': 'PARTIDO_VENCEDOR'
        })

        resultados_municipios = resultados_municipios[['MUNICIPIO', 'LATITUDE', 'LONGITUDE', 'PARTIDO_VENCEDOR']]
        
        mapa_brasil = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)

        partidos_unicos = resultados_municipios['PARTIDO_VENCEDOR'].unique()
        num_partidos = len(partidos_unicos)
        
        colormap = plt.get_cmap('Set1', num_partidos)
        cores_partidos = {partido: mcolors.to_hex(colormap(i / num_partidos)) for i, partido in enumerate(partidos_unicos)}

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
        print("Mapa gerado e salvo como 'output/resultado_eleicoes_mapa.html'")
        
    except Exception as e:
        print(f"Erro no insight 9 - mapa_resultados_eleicao: {e}")


def main():

    data_paths = {
        "candidatos": "./data/candidatos/",
        "candidatos_bens": "./data/candidatos_bens/",
        "candidatos_info_complementar": "./data/candidatos_info_complementar/",
        "candidatos_redes_sociais": "./data/candidatos_redes_sociais/",
        "coligacoes": "./data/coligacoes/",
        "motivo_cassacao": "./data/motivo_cassacao/",
        "vagas": "./data/vagas/"
    }

    dados_candidatos = load_data_from_folder(data_paths["candidatos"])
    dados_bens = load_data_from_folder(data_paths["candidatos_bens"])
    dados_info_complementar = load_data_from_folder(data_paths["candidatos_info_complementar"])
    dados_redes_sociais = load_data_from_folder(data_paths["candidatos_redes_sociais"])
    dados_coligacoes = load_data_from_folder(data_paths["coligacoes"])
    dados_motivo_cassacao = load_data_from_folder(data_paths["motivo_cassacao"])
    dados_vagas = load_data_from_folder(data_paths["vagas"])

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

    municipios_coordenadas = pd.DataFrame({
        'MUNICIPIO': [
            'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Fortaleza',
            'Brasília', 'Curitiba', 'Manaus', 'Recife', 'Porto Alegre',
            'Belém', 'Goiânia', 'São Luís', 'Maceió', 'Natal',
            'Campo Grande', 'Teresina', 'João Pessoa', 'Aracaju', 'Cuiabá',
            'Florianópolis', 'Macapá', 'Rio Branco', 'Palmas', 'Boa Vista'
        ],
        'SG_UF': [
            'SP', 'RJ', 'MG', 'BA', 'CE', 'DF', 'PR', 'AM', 'PE', 'RS',
            'PA', 'GO', 'MA', 'AL', 'RN', 'MS', 'PI', 'PB', 'SE', 'MT',
            'SC', 'AP', 'AC', 'TO', 'RR'
        ],
        'LATITUDE': [
            -23.5505, -22.9068, -19.9245, -12.9714, -3.7172,
            -15.7942, -25.429, -3.119, -8.0476, -30.0346,
            -1.4558, -16.6869, -2.5307, -9.6498, -5.7945,
            -20.4697, -5.0919, -7.1195, -10.9472, -15.601,
            -27.5954, 0.0355, -9.9754, -10.1842, 2.8235
        ],
        'LONGITUDE': [
            -46.6333, -43.1729, -43.9352, -38.5014, -38.5434,
            -47.8822, -49.2671, -60.0217, -34.8783, -51.2177,
            -48.4902, -49.2644, -44.3028, -35.7089, -35.211,
            -54.6201, -42.8018, -34.845, -37.0731, -56.0979,
            -48.548, -51.0664, -67.8243, -48.3277, -60.6758
        ]
    })

    # Executa o insight 1 diretamente
    insight_1_economia_influencia_eleicao(dados_candidatos, dados_bens)
    print("Insight 1 processado com sucesso.")

    # Executa o insight 2 diretamente
    insight_2_coligacoes_disputas_vitoria(dados_candidatos, dados_coligacoes)
    print("Insight 2 processado com sucesso.")

    # Executa o insight 3 diretamente
    insight_3_maior_partido_uf(dados_candidatos)
    print("Insight 3 processado com sucesso.")

    # Executa o insight 4 diretamente
    insight_4_tendencia_regional_partido(dados_candidatos)
    print("Insight 4 processado com sucesso.")

    # Executa o insight 5 diretamente
    insight_5_partido_dominante_cargo(dados_candidatos)
    print("Insight 5 processado com sucesso.")

    # Executa o insight 6 diretamente
    insight_6_candidatos_indigenas_quilombolas(dados_candidatos, dados_info_complementar)
    print("Insight 6 processado com sucesso.")

    # Executa o insight 7 diretamente
    insight_7_rede_social_preferida(dados_redes_sociais)
    print("Insight 7 processado com sucesso.")

    # Executa o insight 8 diretamente
    insight_8_termos_propostas_governo(dados_candidatos)
    print("Insight 8 processado com sucesso.")

    # Executa o insight 9 diretamente
    insight_9_mapa_resultados_eleicao(dados_candidatos, municipios_coordenadas)
    print("Insight 9 processado com sucesso.")

if __name__ == "__main__":
    main()