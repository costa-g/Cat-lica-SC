# scripts/análise_eleitoral.py

import os
import pandas as pd
from glob import glob
import fitz
import nltk
from nltk.corpus import stopwords
import matplotlib.ticker as mticker
from collections import Counter
import folium
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from matplotlib import cm, colors
from matplotlib import colormaps

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

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def insight_1_economia_influencia_eleicao(dados_candidatos, dados_bens):
    print("\nInsight 1: Economia e Influência na Eleição")
    
    try:
        # Verificação para garantir que a pasta output existe
        os.makedirs("output", exist_ok=True)

        # Filtra os candidatos eleitos para prefeito
        prefeitos_eleitos = dados_candidatos[
            (dados_candidatos['DS_CARGO'].str.lower() == 'prefeito') &
            (dados_candidatos['DS_SIT_TOT_TURNO'].str.lower() == 'eleito')
        ][['SQ_CANDIDATO', 'NM_CANDIDATO']]

        # Verificação: se prefeitos_eleitos está vazio
        if prefeitos_eleitos.empty:
            print("Nenhum prefeito eleito encontrado nos dados de candidatos.")
            return
        
        # Converte a coluna de bens para float, substituindo vírgulas por pontos e valores não numéricos por NaN
        dados_bens['VR_BEM_CANDIDATO'] = pd.to_numeric(
            dados_bens['VR_BEM_CANDIDATO'].replace(',', '.', regex=True), errors='coerce'
        )

        # Calcula o total de bens de cada prefeito eleito
        bens_eleitos = dados_bens[dados_bens['SQ_CANDIDATO'].isin(prefeitos_eleitos['SQ_CANDIDATO'])]
        total_bens_por_candidato = bens_eleitos.groupby('SQ_CANDIDATO')['VR_BEM_CANDIDATO'].sum().reset_index()

        # Verificação: se total_bens_por_candidato está vazio
        if total_bens_por_candidato.empty:
            print("Nenhum bem declarado encontrado para os prefeitos eleitos.")
            return

        # Filtra os 10 prefeitos eleitos com maior valor declarado de bens
        top_10_bens = total_bens_por_candidato.nlargest(10, 'VR_BEM_CANDIDATO')
        
        # Junta com o nome dos candidatos para facilitar a visualização
        top_10_bens = top_10_bens.merge(prefeitos_eleitos, on='SQ_CANDIDATO')
        
        # Salva os dados dos 10 maiores em um CSV na pasta output
        top_10_bens.to_csv("output/total_bens_prefeitos_eleitos.csv", index=False)

        # Geração e exibição do gráfico
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
        
        # Salva e exibe o gráfico
        plt.savefig("output/total_bens_prefeitos_eleitos.png")
        plt.show()
        plt.close()

    except Exception as e:
        print(f"Erro no insight 1 - economia_influencia_eleicao: {e}")


def insight_2_coligacoes_disputas_vitoria(dados_candidatos, dados_coligacoes):
    print("Insight 2: Coligações e Disputas de Vitória")
    try:
        # Verifica se a pasta de output existe
        os.makedirs("output", exist_ok=True)
        
        # Calcula o número de partidos em cada coligação
        dados_coligacoes['NUMERO_PARTIDOS'] = dados_coligacoes['DS_COMPOSICAO_FEDERACAO'].str.count(',') + 1
        
        # Filtra apenas os candidatos eleitos e agrupa por coligação
        coligacoes_eleitos = dados_candidatos[dados_candidatos['DS_SIT_TOT_TURNO'].str.lower() == 'eleito']
        coligacoes_resultados = coligacoes_eleitos.groupby(['SQ_COLIGACAO', 'SG_UF']).size().reset_index(name='NUM_ELEITOS')
        
        # Junta as informações de coligação com os resultados de eleição e UF
        coligacoes_detalhadas = dados_coligacoes.merge(coligacoes_resultados, on='SQ_COLIGACAO', how='left')
        coligacoes_detalhadas['NUM_ELEITOS'] = coligacoes_detalhadas['NUM_ELEITOS'].fillna(0)
        
        # Ordena pelos maiores valores de coligação e número de eleitos
        coligacoes_detalhadas = coligacoes_detalhadas.sort_values(by=['NUMERO_PARTIDOS', 'NUM_ELEITOS'], ascending=[False, False])
        
        # Salva o resultado em um CSV
        coligacoes_detalhadas.to_csv("output/coligacoes_detalhadas.csv", index=False)
        
        # Visualização com dispersão e tamanhos variáveis
        plt.figure(figsize=(14, 8))
        
        # Gráfico de dispersão com tamanho do ponto proporcional ao número de eleitos, com UF como hue
        scatter = sns.scatterplot(
            data=coligacoes_detalhadas, 
            x='NUMERO_PARTIDOS', 
            y='NUM_ELEITOS', 
            size='NUM_ELEITOS', 
            hue='SG_UF_x', 
            sizes=(40, 400),  # Define tamanho dos pontos
            alpha=0.7, 
            legend='full'
        )

        # Ajustes no gráfico
        plt.title("Coligações: Número de Eleitos por Número de Partidos e UF")
        plt.xlabel("Número de Partidos na Coligação")
        plt.ylabel("Número de Eleitos")
        scatter.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # Eixos em inteiros

        # Salvando as figuras
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


# Função para gerar o mapa das eleições
def insight_9_mapa_resultados_eleicao(dados_candidatos, dados_municipios):
    try:
        # Verifica se as colunas necessárias estão presentes em ambos os DataFrames
        required_columns_candidatos = {'NM_UE', 'SG_PARTIDO', 'SG_UF', 'DS_SIT_TOT_TURNO'}
        required_columns_municipios = {'MUNICIPIO', 'LATITUDE', 'LONGITUDE', 'SG_UF'}
        
        if not required_columns_candidatos.issubset(dados_candidatos.columns):
            print("Colunas disponíveis em dados_candidatos:", dados_candidatos.columns)
            raise ValueError(f"Colunas necessárias faltando em dados_candidatos: {required_columns_candidatos - set(dados_candidatos.columns)}")

        if not required_columns_municipios.issubset(dados_municipios.columns):
            print("Colunas disponíveis em dados_municipios:", dados_municipios.columns)
            raise ValueError(f"Colunas necessárias faltando em dados_municipios: {required_columns_municipios - set(dados_municipios.columns)}")

        # Filtra candidatos eleitos para cada município
        candidatos_eleitos = dados_candidatos[dados_candidatos['DS_SIT_TOT_TURNO'].str.lower() == 'eleito']

        # Mescla com dados dos municípios para obter as coordenadas
        resultados_municipios = pd.merge(
            candidatos_eleitos[['NM_UE', 'SG_PARTIDO', 'SG_UF']],
            dados_municipios[['MUNICIPIO', 'LATITUDE', 'LONGITUDE', 'SG_UF']],
            left_on=['NM_UE', 'SG_UF'],
            right_on=['MUNICIPIO', 'SG_UF']
        )

        # Gera uma lista única de partidos vencedores para atribuir cores
        partidos = resultados_municipios['SG_PARTIDO'].unique()
        
        # Cria o colormap e uma lista de cores baseada no número de partidos
        cmap = colormaps['tab20']
        colors = [cmap(i / len(partidos)) for i in range(len(partidos))]
        
        # Mapeia partidos a cores
        partido_cores = {partido: colors[i] for i, partido in enumerate(partidos)}

        # Inicializa o mapa centrado no Brasil
        mapa_brasil = folium.Map(location=[-15.7801, -47.9292], zoom_start=4)

        # Adiciona um marcador para cada município vencedor
        for _, row in resultados_municipios.iterrows():
            partido = row['SG_PARTIDO']
            color = partido_cores.get(partido, 'gray')  # 'gray' para partidos sem cor mapeada
            folium.CircleMarker(
                location=(row['LATITUDE'], row['LONGITUDE']),
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['MUNICIPIO']} ({partido})"
            ).add_to(mapa_brasil)

        # Salva o mapa como um arquivo HTML
        mapa_brasil.save("output/resultado_eleicoes_mapa.html")
        print("Mapa gerado e salvo como resultado_eleicoes_mapa_brasil.html")
        
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

    # Exemplo de estrutura do DataFrame de coordenadas dos municípios
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
    # insight_1_economia_influencia_eleicao(dados_candidatos, dados_bens)
    # print("Insight 1 processado com sucesso.")

    # Executa o insight 2 diretamente
    # insight_2_coligacoes_disputas_vitoria(dados_candidatos, dados_coligacoes)
    # print("Insight 2 processado com sucesso.")

    # Executa o insight 3 diretamente
    # insight_3_maior_partido_uf(dados_candidatos)
    # print("Insight 3 processado com sucesso.")

    # Executa o insight 4 diretamente
    # insight_4_tendencia_regional_partido(dados_candidatos)
    # print("Insight 4 processado com sucesso.")

    # Executa o insight 5 diretamente
    # insight_5_partido_dominante_cargo(dados_candidatos)
    # print("Insight 5 processado com sucesso.")

    # Executa o insight 6 diretamente
    # insight_6_candidatos_indigenas_quilombolas(dados_candidatos, dados_info_complementar)
    # print("Insight 6 processado com sucesso.")

    # Executa o insight 7 diretamente
    # insight_7_rede_social_preferida(dados_redes_sociais)
    # print("Insight 7 processado com sucesso.")

    # Executa o insight 8 diretamente
    # insight_8_termos_propostas(dados_candidatos)
    # print("Insight 8 processado com sucesso.")

    # Executa o insight 9 diretamente
    insight_9_mapa_resultados_eleicao(dados_candidatos, municipios_coordenadas)
    print("Insight 9 processado com sucesso.")

if __name__ == "__main__":
    main()