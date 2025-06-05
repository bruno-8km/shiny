from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import plotly.express as px

CSV_URL = "https://raw.githubusercontent.com/bruno-8km/shiny/refs/heads/master/points_per_health.csv"

COLORS = {
    "Água": "#007ED5", "Ar": "#58BDDC", "Nutrição": "#B2DA51",
    "Luz Solar": "#FBDC61", "Temperança": "#F8A754", "Exercício Físico": "#F26E52",
    "Descanso": "#E16BA8", "Confiança em Deus": "#AA63A7"
}

MAXIMOS = {
    "Água": 10, "Ar": 10, "Nutrição": 10,
    "Luz Solar": 10, "Temperança": 10,
    "Exercício Físico": 10, "Descanso": 10,
    "Confiança em Deus": 10
}

CLASSIFICACAO_IMC = [
    "Magreza grave (<16)",
    "Magreza moderada (16-16.9)",
    "Magreza leve (17-18.4)",
    "Peso normal (18.5-24.9)",
    "Sobrepeso (25-29.9)",
    "Obesidade grau I (30-34.9)",
    "Obesidade grau II (35-39.9)", 
    "Obesidade grau III (≥40)"
]

CLASSIFICACAO_PRESSAO = [
    "Hipotensão (PAS<90 ou PAD<60)",
    "Normal (PAS<120 e PAD<80)",
    "Elevada (PAS 120-129 e PAD<80)",
    "Hipertensão Estágio 1 (PAS 130-139 ou PAD 80-89)",
    "Hipertensão Estágio 2 (PAS≥140 ou PAD≥90)",
    "Crise Hipertensiva (PAS≥180 ou PAD≥120)"
]

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Filtros e Controles", class_="text-center"),
        ui.hr(),
        
        # Contador de gêneros com emojis
        ui.output_ui("contador_generos"),
        
        # Filtro de gênero com labels modificados
        ui.input_selectize(
            "filtro_genero",
            "Filtrar por Gênero:",
            choices={"M": "Masculino", "F": "Feminino", "N": "Não Informado"},
            selected=["M", "F", "N"],
            multiple=True
        ),
        
        ui.input_slider(
            "filtro_imc",
            "Classificação de IMC:",
            min=0,
            max=7,
            value=[0, 7],
            step=1,
            ticks=False
        ),
        ui.output_text_verbatim("texto_classificacao_imc"),
        
        ui.input_slider(
            "filtro_pressao",
            "Classificação de Pressão:",
            min=0,
            max=5,
            value=[0, 5],
            step=1,
            ticks=False
        ),
        ui.output_text_verbatim("texto_classificacao_pressao"),
        
        ui.hr(),
        
        ui.input_slider(
            "num_pessoas",
            "Número de Pessoas:",
            min=1,
            max=300,
            value=50,
            step=1
        ),
        
        ui.input_checkbox_group(
            "remedios_selecionados",
            "Filtro dos 8 Remédios Naturais:",
            choices=list(COLORS.keys()),
            selected=list(COLORS.keys())
        ),
        
        ui.input_radio_buttons(
            "tipo_eixo_x",
            "Controles do gráfico dos 8 Remédios:",
            {"pontos": "Número de Pontos",
             "porcentagem": "Porcentagem Total"}
        ),
        
        width=300,
        open="closed"
    ),
    
    # Sistema de abas alternativo
    ui.tags.div(
        ui.tags.ul(
            ui.tags.li(
                ui.tags.a("Gráfico dos 8 Remédios Naturais", href="#", onclick="showTab('grafico-principal')"),
                class_="active"
            ),
            ui.tags.li(
                ui.tags.a("Gráfico do IMC", href="#", onclick="showTab('grafico-imc')")
            ),
            ui.tags.li(
                ui.tags.a("Gráfico de Pressão", href="#", onclick="showTab('grafico-pressao')")
            ),
            ui.tags.li(
                ui.tags.a("Tabela de Dados do App", href="#", onclick="showTab('tabela-dados')")
            ),
            class_="nav nav-tabs"
        ),
        
        ui.tags.div(
            ui.tags.div(
                ui.output_ui("plotly_container"),
                id="grafico-principal",
                class_="tab-pane active"
            ),
            ui.tags.div(
                ui.output_ui("grafico_imc"),
                id="grafico-imc",
                class_="tab-pane"
            ),
            ui.tags.div(
                ui.output_ui("grafico_pressao"),
                id="grafico-pressao",
                class_="tab-pane"
            ),
            ui.tags.div(
                ui.output_data_frame("tabela_dados_saude"),
                id="tabela-dados",
                class_="tab-pane"
            ),
            class_="tab-content"
        ),
        
        ui.tags.script("""
            function showTab(tabId) {
                // Esconde todos os conteúdos
                document.querySelectorAll('.tab-pane').forEach(function(tab) {
                    tab.classList.remove('active');
                });
                
                // Remove a classe active de todas as abas
                document.querySelectorAll('.nav-tabs li').forEach(function(tab) {
                    tab.classList.remove('active');
                });
                
                // Mostra o conteúdo selecionado
                document.getElementById(tabId).classList.add('active');
                
                // Ativa a aba clicada
                event.currentTarget.parentElement.classList.add('active');
                
                // Previne o comportamento padrão do link
                return false;
            }
        """)
    )
)

def server(input, output, session):
    # Função para classificar o IMC
    def classificar_imc(imc):
        if imc < 16.0: return 0
        if 16.0 <= imc <= 16.9: return 1
        if 17.0 <= imc <= 18.4: return 2
        if 18.5 <= imc <= 24.9: return 3
        if 25.0 <= imc <= 29.9: return 4
        if 30.0 <= imc <= 34.9: return 5
        if 35.0 <= imc <= 39.9: return 6
        if imc >= 40.0: return 7
        return 3  # Default
    
    # Função para classificar a pressão arterial
    def classificar_pressao(pas, pad):
        if pas < 90 or pad < 60: return 0  # Hipotensão
        if pas >= 180 or pad >= 120: return 5  # Crise Hipertensiva
        if pas >= 140 or pad >= 90: return 4  # Hipertensão Estágio 2
        if (130 <= pas <= 139) or (80 <= pad <= 89): return 3  # Hipertensão Estágio 1
        if (120 <= pas <= 129) and (pad < 80): return 2  # Elevada
        if pas < 120 and pad < 80: return 1  # Normal
        return 1  # Default para Normal

    # Texto explicativo para o slider de IMC
    @output
    @render.text
    def texto_classificacao_imc():
        min_imc = input.filtro_imc()[0]
        max_imc = input.filtro_imc()[1]
        return f"IMC: {CLASSIFICACAO_IMC[min_imc]} a {CLASSIFICACAO_IMC[max_imc]}"

    # Texto explicativo para o slider de pressão
    @output
    @render.text
    def texto_classificacao_pressao():
        min_p = input.filtro_pressao()[0]
        max_p = input.filtro_pressao()[1]
        return f"Pressão: {CLASSIFICACAO_PRESSAO[min_p]} a {CLASSIFICACAO_PRESSAO[max_p]}"

    # Contador de gêneros com emojis - agora usa dados_filtrados() em vez de dados_completos()
    @output
    @render.ui
    def contador_generos():
        df = dados_filtrados()  # Alterado para usar dados_filtrados() que considera todos os filtros
        if df.empty:
            return ui.div("Nenhum dado para exibir com os filtros atuais", style="color: red;")
            
        counts = df['Gênero'].value_counts()
        
        # Contagem para cada gênero (usando get para evitar KeyError)
        m_count = counts.get('M', 0)
        f_count = counts.get('F', 0)
        n_count = counts.get('N', 0)
        
        return ui.tags.div(
            ui.tags.div(
                ui.tags.span("🚹", style="font-size: 1.5em; margin-right: 8px;"),
                ui.tags.span(f"Masculino: {m_count}"),
                style="display: flex; align-items: center; margin-bottom: 8px;"
            ),
            ui.tags.div(
                ui.tags.span("🚺", style="font-size: 1.5em; margin-right: 8px;"),
                ui.tags.span(f"Feminino: {f_count}"),
                style="display: flex; align-items: center; margin-bottom: 8px;"
            ),
            ui.tags.div(
                ui.tags.span("🔍", style="font-size: 1.5em; margin-right: 8px;"),
                ui.tags.span(f"Não Informado: {n_count}"),
                style="display: flex; align-items: center;"
            ),
            ui.tags.div(
                ui.tags.span("👥", style="font-size: 1.5em; margin-right: 8px;"),
                ui.tags.span(f"Total: {len(df)}"),
                style="display: flex; align-items: center; margin-top: 8px; font-weight: bold;"
            ),
            style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;"
        )

    @reactive.calc
    def dados_completos():
        try:
            df = pd.read_csv(CSV_URL)
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
            df['height'] = pd.to_numeric(df['height'], errors='coerce')
            df['IMC'] = df.apply(lambda row: row['weight'] / (row['height']**2) if row['height'] > 0 else np.nan, axis=1)
            df['IMC'] = df['IMC'].round(2)
            df['Classificacao_IMC'] = df['IMC'].apply(classificar_imc)
            df['PAS'] = np.random.randint(90, 141, size=len(df))
            df['PAD'] = np.random.randint(60, 91, size=len(df))
            df['Classificacao_Pressao'] = df.apply(lambda row: classificar_pressao(row['PAS'], row['PAD']), axis=1)
            
            df = df.rename(columns={
                'name': 'Nome', 'age': 'Idade', 'sex': 'Gênero',
                'weight': 'Peso', 'height': 'Altura', 'blood': 'Pressão',
                'water': 'Água', 'air': 'Ar', 'nutrition': 'Nutrição',
                'sun': 'Luz Solar', 'temperance': 'Temperança',
                'exercise': 'Exercício Físico', 'rest': 'Descanso',
                'trust': 'Confiança em Deus'
            })
            
            return df
        except Exception as e:
            print(f"Erro: {e}")
            return pd.DataFrame({"Erro": [str(e)]})

    @reactive.calc
    def dados_filtrados():
        df = dados_completos()
        
        # Filtro por gênero
        if input.filtro_genero():
            df = df[df['Gênero'].isin(input.filtro_genero())]
        else:
            return pd.DataFrame()
        
        # Filtro por IMC
        imc_min, imc_max = input.filtro_imc()
        df = df[(df['Classificacao_IMC'] >= imc_min) & (df['Classificacao_IMC'] <= imc_max)]
        
        # Filtro por pressão arterial
        pressao_min, pressao_max = input.filtro_pressao()
        df = df[(df['Classificacao_Pressao'] >= pressao_min) & (df['Classificacao_Pressao'] <= pressao_max)]
        
        # Amostragem - agora este filtro é considerado no contador de gêneros
        n = min(input.num_pessoas(), len(df))
        return df.sample(n) if n < len(df) else df

    @render.data_frame
    def tabela_dados_saude():
        df = dados_filtrados()
        if df.empty:
            return render.DataGrid(pd.DataFrame({"Mensagem": ["Nenhum dado para exibir com os filtros atuais"]}))
        return render.DataGrid(df)

    @render.ui
    def plotly_container():
        df = dados_filtrados()
        try:
            if df.empty:
                return ui.div("Nenhum dado para exibir com os filtros atuais", style="color: red;")
            
            remedios_ativos = input.remedios_selecionados()
            if not remedios_ativos:
                remedios_ativos = list(COLORS.keys())
            
            df_melted = pd.melt(df, 
                               value_vars=remedios_ativos, 
                               var_name="Remedio", 
                               value_name="Pontos")
            
            df_melted['Pontos'] = pd.to_numeric(df_melted['Pontos'], errors='coerce')
            
            if input.tipo_eixo_x() == "porcentagem":
                df_melted['Valor'] = df_melted.apply(
                    lambda row: (row['Pontos'] / MAXIMOS[row['Remedio']]) * 100, 
                    axis=1
                )
                x_label = "Porcentagem do Máximo (%)"
            else:
                df_melted['Valor'] = df_melted['Pontos']
                x_label = "Pontos dos 8 Remédios"
            
            df_counts = df_melted.groupby(['Remedio', 'Valor']).size().reset_index(name='NumPessoas')
            
            cores_filtradas = {k: v for k, v in COLORS.items() if k in remedios_ativos}
            
            fig = px.area(
                df_counts,
                x="Valor",
                y="NumPessoas",
                color="Remedio",
                title=f"Gráfico dos 8 Remédios Naturais ({len(df)} pessoas analisadas)",
                labels={"Valor": x_label, "NumPessoas": "Número de Pessoas", "Remedio": "8 Remédios Naturais"},
                color_discrete_map=cores_filtradas,
            )
            
            fig.update_layout(
                hovermode="x unified",
                xaxis_title=x_label,
                yaxis_title="Número de Pessoas",
                showlegend=True
            )
            
            if input.tipo_eixo_x() == "porcentagem":
                fig.update_xaxes(range=[0, 100])
            
            return ui.HTML(fig.to_html(full_html=False))
            
        except Exception as e:
            print(f"Erro no gráfico: {e}")
            return ui.p(f"Erro ao gerar gráfico: {e}")

    @render.ui
    def grafico_imc():
        df = dados_filtrados()
        try:
            if df.empty:
                return ui.div("Nenhum dado para exibir com os filtros atuais", style="color: red;")
            
            # Contagem por classificação de IMC
            df_counts = df['Classificacao_IMC'].value_counts().reset_index()
            df_counts.columns = ['Classificacao', 'NumPessoas']
            df_counts = df_counts.sort_values('Classificacao')
            
            # Mapear números para rótulos
            df_counts['Classificacao_Label'] = df_counts['Classificacao'].apply(lambda x: CLASSIFICACAO_IMC[x])
            
            fig = px.bar(
                df_counts,
                x="Classificacao_Label",
                y="NumPessoas",
                title=f"Gráfico do IMC ({len(df)} pessoas analisadas)",
                labels={"Classificacao_Label": "Classificação de IMC", "NumPessoas": "Número de Pessoas"},
                color="Classificacao_Label",
                color_discrete_sequence=px.colors.sequential.Blues
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Número de Pessoas",
                showlegend=False
            )
            
            return ui.HTML(fig.to_html(full_html=False))
            
        except Exception as e:
            print(f"Erro no gráfico de IMC: {e}")
            return ui.p(f"Erro ao gerar gráfico de IMC: {e}")

    @render.ui
    def grafico_pressao():
        df = dados_filtrados()
        try:
            if df.empty:
                return ui.div("Nenhum dado para exibir com os filtros atuais", style="color: red;")
            
            # Contagem por classificação de pressão
            df_counts = df['Classificacao_Pressao'].value_counts().reset_index()
            df_counts.columns = ['Classificacao', 'NumPessoas']
            df_counts = df_counts.sort_values('Classificacao')
            
            # Mapear números para rótulos
            df_counts['Classificacao_Label'] = df_counts['Classificacao'].apply(lambda x: CLASSIFICACAO_PRESSAO[x])
            
            fig = px.bar(
                df_counts,
                x="Classificacao_Label",
                y="NumPessoas",
                title=f"Distribuição de Pressão Arterial ({len(df)} pessoas analisadas)",
                labels={"Classificacao_Label": "Classificação de Pressão", "NumPessoas": "Número de Pessoas"},
                color="Classificacao_Label",
                color_discrete_sequence=px.colors.sequential.Reds
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Número de Pessoas",
                showlegend=False
            )
            
            return ui.HTML(fig.to_html(full_html=False))
            
        except Exception as e:
            print(f"Erro no gráfico de Pressão: {e}")
            return ui.p(f"Erro ao gerar gráfico de Pressão: {e}")

app = App(app_ui, server)
