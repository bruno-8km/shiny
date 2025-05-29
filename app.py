from shiny import ui, reactive, render, App
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# === Funções de classificação ===
def get_bp_classification_value(pas, pad):
    """Retorna um valor numérico para a classificação da pressão arterial."""
    if pas < 120 and pad < 80:
        return 0  # Ótima
    elif 120 <= pas <= 129 and pad < 80:
        return 1  # Normal
    elif 130 <= pas <= 139 or 80 <= pad <= 89:
        return 2  # Limítrofe (Pré-hipertensão)
    elif 140 <= pas <= 159 or 90 <= pad <= 99:
        return 3  # Hipertensão Estágio 1
    elif 160 <= pas <= 179 or 100 <= pad <= 109:
        return 4  # Hipertensão Estágio 2
    elif pas >= 180 or pad >= 110:
        return 5  # Crise Hipertensiva
    return 1 # Default para Normal se algo estranho acontecer

def get_bmi_classification_value(bmi):
    """Retorna um valor numérico para a classificação do IMC."""
    if bmi < 16.0: return 0
    if 16.0 <= bmi <= 16.9: return 1
    if 17.0 <= bmi <= 18.4: return 2
    if 18.5 <= bmi <= 24.9: return 3
    if 25.0 <= bmi <= 29.9: return 4
    if 30.0 <= bmi <= 34.9: return 5
    if 35.0 <= bmi <= 39.9: return 6
    if bmi >= 40.0: return 7
    return 3 # Default para Peso Normal

# === Carregamento e preparação do dataframe df ===
# Carrega dados de saúde personalizados
try:
    df = pd.read_csv("https://raw.githubusercontent.com/bruno-8km/meuprojeto/master/points_per_health.csv")
except Exception as e:
    print(f"Erro ao carregar o CSV: {e}")
    # Criar um DataFrame de exemplo em caso de falha no carregamento para teste
    data_exemplo = {
        'ID': range(1, 21),
        'Name': [f'Pessoa {i}' for i in range(1, 21)],
        'Age': np.random.randint(18, 70, 20),
        'Sex': np.random.choice(['M', 'F', 'N'], 20),
        'Weight': np.random.uniform(50, 120, 20).round(1),
        'Height': np.random.randint(150, 200, 20),
        'Water': np.random.randint(0, 10, 20),
        'Air': np.random.randint(0, 10, 20),
        'Nutrition': np.random.randint(0, 10, 20),
        'Sun': np.random.randint(0, 10, 20),
        'Temperance': np.random.randint(0, 10, 20),
        'Exercise': np.random.randint(0, 10, 20),
        'Rest': np.random.randint(0, 10, 20),
        'Trust': np.random.randint(0, 10, 20),
    }
    df = pd.DataFrame(data_exemplo)
    print("Usando DataFrame de exemplo.")


# --- ADIÇÃO DAS COLUNAS FALTANTES ---
# 1. Calcular IMC: Peso (kg) / (Altura (m))^2
# Certifique-se que as colunas 'Weight' e 'Height' existem no seu CSV
if 'Weight' in df.columns and 'Height' in df.columns:
    df['IMC'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['IMC'] = df['IMC'].round(2) # Arredondar para 2 casas decimais
else:
    print("Aviso: Colunas 'Weight' ou 'Height' não encontradas. IMC não pôde ser calculado.")
    df['IMC'] = np.nan # Adiciona a coluna com NaN se não puder calcular

# 2. "Inventar" PAS e PAD (Pressão Arterial Sistólica e Diastólica)
# Vamos gerar valores aleatórios dentro de um intervalo plausível
np.random.seed(42) # Para reprodutibilidade
num_rows = len(df)
df['PAS'] = np.random.randint(100, 180, size=num_rows)
df['PAD'] = np.random.randint(60, 110, size=num_rows)

# Certifique-se que PAD seja geralmente menor que PAS
for index, row in df.iterrows():
    if row['PAD'] >= row['PAS']:
        df.loc[index, 'PAD'] = row['PAS'] - np.random.randint(10, 40)
        if df.loc[index, 'PAD'] < 50: # Evitar PAD muito baixo
             df.loc[index, 'PAD'] = 50


# === Dicionários e configurações ===
colors = {
    "Water": "#007ED5", "Air": "#58BDDC", "Nutrition": "#B2DA51",
    "Sun": "#FBDC61", "Temperance": "#F8A754", "Exercise": "#F26E52",
    "Rest": "#E16BA8", "Trust": "#AA63A7",
}
health_factors_en = list(colors.keys())
factor_display_names = {
    "Water": "Água", "Air": "Ar Puro", "Nutrition": "Nutrição", "Sun": "Luz Solar",
    "Temperance": "Temperança", "Exercise": "Exercício", "Rest": "Repouso", "Trust": "Confiança em Deus",
}
column_name_translations = {
    "ID": "ID", "Name": "Nome", "Age": "Idade", "Sex": "Sexo",
    "PAS": "PAS (mmHg)", "PAD": "PAD (mmHg)", "Weight": "Peso (kg)", "Height": "Altura (cm)",
    # "Blood": "Pressão Sanguínea", # Esta coluna não existe, usamos PAS e PAD
    "IMC": "IMC (kg/m²)",
    **factor_display_names
}

# BP Slider labels (usando os retornos da função get_bp_classification_value)
bp_slider_marks_dict = {
    0: "Ótima (<120/<80)",
    1: "Normal (120-129/<80)",
    2: "Limítrofe (130-139/80-89)",
    3: "Hipert. Estágio 1 (140-159/90-99)",
    4: "Hipert. Estágio 2 (160-179/100-109)",
    5: "Crise Hipert. (>=180/>=110)"
}
bp_slider_min_val = 0
bp_slider_max_val = len(bp_slider_marks_dict) - 1 # 5
bp_slider_default_selected = 1

bmi_slider_marks_dict_full = {
    0: "Magreza grave (< 16.0)",
    1: "Magreza moderada (16.0 – 16.9)",
    2: "Magreza leve (17.0 – 18.4)",
    3: "Peso normal (18.5 – 24.9)",
    4: "Sobrepeso (25.0 – 29.9)",
    5: "Obesidade grau I (30.0 – 34.9)",
    6: "Obesidade grau II (35.0 – 39.9)",
    7: "Obesidade grau III (>= 40.0)"
}
bmi_slider_min_val = 0
bmi_slider_max_val = len(bmi_slider_marks_dict_full) - 1
bmi_slider_default_selected = 3

# === UI ===
sidebar = ui.sidebar(
    ui.input_radio_buttons("x_mode", "Eixo X:", {"score": "Pontuação", "percent": "Porcentagem"}, selected="score"),
    ui.input_slider("n_pessoas", "Número de pessoas a incluir:", min=1, max=len(df), value=len(df), step=1),
    ui.input_selectize(
        "sex_select", "Gênero:",
        choices={"All": "Todos", "M": "Masculino", "F": "Feminino", "N": "Não Informado"},
        multiple=True, selected=["All"] # Alterado para lista para corresponder à lógica de seleção
    ),
    ui.tags.hr(),
    ui.input_switch("apply_bp_filter", "Filtrar por Pressão Arterial", False),
    ui.panel_conditional("input.apply_bp_filter",
        ui.input_slider(
            "bp_level_slider",
            "Nível de Pressão Arterial:",
            min=bp_slider_min_val,
            max=bp_slider_max_val,
            value=bp_slider_default_selected,
            step=1,
            # Adicionando marks para melhor visualização se o Shiny suportar diretamente
            # ou você pode adicionar um ui.output_text para mostrar o label correspondente
        ),
        # Para exibir o label correspondente ao valor do slider de PA:
        ui.output_ui("bp_level_label_ui")
    ),
    ui.tags.hr(),
    ui.input_switch("apply_bmi_filter", "Filtrar por IMC", False),
    ui.panel_conditional("input.apply_bmi_filter",
        ui.input_slider(
            "bmi_level_slider",
            "Nível de IMC:",
            min=bmi_slider_min_val,
            max=bmi_slider_max_val,
            value=bmi_slider_default_selected,
            step=1,
        ),
        # Para exibir o label correspondente ao valor do slider de IMC:
        ui.output_ui("bmi_level_label_ui")
    ),
    open="desktop"
)

app_ui = ui.page_sidebar(
    sidebar,
    ui.navset_pill(
        ui.nav_panel("Home",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Controle dos Remédios Naturais"),
                    ui.tags.h5("Os 8 Remédios Naturais"),
                    ui.input_checkbox("chk_all_factors", "Selecionar/Deselecionar Todos", True), # Checkbox "Selecionar Todos"
                    ui.tags.hr(),
                    ui.tags.style("""
                        .checkbox-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2px 10px; }
                        .checkbox-grid .form-check { margin-bottom: 0.1rem; padding-left: 1.5em; }
                        .checkbox-grid .form-check-input { margin-top: 0.2em; }
                    """),
                    ui.tags.div(
                        *[
                            ui.input_checkbox(f"chk_{factor_en}", factor_display_names[factor_en], True)
                            for factor_en in health_factors_en
                        ],
                        class_="checkbox-grid"
                    )
                ),
                ui.card(
                    ui.card_header("Distribuição por Remédio Natural"),
                    ui.output_plot("area_plot")
                ),
                ui.card(
                    ui.card_header("Tabela dos Dados de Remédios Naturais"),
                    ui.output_data_frame("table")
                ),
                col_widths=[4, 8, 12] # Ajuste de layout para melhor visualização
            )
        ),
        ui.nav_panel("Gráficos",
            ui.layout_columns( # Usando layout_columns para colocar lado a lado se desejado
                ui.card(
                    ui.card_header("Distribuição do IMC"),
                    ui.output_plot("bmi_plot")
                ),
                ui.card(
                    ui.card_header("Distribuição da Pressão Arterial"),
                    ui.output_plot("bp_plot")
                ),
             col_widths=[6,6] # Cada card ocupa metade da largura
            )
        )
    )
)

# === SERVER ===
def server(input, output, session):

    @reactive.Effect
    @reactive.event(input.chk_all_factors)
    def _sync_health_factors_action():
        is_checked = input.chk_all_factors()
        for factor_en in health_factors_en:
            ui.update_checkbox(session, f"chk_{factor_en}", value=is_checked) # Adicionado `session`

    @reactive.Effect
    def _update_chk_all_factors_status():
        all_selected = all(input[f"chk_{factor_en}"]() for factor_en in health_factors_en)
        # Apenas atualiza se o estado do "chk_all_factors" for diferente do estado real
        # Isso evita loops reativos.
        current_all_factors_value = input.chk_all_factors()
        if current_all_factors_value is not None and current_all_factors_value != all_selected:
             ui.update_checkbox(session, "chk_all_factors", value=all_selected) # Adicionado `session`

    @reactive.Calc
    def selected_factors_calc():
        return [factor_en for factor_en in health_factors_en if input[f"chk_{factor_en}"]()]

    @reactive.Calc
    def filtered_df():
        # Começa com uma cópia do df global que já tem IMC, PAS, PAD
        df_copy = df.copy()
        
        # Aplicar filtro de número de pessoas
        df_filtered = df_copy.head(input.n_pessoas())

        # Aplicar filtro de sexo
        selected_sex_options = input.sex_select()
        # Verifica se selected_sex_options não é None e não contém "All"
        if selected_sex_options and "All" not in selected_sex_options:
            df_filtered = df_filtered[df_filtered['Sex'].isin(selected_sex_options)]

        # Aplicar filtro de Pressão Arterial
        if input.apply_bp_filter():
            selected_bp_level = input.bp_level_slider()
            # Certifique-se de que PAS e PAD existam antes de aplicar o filtro
            if 'PAS' in df_filtered.columns and 'PAD' in df_filtered.columns:
                df_filtered = df_filtered[
                    df_filtered.apply(lambda row: get_bp_classification_value(row['PAS'], row['PAD']) == selected_bp_level, axis=1)
                ]

        # Aplicar filtro de IMC
        if input.apply_bmi_filter():
            selected_bmi_level = input.bmi_level_slider()
            # Certifique-se de que IMC exista antes de aplicar o filtro
            if 'IMC' in df_filtered.columns:
                 # Lidar com NaNs no IMC, caso contrário a comparação falha
                df_filtered_no_nan_imc = df_filtered.dropna(subset=['IMC'])
                df_filtered_nan_imc = df_filtered[df_filtered['IMC'].isna()]

                df_filtered_no_nan_imc = df_filtered_no_nan_imc[
                    df_filtered_no_nan_imc['IMC'].apply(lambda v: get_bmi_classification_value(v) == selected_bmi_level)
                ]
                df_filtered = pd.concat([df_filtered_no_nan_imc, df_filtered_nan_imc])


        return df_filtered

    @output
    @render.plot()
    def area_plot():
        current_plot_df = filtered_df()
        selected_factors_en = selected_factors_calc()
        x_mode = input.x_mode()
        fig, ax = plt.subplots(figsize=(10, 6))

        if current_plot_df.empty or not selected_factors_en:
            msg = "Nenhum dado disponível para a seleção atual."
            ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([]) # Limpa ticks do eixo x
            ax.set_yticks([]) # Limpa ticks do eixo y
            return fig

        max_score_overall = 0
        # Encontrar a pontuação máxima entre todos os fatores selecionados para o modo porcentagem
        if x_mode == "percent":
            for factor_en in selected_factors_en:
                if factor_en in current_plot_df.columns:
                    max_val = current_plot_df[factor_en].max()
                    if pd.notna(max_val) and max_val > max_score_overall:
                        max_score_overall = max_val
            if max_score_overall == 0: # Evitar divisão por zero
                max_score_overall = 10 # Default para 10 se não houver dados ou todos forem 0


        for factor_en in selected_factors_en:
            if factor_en not in current_plot_df.columns: continue
            
            # Remove NaNs para value_counts e interpolação
            factor_data = current_plot_df[factor_en].dropna()
            if factor_data.empty: continue

            counts = factor_data.value_counts().sort_index()
            if counts.empty: continue
            
            xs_raw = np.array(counts.index)
            ys_raw = counts.values

            if len(xs_raw) < 2: # Não é possível interpolar com menos de 2 pontos
                xs_smooth, ys_smooth = xs_raw, ys_raw
            else:
                k_spline = min(3, len(xs_raw) - 1)
                if k_spline < 1: k_spline = 1 # k deve ser pelo menos 1

                try:
                    xs_smooth = np.linspace(xs_raw.min(), xs_raw.max(), 300)
                    spline = make_interp_spline(xs_raw, ys_raw, k=k_spline)
                    ys_smooth = spline(xs_smooth)
                except Exception as e:
                    # print(f"Erro na interpolação para {factor_en}: {e}. Usando dados brutos.")
                    xs_smooth, ys_smooth = xs_raw, ys_raw
            
            label_display = factor_display_names.get(factor_en, factor_en)

            if x_mode == "score":
                ax.set_xlabel("Pontuação")
                ax.plot(xs_smooth, np.maximum(0, ys_smooth), color=colors[factor_en], label=label_display)
                ax.fill_between(xs_smooth, np.maximum(0, ys_smooth), alpha=0.3, color=colors[factor_en])
            else: # x_mode == "percent"
                # Normaliza xs_smooth pela pontuação máxima *geral* encontrada
                xs_percent = (xs_smooth / max_score_overall) * 100
                ax.plot(xs_percent, np.maximum(0, ys_smooth), color=colors[factor_en], label=label_display)
                ax.fill_between(xs_percent, np.maximum(0, ys_smooth), alpha=0.3, color=colors[factor_en])
                ax.set_xlabel("Porcentagem da pontuação máxima (%)")
                ax.set_xlim(0, 100)


        ax.set_ylabel("Número de Pessoas")
        ax.set_title("Distribuição por Remédio Natural")
        ax.legend()
        ax.grid(True)
        return fig

    @output
    @render.data_frame
    def table():
        df_current = filtered_df().copy()
        # Traduz apenas as colunas que existem no DataFrame
        existing_cols = df_current.columns
        translated_cols = [column_name_translations.get(col, col) for col in existing_cols]
        df_current.columns = translated_cols
        return render.DataGrid(df_current, selection_mode="none") # Desabilitar seleção na tabela

    @output
    @render.plot()
    def bmi_plot():
        # Usa filtered_df() para refletir as seleções do usuário
        current_plot_df = filtered_df()
        fig, ax = plt.subplots(figsize=(10, 6))

        if 'IMC' not in current_plot_df.columns or current_plot_df['IMC'].dropna().empty:
            msg = "Nenhum dado de IMC disponível para a seleção atual."
            ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Remove NaNs antes de contar e plotar
        imc_data = current_plot_df['IMC'].dropna()
        if imc_data.empty:
            msg = "Nenhum dado de IMC válido para a seleção atual."
            ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        imc_counts = imc_data.value_counts().sort_index()
        
        # Para um gráfico de barras das categorias de IMC:
        imc_categories_values = imc_data.apply(get_bmi_classification_value)
        category_counts = pd.Series(imc_categories_values).value_counts().sort_index()
        
        # Mapear os índices numéricos para os labels de texto para o eixo x
        category_labels = [bmi_slider_marks_dict_full.get(i, str(i)) for i in category_counts.index]

        ax.bar(category_labels, category_counts.values, color='skyblue', alpha=0.7, width=0.8)
        ax.set_xlabel('Categoria de IMC')
        ax.set_ylabel('Número de Pessoas')
        ax.set_title('Distribuição das Categorias de IMC')
        plt.xticks(rotation=45, ha="right") # Rotaciona os labels para melhor visualização
        plt.tight_layout() # Ajusta o layout para não cortar os labels
        ax.grid(True, axis='y')
        return fig

    @output
    @render.plot()
    def bp_plot():
        # Usa filtered_df() para refletir as seleções do usuário
        current_plot_df = filtered_df()
        fig, ax = plt.subplots(figsize=(10, 6))

        if not ('PAS' in current_plot_df.columns and 'PAD' in current_plot_df.columns) or \
           current_plot_df[['PAS', 'PAD']].dropna().empty:
            msg = "Nenhum dado de Pressão Arterial disponível para a seleção atual."
            ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        # Remove linhas onde PAS ou PAD são NaN antes de aplicar a classificação
        bp_data = current_plot_df[['PAS', 'PAD']].dropna()
        if bp_data.empty:
            msg = "Nenhum dado de Pressão Arterial válido para a seleção atual."
            ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            return fig

        bp_categories = bp_data.apply(lambda row: get_bp_classification_value(row['PAS'], row['PAD']), axis=1)
        bp_counts = pd.Series(bp_categories).value_counts().sort_index()
        
        # Mapear os índices numéricos para os labels de texto para o eixo x
        category_labels = [bp_slider_marks_dict.get(i, str(i)) for i in bp_counts.index]

        ax.bar(category_labels, bp_counts.values, color='salmon', alpha=0.7, width=0.8)
        ax.set_xlabel('Classificação da Pressão Arterial')
        ax.set_ylabel('Número de Pessoas')
        ax.set_title('Distribuição da Pressão Arterial')
        plt.xticks(rotation=30, ha="right") # Rotaciona os labels para melhor visualização
        plt.tight_layout() # Ajusta o layout para não cortar os labels
        ax.grid(True, axis='y')
        return fig

    # UI Outputs para labels dos sliders
    @output
    @render.ui
    def bp_level_label_ui():
        selected_bp_value = input.bp_level_slider()
        label = bp_slider_marks_dict.get(selected_bp_value, "Desconhecido")
        return ui.tags.div(
            ui.tags.strong("Categoria Selecionada: "),
            label,
            style="margin-top: 5px; font-size: 0.9em;"
        )

    @output
    @render.ui
    def bmi_level_label_ui():
        selected_bmi_value = input.bmi_level_slider()
        label = bmi_slider_marks_dict_full.get(selected_bmi_value, "Desconhecido")
        return ui.tags.div(
            ui.tags.strong("Categoria Selecionada: "),
            label,
            style="margin-top: 5px; font-size: 0.9em;"
        )

# === INICIALIZAÇÃO ===
app = App(app_ui, server)
