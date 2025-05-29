from shiny import reactive, render, req
from shiny.express import input, ui
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


# Função de Classificação da Pressão Arterial
def get_bp_classification_value(pas, pad):
    """
    Classifica a pressão arterial com base nos valores de PAS (Pressão Arterial Sistólica)
    e PAD (Pressão Arterial Diastólica) de acordo com as categorias padrão.
    Retorna um valor numérico de 0 a 5, onde:
    0: Hipotensão
    1: Normal
    2: Elevada
    3: Hipertensão Estágio 1
    4: Hipertensão Estágio 2
    5: Crise Hipertensiva
    """
    if pas < 90 or pad < 60: return 0  # Hipotensão
    if pas >= 180 or pad >= 120: return 5  # Crise Hipertensiva
    if pas >= 140 or pad >= 90: return 4  # Hipertensão Estágio 2
    if (130 <= pas <= 139) or (80 <= pad <= 89): return 3  # Hipertensão Estágio 1
    if (120 <= pas <= 129) and (pad < 80): return 2  # Elevada
    if pas < 120 and pad < 80: return 1  # Normal
    return 1  # Default para Normal


# Função de Classificação do IMC
def get_bmi_classification_value(imc):
    """
    Classifica o IMC (Índice de Massa Corporal) de acordo com a tabela da OMS para adultos.
    Retorna um valor numérico de 0 a 7, onde:
    0: Magreza grave
    1: Magreza moderada
    2: Magreza leve
    3: Peso normal
    4: Sobrepeso
    5: Obesidade grau I
    6: Obesidade grau II
    7: Obesidade grau III
    """
    if imc < 16.0: return 0
    if 16.0 <= imc <= 16.9: return 1
    if 17.0 <= imc <= 18.4: return 2
    if 18.5 <= imc <= 24.9: return 3
    if 25.0 <= imc <= 29.9: return 4
    if 30.0 <= imc <= 34.9: return 5
    if 35.0 <= imc <= 39.9: return 6
    if imc >= 40.0: return 7
    return 3  # Default para Peso normal


# Carrega dados de saúde personalizados
#df = pd.read_csv("https://raw.githubusercontent.com/bruno-8km/meuprojeto/master/points_per_health.csv")
df = pd.read_csv("https://raw.githubusercontent.com/bruno-8km/shiny/refs/heads/master/points_per_health.csv")

# Adiciona colunas de exemplo 'PAS', 'PAD', e 'IMC'
if 'PAS' not in df.columns or 'PAD' not in df.columns:
    num_rows = len(df)
    np.random.seed(42)
    df['PAS'] = np.random.normal(loc=125, scale=20, size=num_rows).astype(int)
    df['PAD'] = np.random.normal(loc=75, scale=15, size=num_rows).astype(int)
    df['PAS'] = np.clip(df['PAS'], 70, 220)
    df['PAD'] = np.clip(df['PAD'], 40, 140)
    mask_hipo = df.index < int(num_rows * 0.05)
    df.loc[mask_hipo, 'PAS'] = np.random.randint(70, 90, size=mask_hipo.sum())
    df.loc[mask_hipo, 'PAD'] = np.random.randint(40, 60, size=mask_hipo.sum())
    mask_crise_start = int(num_rows * 0.95)
    mask_crise = df.index >= mask_crise_start
    df.loc[mask_crise, 'PAS'] = np.random.randint(180, 220, size=mask_crise.sum())
    df.loc[mask_crise, 'PAD'] = np.random.randint(120, 140, size=mask_crise.sum())

    # Injetar dados para garantir representação em categorias extremas de IMC
    # Magreza grave (cerca de 2% dos dados)
    mask_magreza_grave = df.index < int(num_rows * 0.02)
    df.loc[mask_magreza_grave, 'Weight'] = np.random.randint(40, 50, size=mask_magreza_grave.sum())  # Baixo peso
    df.loc[mask_magreza_grave, 'Height'] = np.random.randint(160, 180, size=mask_magreza_grave.sum())  # Altura normal
    # Obesidade grau III (cerca de 2% dos dados)
    mask_obesidade_grau3 = df.index >= int(num_rows * 0.98)
    df.loc[mask_obesidade_grau3, 'Weight'] = np.random.randint(100, 150, size=mask_obesidade_grau3.sum())  # Alto peso
    df.loc[mask_obesidade_grau3, 'Height'] = np.random.randint(150, 170,
                                                               size=mask_obesidade_grau3.sum())  # Altura normal/baixa para simular alto IMC

if 'IMC' not in df.columns:
    # Calcula IMC: Peso (kg) / (Altura (m))^2
    df['IMC'] = df['Weight'] / ((df['Height']) ** 2)
    df['IMC'] = df['IMC'].round(1)  # Arredonda para uma casa decimal

# Dicionário de cores para os fatores de saúde
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

# Mapeamento de nomes de colunas para exibição na tabela
column_name_translations = {
    "ID": "ID", "Name": "Nome", "Age": "Idade", "Sex": "Sexo",
    "PAS": "PAS (mmHg)", "PAD": "PAD (mmHg)", "Weight": "Peso (kg)", "Height": "Altura (cm)",
    "Blood": "Pressão Sanguínea", "IMC": "IMC (kg/m²)",
    **factor_display_names
}

# Definições para o slider de Pressão Arterial
bp_slider_min_val = 0
bp_slider_max_val = 5
bp_slider_default_selected = 1

# Definições para o slider de IMC
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
bmi_slider_default_selected = 3  # Padrão para "Peso normal"

# Título do aplicativo e configurações da página
ui.page_opts(title="Distribuição dos Remédios Naturais", fillable=True)

# Barra lateral com controles de filtro
with ui.sidebar(open="desktop"):
    ui.input_radio_buttons("x_mode", "Eixo X:", {"score": "Pontuação", "percent": "Porcentagem"}, selected="score")
    ui.input_slider("n_pessoas", "Número de pessoas a incluir:", min=1, max=len(df), value=len(df), step=1)
    ui.input_selectize(
        "sex_select", "Gênero:",
        choices={"All": "Todos", "M": "Masculino", "F": "Feminino", "N": "Não Informado"},
        multiple=True, selected="All"
    )
    ui.tags.hr()

    # Filtro por Pressão Arterial
    ui.input_switch("apply_bp_filter", "Índice de Pressão Arterial", False)
    with ui.panel_conditional("input.apply_bp_filter"):
        ui.input_slider(
            "bp_level_slider",
            "Nível de Pressão Arterial:",
            min=bp_slider_min_val,
            max=bp_slider_max_val,
            value=bp_slider_default_selected,
            step=1,
        )


        @render.text
        def selected_bp_category_label():
            bp_slider_marks_dict_full_bp = {
                0: "Hipotensão (<90 ou <60 mmHg)", 1: "Normal (<120 e <80 mmHg)",
                2: "Elevada (120-129 e <80 mmHg)", 3: "Hipert. Est.1 (130-139 ou 80-89 mmHg)",
                4: "Hipert. Est.2 (≥140 ou ≥90 mmHg)", 5: "Crise Hipert. (≥180 e/ou ≥120 mmHg)"
            }
            if input.apply_bp_filter():
                return f"Selecionado: {bp_slider_marks_dict_full_bp.get(input.bp_level_slider(), 'N/A')}"
            return ""

    ui.tags.hr()

    # Novo Filtro por IMC
    ui.input_switch("apply_bmi_filter", "Índice do IMC", False)
    with ui.panel_conditional("input.apply_bmi_filter"):
        ui.input_slider(
            "bmi_level_slider",
            "Nível de IMC:",
            min=bmi_slider_min_val,
            max=bmi_slider_max_val,
            value=bmi_slider_default_selected,
            step=1,
        )


        @render.text
        def selected_bmi_category_label():
            if input.apply_bmi_filter():
                return f"Selecionado: {bmi_slider_marks_dict_full.get(input.bmi_level_slider(), 'N/A')}"
            return ""


@reactive.Effect
@reactive.event(input.chk_all_factors)
def _sync_health_factors_action():
    is_checked = input.chk_all_factors()
    for factor_en in health_factors_en: ui.update_checkbox(f"chk_{factor_en}", value=is_checked)


@reactive.Effect
def _update_chk_all_factors_status():
    all_selected = all(input[f"chk_{factor_en}"]() for factor_en in health_factors_en)
    if input.chk_all_factors() != all_selected: ui.update_checkbox("chk_all_factors", value=all_selected)


@reactive.Calc
def filtered_df():
    df_copy = df.copy()

    # Filtro por N Pessoas
    df_filtered = df_copy.head(input.n_pessoas())

    # Filtro por Sexo
    if 'Sex' in df_filtered.columns:
        selected_sex_options = input.sex_select()
        if not selected_sex_options:
            df_filtered = df_filtered.iloc[0:0]
        elif "All" in selected_sex_options:
            pass  # Não filtra
        else:
            actual_sexes_to_filter = [s for s in selected_sex_options if s in ["M", "F", "N"]]
            if not actual_sexes_to_filter:
                df_filtered = df_filtered.iloc[0:0]
            else:
                df_filtered = df_filtered[df_filtered['Sex'].isin(actual_sexes_to_filter)]

    # Filtro por Pressão Arterial (se ativo)
    if input.apply_bp_filter() and 'PAS' in df_filtered.columns and 'PAD' in df_filtered.columns:
        selected_bp_level_val = input.bp_level_slider()
        bp_categories = df_filtered.apply(
            lambda row: get_bp_classification_value(row['PAS'], row['PAD']), axis=1
        )
        df_filtered = df_filtered[bp_categories == selected_bp_level_val]

    # Filtro por IMC (se ativo)
    if input.apply_bmi_filter() and 'IMC' in df_filtered.columns:
        selected_bmi_level_val = input.bmi_level_slider()
        bmi_categories = df_filtered.apply(
            lambda row: get_bmi_classification_value(row['IMC']), axis=1
        )
        df_filtered = df_filtered[bmi_categories == selected_bmi_level_val]

    return df_filtered


ui.page_opts(fillable=True)

with ui.navset_pill(id="tab"):
    with ui.nav_panel("Home"):
        ""

        with ui.layout_columns(col_widths=[12, 12, 12]):
            with ui.card(full_screen=False):
                ui.card_header("Controle dos Remédios Naturais")
                ui.tags.h5("Os 8 Remédios Naturais")
                # ui.input_checkbox("chk_all_factors", "Selecionar Todos/Nenhum", True)
                ui.tags.hr()
                ui.tags.style("""
                    .checkbox-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2px 10px; }
                    .checkbox-grid .form-check { margin-bottom: 0.1rem; padding-left: 1.5em; }
                    .checkbox-grid .form-check-input { margin-top: 0.2em; }
                """)
                with ui.tags.div(class_="checkbox-grid"):
                    for factor_en in health_factors_en:
                        ui.input_checkbox(f"chk_{factor_en}", factor_display_names[factor_en], True)

            with ui.card(full_screen=True):
                ui.card_header("Distribuição por Remédio Natural")


                @reactive.Calc
                def selected_factors_calc():
                    return [factor_en for factor_en in health_factors_en if input[f"chk_{factor_en}"]()]


                @render.plot()
                def area_plot():
                    current_plot_df = filtered_df()
                    selected_factors_en = selected_factors_calc()
                    x_mode = input.x_mode()
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if current_plot_df.empty or not selected_factors_en:
                        msg = "Nenhum dado disponível para a seleção atual."
                        if not selected_factors_en and not current_plot_df.empty:
                            msg = "Nenhum remédio natural selecionado."
                        elif current_plot_df.empty and selected_factors_en:
                            msg = "Nenhum dado de pessoas para os filtros e remédios selecionados."
                        ax.text(0.5, 0.5, msg, horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=12, color='gray')
                        ax.set_xticks([]);
                        ax.set_yticks([])
                        ax.set_ylabel("Número de Pessoas por Ponto")
                        title_text_val = 'Pontuação de cada Remédio' if x_mode == 'score' else 'Porcentagem da Pontuação Máxima'
                        ax.set_title(title_text_val)
                        if x_mode == "score":
                            ax.set_xlabel("Pontuação de cada Remédio")
                        else:
                            ax.set_xlabel("Porcentagem da pontuação máxima (%)")
                            ax.set_xlim(0, 100)
                        return fig

                    for factor_en in selected_factors_en:
                        if factor_en not in current_plot_df.columns: continue
                        counts = current_plot_df[factor_en].value_counts().sort_index()
                        if counts.empty: continue
                        max_score = counts.index.max()
                        xs_raw = np.array(counts.index);
                        ys_raw = counts.values
                        if len(xs_raw) < 4:
                            k_spline = len(xs_raw) - 1 if len(xs_raw) > 1 else 1
                        else:
                            k_spline = 3
                        valid_spline = k_spline > 0 and len(xs_raw) > 1

                        if x_mode == "score":
                            xs = xs_raw
                            if valid_spline:
                                xs_smooth = np.linspace(xs.min(), xs.max(), 300)
                                try:
                                    spline = make_interp_spline(xs, ys_raw, k=k_spline);
                                    ys_smooth = spline(xs_smooth)
                                except ValueError:
                                    xs_smooth, ys_smooth = xs, ys_raw
                            else:
                                xs_smooth, ys_smooth = xs, ys_raw
                            ax.fill_between(xs_smooth, np.maximum(0, ys_smooth), alpha=0.3, color=colors[factor_en],
                                            label=factor_display_names[factor_en])
                            ax.plot(xs_smooth, np.maximum(0, ys_smooth), color=colors[factor_en])
                            ax.set_xlabel("Pontuação de cada Remédio")
                        else:  # x_mode == "percent"
                            if max_score == 0: max_score = 1
                            xs = (xs_raw / max_score) * 100
                            unique_xs = np.unique(xs)
                            valid_spline_percent = valid_spline and len(unique_xs) >= 2
                            if valid_spline_percent:
                                xs_smooth = np.linspace(unique_xs.min(), unique_xs.max(), 300)
                                try:
                                    k_actual_spline = k_spline
                                    if len(unique_xs) <= k_spline: k_actual_spline = len(unique_xs) - 1 if len(
                                        unique_xs) > 1 else 1
                                    if k_actual_spline > 0:
                                        temp_df_for_spline = pd.DataFrame({'x': xs, 'y': ys_raw}).groupby('x')[
                                            'y'].mean().reset_index()
                                        spline = make_interp_spline(temp_df_for_spline['x'], temp_df_for_spline['y'],
                                                                    k=k_actual_spline)
                                        ys_smooth = spline(xs_smooth)
                                    else:
                                        xs_smooth, ys_smooth = xs, ys_raw
                                except ValueError:
                                    xs_smooth, ys_smooth = xs, ys_raw
                            else:
                                xs_smooth, ys_smooth = xs, ys_raw
                            ax.plot(xs_smooth, np.maximum(0, ys_smooth), color=colors[factor_en],
                                    label=factor_display_names[factor_en])
                            ax.set_xlabel("Porcentagem da pontuação máxima (%)");
                            ax.set_xlim(0, 100)
                        ax.set_ylabel("Número de Pessoas por Ponto")
                        title_text_val = 'Distribuição por Pontuação' if x_mode == 'score' else 'Distribuição por Porcentagem da Pontuação Máxima'
                        ax.set_title(title_text_val)
                        if any(ax.get_lines()): ax.legend(title="Remédios")
                        ax.grid(True, linestyle="--", alpha=0.5);
                        ax.set_ylim(bottom=0)
                        return fig

        with ui.card(full_screen=True):
            ui.card_header("Tabela dos Dados de Remédios Naturais")


            @render.data_frame
            def table():
                current_display_df = filtered_df()
                df_renamed = current_display_df.copy()
                df_renamed.columns = [column_name_translations.get(col, col) for col in df_renamed.columns]
                print({col: len(df_renamed[col]) for col in df_renamed.columns})
                return render.DataGrid(df_renamed)

    with ui.nav_panel("Gráficos"):
        # Gráfico IMC (Área Empilhada)
        with ui.card(full_screen=True):
            ui.card_header("Distribuição do IMC")


            @render.plot()
            def bmi_plot():
                fig, ax = plt.subplots(figsize=(10, 6))
                imc_counts = df['IMC'].value_counts().sort_index()
                ax.fill_between(imc_counts.index, imc_counts.values, color='skyblue', alpha=0.5)
                ax.set_xlabel('IMC (kg/m²)')
                ax.set_ylabel('Número de Pessoas')
                ax.set_title('Distribuição do IMC')
                return fig

        # Gráfico de Pressão Arterial (Área Empilhada)
        with ui.card(full_screen=True):
            ui.card_header("Distribuição da Pressão Arterial")


            @render.plot()
            def bp_plot():
                fig, ax = plt.subplots(figsize=(10, 6))
                bp_categories = df.apply(
                    lambda row: get_bp_classification_value(row['PAS'], row['PAD']), axis=1
                )
                bp_counts = pd.Series(bp_categories).value_counts().sort_index()
                ax.fill_between(bp_counts.index, bp_counts.values, color='salmon', alpha=0.5)
                ax.set_xlabel('Índice de Pressão Arterial')
                ax.set_ylabel('Número de Pessoas')
                ax.set_title('Distribuição da Pressão Arterial')
                return fig
