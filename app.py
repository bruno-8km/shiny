from shiny import ui, reactive, render, App
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# === Funções de classificação (sem mudanças) ===
# ... [mesmas funções get_bp_classification_value e get_bmi_classification_value aqui] ...

# === Carregamento e preparação do dataframe df (sem mudanças) ===
# Carrega dados de saúde personalizados
#df = pd.read_csv("https://raw.githubusercontent.com/bruno-8km/meuprojeto/master/points_per_health.csv")
df = pd.read_csv("https://raw.githubusercontent.com/bruno-8km/shiny/refs/heads/master/points_per_health.csv")

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
    "Blood": "Pressão Sanguínea", "IMC": "IMC (kg/m²)",
    **factor_display_names
}

bp_slider_min_val = 0
bp_slider_max_val = 5
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

# === CORRIGIDO: UI ===
sidebar = ui.sidebar(
    ui.input_radio_buttons("x_mode", "Eixo X:", {"score": "Pontuação", "percent": "Porcentagem"}, selected="score"),
    ui.input_slider("n_pessoas", "Número de pessoas a incluir:", min=1, max=len(df), value=len(df), step=1),
    ui.input_selectize(
        "sex_select", "Gênero:",
        choices={"All": "Todos", "M": "Masculino", "F": "Feminino", "N": "Não Informado"},
        multiple=True, selected="All"
    ),
    ui.tags.hr(),
    ui.input_switch("apply_bp_filter", "Índice de Pressão Arterial", False),
    ui.panel_conditional("input.apply_bp_filter",
        ui.input_slider(
            "bp_level_slider",
            "Nível de Pressão Arterial:",
            min=bp_slider_min_val,
            max=bp_slider_max_val,
            value=bp_slider_default_selected,
            step=1,
        ),
    ),
    ui.tags.hr(),
    ui.input_switch("apply_bmi_filter", "Índice do IMC", False),
    ui.panel_conditional("input.apply_bmi_filter",
        ui.input_slider(
            "bmi_level_slider",
            "Nível de IMC:",
            min=bmi_slider_min_val,
            max=bmi_slider_max_val,
            value=bmi_slider_default_selected,
            step=1,
        ),
    ),
    open="desktop"
)

# === UI principal com sidebar e navsets ===
app_ui = ui.page_sidebar(
    sidebar,
    ui.navset_pill(
        ui.nav_panel("Home",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Controle dos Remédios Naturais"),
                    ui.tags.h5("Os 8 Remédios Naturais"),
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
                )
            )
        ),
        ui.nav_panel("Gráficos",
            ui.card(
                ui.card_header("Distribuição do IMC"),
                ui.output_plot("bmi_plot")
            ),
            ui.card(
                ui.card_header("Distribuição da Pressão Arterial"),
                ui.output_plot("bp_plot")
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
            ui.update_checkbox(f"chk_{factor_en}", value=is_checked)

    @reactive.Effect
    def _update_chk_all_factors_status():
        all_selected = all(input[f"chk_{factor_en}"]() for factor_en in health_factors_en)
        if input.chk_all_factors() != all_selected:
            ui.update_checkbox("chk_all_factors", value=all_selected)

    @reactive.Calc
    def selected_factors_calc():
        return [factor_en for factor_en in health_factors_en if input[f"chk_{factor_en}"]()]

    @reactive.Calc
    def filtered_df():
        df_copy = df.copy()
        df_filtered = df_copy.head(input.n_pessoas())

        selected_sex_options = input.sex_select()
        if selected_sex_options and "All" not in selected_sex_options:
            df_filtered = df_filtered[df_filtered['Sex'].isin(selected_sex_options)]

        if input.apply_bp_filter():
            selected = input.bp_level_slider()
            df_filtered = df_filtered[
                df_filtered.apply(lambda row: get_bp_classification_value(row['PAS'], row['PAD']) == selected, axis=1)
            ]

        if input.apply_bmi_filter():
            selected = input.bmi_level_slider()
            df_filtered = df_filtered[
                df_filtered['IMC'].apply(lambda v: get_bmi_classification_value(v) == selected)
            ]

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
            return fig

        for factor_en in selected_factors_en:
            if factor_en not in current_plot_df.columns: continue
            counts = current_plot_df[factor_en].value_counts().sort_index()
            if counts.empty: continue
            xs_raw = np.array(counts.index)
            ys_raw = counts.values
            k_spline = min(3, len(xs_raw) - 1)
            try:
                xs_smooth = np.linspace(xs_raw.min(), xs_raw.max(), 300)
                spline = make_interp_spline(xs_raw, ys_raw, k=k_spline)
                ys_smooth = spline(xs_smooth)
            except Exception:
                xs_smooth, ys_smooth = xs_raw, ys_raw

            if x_mode == "score":
                ax.set_xlabel("Pontuação")
            else:
                xs_smooth = (xs_smooth / xs_raw.max()) * 100
                ax.set_xlabel("Porcentagem da pontuação máxima (%)")
                ax.set_xlim(0, 100)

            ax.fill_between(xs_smooth, np.maximum(0, ys_smooth), alpha=0.3, color=colors[factor_en],
                            label=factor_display_names[factor_en])
            ax.plot(xs_smooth, np.maximum(0, ys_smooth), color=colors[factor_en])
        ax.set_ylabel("Número de Pessoas")
        ax.set_title("Distribuição por Remédio Natural")
        ax.legend()
        ax.grid(True)
        return fig

    @output
    @render.data_frame
    def table():
        df_current = filtered_df().copy()
        df_current.columns = [column_name_translations.get(col, col) for col in df_current.columns]
        return render.DataGrid(df_current)

    @output
    @render.plot()
    def bmi_plot():
        fig, ax = plt.subplots(figsize=(10, 6))
        imc_counts = df['IMC'].value_counts().sort_index()
        ax.fill_between(imc_counts.index, imc_counts.values, color='skyblue', alpha=0.5)
        ax.set_xlabel('IMC (kg/m²)')
        ax.set_ylabel('Número de Pessoas')
        ax.set_title('Distribuição do IMC')
        return fig

    @output
    @render.plot()
    def bp_plot():
        fig, ax = plt.subplots(figsize=(10, 6))
        bp_categories = df.apply(lambda row: get_bp_classification_value(row['PAS'], row['PAD']), axis=1)
        bp_counts = pd.Series(bp_categories).value_counts().sort_index()
        ax.fill_between(bp_counts.index, bp_counts.values, color='salmon', alpha=0.5)
        ax.set_xlabel('Índice de Pressão Arterial')
        ax.set_ylabel('Número de Pessoas')
        ax.set_title('Distribuição da Pressão Arterial')
        return fig

# === INICIALIZAÇÃO ===
app = App(app_ui, server)
