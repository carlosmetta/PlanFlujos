import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go
from openpyxl import Workbook
import io
import numpy as np
import numpy_financial as npf


st.set_page_config(page_title="Distribución de Gasto", layout="wide")

# ------------------------
# Inicializar memoria
# ------------------------
if "flujos" not in st.session_state:
    st.session_state.flujos = []

# ------------------------
# Tabs de navegación
# ------------------------

tab1, tab2,tab3 = st.tabs(["🔧 Crear Flujo", "📊 Flujo Acumulado","📊 Resumen Financiero"])
with tab1:
    st.title("📊 Distribución de Gasto Suavizada por Trimestre")

    # Entradas del usuario
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("📅 Fecha de inicio", datetime(2025, 1, 1))
    with col2:
        end_date = st.date_input("📅 Fecha de fin", datetime(2025, 12, 31))
    with col3:
        total_amount = st.number_input("💰 Monto total a distribuir", value=100000.0)

    if start_date >= end_date:
        st.error("❌ La fecha de inicio debe ser anterior a la fecha de fin.")
        st.stop()

    # Trimestres y etiquetas
    quarters = pd.date_range(start=start_date, end=end_date, freq='Q')
    quarter_labels = [f"Q{((q.month - 1) // 3) + 1} {q.year}" for q in quarters]
    num_quarters = len(quarter_labels)
    q_indices = np.arange(num_quarters)

    # Selección del método de distribución
    st.subheader("🎯 Cómo quieres asignar los pesos por trimestre")
    método = st.selectbox("Tipo de distribución", ["Manual (picos definidos)", "Uniforme", "Gaussiana", "Lineal"])

    weights = np.zeros(num_quarters)

    if método == "Manual (picos definidos)":
        n_picos = st.slider("Número de picos", 1, 5, 2)
        st.markdown("### Configura cada pico")
        picos = []
        for i in range(n_picos):
            col1, col2 = st.columns([2, 3])
            with col1:
                pico_idx = st.slider(f"🎯 Posición del pico {i+1}", 0, num_quarters - 1, i, key=f"pico_{i}")
            with col2:
                altura = st.slider(f"📈 Altura del pico {i+1}", 0.0, 1.0, 1.0, key=f"altura_{i}")
            picos.append((pico_idx, altura))
        for idx, altura in picos:
            kernel = np.exp(-0.5 * ((q_indices - idx) / 1.0)**2)
            weights += altura * kernel
        weights /= weights.sum()

    elif método == "Uniforme":
        weights = np.ones(num_quarters) / num_quarters

    elif método == "Gaussiana":
        x = np.linspace(-2, 2, num_quarters)
        weights = np.exp(-0.5 * x**2)
        weights /= weights.sum()

    elif método == "Lineal":
        sentido = st.radio("📈 Sentido de la distribución", ["Ascendente", "Descendente"], horizontal=True)
        if sentido == "Ascendente":
            weights = np.linspace(0.1, 1.0, num_quarters)
        else:
            weights = np.linspace(1.0, 0.1, num_quarters)
        weights /= weights.sum()

    # Gráfico de pesos trimestrales
    fig = go.Figure(go.Bar(x=quarter_labels, y=weights, marker_color='skyblue'))
    fig.update_layout(
        title="🔵 Distribución de Pesos por Trimestre",
        yaxis_title="Peso",
        xaxis_title="Trimestre"
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # 🧮 CÁLCULO MENSUAL
    # =========================
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    month_df = pd.DataFrame({'Month': months})
    month_df['Quarter'] = month_df['Month'].dt.to_period("Q")
    month_df['MonthIndex'] = np.arange(len(month_df))
    month_df['Weight'] = 0.0

    # Slider para suavizado (solo si aplica)
    if método in ["Manual (picos definidos)", "Gaussiana"]:
        suavizado_sigma = st.slider(
            "🎛️ Ajusta el grado de suavizado (σ)",
            min_value=0.1, max_value=5.0,
            value=1.2, step=0.1,
            help="Controla la suavidad de la curva mensual. Valores bajos generan curvas más picudas."
        )
    else:
        suavizado_sigma = None

    if método == "Lineal":
        # Distribución mensual lineal directa
        if sentido == "Ascendente":
            weights = np.linspace(0.1, 1.0, len(month_df))
        else:
            weights = np.linspace(1.0, 0.1, len(month_df))
        weights /= weights.sum()
        month_df['Weight'] = weights

    elif método == "Uniforme":
        # Distribución mensual uniforme
        month_df['Weight'] = 1 / len(month_df)

    else:
        # Suavizado Gaussiano mensual basado en pesos trimestrales
        quarter_centers = []
        for q in quarters:
            q_period = q.to_period("Q")
            q_months = month_df[month_df['Quarter'] == q_period]
            if not q_months.empty:
                center_idx = q_months['MonthIndex'].mean()
                quarter_centers.append(center_idx)
        for i, center in enumerate(quarter_centers):
            kernel = norm.pdf(month_df['MonthIndex'], loc=center, scale=suavizado_sigma if suavizado_sigma else 1.2)
            kernel /= kernel.sum()
            month_df['Weight'] += kernel * weights[i]

    # Monto final mensual
    month_df['Amount'] = month_df['Weight'] * total_amount


    # =========================
    # 📊 VISUALIZACIÓN MENSUAL
    # =========================
    st.subheader("📅 Distribución de Gasto Mensual Estimada")
    col_table, col_chart = st.columns([1, 2])

    with col_chart:
        fig2 = go.Figure(go.Scatter(
            x=month_df['Month'],
            y=month_df['Amount'],
            mode='lines+markers',
            line=dict(shape='spline')
        ))
        fig2.update_layout(
            title="📈 Gasto Estimado por Mes (Curva Suavizada)",
            yaxis_title="Monto ($)",
            xaxis_title="Mes"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_table:
        st.dataframe(
            month_df[['Month', 'Amount']].set_index('Month').style.format("${:,.2f}")
        )

    # =========================
    # 💾 GUARDAR FLUJO
    # =========================
    st.subheader("✅ Guardar Flujo Actual")
    with st.form("guardar_flujo"):
        nombre = st.text_input("🔖 Nombre del flujo")
        categoria = st.selectbox("📁 Categoría", ["Ingresos", "Gastos", "Financiero"])
        signo = st.radio("💡 Tipo de flujo", ["Positivo", "Negativo"], horizontal=True)
        guardar = st.form_submit_button("💾 Agregar al Total")

    if guardar:
        if not nombre:
            st.warning("⚠️ Debes ingresar un nombre.")
        else:
            signed_amounts = month_df['Amount'] * (1 if signo == "Positivo" else -1)
            st.session_state.flujos.append({
                "nombre": nombre,
                "categoria": categoria,
                "metodo": método,
                "sigma": suavizado_sigma,
                "data": pd.DataFrame({
                    "Month": month_df['Month'],
                    "Amount": signed_amounts
                })
            })
            st.success(f"✅ Flujo '{nombre}' agregado correctamente.")
            st.rerun()

# TAB 2: Visualización de Flujo Total Acumulado
# ================================================
with tab2:
    st.title("📊 Visualizador de Flujo Total Acumulado")

    if not st.session_state.flujos:
        st.info("🔔 No hay flujos guardados todavía.")
    else:
        # Combinar todos los flujos
        all_data = pd.DataFrame()
        for flujo in st.session_state.flujos:
            df = flujo['data'].copy()
            df['Nombre'] = flujo['nombre']
            df['Categoría'] = flujo['categoria']
            all_data = pd.concat([all_data, df], ignore_index=True)

        all_data = all_data.groupby(['Month', 'Categoría']).sum().reset_index()
        total = all_data.groupby('Month')['Amount'].sum().reset_index()

        st.subheader("🧾 Tabla Consolidada por Categoría")
        st.dataframe(all_data.pivot_table(index='Month', columns='Categoría', values='Amount', aggfunc='sum').fillna(0).style.format("${:,.2f}"))

        st.subheader("📈 Gráfico Acumulado Total")
        fig = go.Figure()
        for categoria in all_data['Categoría'].unique():
            df_cat = all_data[all_data['Categoría'] == categoria]
            fig.add_trace(go.Scatter(
                x=df_cat['Month'],
                y=df_cat['Amount'].cumsum(),
                mode='lines+markers',
                name=categoria
            ))
        fig.add_trace(go.Scatter(
            x=total['Month'],
            y=total['Amount'].cumsum(),
            mode='lines',
            name="Total Neto",
            line=dict(width=4, dash='dot')
        ))
        fig.update_layout(
            title="📊 Acumulado por Categoría y Total",
            xaxis_title="Mes",
            yaxis_title="Monto Acumulado ($)"
        )
        st.plotly_chart(fig, use_container_width=True)
                # =============================
        # OPCIONES: MODIFICAR / ELIMINAR
        # =============================
        st.subheader("🧩 Flujos Guardados")

        for i, flujo in enumerate(st.session_state.flujos):
            with st.expander(f"📌 {flujo['nombre']} ({flujo['categoria']})"):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    nuevo_nombre = st.text_input("✏️ Cambiar nombre", flujo['nombre'], key=f"edit_nombre_{i}")
                with col2:
                    nueva_categoria = st.selectbox("📁 Cambiar categoría", ["Ingresos", "Gastos", "Financiero"], index=["Ingresos", "Gastos", "Financiero"].index(flujo['categoria']), key=f"edit_cat_{i}")
                with col3:
                    nuevo_signo = st.radio("💡 Signo", ["Positivo", "Negativo"],
                                           index=0 if flujo['data']['Amount'].sum() >= 0 else 1,
                                           horizontal=True, key=f"edit_signo_{i}")

                col4, col5 = st.columns([1, 1])
                with col4:
                    if st.button("💾 Guardar cambios", key=f"guardar_edit_{i}"):
                        # Actualizar nombre y categoría
                        flujo['nombre'] = nuevo_nombre
                        flujo['categoria'] = nueva_categoria
                        # Actualizar signo
                        signo_valor = 1 if nuevo_signo == "Positivo" else -1
                        flujo['data']['Amount'] = flujo['data']['Amount'].abs() * signo_valor
                        st.success(f"✅ Flujo '{nuevo_nombre}' actualizado.")
                        st.rerun()
                with col5:
                    if st.button("🗑️ Eliminar flujo", key=f"eliminar_{i}"):
                        st.session_state.flujos.pop(i)
                        st.warning(f"🗑️ Flujo eliminado.")
                        st.rerun()

   




    st.subheader("📤 Exportar Flujos a Excel")

    # Combinar todos los flujos individuales en un solo DataFrame
    flujos_export = pd.DataFrame()

    for flujo in st.session_state.flujos:
        df = flujo['data'].copy()
        df['Nombre'] = flujo['nombre']
        df['Categoría'] = flujo['categoria']
        flujos_export = pd.concat([flujos_export, df], ignore_index=True)

    # Asegurarse del orden correcto
    flujos_export = flujos_export[['Month', 'Nombre', 'Categoría', 'Amount']]
    flujos_export['Amount'] = flujos_export['Amount'].round(2)

    # Flujo total
    export_total = total.copy()
    export_total.columns = ['Month', 'TotalNet']

    # Crear archivo Excel en memoria
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja 1: todos los flujos combinados
        flujos_export.to_excel(writer, index=False, sheet_name="FlujosMensuales")
        # Hoja 2: flujo total acumulado
        export_total.to_excel(writer, index=False, sheet_name="FlujoTotal")

    # Botón de descarga
    st.download_button(
        label="📥 Descargar Excel",
        data=output.getvalue(),
        file_name="FlujosFinancieros.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


with tab3:
    st.title("📊 Resumen Financiero")

    if not st.session_state.flujos:
        st.info("🔔 No hay flujos guardados todavía.")
    else:
        # 1. Calcular flujo neto por mes
        all_data = pd.DataFrame()
        for flujo in st.session_state.flujos:
            df = flujo['data'].copy()
            all_data = pd.concat([all_data, df], ignore_index=True)

        flujo_mensual = all_data.groupby('Month')['Amount'].sum().reset_index()

        # 2. Estado de resultados
        ingresos = all_data[all_data['Amount'] > 0]['Amount'].sum()
        egresos = all_data[all_data['Amount'] < 0]['Amount'].sum()  # ya es negativo
        margen = (ingresos + egresos) / ingresos if ingresos > 0 else 0  # (Ingreso - |Egreso|) / Ingreso

        st.subheader("📑 Estado de Resultados")
        st.metric("Ingresos Totales", f"${ingresos:,.2f}")
        st.metric("Egresos Totales", f"${egresos:,.2f}")
        st.metric("Margen", f"{margen * 100:.2f}%")

        # 3. Capital máximo (lo más negativo del acumulado)
        flujo_mensual['Acumulado'] = flujo_mensual['Amount'].cumsum()
        capital_max = abs(min(0, flujo_mensual['Acumulado'].min()))
        st.subheader("💰 Capital Máximo Necesario")
        st.metric("Capital Requerido", f"${capital_max:,.2f}")

        # 4. VPN y TIR
        st.subheader("📈 VPN y TIR")
        tasa_descuento = st.number_input("Tasa de descuento anual (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        tasa_mensual = (1 + tasa_descuento / 100)**(1/12) - 1  # conversión a mensual

        vpn = npf.npv(tasa_mensual, flujo_mensual['Amount'])
        try:
            tir = npf.irr(flujo_mensual['Amount']) * 12 * 100  # anualizar y pasar a %
        except:
            tir = None

        st.metric("VPN", f"${vpn:,.2f}")
        st.metric("TIR", f"{tir:.2f}%" if tir is not None else "N/A")

        # 5. Gráfico del flujo acumulado
        st.subheader("📊 Flujo Neto Acumulado")
        fig3 = go.Figure(go.Scatter(
            x=flujo_mensual['Month'],
            y=flujo_mensual['Acumulado'],
            mode='lines+markers',
            line=dict(shape='linear')
        ))
        fig3.update_layout(
            title="Flujo Neto Acumulado",
            yaxis_title="Monto ($)",
            xaxis_title="Mes"
        )
        st.plotly_chart(fig3, use_container_width=True)
