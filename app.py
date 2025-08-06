# ===============================
# CALCULADORA MAESTRA INTEGRADA
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go
import numpy_financial as npf
import io
import json

st.set_page_config(page_title="Calculadora Maestra Inmobiliaria", layout="wide")

# -------------------------------
# INICIALIZAR SESIONES GLOBALES
# -------------------------------
if "costos" not in st.session_state:
    st.session_state.costos = []
if "tipos_unidades" not in st.session_state:
    st.session_state.tipos_unidades = []
if "flujos" not in st.session_state:
    st.session_state.flujos = []
if "flujos_base" not in st.session_state:
    st.session_state.flujos_base = {}

# -------------------------------
# TABS PRINCIPALES
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Calculadora Maestra",
    "🗓️ Temporalizar Flujos",
    "📊 Flujo Acumulado",
    "📈 Resumen Financiero"
])

# =========================================
# 📘 TAB 1: CALCULADORA MAESTRA INMOBILIARIA
# =========================================
with tab1:
    st.title("📘 Calculadora Maestra Inmobiliaria")
    st.header("📂 Cargar Proyecto")

    archivo_cargado = st.file_uploader("📁 Selecciona archivo JSON del proyecto", type=["json"])
    if archivo_cargado and not st.session_state.get("proyecto_cargado"):
        proyecto = json.load(archivo_cargado)

        st.session_state.costos = proyecto.get("costos", [])
        st.session_state.tipos_unidades = proyecto.get("tipos_unidades", [])
        st.session_state.flujos_base = proyecto.get("flujos_base", {})
        st.session_state.unidades_disponibles = proyecto.get("unidades_disponibles", 0)
        st.session_state.m2_construibles = proyecto.get("m2_construibles", 0)
        st.session_state.precio_sugerido_m2 = proyecto.get("precio_sugerido_m2", 0)
        st.session_state.venta_total = proyecto.get("venta_total", 0)
        st.session_state.costo_total = proyecto.get("costo_total", 0)

        # Reconstrucción segura de flujos desde JSON
        flujos = []
        for f in proyecto.get("flujos", []):
            data_dict = f.get("data", {})
            months = list(data_dict.get("Month", {}).values())
            amounts = list(data_dict.get("Amount", {}).values())

            df = pd.DataFrame({
                "Month": pd.to_datetime(months),
                "Amount": pd.to_numeric(amounts)
            })

            flujos.append({
                "nombre": f["nombre"],
                "categoria": f["categoria"],
                "data": df
            })

        st.session_state.flujos = flujos
        st.session_state.proyecto_cargado = True  # ← Marca que ya se cargó
        st.success("✅ Proyecto cargado correctamente.")
        st.rerun()

    # Inicialización
    if "costos" not in st.session_state:
        st.session_state.costos = []
    if "tipos_unidades" not in st.session_state:
        st.session_state.tipos_unidades = []

    # 1️⃣ DATOS GENERALES
    st.header("1️⃣ Datos Generales del Proyecto")
    col1, col2 = st.columns(2)
    with col1:
        unidades_disponibles = st.number_input("🔢 Unidades máximas disponibles", min_value=1, value=100)
    with col2:
        m2_por_unidad = st.number_input("📐 m² promedio por unidad", min_value=1.0, value=60.0)

    m2_construibles = unidades_disponibles * m2_por_unidad
    st.session_state.unidades_disponibles = unidades_disponibles
    st.session_state.m2_construibles = m2_construibles

    st.markdown(f"**📏 m² construibles totales:** {m2_construibles:,.2f} m²")

    # 2️⃣ COSTOS
    st.header("2️⃣ Costos del Proyecto")
    with st.form("form_costos"):
        nombre = st.text_input("Nombre del costo", placeholder="Ej: Construcción, Tramitología, Terreno")
        metodo = st.selectbox("Tipo de cálculo", ["Total", "Por unidad", "Por m²"])
        valor = st.number_input("Valor ($)", min_value=0.0, step=100.0)
        submit_costo = st.form_submit_button("➕ Agregar costo")

    if submit_costo and nombre:
        st.session_state.costos.append({"nombre": nombre, "metodo": metodo, "valor": valor})
        st.success(f"✅ Costo '{nombre}' agregado.")

    df_costos = pd.DataFrame(st.session_state.costos)
    costo_total = 0
    if not df_costos.empty:
        def calcular_total(row):
            if row['metodo'] == 'Total': return row['valor']
            elif row['metodo'] == 'Por unidad': return row['valor'] * unidades_disponibles
            elif row['metodo'] == 'Por m²': return row['valor'] * m2_construibles
        df_costos["Total"] = df_costos.apply(calcular_total, axis=1)
        costo_total = df_costos["Total"].sum()
        st.session_state.costo_total = costo_total

        st.markdown("### 🧾 Lista de Costos")
        for i, row in df_costos.iterrows():
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
            col1.markdown(f"**{row['nombre']}**")
            col2.markdown(f"{row['metodo']}")
            col3.markdown(f"$ {row['valor']:,.2f}")
            col4.markdown(f"$ {row['Total']:,.2f}")
            if col5.button("❌", key=f"delete_costo_{i}"):
                st.session_state.costos.pop(i)
                st.rerun()

        st.markdown(f"**💸 Costo total del proyecto:** $ {costo_total:,.2f}")
    else:
        st.info("Agrega al menos un costo para continuar.")

    # 3️⃣ INGRESOS POR MARGEN
    st.header("3️⃣ Ingresos Esperados por Margen")
    margen_deseado = st.slider("🎯 Margen deseado (%)", 0.0, 100.0, 30.0)
    if m2_construibles > 0:
        precio_sugerido_m2 = costo_total / m2_construibles / (1 - margen_deseado / 100)
    else:
        precio_sugerido_m2 = 0
    venta_total = precio_sugerido_m2 * m2_construibles

    st.session_state.precio_sugerido_m2 = precio_sugerido_m2
    st.session_state.venta_total = venta_total

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Precio sugerido por m²", f"$ {precio_sugerido_m2:,.2f}")
    col2.metric("💼 Venta total esperada", f"$ {venta_total:,.2f}")
    col3.metric("📈 Margen aplicado", f"{margen_deseado:.2f}%")

    # 4️⃣ TIPOS DE UNIDADES
    st.header("4️⃣ Definición de Tipos de Unidad")
    with st.form("form_unidades"):
        tipo = st.text_input("Nombre del tipo de unidad", placeholder="Ej: Tipo A, Tipo B")
        etapa = st.text_input("Etapa", placeholder="Ej: Etapa 1")
        cantidad = st.number_input("Cantidad de unidades", min_value=1, value=10)
        m2_unidad = st.number_input("m² por unidad", min_value=1.0, value=60.0)
        agregar_tipo = st.form_submit_button("➕ Agregar tipo")

    if agregar_tipo and tipo:
        st.session_state.tipos_unidades.append({
            "tipo": tipo,
            "etapa": etapa,
            "cantidad": cantidad,
            "m2": m2_unidad
        })
        st.success(f"✅ Tipo de unidad '{tipo}' agregado.")

    df_unidades = pd.DataFrame(st.session_state.tipos_unidades)
    if not df_unidades.empty:
        df_unidades["Total m²"] = df_unidades["cantidad"] * df_unidades["m2"]
        df_unidades["Ingreso por unidad"] = df_unidades["m2"] * precio_sugerido_m2
        df_unidades["Ingreso total"] = df_unidades["Ingreso por unidad"] * df_unidades["cantidad"]
        total_m2_asignado = df_unidades["Total m²"].sum()

        if total_m2_asignado > m2_construibles:
            st.warning(f"⚠️ Te estás excediendo por {total_m2_asignado - m2_construibles:.2f} m² sobre los {m2_construibles} disponibles.")

        st.markdown("### 🧾 Tipos de Unidad")
        for i, row in df_unidades.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2, 2, 1])
            col1.markdown(f"**{row['tipo']}**")
            col2.markdown(f"{row['etapa']}")
            col3.markdown(f"{row['cantidad']}")
            col4.markdown(f"{row['m2']:,.2f} m²")
            col5.markdown(f"$ {row['Ingreso por unidad']:,.2f}")
            col6.markdown(f"$ {row['Ingreso total']:,.2f}")
            if col7.button("❌", key=f"delete_unidad_{i}"):
                st.session_state.tipos_unidades.pop(i)
                st.rerun()

        st.markdown(f"**🏠 Total m² asignados:** {total_m2_asignado:,.2f} / {m2_construibles:,.2f} m²")

    # 5️⃣ AJUSTES DE PRECIO
    st.header("5️⃣ Ajustes de Precio y Evaluación")
    nuevo_precio_m2 = st.number_input("💵 Precio alternativo por m²", min_value=0.0, value=float(precio_sugerido_m2), step=100.0)
    venta_ajustada = nuevo_precio_m2 * m2_construibles
    margen_actual = (venta_ajustada - costo_total) / venta_ajustada * 100 if venta_ajustada > 0 else 0
    costo_necesario = venta_ajustada * (1 - margen_deseado / 100)
    ajuste_costo = costo_total - costo_necesario

    col1, col2, col3 = st.columns(3)
    col1.metric("🔁 Venta ajustada", f"$ {venta_ajustada:,.2f}")
    col2.metric("📉 Margen actual", f"{margen_actual:.2f}%")
    col3.metric("📌 Costo necesario", f"$ {costo_necesario:,.2f}")
    st.metric("📉 Ajuste necesario al costo", f"$ {ajuste_costo:,.2f}")

    if not df_unidades.empty:
        st.subheader("💹 Precio por unidad con precio ajustado")
        df_ajuste = df_unidades.copy()
        df_ajuste["Precio ajustado por unidad"] = df_ajuste["m2"] * nuevo_precio_m2
        st.dataframe(df_ajuste[["tipo", "etapa", "m2", "cantidad", "Precio ajustado por unidad"]].style.format({
            "m2": "{:,.2f}",
            "Precio ajustado por unidad": "$ {:,.2f}"
        }), use_container_width=True)

    st.session_state.flujos_base = {
        "Ingresos del Proyecto": venta_total,
        "Costos del Proyecto": costo_total
    }






# =========================================
# 🗓️ TAB 2: TEMPORIZAR Y AGREGAR FLUJOS
# =========================================
with tab2:
    st.title("🗓️ Temporalizar y Agregar Flujos")

    tipo_flujo = st.radio("Tipo de flujo a crear o temporalizar", ["Desde Calculadora", "Independiente", "Dependiente"], horizontal=True)

    if tipo_flujo == "Desde Calculadora":
        opciones = list(st.session_state.flujos_base.keys())
        seleccion = st.selectbox("Selecciona un flujo base para temporalizar:", opciones)

        if seleccion == "Costos del Proyecto":
            st.markdown("### Selecciona el subcosto a temporalizar")
            if st.session_state.costos:
                subcosto = st.selectbox("Costo específico", [c["nombre"] for c in st.session_state.costos])
                costos_df = pd.DataFrame(st.session_state.costos)
                selected_row = costos_df[costos_df["nombre"] == subcosto].iloc[0]
                metodo = selected_row["metodo"]
                valor = selected_row["valor"]

                if metodo == "Total":
                    monto_total = valor
                elif metodo == "Por unidad":
                    monto_total = valor * st.session_state.get("unidades_disponibles", 100)
                elif metodo == "Por m²":
                    m2 = st.session_state.get("m2_construibles", 6000)
                    monto_total = valor * m2
                nombre_flujo = f"Costo - {subcosto}"
            else:
                st.warning("⚠️ No hay costos definidos.")
                st.stop()

        elif seleccion == "Ingresos del Proyecto":
            st.markdown("### Selecciona la etapa de ingreso a temporalizar")
            if st.session_state.tipos_unidades:
                etapas_disponibles = list(set([u["etapa"] for u in st.session_state.tipos_unidades]))
                etapa_seleccionada = st.selectbox("Etapa", etapas_disponibles)

                # Calcular ingreso total de esa etapa
                df_unidades = pd.DataFrame(st.session_state.tipos_unidades)
                df_etapa = df_unidades[df_unidades["etapa"] == etapa_seleccionada]
                df_etapa["Ingreso total"] = df_etapa["cantidad"] * df_etapa["m2"] * st.session_state.precio_sugerido_m2
                monto_total = df_etapa["Ingreso total"].sum()
                nombre_flujo = f"Ingreso - {etapa_seleccionada}"
            else:
                st.warning("⚠️ No hay tipos de unidad definidos.")
                st.stop()

        else:
            nombre_flujo = seleccion
            monto_total = st.session_state.flujos_base[seleccion]

    elif tipo_flujo == "Independiente":
        nombre_flujo = st.text_input("🔖 Nombre del nuevo flujo")
        monto_total = st.number_input("💰 Monto total a distribuir", value=0.0)

    elif tipo_flujo == "Dependiente":
        flujo_referencia = st.selectbox("Selecciona flujo base", [f['nombre'] for f in st.session_state.flujos])
        porcentaje = st.number_input("📊 Porcentaje (%)", min_value=-100.0, max_value=100.0, value=5.0)
        nombre_flujo = st.text_input("🔖 Nombre del nuevo flujo dependiente")
        categoria = st.selectbox("📁 Categoría del nuevo flujo", ["Ingreso", "Costo", "Financiero", "Operación"], key="dep_cat")

        if flujo_referencia and nombre_flujo:
            flujo_base = next(f for f in st.session_state.flujos if f['nombre'] == flujo_referencia)
            df_base = flujo_base['data'].copy()
            df_dep = df_base.copy()
            df_dep['Amount'] = df_base['Amount'] * (porcentaje / 100)

            if st.button("💾 Guardar flujo dependiente"):
                st.session_state.flujos.append({
                    "nombre": nombre_flujo,
                    "categoria": categoria,
                    "data": df_dep
                })
                st.success(f"✅ Flujo dependiente '{nombre_flujo}' creado como {porcentaje}% de '{flujo_referencia}'.")
        st.stop()

    if nombre_flujo:
        categoria = st.selectbox("📁 Categoría del flujo", ["Ingreso", "Costo", "Financiero", "Operación"], key="cat_flujo")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("📅 Fecha de inicio", datetime(2025, 1, 1), key="fecha_inicio")
        with col2:
            end_date = st.date_input("📅 Fecha de fin", datetime(2025, 12, 31), key="fecha_fin")

        if start_date < end_date:
            quarters = pd.date_range(start=start_date, end=end_date, freq='Q')
            quarter_labels = [f"Q{((q.month - 1) // 3) + 1} {q.year}" for q in quarters]
            num_quarters = len(quarter_labels)
            q_indices = np.arange(num_quarters)

            st.subheader("🎯 Distribución Temporal")
            metodo = st.selectbox("Tipo de distribución", ["Manual (picos definidos)", "Uniforme", "Gaussiana", "Lineal"])
            weights = np.zeros(num_quarters)

            if metodo == "Manual (picos definidos)":
                n_picos = st.slider("Número de picos", 1, 5, 2)
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

            elif metodo == "Uniforme":
                weights = np.ones(num_quarters) / num_quarters

            elif metodo == "Gaussiana":
                x = np.linspace(-2, 2, num_quarters)
                weights = np.exp(-0.5 * x**2)
                weights /= weights.sum()

            elif metodo == "Lineal":
                sentido = st.radio("📈 Sentido de la distribución", ["Ascendente", "Descendente"], horizontal=True)
                if sentido == "Ascendente":
                    weights = np.linspace(0.1, 1.0, num_quarters)
                else:
                    weights = np.linspace(1.0, 0.1, num_quarters)
                weights /= weights.sum()

            fig = go.Figure(go.Bar(x=quarter_labels, y=weights, marker_color='skyblue'))
            fig.update_layout(title="Distribución de Pesos por Trimestre", xaxis_title="Trimestre", yaxis_title="Peso")
            st.plotly_chart(fig, use_container_width=True)

            months = pd.date_range(start=start_date, end=end_date, freq='MS')
            month_df = pd.DataFrame({'Month': months})
            month_df['Quarter'] = month_df['Month'].dt.to_period("Q")
            month_df['MonthIndex'] = np.arange(len(month_df))
            month_df['Weight'] = 0.0
            quarter_centers = []
            sigma = st.slider("🎛️ Grado de suavizado (σ)", min_value=0.1, max_value=5.0, value=1.2, step=0.1) if metodo in ["Manual (picos definidos)", "Gaussiana"] else 1.2

            
            for q in quarters:
                q_period = q.to_period("Q")
                q_months = month_df[month_df['Quarter'] == q_period]
                if not q_months.empty:
                    center_idx = q_months['MonthIndex'].mean()
                    quarter_centers.append(center_idx)

            month_df['Weight'] = 0.0  # Reset

            if metodo in ["Manual (picos definidos)", "Gaussiana"]:
                for i, center in enumerate(quarter_centers):
                    kernel = norm.pdf(month_df['MonthIndex'], loc=center, scale=sigma)
                    kernel /= kernel.sum()
                    month_df['Weight'] += kernel * weights[i]

            elif metodo == "Uniforme":
                month_df['Weight'] = 1 / len(month_df)

            elif metodo == "Lineal":
                sentido = st.session_state.get("sentido_lineal", "Ascendente")
                if sentido == "Ascendente":
                    month_df['Weight'] = np.linspace(0.1, 1.0, len(month_df))
                else:
                    month_df['Weight'] = np.linspace(1.0, 0.1, len(month_df))
                month_df['Weight'] /= month_df['Weight'].sum()

            # Finalmente:
            month_df['Amount'] = month_df['Weight'] * monto_total

            st.subheader("📅 Distribución de Gasto Mensual Estimada")
            col_table, col_chart = st.columns([1, 2])
            with col_chart:
                fig2 = go.Figure(go.Scatter(
                    x=month_df['Month'],
                    y=month_df['Amount'],
                    mode='lines+markers',
                    line=dict(shape='spline')
                ))
                fig2.update_layout(title="📈 Gasto Estimado por Mes (Curva Suavizada)", yaxis_title="Monto ($)", xaxis_title="Mes")
                st.plotly_chart(fig2, use_container_width=True)
            with col_table:
                st.dataframe(month_df[['Month', 'Amount']].set_index('Month').style.format("${:,.2f}"))

            # 👉 Nuevo selector de signo
            signo_flujo = st.radio("💡 ¿Este flujo debe ser positivo o negativo?", ["Positivo", "Negativo"], horizontal=True)

            if st.button("💾 Guardar flujo mensualizado"):
                signo = 1 if signo_flujo == "Positivo" else -1
                month_df["Amount"] = month_df["Amount"].abs() * signo

                st.session_state.flujos.append({
                    "nombre": nombre_flujo,
                    "categoria": categoria,
                    "data": month_df[['Month', 'Amount']]
                })
                st.success(f"✅ Flujo '{nombre_flujo}' guardado correctamente.")

        else:
            st.error("❌ La fecha de inicio debe ser menor a la de fin.")






# =========================================
# 📊 TAB 3: FLUJO ACUMULADO Y HERRAMIENTAS
# =========================================
with tab3:
    st.title("📊 Resumen de Flujos y Herramientas Administrativas")

    if not st.session_state.flujos:
        st.info("🔔 No hay flujos definidos aún. Ve al Tab 2 para crearlos.")
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

        # Tabla Pivot por Categoría
        st.subheader("📑 Resumen Mensual por Categoría")
        pivot = all_data.pivot_table(index='Month', columns='Categoría', values='Amount', aggfunc='sum').fillna(0)
        st.dataframe(pivot.style.format("${:,.2f}"))

        # Gráfico acumulado
        st.subheader("📈 Gráfico de Flujo Acumulado por Categoría")
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
            line=dict(width=4, dash='dot', color='black')
        ))
        fig.update_layout(
            title="📊 Flujo Acumulado por Categoría",
            xaxis_title="Mes",
            yaxis_title="Monto Acumulado ($)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # =============================
        # OPCIONES ADMINISTRATIVAS
        # =============================
        st.subheader("🧩 Flujos Guardados")

        for i, flujo in enumerate(st.session_state.flujos):
            with st.expander(f"📌 {flujo['nombre']} ({flujo['categoria']})"):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    nuevo_nombre = st.text_input("✏️ Cambiar nombre", flujo['nombre'], key=f"edit_nombre_{i}")
                with col2:
                    nueva_categoria = st.selectbox("📁 Cambiar categoría", ["Ingreso", "Costo", "Financiero", "Operación"], index=["Ingreso", "Costo", "Financiero", "Operación"].index(flujo['categoria']), key=f"edit_cat_{i}")
                with col3:
                    nuevo_signo = st.radio("💡 Signo", ["Positivo", "Negativo"], index=0 if flujo['data']['Amount'].sum() >= 0 else 1, horizontal=True, key=f"edit_signo_{i}")

                col4, col5 = st.columns([1, 1])
                with col4:
                    if st.button("💾 Guardar cambios", key=f"guardar_edit_{i}"):
                        flujo['nombre'] = nuevo_nombre
                        flujo['categoria'] = nueva_categoria
                        signo_valor = 1 if nuevo_signo == "Positivo" else -1
                        flujo['data']['Amount'] = flujo['data']['Amount'].abs() * signo_valor
                        st.success(f"✅ Flujo '{nuevo_nombre}' actualizado.")
                        st.rerun()
                with col5:
                    if st.button("🗑️ Eliminar flujo", key=f"eliminar_{i}"):
                        st.session_state.flujos.pop(i)
                        st.warning(f"🗑️ Flujo eliminado.")
                        st.rerun()

        # Exportar a Excel
        st.subheader("📤 Exportar Flujos a Excel")
        export_data = pd.DataFrame()
        for flujo in st.session_state.flujos:
            df = flujo['data'].copy()
            df['Nombre'] = flujo['nombre']
            df['Categoría'] = flujo['categoria']
            export_data = pd.concat([export_data, df], ignore_index=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_data.to_excel(writer, index=False, sheet_name="FlujosMensuales")
            total.to_excel(writer, index=False, sheet_name="TotalNeto")

        st.download_button(
            label="📥 Descargar Excel",
            data=output.getvalue(),
            file_name="Resumen_Flujos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================================
# 📈 TAB 4: RESUMEN FINANCIERO COMPLETO
# =========================================
with tab4:
    st.title("📈 Resumen Financiero y Estado de Resultados")

    if not st.session_state.flujos:
        st.info("🔔 No hay flujos definidos aún. Ve al Tab 2 para crearlos.")
    else:
        # Preparar datos
        all_data = pd.DataFrame()
        for flujo in st.session_state.flujos:
            df = flujo['data'].copy()
            df['Nombre'] = flujo['nombre']
            df['Categoría'] = flujo['categoria']
            all_data = pd.concat([all_data, df], ignore_index=True)

        flujo_mensual = all_data.groupby(['Month', 'Categoría']).sum().reset_index()
        pivot_total = flujo_mensual.pivot(index='Month', columns='Categoría', values='Amount').fillna(0)
        pivot_total['Total Neto'] = pivot_total.sum(axis=1)

        # ========================
        # ESTADO DE RESULTADOS
        # ========================
        st.subheader("📊 Estado de Resultados Vertical (Total Proyecto)")

        # Consolidar todos los flujos
        all_data = pd.DataFrame()
        for flujo in st.session_state.flujos:
            df = flujo['data'].copy()
            df['Nombre'] = flujo['nombre']
            df['Categoría'] = flujo['categoria']
            all_data = pd.concat([all_data, df], ignore_index=True)

        all_data['Amount'] = pd.to_numeric(all_data['Amount'], errors='coerce')

        # Sumar por categoría general
        total_ingresos = all_data.query("Categoría == 'Ingreso'")['Amount'].sum()
        total_costos = all_data.query("Categoría == 'Costo'")['Amount'].sum()
        utilidad_bruta = total_ingresos + total_costos  # costos son negativos
        margen_bruto = (utilidad_bruta / total_ingresos * 100) if total_ingresos else 0

        total_gasto_operacion = all_data.query("Categoría == 'Operación'")['Amount'].sum()
        utilidad_operativa = utilidad_bruta + total_gasto_operacion
        margen_operativo = (utilidad_operativa / total_ingresos * 100) if total_ingresos else 0

        total_gasto_financiero = all_data.query("Categoría == 'Financiero'")['Amount'].sum()
        utilidad_neta = utilidad_operativa + total_gasto_financiero
        margen_neto = (utilidad_neta / total_ingresos * 100) if total_ingresos else 0

        # Mostrar en tabla vertical
        def fmt(monto): return f"$ {monto:,.2f}"

        resultado_dict = {
            "Ingresos": total_ingresos,
            "Costos": total_costos,
            "Utilidad Bruta": utilidad_bruta,
            "Margen Bruto": f"{margen_bruto:.2f}%",
            "Gastos de Operación": total_gasto_operacion,
            "Utilidad Operativa": utilidad_operativa,
            "Margen Operativo": f"{margen_operativo:.2f}%",
            "Gastos Financieros": total_gasto_financiero,
            "Utilidad Neta": utilidad_neta,
            "Margen Neto": f"{margen_neto:.2f}%"
        }

        for k, v in resultado_dict.items():
            if isinstance(v, (int, float)):
                st.markdown(f"**{k}:** {fmt(v)}")
            else:
                st.markdown(f"**{k}:** {v}")

        # ========================
        # DETALLE EXPANDIBLE POR FLUJO
        # ========================
        st.subheader("🔍 Detalle por Flujo")

        detalle = all_data.groupby(["Categoría", "Nombre"])["Amount"].sum().reset_index()
        detalle['Amount'] = detalle['Amount'].round(2)

        with st.expander("📋 Ver detalle completo por categoría y flujo"):
            st.dataframe(detalle.style.format({"Amount": "$ {:,.2f}"}), use_container_width=True)


        # ========================
        # METRICAS FINANCIERAS
        # ========================
        st.subheader("📈 Indicadores Financieros")
        flujo_neto = pivot_total['Total Neto']
        flujo_neto_acumulado = flujo_neto.cumsum()

        capital_requerido = abs(min(0, flujo_neto_acumulado.min()))
        mes_min = flujo_neto.idxmin()
        mes_max = flujo_neto.idxmax()

        tasa_descuento = st.number_input("Tasa de descuento anual (%)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        tasa_mensual = (1 + tasa_descuento / 100)**(1/12) - 1

        vpn = npf.npv(tasa_mensual, flujo_neto.values)
        try:
            tir = npf.irr(flujo_neto.values) * 12 * 100
        except:
            tir = None


        resultado_neto = utilidad_neta  # Definirlo claramente

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VPN", f"${vpn:,.2f}")
        col2.metric("TIR", f"{tir:.2f}%" if tir is not None else "N/A")
        col3.metric("Capital Requerido", f"${capital_requerido:,.0f}")
        col4.metric("Margen Total", f"{(resultado_neto / total_ingresos * 100):.2f}%" if total_ingresos else "0.00%")





        st.markdown(f"📉 Mes con menor flujo: **{mes_min.strftime('%B %Y')}**  :${flujo_neto[mes_min]:,.0f}")
        st.markdown(f"📈 Mes con mayor flujo: **{mes_max.strftime('%B %Y')}**  : ${flujo_neto[mes_max]:,.0f}")

        # ========================
        # GRAFICO FINAL
        # ========================
        st.subheader("📊 Flujo Neto Acumulado")
        fig4 = go.Figure(go.Scatter(
            x=flujo_neto.index,
            y=flujo_neto_acumulado,
            mode='lines+markers',
            line=dict(shape='linear')
        ))
        fig4.update_layout(
            title="Flujo Neto Acumulado",
            yaxis_title="Monto ($)",
            xaxis_title="Mes"
        )
        st.plotly_chart(fig4, use_container_width=True)
      
        # 📊 Gráfico de flujo mensual por categoría con total neto
        st.subheader("📊 Flujo Mensual por Categoría (No Acumulado) con Total Neto")

        fig_mensual = go.Figure()

        # Barras por categoría
        for categoria in all_data['Categoría'].unique():
            df_cat = all_data[all_data['Categoría'] == categoria]
            fig_mensual.add_trace(go.Bar(
                x=df_cat['Month'],
                y=df_cat['Amount'],
                name=categoria
            ))

        # Línea del total neto mensual
        fig_mensual.add_trace(go.Scatter(
            x=total['Month'],
            y=total['Amount'],
            name="Total Neto",
            mode="lines+markers",
            line=dict(color="black", width=3, dash="dot")
        ))

        fig_mensual.update_layout(
            barmode='relative',
            title="Flujo Mensual por Categoría + Total Neto",
            xaxis_title="Mes",
            yaxis_title="Monto ($)",
            legend_title="Categoría / Total",
            height=500
        )

        st.plotly_chart(fig_mensual, use_container_width=True)





        st.header("💾 Guardar Proyecto")

        if st.button("📥 Descargar proyecto como JSON"):
            proyecto = {
                "costos": st.session_state.get("costos", []),
                "tipos_unidades": st.session_state.get("tipos_unidades", []),
                "flujos": [
                    {
                        "nombre": f["nombre"],
                        "categoria": f["categoria"],
                        "data": f["data"].copy().assign(Month=f["data"]["Month"].astype(str)).to_dict()
                    } for f in st.session_state.get("flujos", [])
                ],
                "flujos_base": st.session_state.get("flujos_base", {}),
                "unidades_disponibles": st.session_state.get("unidades_disponibles", 0),
                "m2_construibles": st.session_state.get("m2_construibles", 0),
                "precio_sugerido_m2": st.session_state.get("precio_sugerido_m2", 0),
                "venta_total": st.session_state.get("venta_total", 0),
                "costo_total": st.session_state.get("costo_total", 0),
            }

            json_bytes = json.dumps(proyecto).encode("utf-8")
            st.download_button("⬇️ Descargar JSON", data=json_bytes, file_name="proyecto_inmobiliario.json", mime="application/json")
