# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:43:02 2025

@author: olivi
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from modele_vehicule import simuler_vehicule_et_calculer_conso, distance, pos_x, pos_y, heading_interp

st.set_page_config(layout="wide")
st.title("Simulateur Shell Eco Marathon \U0001F697\U0001F4A8")

# --- Paramètres globaux de simulation ---
st.sidebar.header("Paramètres de simulation")
vent = st.sidebar.checkbox("Activer le vent", value=False, key="vent_checkbox")
vitesse_vent = st.sidebar.slider("Vitesse du vent (m/s)", 0.0, 10.0, 2.57, step=0.1, key="vent_vitesse")
angle_vent_deg = st.sidebar.slider("Angle du vent (degrés)", 0, 360, 135, key="vent_angle")
wind_angle_global = np.deg2rad(angle_vent_deg)
aero = st.sidebar.checkbox("Activer l'aérodynamique", value=True)
gravite = st.sidebar.checkbox("Activer la gravité", value=True)
enviolo = st.sidebar.checkbox("Utiliser Enviolo", value=True)
moteur_elec = st.sidebar.checkbox("Activer moteur électrique", value=False)

# --- Animation du vent ---
with st.expander("🛲️ Animation du vent autour du circuit"):
    wind_speed = vitesse_vent * 0.514
    C_wind = 0.5

    wind_vector = np.array([
        wind_speed * np.cos(wind_angle_global),
        wind_speed * np.sin(wind_angle_global)
    ])

    base_trace = [
        go.Scatter(x=pos_x, y=pos_y, mode="lines", line=dict(color="black"), name="Circuit", showlegend=True)
    ]

    frames = []
    for i in range(0, len(pos_x), max(len(pos_x) // 150, 1)):
        x = pos_x[i]
        y = pos_y[i]
        heading = heading_interp(distance[i] if not hasattr(distance, 'iloc') else distance.iloc[i])
        car_dir = np.array([np.cos(heading), np.sin(heading)])
        v_wind_along = np.dot(wind_vector, car_dir)
        eff_wind = v_wind_along * car_dir

        frames.append(go.Frame(name=str(i), data=base_trace + [
            go.Scatter(x=[x], y=[y], mode="markers", marker=dict(color="blue", size=10), name="Voiture"),
            go.Scatter(x=[x, x + wind_vector[0] * 5], y=[y, y + wind_vector[1] * 5],
                       mode="lines+markers", name="Vent global", line=dict(color="red", width=3)),
            go.Scatter(x=[x, x + eff_wind[0] * 5], y=[y, y + eff_wind[1] * 5],
                       mode="lines+markers", name="Vent (proj.)", line=dict(color="green", width=3)),
            go.Scatter(x=[x, x + car_dir[0] * 5], y=[y, y + car_dir[1] * 5],
                       mode="lines+markers", name="Cap voiture", line=dict(color="blue", dash="dot"))
        ]))

    initial_data = base_trace.copy()
    if frames:
        initial_data += frames[0].data

    fig_wind = go.Figure(
        data=initial_data,
        layout=go.Layout(
            title="Vecteurs de vent et cap de la voiture",
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)", scaleanchor="x", scaleratio=1),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), fromcurrent=True)])
                ]
            )]
        ),
        frames=frames
    )

    st.plotly_chart(fig_wind, use_container_width=True)

# --- Paramètres de simulation spécifiques ---
borne_min1 = st.sidebar.slider("Borne min phase 1", 0.0, 20.0, 0.0, step=0.1)
borne_max1 = st.sidebar.slider("Borne max phase 1", 0.0, 20.0, 8.39, step=0.1)
borne_min2 = st.sidebar.slider("Borne min phase 2", 0.0, 20.0, 5.8, step=0.1)
borne_max2 = st.sidebar.slider("Borne max phase 2", 0.0, 20.0, 7.7, step=0.1)
borne_min3 = st.sidebar.slider("Borne min phase 3", 0.0, 20.0, 7.0, step=0.1)
borne_max3 = st.sidebar.slider("Borne max phase 3", 0.0, 20.0, 7.3, step=0.1)
distance_totale = distance.iloc[-1]
temps_max = st.sidebar.slider("Temps max de simulation (s)", 100, 600, 228)

if st.button("🌟 Lancer la simulation"):
    st.info("Simulation en cours...")
    res = simuler_vehicule_et_calculer_conso(
        distance_totale=distance_totale,
        borne_min1=borne_min1,
        borne_max1=borne_max1,
        borne_min2=borne_min2,
        borne_max2=borne_max2,
        borne_min3=borne_min3,
        borne_max3=borne_max3,
        temps_max=temps_max,
        vent_active=vent,
        vitesse_vent=vitesse_vent,
        wind_angle_global=wind_angle_global,
        aero_active=aero,
        gravite_active=gravite,
        enviolo_on=enviolo,
        moteur_elec=moteur_elec
    )

    st.success("Simulation terminée ✅")
    t_vals, pos_vals, vit_vals, forces, regimes, conso_g, conso_ml, ratios, conso_inst = res

    st.metric("Consommation totale (g)", f"{conso_g:.2f}")
    st.metric("Consommation totale (ml)", f"{conso_ml:.2f}")

    # --- Graphiques Plotly ---
    st.subheader("📊 Données simulées")

    fig_vitesse = go.Figure()
    fig_vitesse.add_trace(go.Scatter(x=t_vals, y=vit_vals, name="Vitesse (m/s)", line=dict(color="green")))
    fig_vitesse.update_layout(title="Vitesse au cours du temps", xaxis_title="Temps (s)", yaxis_title="Vitesse (m/s)")
    st.plotly_chart(fig_vitesse, use_container_width=True)

    fig_position = go.Figure()
    fig_position.add_trace(go.Scatter(x=t_vals, y=pos_vals, name="Position (m)", line=dict(color="blue")))
    fig_position.update_layout(title="Position au cours du temps", xaxis_title="Temps (s)", yaxis_title="Position (m)")
    st.plotly_chart(fig_position, use_container_width=True)

    fig_forces = go.Figure()
    fig_forces.add_trace(go.Scatter(x=t_vals, y=forces["aero"], name="Aéro"))
    fig_forces.add_trace(go.Scatter(x=t_vals, y=forces["rolling"], name="Roulement"))
    fig_forces.add_trace(go.Scatter(x=t_vals, y=forces["gravity"], name="Gravité"))
    fig_forces.add_trace(go.Scatter(x=t_vals, y=forces["wind"], name="Vent"))
    fig_forces.update_layout(title="Forces appliquées", xaxis_title="Temps (s)", yaxis_title="Force (N)")
    st.plotly_chart(fig_forces, use_container_width=True)

    fig_rpm = go.Figure()
    fig_rpm.add_trace(go.Scatter(x=t_vals, y=regimes, name="Régime moteur (RPM)", line=dict(color="orange")))
    fig_rpm.update_layout(title="Régime moteur", xaxis_title="Temps (s)", yaxis_title="RPM")
    st.plotly_chart(fig_rpm, use_container_width=True)

    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(x=t_vals, y=ratios, name="Rapport Enviolo"))
    fig_ratio.update_layout(title="Rapport Enviolo", xaxis_title="Temps (s)", yaxis_title="Ratio")
    st.plotly_chart(fig_ratio, use_container_width=True)

    fig_motor_force = go.Figure()
    fig_motor_force.add_trace(go.Scatter(x=t_vals, y=forces["motor"], name="Force motrice"))
    fig_motor_force.update_layout(title="Force motrice", xaxis_title="Temps (s)", yaxis_title="Force (N)")
    st.plotly_chart(fig_motor_force, use_container_width=True)

    fig_conso = go.Figure()
    fig_conso.add_trace(go.Scatter(x=t_vals, y=conso_inst / 0.75, name="Consommation cumulative (ml)", line=dict(color="purple")))
    fig_conso.update_layout(title="Consommation en carburant", xaxis_title="Temps (s)", yaxis_title="ml")
    st.plotly_chart(fig_conso, use_container_width=True)

    # --- Comparaison avec données réelles ---
    try:
        lap_data = pd.read_csv("lap_4_data.csv")
        lap_data.columns = [col.lower() for col in lap_data.columns]
        lap_data.ffill(inplace=True)

        time_real = lap_data["lap_obc_timestamp"]
        velocity_real = lap_data["gps_speed"] / 3.6  # km/h -> m/s
        position_real = lap_data["lap_dist"]

        fig_compare_speed = go.Figure()
        fig_compare_speed.add_trace(go.Scatter(x=pos_vals, y=vit_vals, name="Vitesse simulée", line=dict(color="green")))
        fig_compare_speed.add_trace(go.Scatter(x=position_real, y=velocity_real, name="Vitesse réelle", line=dict(color="red", dash="dash")))
        fig_compare_speed.update_layout(title="Comparaison Vitesse Simulée vs Réelle",
                                        xaxis_title="Position (m)", yaxis_title="Vitesse (m/s)")
        st.plotly_chart(fig_compare_speed, use_container_width=True)

        fig_compare_pos = go.Figure()
        fig_compare_pos.add_trace(go.Scatter(x=t_vals, y=pos_vals, name="Position simulée", line=dict(color="green")))
        fig_compare_pos.add_trace(go.Scatter(x=time_real, y=position_real, name="Position réelle", line=dict(color="red", dash="dash")))
        fig_compare_pos.update_layout(title="Comparaison Position Simulée vs Réelle",
                                      xaxis_title="Temps (s)", yaxis_title="Position (m)")
        st.plotly_chart(fig_compare_pos, use_container_width=True)

    except Exception as e:
        st.warning(f"Données réelles non disponibles ou erreur lors du chargement : {e}")

