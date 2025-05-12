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
st.title("Simulateur Shell Eco Marathon")

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

# --- Animation du vent en Plotly ---
with st.expander("Animation du vent"):
    frames = []
    arrow_scale = 6
    for i in range(0, len(pos_x), max(len(pos_x)//60, 1)):
        x, y = pos_x[i], pos_y[i]
        heading = heading_interp(distance.iloc[i])
        car_dir = np.array([np.cos(heading), np.sin(heading)])
        wind_vector = np.array([
            vitesse_vent * 0.514 * np.cos(wind_angle_global),
            vitesse_vent * 0.514 * np.sin(wind_angle_global)
        ])
        v_proj = np.dot(wind_vector, car_dir)
        proj_vec = v_proj * car_dir
        data = [
            go.Scatter(x=pos_x, y=pos_y, mode="lines", name="Circuit", line=dict(color="black")),
            go.Scatter(x=[x, x + car_dir[0]*arrow_scale], y=[y, y + car_dir[1]*arrow_scale],
                       mode="lines+markers", name="Cap", line=dict(color="blue", width=3)),
            go.Scatter(x=[x, x + wind_vector[0]*arrow_scale], y=[y, y + wind_vector[1]*arrow_scale],
                       mode="lines+markers", name="Vent", line=dict(color="red", width=3)),
            go.Scatter(x=[x, x + proj_vec[0]*arrow_scale], y=[y, y + proj_vec[1]*arrow_scale],
                       mode="lines+markers", name="Vent proj.", line=dict(color="green", width=3)),
        ]
        frames.append(go.Frame(data=data, name=str(i)))


    fig_vec = go.Figure(data=frames[0].data, frames=frames)
    fig_vec.update_layout(
            title="Animation vecteurs : cap, vent, projection",
            xaxis=dict(title="X (m)"),
            yaxis=dict(title="Y (m)", scaleanchor="x", scaleratio=1),
            updatemenus=[dict(type="buttons",showactive=True,buttons=[
            dict(label="Play", method="animate",
             args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
            dict(label="Pause", method="animate",
             args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])
            ])]

    )
    st.plotly_chart(fig_vec, use_container_width=True)

# --- Paramètres de simulation spécifiques ---
borne_min1 = st.sidebar.slider("Borne min phase 1", 0.0, 20.0, 0.0, step=0.05, format="%.2f")
borne_max1 = st.sidebar.slider("Borne max phase 1", 0.0, 20.0, 8.39, step=0.05, format="%.2f")
borne_min2 = st.sidebar.slider("Borne min phase 2", 0.0, 20.0, 5.8, step=0.05, format="%.2f")
borne_max2 = st.sidebar.slider("Borne max phase 2", 0.0, 20.0, 7.7, step=0.05, format="%.2f")
borne_min3 = st.sidebar.slider("Borne min phase 3", 0.0, 20.0, 7.0, step=0.05, format="%.2f")
borne_max3 = st.sidebar.slider("Borne max phase 3", 0.0, 20.0, 7.3, step=0.05, format="%.2f")
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

    # --- Préchargement des données réelles ---

    # --- Comparaison avec données réelles ---
    try:
        lap_data = pd.read_csv("lap_4_data.csv")
        lap_data.columns = [col.lower() for col in lap_data.columns]
        lap_data.ffill(inplace=True)
      

# Première vraie valeur
        v_init = lap_data["gps_speed"].iloc[0] / 3.6  # m/s
        d_init = lap_data["lap_dist"].iloc[1]
        t_acc = 3 # secondes d'accélération

# Création des points interpolés de 0 à v_init
        n_interp = 10
        t_interp = np.linspace(0, t_acc, n_interp)
        v_interp = np.linspace(0, v_init, n_interp)
        d_interp = np.linspace(0, d_init, n_interp)

        interp_df = pd.DataFrame({
            "lap_obc_timestamp": t_interp,
            "gps_speed": v_interp * 3.6,  # on repasse en km/h
            "lap_dist": d_interp
        })

# Fusionner et réordonner
        lap_data = pd.concat([interp_df, lap_data], ignore_index=True)
        lap_data.sort_values("lap_obc_timestamp", inplace=True)
        lap_data.reset_index(drop=True, inplace=True)

# Extraire les colonnes finales
        time_real = lap_data["lap_obc_timestamp"].values
        velocity_real = lap_data["gps_speed"].values / 3.6  # km/h → m/s
        position_real = lap_data["lap_dist"].values

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

from scipy.interpolate import interp1d

with st.expander("🚗 Animation : Simulé vs Réel en fonction des vitesses"):

        try:
            lap_data = pd.read_csv("lap_4_data.csv")
            lap_data.columns = [col.lower() for col in lap_data.columns]
            lap_data.ffill(inplace=True)
      

# Première vraie valeur
            v_init = lap_data["gps_speed"].iloc[0] / 3.6  # m/s
            d_init = lap_data["lap_dist"].iloc[1]
            t_acc = 3  # secondes d'accélération

# Création des points interpolés de 0 à v_init
            n_interp = 40
            t_interp = np.linspace(0, t_acc, n_interp)
            v_interp = np.linspace(0, v_init, n_interp)
            d_interp = np.linspace(0, d_init, n_interp)

            interp_df = pd.DataFrame({
                "lap_obc_timestamp": t_interp,
                "gps_speed": v_interp * 3.6,  # on repasse en km/h
                "lap_dist": d_interp
            })

# Fusionner et réordonner
            lap_data = pd.concat([interp_df, lap_data], ignore_index=True)
            lap_data.sort_values("lap_obc_timestamp", inplace=True)
            lap_data.reset_index(drop=True, inplace=True)

# Extraire les colonnes finales
            time_real = lap_data["lap_obc_timestamp"].values
            velocity_real = lap_data["gps_speed"].values / 3.6  # km/h → m/s
            position_real = lap_data["lap_dist"].values

        # Interpolateurs x/y en fonction de la distance
            interp_x = interp1d(distance, pos_x, kind='linear', fill_value="extrapolate")
            interp_y = interp1d(distance, pos_y, kind='linear', fill_value="extrapolate")

        # Interpolation de distance(t)
            interp_sim_dist = interp1d(t_vals, pos_vals, bounds_error=False, fill_value="extrapolate")
            interp_real_dist = interp1d(time_real, position_real, bounds_error=False, fill_value="extrapolate")

            common_times = np.linspace(0, min(t_vals[-1], time_real[-1]), 150)
            frames = []

            for t in common_times:
            # Simulation
                d_sim = interp_sim_dist(t)
                x_sim, y_sim = interp_x(d_sim), interp_y(d_sim)

            # Réel
                d_real = interp_real_dist(t)
                x_real, y_real = interp_x(d_real), interp_y(d_real)

                frames.append(go.Frame(data=[
                    go.Scatter(x=pos_x, y=pos_y, mode="lines", line=dict(color="black"), name="Circuit"),
                    go.Scatter(x=[x_sim], y=[y_sim], mode="markers", marker=dict(color="green", size=12), name="Simulation"),
                    go.Scatter(x=[x_real], y=[y_real], mode="markers", marker=dict(color="red", size=12), name="Réel")
                ], name=str(round(t, 1))))

        # Initialisation
            fig_anim = go.Figure(
                data=frames[0].data,
                frames=frames
            )

            fig_anim.update_layout(
                title="Animation : Véhicule simulé vs réel (vitesse indépendante)",
                xaxis=dict(title="X (m)"),
                yaxis=dict(title="Y (m)", scaleanchor="x", scaleratio=1),
                updatemenus=[dict(
                    type="buttons",
                    showactive=True,
                    buttons=[
                        dict(label="▶️ Play", method="animate",
                             args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
                    ]
                )],
                    sliders=[dict(
                    steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0))],
                                label=f.name) for f in frames],
                    transition=dict(duration=0),
                    x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Temps (s): ", visible=True),
                    len=1.0
                )]
            )

            st.plotly_chart(fig_anim, use_container_width=True)

        except Exception as e:
            st.warning(f"Erreur lors de l'animation comparative : {e}")

with st.expander("🌍 Animation 3D du circuit"):

        try:
        # Charger les coordonnées 3D avec altitude
            df_3d = pd.read_csv("xyz_coordinates_lap4.csv")
            df_3d.columns = [col.lower() for col in df_3d.columns]

        # Interpolateurs 3D en fonction de la distance cumulée
            interp_x_3d = interp1d(df_3d["cumulative_distance"], df_3d["x"], kind='linear', fill_value="extrapolate")
            interp_y_3d = interp1d(df_3d["cumulative_distance"], df_3d["y"], kind='linear', fill_value="extrapolate")
            interp_z_3d = interp1d(df_3d["cumulative_distance"], df_3d["z"], kind='linear', fill_value="extrapolate")

        # Interpolation distance(t)
            interp_sim_dist = interp1d(t_vals, pos_vals, bounds_error=False, fill_value="extrapolate")
            interp_real_dist = interp1d(time_real, position_real, bounds_error=False, fill_value="extrapolate")

            common_times = np.linspace(0, min(t_vals[-1], time_real[-1]), 150)
            frames = []

            for t in common_times:
            # Véhicule simulé
                alt_factor = 5

                d_sim = interp_sim_dist(t)
                x_sim, y_sim, z_sim = interp_x_3d(d_sim), interp_y_3d(d_sim), interp_z_3d(d_sim)
                
            # Véhicule réel
                d_real = interp_real_dist(t)
                x_real, y_real, z_real = interp_x_3d(d_real), interp_y_3d(d_real), interp_z_3d(d_real)
                z_sim = interp_z_3d(d_sim) * alt_factor
                z_real = interp_z_3d(d_real) * alt_factor
                frames.append(go.Frame(data=[
                    go.Scatter3d(
                        x=df_3d["x"], y=df_3d["y"], z=df_3d["z"]*alt_factor,
                        mode="lines", line=dict(color="black", width=4), name="Circuit"
                    ),
                    go.Scatter3d(
                        x=[x_sim], y=[y_sim], z=[z_sim],
                        mode="markers", marker=dict(color="green", size=6), name="Simulation"
                    ),
                    go.Scatter3d(
                        x=[x_real], y=[y_real], z=[z_real],
                        mode="markers", marker=dict(color="red", size=6), name="Réel"
                    )
                ], name=str(round(t, 1))))

        # Graphique 3D avec animation
            fig_3d = go.Figure(data=frames[0].data, frames=frames)
            fig_3d.update_layout(
                title="Animation 3D : Simulé vs Réel",
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Altitude (m)',
                    aspectmode='data'
                ),
                updatemenus=[dict(
                    type="buttons", showactive=True,
                    buttons=[
                        dict(label="▶️ Play", method="animate",
                             args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])
                    ]
                )],
                    sliders=[dict(
                        steps=[dict(method="animate", args=[[f.name], dict(mode="immediate", frame=dict(duration=0))],
                            label=f.name) for f in frames],
                        transition=dict(duration=0),
                        x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Temps (s): ", visible=True),
                        len=1.0
                    )]
            )

            st.plotly_chart(fig_3d, use_container_width=True)

        except Exception as e:
            st.warning(f"Erreur lors de l'animation 3D : {e}")



