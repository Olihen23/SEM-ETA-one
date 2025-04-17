# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 09:40:43 2025

@author: olivi
"""

# Ce fichier est une version nettoyée du script de simulation pour être utilisé comme module
# Toutes les fonctions, classes et interpolateurs sont gardés, mais le code exécuté en direct est supprimé

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Chargement et interpolation des données moteur thermique
moteur_consomini = pd.read_excel(r"Mesures moteur Consomini.xlsx", "Moteur consomini")
moteur_consomini.columns = [col.lower() for col in moteur_consomini.columns]
moteur_consomini.ffill(inplace=True)
moteur_consomini = moteur_consomini.dropna(subset=["n moteur", "m corr"])
moteur_consomini["n moteur"] = pd.to_numeric(moteur_consomini["n moteur"], errors="coerce")
moteur_consomini = moteur_consomini.dropna(subset=["n moteur"])

new_points = pd.DataFrame({
    "n moteur": [0, 1, 1489],
    "m corr": [2.37, 2.37, 2.37],
    "csp": [301, 301, 301]
})
additional_points = pd.DataFrame({
    "n moteur": np.linspace(1, 1489, 1488),
    "m corr": [2.37] * 1488,
    "csp": [301] * 1488
})
moteur_consomini = pd.concat([new_points, additional_points, moteur_consomini], ignore_index=True)
moteur_consomini = moteur_consomini.groupby("n moteur", as_index=False).mean().sort_values("n moteur")

x = moteur_consomini["n moteur"]
f_interpol_couple = interp1d(x, moteur_consomini["m corr"] * 0.61, kind="cubic", fill_value="extrapolate")
f_interpol_csp = interp1d(x, moteur_consomini["csp"], kind="linear", fill_value="extrapolate")

# Moteur électrique
data_elec = pd.read_excel("Caracterisation Moteur.xlsx", "Test moteur")
data_elec.columns = [col.lower() for col in data_elec.columns]
data_elec.ffill(inplace=True)
data_elec.dropna(subset=["couple", "vitesse rotor"], inplace=True)
data_elec = data_elec.groupby("vitesse rotor", as_index=False).mean()

x_elec = data_elec["vitesse rotor"]
f_interpol_couple_elec = interp1d(x_elec, data_elec["couple"], kind="linear", fill_value="extrapolate")
f_interpol_P_elec = interp1d(x_elec, data_elec["pelec calc"], kind="linear", fill_value="extrapolate")

# Enviolo
data_enviolo = pd.read_excel("calcul enviolo-reel-26-04.xlsx", skiprows=12)
data_enviolo.columns = [col.lower() for col in data_enviolo.columns]
data_enviolo.ffill(inplace=True)
data_enviolo = data_enviolo.dropna(subset=["rapport enviolo", "vitesse moteur"])
data_enviolo = pd.concat([
    pd.DataFrame({"vitesse moteur": [0, 1, 1499], "rapport enviolo": [0.5, 0.5, 0.5]}),
    pd.DataFrame({"vitesse moteur": np.linspace(0, 1499, 1498), "rapport enviolo": [0.5]*1498}),
    data_enviolo
], ignore_index=True)
data_enviolo = data_enviolo.groupby("vitesse moteur", as_index=False).mean()

x_env = data_enviolo["vitesse moteur"]
y_env = data_enviolo["rapport enviolo"]
f_interp_env = interp1d(x_env, y_env, kind="linear", fill_value="extrapolate")

# Piste
xyz = pd.read_csv('xyz_coordinates_lap4.csv')
xyz.columns = [col.lower() for col in xyz.columns]
distance = xyz["cumulative_distance"]
altitude = xyz["z"]
pos_x = xyz["x"]
pos_y = xyz["y"]

heading = np.arctan2(np.gradient(pos_y), np.gradient(pos_x))
heading_interp = interp1d(distance, heading, fill_value="extrapolate")

rayon = pd.read_csv("xy_rayon_de_courbure.csv")
rayon.columns = [col.lower() for col in rayon.columns]
distance_2 = rayon["cumulative_distance_interp"]

angle_1 = pd.read_csv("angle_rayon.csv")
angle_1.columns = [col.lower() for col in angle_1.columns]

# --- Classes, fonctions et simulation principale ---
from modele_vehicule_simulation_core import ParametresVehicule, ProfilPiste, Moteur, DynamiqueVehicule, simuler_vehicule_et_calculer_conso

# Aucun code exécuté directement ici

