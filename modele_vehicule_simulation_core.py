# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:36:49 2025

@author: olivi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# =============================================================================
# 1) Importation et interpolation des données moteur
# =============================================================================

moteur_consomini = pd.read_excel("Mesures moteur Consomini.xlsx", sheet_name="Moteur consomini")
moteur_consomini.columns = [col.lower() for col in moteur_consomini.columns]
moteur_consomini.reset_index(drop=True, inplace=True)
moteur_consomini.ffill(inplace=True)

moteur_consomini["n moteur"] = pd.to_numeric(moteur_consomini["n moteur"], errors="coerce")
moteur_consomini["m corr"] = pd.to_numeric(moteur_consomini["m corr"], errors="coerce")
moteur_consomini["csp"] = pd.to_numeric(moteur_consomini["csp"], errors="coerce")
moteur_consomini.dropna(subset=["n moteur", "m corr"], inplace=True)

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
moteur_consomini = pd.concat([new_points, additional_points, moteur_consomini],
                             ignore_index=True).sort_values("n moteur").reset_index(drop=True)
moteur_consomini = moteur_consomini.groupby("n moteur", as_index=False).mean()
#*0.61*1.06
x = moteur_consomini["n moteur"]
f_interpol_couple = interp1d(x, moteur_consomini["m corr"]*0.61*1.06, kind="cubic", fill_value="extrapolate")
f_interpol_csp = interp1d(x, moteur_consomini["csp"]*1.55, kind="linear", fill_value="extrapolate")
# =============================================================================
# 2) Importation des moteur électrique 
# =============================================================================
moteur_elec = pd.read_excel("Caracterisation Moteur.xlsx", "Test moteur")
moteur_elec.columns = [col.lower() for col in moteur_elec.columns]
moteur_elec.reset_index(drop=True, inplace=True)
moteur_elec.ffill(inplace=True)

# Convertir les colonnes "couple" et "vitesse rotor" en numérique 
moteur_elec["couple"] = pd.to_numeric(moteur_elec["couple"], errors="coerce")
moteur_elec["vitesse rotor"] = pd.to_numeric(moteur_elec["vitesse rotor"], errors="coerce")
#moteur_elec[""] = pd.to_numeric(moteur_elec["vitesse rotor"], errors="coerce")
moteur_elec.dropna(subset=["couple", "vitesse rotor"], inplace=True)

# Suppression ou agrégation des doublons sur la colonne "couple"
cols_numeric = moteur_elec.select_dtypes(include='number').columns
moteur_elec = moteur_elec.groupby("couple", as_index=False)[cols_numeric].mean()
cols_num = data_elec.select_dtypes(include='number').columns
data_elec = data_elec.groupby("vitesse rotor", as_index=False)[cols_num].mean()

# Définir x_elec comme étant les valeurs de "couple"
x_elec = moteur_elec["vitesse rotor"]

# Création de l'interpolateur pour le moteur électrique en utilisant "couple" comme variable indépendante
f_interpol_couple_elec = interp1d(x_elec, moteur_elec["couple"], kind="linear", fill_value="extrapolate")
f_interpol_P_elec=interp1d(x_elec,moteur_elec["pelec calc"],kind="linear",fill_value="extrapolate")

# =============================================================================
# 3) Importation des données enviolo
# =============================================================================
data_enviolo = pd.read_excel("calcul enviolo-reel-26-04.xlsx", skiprows=12)
data_enviolo.columns = [col.lower() for col in data_enviolo.columns]
data_enviolo.reset_index(drop=True, inplace=True)
data_enviolo.ffill(inplace=True)
#rendement_enviolo=pd.read_excel("Caractérisation enviolo")

data_enviolo["rapport enviolo"] = pd.to_numeric(data_enviolo["rapport enviolo"], errors="coerce")
data_enviolo["vitesse moteur"] = pd.to_numeric(data_enviolo["vitesse moteur"], errors="coerce")
#rendement=rendement_enviolo["Rendement P"]

#f_interp_rendement=interp1d(rendement,rendement_enviolo["SpeedIn"])
new_points = pd.DataFrame({
    "vitesse moteur": [0, 1, 1499],
    "rapport enviolo": [0.5, 0.5, 0.5]
})
additional_points = pd.DataFrame({
    "vitesse moteur": np.linspace(0, 1499, 1498),
    "rapport enviolo": [0.5] * 1498
})
data_enviolo = pd.concat([new_points, additional_points, data_enviolo],
                         ignore_index=True).sort_values("vitesse moteur").reset_index(drop=True)
data_enviolo = data_enviolo.groupby("vitesse moteur", as_index=False).mean()
x_env = data_enviolo["vitesse moteur"]
y_env = data_enviolo["rapport enviolo"]

f_interp_env = interp1d(x_env, y_env, kind="linear", fill_value="extrapolate")

# =============================================================================
# 4) Chargement des coordonnées de la piste
# =============================================================================

xyz = pd.read_csv('xyz_coordinates_lap4.csv')
xyz.columns = [col.lower() for col in xyz.columns]
distance = xyz["cumulative_distance"]
altitude = xyz["z"]
pos_x = xyz["x"]
pos_y = xyz["y"]

heading = np.arctan2(np.gradient(pos_y), np.gradient(pos_x))
heading_interp = interp1d(distance, heading, fill_value="extrapolate")

rayon=pd.read_csv("xy_rayon_de_courbure.csv")
rayon.columns = [col.lower() for col in rayon.columns]
rayon_courb=rayon["radius_of_curvature_polyfit_interp"]
distance_2=rayon["cumulative_distance_interp"]
angle_1=pd.read_csv("angle_rayon.csv")
angle_1.columns = [col.lower() for col in angle_1.columns]
rayon_1=angle_1["rayon"]
angle_ray=angle_1["angle"]
# =============================================================================
# 5)Chargement du Cx variable
# =============================================================================
data_cx=pd.read_excel("Cx_voiture_vent.xlsx",skiprows=3)
data_cx.columns = [col.lower() for col in data_cx.columns]
angle=data_cx["angle (°)"]
Cx=data_cx["cx (-)"]
f_interp_cx = interp1d(angle,Cx, kind="linear", fill_value="extrapolate")



# =============================================================================
# 6) Paramètres embrayage dynamique
# =============================================================================

m_mass   = 0.0774       # Masse d'une masselotte [kg]
r_mass   = 0.0316       # Rayon effectif [m]
N_mass   = 4            # Nombre de masselottes
mu_clutch = 0.3         # Coeff de frottement
R_cloche = 0.045        # Bras de levier
F_ressort =28.23       # Force ressort
masse = 217
rapport_chaine1 = 95/11
rapport_chaine2 = 2.4
rayon_roue = 0.279


# =============================================================================
# 7) Classes réutilisables
# =============================================================================

class ParametresVehicule:
    def __init__(self):
        self.resistance_roulement = 0.0013
        self.coefficient_trainee = 0.194
        self.surface_frontale = 0.789
        self.masse = 217
        self.rayon_roue = 0.279
        self.rendement_chaine1 = 0.97
        self.rendement_chaine2 = 0.97
        self.rendement_transmission = 0.83
        self.rapport_chaine1 = 95/11
        self.rapport_chaine2 = 2.4
        self.g = 9.81
        self.densite_air = 1.225
        


class ProfilPiste:
    def __init__(self, distance, altitude, rayon_courbure=None,angle_courbure=None):
        self.distance = distance
        self.altitude = altitude
        self.interpolateur_pente = self.calculer_pente()
        # Si les données du rayon sont fournies, créer un interpolateur
        if rayon_courbure is not None:
            self.interpolateur_rayon = interp1d(distance_2, rayon_courbure, fill_value="extrapolate")
        else:
            self.interpolateur_rayon = None
        if angle_courbure is not None:
            self.interpolateur_angle=interp1d(rayon_1,angle_courbure,fill_value="extrapolate")
        else:
            self.interpolateur_angle= None

    def calculer_pente(self):
        delta_distance = np.diff(self.distance)
        delta_altitude = np.diff(self.altitude)
        angle_pente = np.arctan(delta_altitude / delta_distance)
        return interp1d(self.distance[:-1], angle_pente, fill_value="extrapolate")

    def obtenir_pente(self, position):
        return self.interpolateur_pente(position)
    
    def obtenir_rayon(self, position):
        if self.interpolateur_rayon is not None:
            return self.interpolateur_rayon(position)
        else:
            # Si aucune donnée n'est disponible, on considère une route en ligne droite (R=∞)
            return np.inf
    def obtenir_angle(self,rayon):
        if self.interpolateur_angle is not None:
            return self.interpolateur_angle(rayon)
        else:
            return np.inf

class Aerodynamique:
    def __init__(self, surface_frontale, densite_air, f_interp_cx):
        self.S = surface_frontale
        self.rho = densite_air
        self.f_interp_cx = f_interp_cx
        
class Moteur:
    def __init__(self, f_interpol_couple, f_interp_env):
        self.f_interpol_couple = f_interpol_couple
        self.f_interp_env = f_interp_env


class DynamiqueVehicule:
    def __init__(self, params, piste, propulsion,
                 vent_active=False, vitesse_vent=0,
                 aero_active=True, gravite_active=True, wind_angle_global=0):
        self.params = params
        self.piste = piste
        self.propulsion = propulsion
        self.vent_active = vent_active
        self.aero_active = aero_active
        self.gravite_active = gravite_active
        self.vitesse_vent=vitesse_vent
        self.wind_angle_global=wind_angle_global

        
        
    def calculer_forces(self, position, vitesse, couple_moteur,
                        ratio_enviolo, couple_transmis=None):
        if couple_transmis is not None:
            F_moteur = (
                couple_transmis
                * self.params.rapport_chaine1
                * self.params.rapport_chaine2
                / ratio_enviolo
                * self.params.rendement_chaine1
                * self.params.rendement_chaine2
                * self.params.rendement_transmission
            ) / self.params.rayon_roue
        else:
            F_moteur = 0
            if couple_moteur > 0:
                F_moteur = (
                    couple_moteur
                    * self.params.rapport_chaine1
                    * self.params.rapport_chaine2
                    / ratio_enviolo
                    * self.params.rendement_chaine1
                    * self.params.rendement_chaine2
                    * self.params.rendement_transmission
                ) / self.params.rayon_roue

        if self.aero_active:
            vitesse_relative = vitesse
            F_aero = 0.5 * self.params.coefficient_trainee * self.params.surface_frontale * self.params.densite_air * (vitesse_relative**2)
        else:
            F_aero = 0

        F_roulement = self.params.resistance_roulement * self.params.masse * self.params.g

        if self.gravite_active:
            angle_pente = self.piste.obtenir_pente(position)
            F_gravite = self.params.masse * self.params.g * np.sin(angle_pente)
        else:
            F_gravite = 0
            
        if self.vent_active:
            # cap du véhicule
            heading = heading_interp(position)

    # vent global
            vx = self.vitesse_vent * np.cos(self.wind_angle_global)
            vy = self.vitesse_vent * np.sin(self.wind_angle_global)

    # projection du vent sur l'axe longitudinal
            v_wind_along = vx * np.cos(heading) + vy * np.sin(heading)

    # angle entre vent et cap (rad → °)
            phi = np.arctan2(vy, vx) - heading
            phi = (phi + np.pi) % (2*np.pi) - np.pi
            angle_rel_deg = abs(np.degrees(phi))

    # Cx spécifique au vent venant de cet angle
            Cx_wind = float(f_interp_cx(angle_rel_deg))


    # calcul de la force de traînée du vent
            F_wind = 0.5 * Cx_wind \
               * self.params.surface_frontale \
               * self.params.densite_air \
               * (v_wind_along**2) \
               * np.sign(v_wind_along)
        else:
                F_wind = 0.0


        return F_moteur, F_aero, F_roulement, F_gravite, F_wind

# =============================================================================
# 8) Simulation principale 
# =============================================================================

def simuler_vehicule_et_calculer_conso(distance_totale,
                                       borne_min1, borne_max1,
                                       borne_min2, borne_max2,
                                       borne_min3, borne_max3,
                                       temps_max=500,
                                       vent_active=False,
                                       vitesse_vent=0,
                                       wind_angle_global=0,
                                       aero_active=True,
                                       gravite_active=True,
                                       enviolo_on=True,
                                       moteur_elec=False,
                                       coef_aero=1.6,
                                       coef_roul=2,
                                       plot=True
                                       ):

    params = ParametresVehicule()
    piste = ProfilPiste(distance=distance, altitude=altitude,
                        rayon_courbure=rayon["radius_of_curvature_polyfit_interp"],
                        angle_courbure=angle_1["angle"])
    propulsion = Moteur(f_interpol_couple, f_interp_env)
    vehicule = DynamiqueVehicule(params, piste, propulsion,
                                 vent_active, vitesse_vent,wind_angle_global,
                                 aero_active, gravite_active)

    # Listes pour sauvegarder les valeurs pour le tracé
    forces_temp = {"motor": [], "aero": [], "rolling": [], "gravity": [],"wind":[],"elec":[]}
    regimes_moteur_temp = []
    regime_mot_elec_temp=[]
    ratios_utilises_temp = []
    forces_calculees_temps = []

    densite_essence = 0.75
    phase = 1
    pente_passe = False
    moteur_actif = False
    moteur_elec_actif=False
    

    def equations_dynamiques(t, etat):
        nonlocal moteur_actif, phase, pente_passe, moteur_elec_actif
        pos, v, omega_m, conso = etat

    # Vérification de la condition terminale
        if pos >= distance_totale:
            F_moteur = 0.0
            F_aero = 0.0
            F_roulement = 0.0
            F_gravite = 0.0
            rpm_moteur = 0
            rpm_mot_elec=0
            ratio_enviolo = 0
            F_wind=0
        # Mettre à jour toutes les listes simultanément
            forces_calculees_temps.append(t)
            regimes_moteur_temp.append(rpm_moteur)
            ratios_utilises_temp.append(ratio_enviolo)
            regime_mot_elec_temp.append(rpm_mot_elec)
            for key in forces_temp:
                forces_temp[key].append(0.0)
            return [0, 0, 0, 0]

        # Sinon, dans le déroulé normal :
        forces_calculees_temps.append(t)
    
        if pos > 600 and not pente_passe:
            pente_passe = True
            phase = 4

        if not pente_passe:
            if phase == 1:
                if v <= borne_min1:
                    moteur_actif = True
                    moteur_elec_actif=True
                elif v >= borne_max1:
                    moteur_actif = False
                    moteur_elec_actif=False
                    phase = 2
            elif phase == 2:
                if v <= borne_min2:
                    moteur_actif = True
                    moteur_elec_actif=True
                    phase = 3
            elif phase == 3:
                if v >= borne_max2:
                    moteur_actif = False
                    moteur_elec_actif=False
        else:
            if phase == 4:
                if v <= borne_min3:
                    moteur_actif = True
                    moteur_elec_actif=True
                    phase = 5
            elif phase == 5:
                if v >= borne_max3:
                    moteur_actif = False
                    moteur_elec_actif=False
        roue_libre_active=False

    # 1) Calcul du couple moteur
        if moteur_actif:
            rpm_moteur_nor= omega_m * 60/(2*np.pi)
            rpm_moteur=rpm_moteur_nor
            couple_moteur = f_interpol_couple(rpm_moteur)
        else:
            rpm_moteur = 0
            couple_moteur = 0.0
        if moteur_elec and moteur_elec_actif:
            omega_roue=v/params.rayon_roue if v>0 else 0
            rpm_roue=max(omega_roue*60/(2*np.pi),0)
            rpm_mot_elec=rpm_roue*1.5
            couple_electrique = f_interpol_couple_elec(rpm_mot_elec)
        else:
            couple_electrique=0
            rpm_mot_elec=0

    # 2) Calcul du couple embrayage max (positif)
        F_centrifuge = m_mass * (omega_m**2) * r_mass
        embrayage_actif = (F_centrifuge > F_ressort)
        if embrayage_actif:
            C_embre_max = N_mass * (F_centrifuge - F_ressort) * mu_clutch * R_cloche
        else:
            C_embre_max = 0.0

    # 3) Détermination du slip et calcul d'un "couple brut"
        if enviolo_on:
            ratio_enviolo = f_interp_env(rpm_moteur)
            
        else:
            ratio_enviolo=1
        effective_ratio = (rapport_chaine1 * params.rendement_chaine1 *
                       (1/ratio_enviolo) * params.rendement_transmission *
                       rapport_chaine2 * params.rendement_chaine2)
        omega_cloche = (v * effective_ratio)/params.rayon_roue if v > 0 else 0
        slip = omega_m - omega_cloche
        amplitude_transmis = min(couple_moteur, C_embre_max)

        if slip > 0:
            C_transmis_brut = amplitude_transmis
        else:
            C_transmis_brut = 0.0
        
        
        F_moteur, F_aero, F_roulement, F_gravite, F_wind = vehicule.calculer_forces(
            pos, v, couple_moteur, ratio_enviolo, couple_transmis=C_transmis_brut
             )
        
        if moteur_elec_actif:
            F_elec=couple_electrique*1.5/params.rayon_roue
        else:
            F_elec=0.0
    # 4) Détermination de l'état transmission/roue libre et calcul de domega_m_dt
        if not moteur_actif and not moteur_elec_actif and v > 1.0:
            roue_libre_active = True
            C_transmis_brut = 0.0 
            couple_electrique=0
            
#coefficient multiplicatif après la pente
            if roue_libre_active:
                F_aero_modif=coef_aero*(F_aero)
                F_roulement_modif=coef_roul*F_roulement
                dv_dt = (-F_roulement_modif-F_aero_modif-F_gravite + F_wind) / params.masse
            omega_idle = 10  # Aucun couple transmis
    # On fixe domega_m_dt à zéro pour éviter toute influence sur le calcul des forces
            domega_m_dt = -20*(omega_m-omega_idle)
        
        else:
            roue_libre_active = False
            dv_dt = (F_moteur+F_elec - F_aero - F_roulement - F_gravite+F_wind) / params.masse
            if not moteur_actif:
                domega_m_dt=-0.1*omega_m
            domega_m_dt = (couple_moteur - C_transmis_brut) / (0.00201 + masse*(rayon_roue**2)/((rapport_chaine1*rapport_chaine2)/ratio_enviolo)**2)

# 5) Calcul unique des forces sur le véhicule
# On utilise la même fonction dans tous les cas, avec des valeurs de couple adaptées

        dpos_dt = v
        rayon_courbe = piste.obtenir_rayon(pos)
        
        """
        if np.isfinite(rayon_courbe):
                mu_lat = 3  # coefficient de friction latérale (à calibrer)
                v_max_virage = np.sqrt(mu_lat * params.g * rayon_courbe)
        if v > v_max_virage:
            k_frott =0.7  # constante de calibrage pour la décélération additionnelle
            extra_decel = k_frott * (v - v_max_virage)
            dv_dt -= extra_decel
            """
    # 6) Calcul de la consommation
        if rpm_moteur > 0 and moteur_actif:
            puissance_meca = (rpm_moteur * couple_moteur * 2*np.pi/60)
            #puissance_vir=dpos_dt*F_roulement*np.sin(angle_courb)
            if moteur_elec:  
                puissance_elec=f_interpol_P_elec(rpm_mot_elec)
            else: 
                puissance_elec=0
            puissance_tot=puissance_elec+puissance_meca
            csp = f_interpol_csp(min(rpm_moteur, max(x)))
            
        else:
            puissance_meca = 0
            csp = 0
        dconso_dt = (puissance_tot/1000) * (csp/3600) if moteur_actif else 0

        regimes_moteur_temp.append(rpm_moteur)
        ratios_utilises_temp.append(ratio_enviolo)
        if moteur_elec:
            regime_mot_elec_temp.append(rpm_mot_elec)
        forces_temp["motor"].append(F_moteur)
        forces_temp["aero"].append(F_aero)
        forces_temp["rolling"].append(F_roulement)
        forces_temp["gravity"].append(F_gravite)
        forces_temp["wind"].append(F_wind)
        forces_temp["elec"].append(F_elec)

        return [dpos_dt, dv_dt, domega_m_dt, dconso_dt]
    

    # Condition initiale : [position, vitesse, ω_moteur, consommation]
    solution = solve_ivp(
        equations_dynamiques,
        [0, temps_max],
        [0, 0, 109.96/1.8, 0],
        method='RK45',
        events=[lambda t, y: y[0] - distance_totale, lambda t, y: t - temps_max],
        max_step=0.005
    )

    # Interpolation des forces, régimes et ratios sur la grille solution.t
    forces = {}
    for key in forces_temp:
        f_interp = interp1d(forces_calculees_temps, forces_temp[key],
                             bounds_error=False, fill_value='extrapolate')
        forces[key] = f_interp(solution.t)

    f_interp_regime = interp1d(forces_calculees_temps, regimes_moteur_temp,
                               bounds_error=False, fill_value='extrapolate')
    regimes_interp = f_interp_regime(solution.t)
    if moteur_elec:
        f_interp_regime_elec=interp1d(forces_calculees_temps, regime_mot_elec_temp,
                                  bounds_error=False, fill_value='extrapolate')
        regime_elec_interp = f_interp_regime_elec(solution.t)
    f_interp_ratio = interp1d(forces_calculees_temps, ratios_utilises_temp,
                              bounds_error=False, fill_value='extrapolate')
    ratios_interp = f_interp_ratio(solution.t)

    # La consommation totale intégrée (en grammes)
    conso_totale = solution.y[3][-1]
    conso_totale_ml = conso_totale / densite_essence

    print(f"\nConsommation totale de carburant : {conso_totale:.2f} g")
    print(f"Consommation totale de carburant : {conso_totale_ml:.2f} ml")

    if moteur_elec and plot: 
        plot_results(solution.t, solution.y[0], solution.y[1],
                     forces, regimes_interp, ratios_interp,
                     solution.y[3],regime_elec_interp)
        
        return (solution.t, solution.y[0], solution.y[1],
                forces, regimes_interp, conso_totale, conso_totale_ml,
                ratios_interp, solution.y[3],regime_elec_interp)
    elif plot :
        plot_results(solution.t, solution.y[0], solution.y[1],
                     forces, regimes_interp, ratios_interp,
                     solution.y[3])
        
        return (solution.t, solution.y[0], solution.y[1],
                forces, regimes_interp, conso_totale, conso_totale_ml,
                ratios_interp, solution.y[3]) 

# =============================================================================
# 9) Fonction de tracé des résultats
# =============================================================================

def plot_results(t_eval, position, vitesse, forces,
                 regimes_moteur, ratios_utilises, consommation):#regime_mot_elec):
    fig, axs = plt.subplots(4, 2, figsize=(20, 20))

    # Position
    axs[0,0].plot(t_eval, position, label="Position (m)")
    axs[0,0].set_title("Position en fonction du temps")
    axs[0,0].set_xlabel("Temps (s)")
    axs[0,0].set_ylabel("Position (m)")
    axs[0,0].legend()

    # Vitesse
    axs[0,1].plot(t_eval, vitesse, label="Vitesse (m/s)", color="green")
    axs[0,1].set_title("Vitesse en fonction du temps")
    axs[0,1].set_xlabel("Temps (s)")
    axs[0,1].set_ylabel("Vitesse (m/s)")
    axs[0,1].legend()

    # Forces aérodynamique, de roulement et gravitationnelle
    axs[1,0].plot(t_eval, forces["aero"], label="Force aérodynamique (N)")
    axs[1,0].plot(t_eval, forces["rolling"], label="Force de roulement (N)")
    axs[1,0].plot(t_eval, forces["gravity"], label="Force gravitationnelle (N)")
    axs[1,0].plot(t_eval, forces["wind"], label="Force du vent (N)")
    #axs[1,0].plot(t_eval,forces["elec"],label="Force moteur electrique (N)")
    axs[1,0].set_title("Forces appliquées au véhicule")
    axs[1,0].set_xlabel("Temps (s)")
    axs[1,0].set_ylabel("Force (N)")
    axs[1,0].legend()

    # Régime moteur
    axs[1,1].plot(t_eval, regimes_moteur, label="Régime moteur (RPM)", color="orange")
    axs[1,1].set_title("Régime moteur en fonction du temps")
    axs[1,1].set_xlabel("Temps (s)")
    axs[1,1].set_ylabel("RPM")
    axs[1,1].legend()

    # Rapport Enviolo
    axs[2,0].plot(t_eval, ratios_utilises, label="Rapport Enviolo")
    axs[2,0].set_title("Rapport Enviolo en fonction du temps")
    axs[2,0].set_xlabel("Temps (s)")
    axs[2,0].set_ylabel("Rapport")
    axs[2,0].legend()

    # Force motrice
    axs[2,1].plot(t_eval, forces["motor"], label="Force motrice (N)")
    axs[2,1].set_title("Force motrice appliquée au véhicule")
    axs[2,1].set_xlabel("Temps (s)")
    axs[2,1].set_ylabel("Force (N)")
    axs[2,1].legend()

    # Consommation cumulative
    axs[3,0].plot(t_eval, consommation/0.75, label="Consommation cumulative (ml)", color="purple")
    axs[3,0].set_title("Consommation cumulative en fonction du temps")
    axs[3,0].set_xlabel("Temps (s)")
    axs[3,0].set_ylabel("Consommation (ml)")
    axs[3,0].legend()

    """
    axs[3,1].plot(t_eval,regime_mot_elec,label="regime moteur elec")
    axs[3,1].set_title("Regime moteur elec en fonction du temps")
    axs[3,1].set_xlabel("Temps (s)")
    axs[3,1].set_ylabel("regime mot elec (rpm)")
    axs[3,1].legend()
    """
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# 10) Fonction de sauvegarde des résultats dans un fichier CSV
# =============================================================================

def save_simulation_data_to_csv(t_eval, position, vitesse,
                                forces, regimes_moteur, ratios_utilises,
                                consommation, file_name="simulation_results.csv"):
    data = {
        "Time (s)": t_eval,
        "Position (m)": position,
        "Velocity (m/s)": vitesse,
        "Force Aero (N)": forces["aero"],
        "Force Rolling (N)": forces["rolling"],
        "Force Gravity (N)": forces["gravity"],
        "Force Motor (N)": forces["motor"],
        "Engine RPM": regimes_moteur,
        "Enviolo Ratio": ratios_utilises,
        "Cumulative Fuel Consumption (g)": consommation
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False, encoding="utf-8")
    print(f"Simulation data saved to {file_name}")

# =============================================================================
# 11) Exécution principale
# =============================================================================

if __name__=="__main__":
    t_vals, pos_vals, vit_vals, forces_vals, rpm_vals, conso_tot, conso_tot_ml, ratio_vals, conso_instant_vals= simuler_vehicule_et_calculer_conso(
        distance_totale=distance.iloc[-1],
        borne_min1=0,
        borne_max1=8.4,
        borne_min2=5.5,
        borne_max2=7.7,
        borne_min3=6.7,
        borne_max3=7.3,
        temps_max=228,
        vent_active=False,
        vitesse_vent=5*0.514, 
        wind_angle_global=np.deg2rad(10),
        aero_active=True,
        gravite_active=True,
        enviolo_on=True,
        moteur_elec=False
        )
    save_simulation_data_to_csv(
        t_eval=t_vals,
        position=pos_vals,
        vitesse=vit_vals,
        forces=forces_vals,
        regimes_moteur=rpm_vals,
        ratios_utilises=ratio_vals,
        consommation=conso_instant_vals,
        file_name="simulation_results_1.csv"
   )

lap_4_data=pd.read_csv("lap_4_data.csv")
lap_4_data.columns = [col.lower() for col in lap_4_data.columns]
lap_4_data.reset_index(drop=True, inplace=True)
lap_4_data.ffill(inplace=True)

time_real=lap_4_data["lap_obc_timestamp"]
velocity_real=lap_4_data["gps_speed"]/3.6
position_real=lap_4_data["lap_dist"]

plt.figure(figsize=(10, 6))
plt.plot(pos_vals, vit_vals, label='Vitesse Simulée', color='green')
plt.plot(position_real,velocity_real, label='Vitesse Réelle', color='red', linestyle='--')
plt.xlabel("position")
plt.ylabel("Vitesse (m/s)")
plt.title("Comparaison Vitesse Simulée vs. Vitesse Réelle")
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
plt.plot(t_vals, pos_vals, label='position en fonction du temps simulé', color='green')
plt.plot(time_real,position_real,label="position en fonction du temps réel", color="red",linestyle="--")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_vals,rpm_vals)
plt.show()
wind_speed = 5 * 0.514
wind_angle_global = np.deg2rad(0)

wind_vector = np.array([
    wind_speed*np.cos(wind_angle_global),
    wind_speed*np.sin(wind_angle_global)
])

fig, ax = plt.subplots(figsize=(10,8))
ax.plot(pos_x, pos_y, 'k-', label='Circuit')

# Texte pour afficher Cx à chaque frame
text_cx = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                  fontsize=12, color='purple')

# Point voiture et orientation
i0 = 0
x0, y0 = pos_x[i0], pos_y[i0]
car_point, = ax.plot(x0, y0, 'bo', markersize=6, label='Voiture')

heading_quiver = ax.quiver(
    x0, y0, 0, 0,
    angles='xy', scale_units='xy', scale=0.1,
    pivot='mid', color='b', width=0.005,
    label='Cap Voiture'
)

# Vent global (rouge)
wind_quiver = ax.quiver(
    x0, y0,
    wind_vector[0], wind_vector[1],
    angles='xy', scale_units='xy', scale=0.1,
    pivot='mid', color='r', width=0.005,
    label='Vent global'
)

# Projection du vent sur l’axe longitudinal (verte)
proj_quiver = ax.quiver(
    x0, y0, 0, 0,
    angles='xy', scale_units='xy', scale=0.05,
    pivot='mid', color='g', width=0.005,
    label='Vent projeté'
)

# Traînée aérodynamique due à la vitesse (cyan)
aero_quiver = ax.quiver(
    x0, y0, 0, 0,
    angles='xy', scale_units='xy', scale=0.005,
    pivot='mid', color='c', width=0.005,
    label='Traînée vitesse'
)

ax.legend(loc='upper right')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Vecteurs : cap, vent, vent projeté & traînée")
ax.set_xlim(pos_x.min()-5, pos_x.max()+5)
ax.set_ylim(pos_y.min()-5, pos_y.max()+5)
ax.grid(True)

def update(i):
    x, y = pos_x[i], pos_y[i]
    car_point.set_data(x, y)

    # --- cap du véhicule ---
    heading = heading_interp(distance.iloc[i])
    car_dir = np.array([np.cos(heading), np.sin(heading)])
    heading_quiver.set_offsets([[x, y]])
    heading_quiver.set_UVC(car_dir[0], car_dir[1])

    # --- vent global ---
    wind_quiver.set_offsets([[x, y]])
    wind_quiver.set_UVC(wind_vector[0], wind_vector[1])

    # --- projection du vent ---
    v_proj = np.dot(wind_vector, car_dir)
    proj_vec = v_proj * car_dir
    proj_quiver.set_offsets([[x, y]])
    proj_quiver.set_UVC(proj_vec[0], proj_vec[1])

    # --- traînée aérodynamique due à la vitesse ---
    vr = vit_vals[i]
    F_aero = 0.5 * 0.2 * 0.789 * 1.225 * vr**2
    aero_vec = -F_aero * car_dir
    aero_quiver.set_offsets([[x, y]])
    aero_quiver.set_UVC(aero_vec[0], aero_vec[1])

    # --- affichage du Cx pour le vent projeté ---
    phi = np.arctan2(wind_vector[1], wind_vector[0]) - heading
    phi = (phi + np.pi) % (2*np.pi) - np.pi
    angle_rel = abs(np.degrees(phi))
    Cx_wind = float(f_interp_cx(angle_rel))
    text_cx.set_text(f"Cx_wind = {Cx_wind:.3f}")

    # on renvoie TOUT ce qu'on veut redraw
    return car_point, heading_quiver, wind_quiver, proj_quiver, aero_quiver, text_cx

anim = FuncAnimation(fig, update, frames=len(pos_x),
                     interval=50, blit=True)

plt.show()
