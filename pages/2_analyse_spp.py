import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.image as mpimg
import unicodedata

st.set_page_config(page_title="Analyse SPP + Carte UT/CIS", layout="wide")


# --- Chargement des données ---
@st.cache_data()
def load_data():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "spp_final.csv")
    )

    df = pd.read_csv(data_path)

    # Standardiser les noms de colonnes : minuscules, sans espace
    df.columns = df.columns.str.strip().str.lower()

    # Colonnes à convertir de 'xx,yy' → float
    cols_to_fix = [
        "poids",
        "taille",
        "imc",
        "resul ll",
        "tension artérielle systol",
        "tension artérielle diastol",
    ]
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
    df.loc[df["resul ll"] == 0, "niv ll"] = 0

    return df


def niveau_to_couleur(niveau):
    if pd.isna(niveau):
        return "Inconnu"
    if niveau >= 3:
        return "Vert"
    elif niveau == 2:
        return "Orange"
    elif niveau == 1:
        return "Rouge"
    else:
        return "Inconnu"


def score_to_couleur(score):
    if pd.isna(score):
        return "Inconnu"
    if score >= 2.7:
        return "Vert"
    elif score >= 1.5:
        return "Orange"
    else:
        return "Rouge"


def age_to_categorie(age):
    if pd.isna(age):
        return "Inconnu"
    elif age < 30:
        return "16-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age <= 57:
        return "50-57"
    else:
        return "58+"


df = load_data()
df.columns = df.columns.str.strip().str.lower()


# Ajoute dans le chargement si ce n’est pas fait :
if "périmètre abdominal" in df.columns:
    df["périmètre abdominal"] = (
        df["périmètre abdominal"].astype(str).str.replace(",", ".").astype(float)
    )
df["taille"] = df["taille"].astype(str).str.replace(",", ".").astype(float)
df.loc[(df["taille"] <= 100) | (df["taille"] > 250), "taille"] = None
df["taille"] = df["taille"] / 100

# Conversion propre
df["resul ll"] = df["resul ll"].astype(str).str.replace(",", ".").astype(float)

# Correction des erreurs de saisie (ex : 93 → 9.3)
df.loc[df["resul ll"] > 20, "resul ll"] = df["resul ll"] / 10
df["luc_leger_arrondi"] = df["resul ll"].round().astype("Int64")

palier_to_vitesse = {
    0: 8.0,
    1: 8.5,
    2: 9.0,
    3: 9.5,
    4: 10.0,
    5: 10.5,
    6: 11.0,
    7: 11.5,
    8: 12.0,
    9: 12.5,
    10: 13.0,
    11: 13.5,
    12: 14.0,
    13: 14.5,
    14: 15.0,
    15: 15.5,
    16: 16.0,
}
# Convertir sexe_spp en numérique
df["sexe_num"] = (
    df["sexe_spp"].str.upper().map(lambda x: 1 if x == "M" else 0).fillna(0)
)

# Convertir palier Luc Léger en vitesse, 0 si NaN
df["vitesse"] = df["resul ll"].map(palier_to_vitesse).fillna(0)

# Calcul VO2max de la formule avec âge, sexe et vitesse
df["vo2max"] = (
    31.025
    + 3.238 * df["vitesse"]
    - 3.248 * df["age_pro"].fillna(0)
    + 6.318 * df["sexe_num"]
)
df["vo2max"] = df["vo2max"].clip(lower=0)

# Formule de Léger (1988)
df["vo2max_leger"] = (5.857 * df["vitesse"]).fillna(0) - 19.458
df["vo2max_leger"] = df["vo2max_leger"].clip(lower=0)


st.title("Analyse de la Condition Physique et de la Santé (spp)")
with st.expander("📘 Guide d'utilisation de l'application", expanded=False):
    st.markdown(
        """
### 🧭 Guide d'utilisation

Bienvenue dans l'application d'analyse de la condition physique et de la santé.

---

#### 🔍 1. Filtres dynamiques (colonne de gauche)
Utilisez les filtres pour explorer les données :

- **Cie / UT** : sélectionnez une ou plusieurs compagnies ou unités territoriales.
- **Sexe** : filtrez par genre.
- **Aptitude générale** : explorez les performances selon l'aptitude.
- **Âge** : sélection par tranche d'âge (16–29, 30–39, etc.).
- **IMC (Indice de Masse Corporelle)** : sélection par catégorie OMS (normal, surpoids...).
- **Poids** : filtrez les individus selon leur poids (kg).
- **Luc Léger – Paliers** : filtrez par niveau d’endurance (1 à >6).
- **Tension artérielle** :
    - Systolique (mmHg) : filtre par plage personnalisée.
    - Diastolique (mmHg) : filtre par plage personnalisée.
- **VO2max (ml/kg/min)** : filtrez selon la capacité cardio-respiratoire estimée.

⚠️ Tous les graphiques et la carte s’adaptent automatiquement à ces filtres.

---

#### 📊 2. Visualisations proposées
Plusieurs visualisations sont générées à partir des données filtrées :

- **Histogrammes simples** : poids, taille, IMC, VO2max.
- **Histogrammes empilés** :
    - IMC par niveau de Luc Léger.
    - Luc Léger par catégorie d’IMC.
- **Histogrammes et boxplots croisés** :
    - Luc Léger par aptitude ou exposition à l'incendie.
    - Tension artérielle systolique et diastolique (colorées selon les seuils OMS).
- **Corrélations** : carte de chaleur (heatmap) des corrélations entre indicateurs physiques.

---

#### 🗺️ 3. Carte Interactive par UT
- Affiche **l'IMC moyen** par unité territoriale (UT).
- Les cercles sont proportionnels à l'effectif par UT et colorés selon l'IMC moyen.
- Données géographiques automatiquement filtrées selon les sélections ci-dessus.

⚠️ La carte peut prendre quelques secondes à se mettre à jour. Rafraîchissez la page si nécessaire.

---

#### 💾 4. Export des données
- En bas de page, un bouton vous permet de **télécharger les données filtrées** au format CSV.

---

#### 🆘 En cas de problème
- Si un graphique ou une carte ne s'affiche pas, vérifiez que vos filtres ne sont pas trop restrictifs.
- Essayez de **réinitialiser les filtres** ou **rafraîchir la page** du navigateur.
"""
    )


# --- SIDEBAR ---
st.sidebar.header("Filtres dynamiques")
cie = st.sidebar.multiselect("Cie:", df["cie"].dropna().unique())
ut = st.sidebar.multiselect("UT:", df["ut"].dropna().unique())
sexe_options = st.sidebar.multiselect(
    "sexe :", df["sexe_pro"].dropna().unique(), default=df["sexe_pro"].dropna().unique()
)
st.sidebar.markdown("**Abtitude générale**")
aptitude = st.sidebar.multiselect(
    "Aptitude Générale :",
    options=sorted(df["aptitude générale"].dropna().unique()),
    default=sorted(df["aptitude générale"].dropna().unique()),
)

# --- Filtre VO2max ---
if "vo2max" in df.columns:
    st.sidebar.markdown("**VO2max**")
    vo2_min, vo2_max = st.sidebar.slider(
        "Sélectionnez une plage de VO2max :",
        min_value=float(df["vo2max"].min()),
        max_value=float(df["vo2max"].max()),
        value=(float(df["vo2max"].min()), float(df["vo2max"].max())),
        step=1.0,
    )

# --- Filtre VO2max Léger ---
if "vo2max_leger" in df.columns:
    st.sidebar.markdown("**VO2max Léger (Formule 1988)**")
    vo2l_min, vo2l_max = st.sidebar.slider(
        "Plage VO2max (Léger 1988) :",
        min_value=float(df["vo2max_leger"].min()),
        max_value=float(df["vo2max_leger"].max()),
        value=(float(df["vo2max_leger"].min()), float(df["vo2max_leger"].max())),
        step=1.0,
    )


# Slider pour tension artérielle systolique
# Nettoyage des colonnes de tension artérielle
if "tension artérielle systol" in df.columns:
    df["tension artérielle systol"] = (
        df["tension artérielle systol"].astype(str).str.replace(",", ".").astype(float)
    )
    # Correction des valeurs aberrantes : si > 250, on divise par 10
    df.loc[df["tension artérielle systol"] > 250, "tension artérielle systol"] /= 10

if "tension artérielle diastol" in df.columns:
    df["tension artérielle diastol"] = (
        df["tension artérielle diastol"].astype(str).str.replace(",", ".").astype(float)
    )
    # Correction des valeurs aberrantes : si > 150, on divise par 10
    df.loc[df["tension artérielle diastol"] > 150, "tension artérielle diastol"] /= 10

if "tension artérielle systol" in df.columns:
    st.sidebar.markdown("**Tension Artérielle Systolique (mmHg)**")
    sys_min, sys_max = st.sidebar.slider(
        "Sélectionnez une plage pour la tension systolique :",
        min_value=float(df["tension artérielle systol"].min()),
        max_value=float(df["tension artérielle systol"].max()),
        value=(
            float(df["tension artérielle systol"].min()),
            float(df["tension artérielle systol"].max()),
        ),
    )

# Slider pour tension artérielle diastolique
if "tension artérielle diastol" in df.columns:
    st.sidebar.markdown("**Tension Artérielle Diastolique (mmHg)**")
    dia_min, dia_max = st.sidebar.slider(
        "Sélectionnez une plage pour la tension diastolique :",
        min_value=float(df["tension artérielle diastol"].min()),
        max_value=float(df["tension artérielle diastol"].max()),
        value=(
            float(df["tension artérielle diastol"].min()),
            float(df["tension artérielle diastol"].max()),
        ),
    )


st.sidebar.markdown("**Age - Catégories**")
age_category = st.sidebar.multiselect(
    "Selectionnez une catégorie d'Age : ",
    [
        "Tous",
        "16 à 29",
        "30 à 39",
        "40 à 49",
        "50 à 57",
        "plus de 57",
    ],
)
st.sidebar.markdown("**imc - Catégories**")
imc_category = st.sidebar.multiselect(
    "Sélectionnez une catégorie d'imc :",
    [
        "Tous",
        "Normal (18.5 - 24.9)",
        "Surpoids (25.0 - 29.9)",
        "Obésité modérée (30.0 - 34.9)",
        "Obésité sévère (35.0 - 39.9)",
        "Obésité massive (>40)",
    ],
)

# --- Filtre Tour de Taille ---
if "périmètre abdominal" in df.columns:
    st.sidebar.markdown("**Tour de Taille (cm)**")
    tour_min, tour_max = st.sidebar.slider(
        "Sélectionnez une plage pour le tour de taille :",
        min_value=float(df["périmètre abdominal"].min()),
        max_value=float(df["périmètre abdominal"].max()),
        value=(
            float(df["périmètre abdominal"].min()),
            float(df["périmètre abdominal"].max()),
        ),
        step=1.0,
    )

poids_min, poids_max = st.sidebar.slider(
    "poids:", float(df["poids"].min()), float(df["poids"].max()), (0.0, 144.0)
)

st.sidebar.markdown("**Luc Léger - Paliers**")
luc_leger_categories = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs catégories de palier Luc Léger :",
    ["0", "1", "2", "3", "4", "5", "plus de 6"],
)

# --- Application des filtres ---

df_filtered = df.copy()
if cie:
    df_filtered = df_filtered[df_filtered["cie"].isin(cie)]
if ut:
    df_filtered = df_filtered[df_filtered["ut"].isin(ut)]

if aptitude:
    df_filtered = df_filtered[df_filtered["aptitude générale"].isin(aptitude)]


# Filtrage VO2max
df_filtered = df_filtered[
    (df_filtered["vo2max"].fillna(0) >= vo2_min)
    & (df_filtered["vo2max"].fillna(0) <= vo2_max)
]

# ✅ Corrigé ici : filtrage VO2max_Léger avec les bonnes valeurs
df_filtered = df_filtered[
    (df_filtered["vo2max_leger"].fillna(0) >= vo2l_min)
    & (df_filtered["vo2max_leger"].fillna(0) <= vo2l_max)
]

df_filtered["couleur_luc"] = df_filtered["niv ll"].apply(niveau_to_couleur)
df_filtered["couleur_pompes"] = df_filtered["niv pompes"].apply(niveau_to_couleur)
df_filtered["couleur_tractions"] = df_filtered["niv tractions"].apply(niveau_to_couleur)
df_filtered["score_moyen"] = df_filtered[
    ["niv ll", "niv pompes", "niv tractions"]
].mean(axis=1)
df_filtered["couleur_globale"] = df_filtered["score_moyen"].apply(score_to_couleur)
df_filtered["tranche_age"] = df_filtered["age_pro"].apply(age_to_categorie)

if "périmètre abdominal" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["périmètre abdominal"] >= tour_min)
        & (df_filtered["périmètre abdominal"] <= tour_max)
    ]

if luc_leger_categories:
    filtres_luc = []
    for cat in luc_leger_categories:
        if cat == "0":
            filtres_luc.append(df_filtered["resul ll"] == 0)
        elif cat == "1":
            filtres_luc.append(df_filtered["resul ll"] == 1)
        elif cat == "2":
            filtres_luc.append(df_filtered["resul ll"] == 2)
        elif cat == "3":
            filtres_luc.append(df_filtered["resul ll"] == 3)
        elif cat == "4":
            filtres_luc.append(df_filtered["resul ll"] == 4)
        elif cat == "5":
            filtres_luc.append(df_filtered["resul ll"] == 5)
        elif cat == "plus de 6":
            filtres_luc.append(df_filtered["resul ll"] >= 6)

    if filtres_luc:
        df_filtered = df_filtered[pd.concat(filtres_luc, axis=1).any(axis=1)]
# Application des filtres de tension artérielle
if "tension artérielle systol" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["tension artérielle systol"] >= sys_min)
        & (df_filtered["tension artérielle systol"] <= sys_max)
    ]

if "tension artérielle diastol" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["tension artérielle diastol"] >= dia_min)
        & (df_filtered["tension artérielle diastol"] <= dia_max)
    ]


df_filtered = df_filtered[
    (df_filtered["poids"] >= poids_min) & (df_filtered["poids"] <= poids_max)
]


if sexe_options:
    df_filtered = df_filtered[df_filtered["sexe_spp"].isin(sexe_options)]


# Application du filtre imc par classe
if imc_category:
    filtres_imc = []
    for cat in imc_category:
        if "Normal" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 18.5) & (df_filtered["imc"] <= 24.9)
            )
        elif "Surpoids" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 25.0) & (df_filtered["imc"] <= 29.9)
            )
        elif "modérée" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 30.0) & (df_filtered["imc"] <= 34.9)
            )
        elif "sévère" in cat:
            filtres_imc.append(
                (df_filtered["imc"] >= 35.0) & (df_filtered["imc"] <= 39.9)
            )
        elif "massive" in cat:
            filtres_imc.append(df_filtered["imc"] >= 40.0)

    if filtres_imc:
        df_filtered = df_filtered[pd.concat(filtres_imc, axis=1).any(axis=1)]


# --- VISUALISATIONS ---
st.subheader("Statistiques Globales sur les Données Filtrées")
st.write(f"Nombre d'individus: {df_filtered.shape[0]}")


st.subheader("Distribution de l’imc empilée selon le niv ll")

if "imc" in df_filtered.columns and "niv ll" in df_filtered.columns:
    df_imc = df_filtered[["imc", "niv ll"]].dropna()

    # Définir les bins
    bins = np.histogram_bin_edges(df_imc["imc"], bins=20)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Initialiser les comptages pour chaque niveau
    niveaux = [1, 2, 3]
    couleurs = {1: "red", 2: "orange", 3: "green"}
    bar_data = {
        niv: np.histogram(df_imc[df_imc["niv ll"] == niv]["imc"], bins=bins)[0]
        for niv in niveaux
    }

    # Créer le graphique empilé
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros_like(bin_centers)
    for niv in niveaux:
        ax.bar(
            bin_centers,
            bar_data[niv],
            width=np.diff(bins),
            bottom=bottom,
            color=couleurs[niv],
            edgecolor="black",
            label=f"Niveau {niv}",
        )
        bottom += bar_data[niv]

    ax.set_title("Distribution empilée de l’imc par niv ll")
    ax.set_xlabel("imc")
    ax.set_ylabel("Nombre d’individus")
    ax.legend(title="niv ll")
    st.pyplot(fig)
else:
    st.info(
        "Les données nécessaires pour afficher cette visualisation sont incomplètes."
    )

st.subheader("Distribution du Palier Luc Léger par Catégorie d'IMC")

if "resul ll" in df_filtered.columns and "imc" in df_filtered.columns:

    def classify_imc(imc):
        if pd.isna(imc):
            return "Inconnu"
        elif imc < 18.5:
            return "Insuffisance pondérale"
        elif imc < 25:
            return "Normal"
        elif imc < 30:
            return "Surpoids"
        elif imc < 35:
            return "Obésité modérée"
        elif imc < 40:
            return "Obésité sévère"
        else:
            return "Obésité massive"

    ordre_imc = [
        "Insuffisance pondérale",
        "Normal",
        "Surpoids",
        "Obésité modérée",
        "Obésité sévère",
        "Obésité massive",
        "Inconnu",
    ]

    palette = {
        "Insuffisance pondérale": "blue",
        "Normal": "green",
        "Surpoids": "orange",
        "Obésité modérée": "red",
        "Obésité sévère": "darkred",
        "Obésité massive": "black",
        "Inconnu": "gray",
    }

    df_viz = df_filtered[["resul ll", "imc"]].dropna()
    if df_viz.empty:
        st.info("Aucune donnée disponible pour cette combinaison de filtres.")
    else:
        df_viz["imc_cat"] = pd.Categorical(
            df_viz["imc"].apply(classify_imc), categories=ordre_imc, ordered=True
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=df_viz,
            x="resul ll",
            hue="imc_cat",
            multiple="stack",
            palette=palette,
            bins=15,
            edgecolor="white",
            hue_order=ordre_imc,
        )
        ax.set_title("Distribution du Palier Luc Léger par Catégorie d'IMC")
        ax.set_xlabel("Palier Luc Léger")
        ax.set_ylabel("Nombre d'individus")

        # 🔧 Forcer la légende complète
        handles = []
        for cat in ordre_imc:
            if cat in palette:
                patch = plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    label=cat,
                    markerfacecolor=palette[cat],
                    markersize=10,
                )
                handles.append(patch)
        ax.legend(handles=handles, title="Catégorie IMC")

        st.pyplot(fig)
else:
    st.warning("Les colonnes nécessaires 'resul ll' et 'imc' sont manquantes.")


st.subheader("Distribution du Tour de Taille selon le Sexe et les Normes de Santé")

if "périmètre abdominal" in df_filtered.columns and "sexe_spp" in df_filtered.columns:
    df_tour = df_filtered[["périmètre abdominal", "sexe_spp"]].dropna()

    def couleur_tour(row):
        sexe = str(row["sexe_spp"]).lower()
        tour = row["périmètre abdominal"]
        if sexe == "m":
            return "green" if tour < 94 else "red"
        elif sexe == "f":
            return "green" if tour < 80 else "red"
        else:
            return "gray"

    df_tour["couleur"] = df_tour.apply(couleur_tour, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red", "gray"]:
        subset = df_tour[df_tour["couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["périmètre abdominal"],
                bins=15,
                alpha=0.7,
                label=couleur.capitalize(),
                color=couleur,
                edgecolor="black",
            )

    ax.set_title("Distribution du Tour de Taille (coloré selon les seuils OMS)")
    ax.set_xlabel("Tour de Taille (cm)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État de santé")
    st.pyplot(fig)
else:
    st.warning(
        "La colonne 'périmètre abdominal' ou 'sexe_spp' est manquante dans les données."
    )
st.subheader("Distribution de la VO2max")
if "vo2max" in df_filtered.columns and not df_filtered["vo2max"].dropna().empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_filtered["vo2max"], kde=True, bins=20, color="purple")
    ax.set_title("Distribution de la VO2max")
    ax.set_xlabel("VO2max (ml/kg/min)")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)


st.subheader("Nuage de points : VO2max en fonction de l'âge")

if "vo2max" in df_filtered.columns and "age_pro" in df_filtered.columns:
    df_vo2_age = df_filtered[["vo2max", "age_pro"]].dropna()

    if not df_vo2_age.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_vo2_age, x="age_pro", y="vo2max", alpha=0.6)
        sns.regplot(
            data=df_vo2_age,
            x="age_pro",
            y="vo2max",
            scatter=False,
            color="red",
            label="Tendance",
        )
        ax.set_title("Relation entre l'âge et la VO2max")
        ax.set_xlabel("Âge (ans)")
        ax.set_ylabel("VO2max (ml/kg/min)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Aucune donnée VO2max et âge disponible pour l'affichage.")
else:
    st.warning("Les colonnes nécessaires 'vo2max' et 'age_pro' sont manquantes.")


st.subheader("Relation entre l'âge et la VO2max (Formule de Léger 1988)")

if "vo2max_leger" in df_filtered.columns and "age_pro" in df_filtered.columns:
    df_vo2_leger_age = df_filtered[["vo2max_leger", "age_pro"]].dropna()

    if not df_vo2_leger_age.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_vo2_leger_age, x="age_pro", y="vo2max_leger", alpha=0.6)
        sns.regplot(
            data=df_vo2_leger_age,
            x="age_pro",
            y="vo2max_leger",
            scatter=False,
            color="green",
            label="Tendance",
        )
        ax.set_title("Relation entre l'âge et la VO2max (Formule Léger 1988)")
        ax.set_xlabel("Âge (ans)")
        ax.set_ylabel("VO2max (ml/kg/min)")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Aucune donnée disponible pour VO2max (Léger) et âge.")
else:
    st.warning("Les colonnes 'vo2max_leger' et 'age_pro' sont manquantes.")
st.subheader("Distribution de la VO2max - Formule de Léger (1988)")

if (
    "vo2max_leger" in df_filtered.columns
    and not df_filtered["vo2max_leger"].dropna().empty
):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_filtered["vo2max_leger"], kde=True, bins=20, color="teal")
    ax.set_title("Distribution de la VO2max (Formule Léger 1988)")
    ax.set_xlabel("VO2max Léger (ml/kg/min)")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)
else:
    st.info("Aucune donnée VO2max (formule Léger) disponible pour l'affichage.")

features = [
    "imc",
    "taille",
    "poids",
]


for feature in features:
    st.subheader(f"Distribution de {feature.upper()}")
    if feature in df_filtered.columns and not df_filtered[feature].dropna().empty:
        fig, ax = plt.subplots()
        sns.histplot(df_filtered[feature], kde=True, ax=ax)
        ax.set_title(f"Histogramme de {feature.upper()}")
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {feature.upper()}.")

phys_tests = ["resul ll", "resul pompes", "resul tractions"]
for test in phys_tests:
    st.subheader(f"{test.replace('_', ' ').title()} par Cie")
    if not df_filtered.empty and test in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="cie", y=test, data=df_filtered, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {test}.")


st.subheader("Relation entre l'âge et le palier Luc Léger")

if "age_pro" in df_filtered.columns and "resul ll" in df_filtered.columns:
    df_age_luc = df_filtered[["age_pro", "resul ll"]].dropna()
    if not df_age_luc.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            data=df_age_luc,
            x="age_pro",
            y="resul ll",
            scatter_kws={"alpha": 0.5},
            line_kws={"color": "red"},
        )
        ax.set_title("Relation entre l'âge et le palier Luc Léger")
        ax.set_xlabel("Âge")
        ax.set_ylabel("Palier Luc Léger")
        st.pyplot(fig)
    else:
        st.info("Pas de données disponibles pour l'âge ou le palier Luc Léger.")

st.subheader("Distribution de la Tension Artérielle (Systolique & Diastolique)")

if (
    "tension artérielle systol" in df_filtered.columns
    and "tension artérielle diastol" in df_filtered.columns
):
    df_tension = df_filtered[
        ["tension artérielle systol", "tension artérielle diastol"]
    ].dropna()

    # Création des catégories selon les seuils OMS
    df_tension["sys_couleur"] = df_tension["tension artérielle systol"].apply(
        lambda x: "red" if x > 140 else "green"
    )
    df_tension["dia_couleur"] = df_tension["tension artérielle diastol"].apply(
        lambda x: "red" if x > 90 else "green"
    )

    # Histogramme tension systolique
    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red"]:
        subset = df_tension[df_tension["sys_couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["tension artérielle systol"],
                bins=15,
                alpha=0.7,
                label=f"Systolique ({couleur})",
                color=couleur,
                edgecolor="black",
            )
    ax.set_title("Distribution de la Tension Artérielle Systolique")
    ax.set_xlabel("Tension Systolique (mmHg)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État (140 mmHg seuil)")
    st.pyplot(fig)

    # Histogramme tension diastolique
    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red"]:
        subset = df_tension[df_tension["dia_couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["tension artérielle diastol"],
                bins=15,
                alpha=0.7,
                label=f"Diastolique ({couleur})",
                color=couleur,
                edgecolor="black",
            )
    ax.set_title("Distribution de la Tension Artérielle Diastolique")
    ax.set_xlabel("Tension Diastolique (mmHg)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État (90 mmHg seuil)")
    st.pyplot(fig)

else:
    st.warning("Les colonnes de tension artérielle sont manquantes ou incomplètes.")

st.subheader("luc léger selon l'Aptitude Générale et l'Exposition Incendie")

# Histogramme luc léger par Incendie et port de l'ARI, coloré par aptitude
if (
    "luc léger" in df_filtered.columns
    and "aptitude générale" in df_filtered.columns
    and "incendie et port de l'ari toutes missions" in df_filtered.columns
):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="resul ll",
        hue="aptitude générale",
        multiple="stack",
        bins=15,
        palette="Set2",
        kde=False,
    )
    ax.set_title("Répartition du palier luc léger selon l'aptitude générale")
    ax.set_xlabel("Palier luc léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        df_filtered,
        x="resul ll",
        hue="incendie et port de l'ari toutes missions",
        multiple="stack",
        bins=15,
        palette="Set2",
        kde=False,
    )
    ax.set_title(
        "Répartition du palier luc léger selon Incendie et port de l'ARI Toutes missions"
    )
    ax.set_xlabel("Palier luc léger")
    ax.set_ylabel("Nombre d'individus")
    st.pyplot(fig)


# Boxplot luc léger par aptitude et Incendie/ARI
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(
    data=df_filtered,
    x="aptitude générale",
    y="resul ll",
    hue="incendie et port de l'ari toutes missions",
    palette="pastel",
)
ax.set_title("luc léger par Aptitude Générale et Incendie/ARI")
ax.set_ylabel("Palier luc léger")
ax.set_xlabel("Aptitude Générale")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)


st.subheader("🔗 Corrélations avec le Palier luc léger")

# Sélection des colonnes numériques pertinentes
cols_corr = [
    "resul ll",
    "imc",
    "poids",
    "taille",
    "tension artérielle systol",
    "tension artérielle diastol",
    "resul pompes",
    "resul tractions",
    "niv ll",
    "niv pompes",
    "niv tractions",
    "périmètre abdominal",
]

# Filtrage des colonnes existantes dans le dataframe filtré
cols_corr = [col for col in cols_corr if col in df_filtered.columns]
df_corr = df_filtered[cols_corr].dropna()

# Calcul de la matrice de corrélation
corr_matrix = df_corr.corr()

# Affichage d'une heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Matrice de Corrélation - Indicateurs Physiques et luc léger")
st.pyplot(fig)

st.subheader("🎯 Répartition des niveaux ICP - SPP (Filtres appliqués)")

for group_col in ["cie", "ut", "sexe_spp", "tranche_age"]:
    if group_col in df_filtered.columns:
        st.markdown(f"#### Répartition par {group_col}")
        tab = (
            df_filtered.groupby([group_col, "couleur_globale"])
            .size()
            .unstack(fill_value=0)
        )
        tab["Total"] = tab.sum(axis=1)
        for color in ["Vert", "Orange", "Rouge"]:
            if color in tab.columns:
                tab[f"% {color}"] = round(100 * tab[color] / tab["Total"], 1)
        st.dataframe(tab)

st.subheader(
    "📋 Liste des agents avec des résultats manquants ou égaux à 0 aux tests physiques"
)

# --- Nettoyage du matricule ---
if "matricule" in df_filtered.columns:
    df_filtered["matricule"] = (
        df_filtered["matricule"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

# --- Sélection des colonnes à surveiller ---
colonnes_disponibles = [
    "resul ll",
    "niv ll",
    "resul pompes",
    "niv pompes",
    "resul tractions",
    "niv tractions",
    "resul souplesse",
    "resul killy",
]

colonnes_a_surveiller = st.multiselect(
    "🧪 Sélectionnez les tests à surveiller :",
    options=colonnes_disponibles,
    default=colonnes_disponibles,
)

if colonnes_a_surveiller:
    # --- Filtrer les agents avec au moins une valeur manquante ou nulle ---
    conditions = (df_filtered[colonnes_a_surveiller].isna()) | (
        df_filtered[colonnes_a_surveiller] == 0
    )
    agents_incomplets = df_filtered[conditions.any(axis=1)].copy()

    # --- Colonnes à afficher ---
    colonnes_infos = ["matricule", "nom", "prenom", "sexe_spp", "cie", "ut"]
    colonnes_presentes = [
        col for col in colonnes_infos if col in agents_incomplets.columns
    ]
    colonnes_finales = colonnes_presentes + colonnes_a_surveiller

    table_finale = agents_incomplets[colonnes_finales].sort_values(by="matricule")
    table_finale["matricule"] = table_finale["matricule"].astype(str).str.strip()

    if table_finale.empty:
        st.success(
            "✅ Aucun agent avec des résultats manquants ou nuls dans les tests sélectionnés."
        )
    else:
        total_agents = len(df_filtered)
        nb_incomplets = len(table_finale)
        pct_incomplets = round(100 * nb_incomplets / total_agents, 1)

        st.write(
            f"**{nb_incomplets} agents concernés** sur {total_agents} filtrés "
            f"({pct_incomplets} %)."
        )

        st.dataframe(table_finale.reset_index(drop=True), use_container_width=True)

        # --- 📊 Calcul du % de tests manquants/nuls ---
        pourcentages_manquants = {}
        for col in colonnes_a_surveiller:
            nb_manquants = (df_filtered[col].isna() | (df_filtered[col] == 0)).sum()
            pourcentages_manquants[col] = round(100 * nb_manquants / total_agents, 1)

        # Création d'un DataFrame pour affichage clair
        df_pourcentages = pd.DataFrame.from_dict(
            pourcentages_manquants, orient="index", columns=["% manquants/nuls"]
        ).sort_values(by="% manquants/nuls", ascending=False)

        st.subheader("📊 Pourcentage de valeurs manquantes ou nulles par test")
        st.table(df_pourcentages)

        # Export CSV
        csv_incomplets = table_finale.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger le tableau propre au format CSV",
            data=csv_incomplets,
            file_name="agents_incomplets_structures.csv",
            mime="text/csv",
        )
else:
    st.info(
        "Veuillez sélectionner au moins un test pour vérifier les résultats manquants."
    )

# =============== Carte UT (jaune) + CIS (rouge) — SANS UPLOAD ===============

IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "carte_j.jpeg")
)
UT_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ut.csv"))
CIS_REF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cis.csv"))


# Petites utilitaires locales (indépendantes du reste du script)
def _normalise_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .apply(
            lambda x: unicodedata.normalize("NFKD", x)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("-", " ")
        .str.strip()
        .replace({"NAN": np.nan})
    )


def _load_reference(path: str, key_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Référentiel manquant : {os.path.basename(path)}")
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )
    ref = pd.read_csv(path, sep=";")
    needed = {key_col, "x_norm", "y_norm"}
    if not needed.issubset(set(ref.columns)):
        st.error(
            f"{os.path.basename(path)} doit contenir les colonnes : {key_col}, x_norm, y_norm (offset_x/offset_y optionnels)."
        )
        return pd.DataFrame(
            columns=["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]
        )
    ref["key_norm"] = _normalise_series(ref[key_col])
    ref["label"] = ref[key_col]
    for c in ("x_norm", "y_norm", "offset_x", "offset_y"):
        if c not in ref.columns:
            ref[c] = 0
        ref[c] = pd.to_numeric(ref[c], errors="coerce").fillna(0.0)
    return ref[["key_norm", "x_norm", "y_norm", "offset_x", "offset_y", "label"]]


# 1) Normaliser les colonnes UT / CIS dans df_filtered (déjà construit plus haut)
df_map = df_filtered.copy()
df_map.columns = df_map.columns.str.strip().str.lower()

col_ut = next(
    (c for c in ["ut", "compagnie", "unite_territoriale"] if c in df_map.columns), None
)
col_cis = next(
    (c for c in ["cis", "centre", "centre_cis", "nom_cis"] if c in df_map.columns), None
)

if not col_ut and not col_cis:
    st.warning(
        "Aucune colonne UT/Compagnie ni CIS trouvée dans les données — impossible d’afficher la carte."
    )
else:
    if col_ut:
        df_map["ut_norm"] = _normalise_series(df_map[col_ut])
    else:
        df_map["ut_norm"] = np.nan

    if col_cis:
        df_map["cis_norm"] = _normalise_series(df_map[col_cis])
    else:
        df_map["cis_norm"] = np.nan

    # 2) Charger les référentiels (ut.csv et cis.csv)
    ref_ut = _load_reference(UT_REF_PATH, "ut")
    ref_cis = _load_reference(CIS_REF_PATH, "cis")

    # 3) Joindre pour récupérer x_norm / y_norm
    df_ut = df_map.dropna(subset=["ut_norm"]).merge(
        ref_ut, left_on="ut_norm", right_on="key_norm", how="left"
    )
    df_cis = df_map.dropna(subset=["cis_norm"]).merge(
        ref_cis, left_on="cis_norm", right_on="key_norm", how="left"
    )

    # 4) Agréger (effectif + IMC moyen si présent)
    df_ut["imc"] = pd.to_numeric(df_ut.get("imc", np.nan), errors="coerce")
    df_cis["imc"] = pd.to_numeric(df_cis.get("imc", np.nan), errors="coerce")

    agg_ut = df_ut.groupby(
        ["ut_norm", "x_norm", "y_norm", "offset_x", "offset_y"],
        dropna=False,
        as_index=False,
    ).agg(effectif=("ut_norm", "count"), imc_moyen=("imc", "mean"))
    agg_cis = df_cis.groupby(
        ["cis_norm", "x_norm", "y_norm", "offset_x", "offset_y"],
        dropna=False,
        as_index=False,
    ).agg(effectif=("cis_norm", "count"), imc_moyen=("imc", "mean"))

    agg_ut_ok = agg_ut.dropna(subset=["x_norm", "y_norm"])
    agg_cis_ok = agg_cis.dropna(subset=["x_norm", "y_norm"])
    st.sidebar.markdown("**Affichage sur la carte**")
    affichage_points = st.sidebar.radio(
        "Sélectionnez les points à afficher :",
        ("UT uniquement", "CIS uniquement", "UT + CIS"),
        index=2,
    )
    # 5) Afficher la carte
    st.subheader("🗺️ Carte — UT (jaune) et CIS (rouge)")
    if not os.path.exists(IMAGE_PATH):
        st.error(f"Image non trouvée : {IMAGE_PATH}")
    else:
        img = mpimg.imread(IMAGE_PATH)
        H, W = img.shape[0], img.shape[1]

        c1, c2 = st.columns([1, 1])
        with c1:
            ms = st.slider("Taille des points", 4, 20, 10)
            fs = st.slider("Taille du texte", 6, 16, 9)
        with c2:
            show_labels = st.checkbox("Afficher les labels", True)
            show_imc = st.checkbox("Afficher l'IMC moyen", False)

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(img)
        ax.axis("off")

        # UT = jaune
        if affichage_points in ("UT uniquement", "UT + CIS"):
            for _, r in agg_ut_ok.iterrows():
                x = float(r["x_norm"]) * W
                y = float(r["y_norm"]) * H
                dx = float(r.get("offset_x", 0))
                dy = float(r.get("offset_y", 0))
                ax.plot(
                    x,
                    y,
                    "o",
                    markersize=ms,
                    color="yellow",
                    markeredgecolor="black",
                    markeredgewidth=1.2,
                )
                if show_labels:
                    label = f"{r['ut_norm']}\n{int(r['effectif'])} pers"
                    if show_imc and pd.notna(r["imc_moyen"]):
                        label += f"\nIMC {r['imc_moyen']:.1f}"
                    ax.text(
                        x + dx,
                        y + dy,
                        label,
                        fontsize=fs,
                        color="black",
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="white",
                            alpha=0.75,
                            edgecolor="none",
                            boxstyle="round,pad=0.2",
                        ),
                    )

        # CIS = rouge
        if affichage_points in ("CIS uniquement", "UT + CIS"):
            for _, r in agg_cis_ok.iterrows():
                x = float(r["x_norm"]) * W
                y = float(r["y_norm"]) * H
                dx = float(r.get("offset_x", 0))
                dy = float(r.get("offset_y", 0))
                ax.plot(
                    x,
                    y,
                    "o",
                    markersize=ms,
                    color="red",
                    markeredgecolor="white",
                    markeredgewidth=1.2,
                )
                if show_labels:
                    label = f"{r['cis_norm']}\n{int(r['effectif'])} pers"
                    if show_imc and pd.notna(r["imc_moyen"]):
                        label += f"\nIMC {r['imc_moyen']:.1f}"
                    ax.text(
                        x + dx,
                        y + dy,
                        label,
                        fontsize=fs,
                        color="white",
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="black",
                            alpha=0.75,
                            edgecolor="none",
                            boxstyle="round,pad=0.2",
                        ),
                    )

        st.pyplot(fig, use_container_width=True)

        # Infos si des clés manquent dans les référentiels
        ut_missing = sorted(
            set(df_ut["ut_norm"].unique()) - set(ref_ut["key_norm"].dropna().unique())
        )
        cis_missing = sorted(
            set(df_cis["cis_norm"].unique())
            - set(ref_cis["key_norm"].dropna().unique())
        )
        if ut_missing:
            st.warning(
                f"UT sans coordonnées dans ut.csv : {len(ut_missing)} — ex: {ut_missing[:5]}"
            )
        if cis_missing:
            st.warning(
                f"CIS sans coordonnées dans cis.csv : {len(cis_missing)} — ex: {cis_missing[:5]}"
            )
