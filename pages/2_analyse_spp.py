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
        os.path.join(os.path.dirname(__file__), "..", "spp_final_g_i.csv")
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


# 🔹 1) Après avoir standardisé les colonnes (df.columns = df.columns.str.strip().str.lower())
#     on prépare la colonne 'classe_fpt' à partir de 'grade'
def _normalize_txt(s: str) -> str:
    if pd.isna(s):
        return ""
    return (
        unicodedata.normalize("NFKD", str(s))
        .encode("ascii", "ignore")
        .decode("ascii")
        .upper()
        .strip()
        .replace("-", " ")
    )


def grade_to_classe_fpt(grade: str) -> str:
    g = _normalize_txt(grade)

    # Catégorie C : H. du rang + sous-officiers
    if any(
        k in g
        for k in [
            "SAPEUR",
            "1ERE CLASSE",
            "1ERECLASSE",
            "2EME CLASSE",
            "2EMECLASSE",
            "CAPORAL",
            "CAPORAL CHEF",
            "SERGENT",
            "SERGENT CHEF",
            "ADJUDANT",
            "ADJUDANT CHEF",
        ]
    ):
        return "C"

    # Catégorie B : officiers subalternes
    if any(k in g for k in ["LIEUTENANT", "CAPITAINE"]):
        return "B"

    # Catégorie A : officiers supérieurs & direction, SSSM officiers, etc.
    if any(
        k in g
        for k in [
            "COMMANDANT",
            "LT COLONEL",
            "LIEUTENANT COLONEL",
            "COLONEL",
            "CONTROLEUR",
            "MEDECIN",
            "PHARMACIEN",
            "VETERINAIRE",
            "INGENIEUR",
            "DIRECTEUR",
        ]
    ):
        return "A"

    return "Inconnu"


# Crée la colonne même si 'grade' n'existe pas (sécurisé)
if "grade" in df.columns:
    df["classe_fpt"] = df["grade"].apply(grade_to_classe_fpt)
else:
    df["classe_fpt"] = "Inconnu"

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

st.sidebar.markdown("**Classe de grade (FPT A/B/C)**")
classes_disponibles = ["A", "B", "C", "Inconnu"]
classes_choisies = st.sidebar.multiselect(
    "Classe FPT :", options=classes_disponibles, default=classes_disponibles
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

if "classe_fpt" not in df.columns and "grade" in df.columns:
    df["classe_fpt"] = df["grade"].apply(grade_to_classe_fpt)

if "classe_fpt" in df.columns and classes_choisies:
    df_filtered = df_filtered[df_filtered["classe_fpt"].isin(classes_choisies)]

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

import re

st.header("Durée d'engagement — Garde vs Intervention (en heures, sans filtrage)")

# 1) Détection tolérante des colonnes
COL_GARDE_CAND = [
    "durée engagement garde",
    "duree engagement garde",
    "duree_garde",
    "duree garde",
    "duree engag garde",
    "duree en garde",
]
COL_INTER_CAND = [
    "durée engagement inter",
    "duree engagement inter",
    "duree_inter",
    "duree intervention",
    "duree engag inter",
    "duree en inter",
]


def _norm(x: str) -> str:
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).lower().replace("_", " ").strip()


def pick_column(df, candidates):
    normmap = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in normmap:
            return normmap[key]
        for k, orig in normmap.items():
            if all(tok in k for tok in key.split()):
                return orig
    return None


col_garde = pick_column(df, COL_GARDE_CAND)
col_inter = pick_column(df, COL_INTER_CAND)

if not col_garde or not col_inter:
    st.warning("Colonnes non détectées automatiquement. Sélectionne-les ci-dessous.")
    col_garde = st.selectbox("Colonne 'garde' :", options=df.columns)
    col_inter = st.selectbox("Colonne 'inter' :", options=df.columns)

st.caption(
    f"Colonnes utilisées : **{col_garde}** et **{col_inter}** (brut, sans filtre)"
)


# 2) Conversion en heures
def to_hours(series):
    td = pd.to_timedelta(series, errors="coerce")
    hrs = td.dt.total_seconds() / 3600
    return hrs


garde_h = to_hours(df[col_garde])
inter_h = to_hours(df[col_inter])

sub = pd.DataFrame({"duree_garde_h": garde_h, "duree_inter_h": inter_h}).dropna()

# 3) Diagnostics


if sub.empty:
    st.error("Aucune donnée exploitable après conversion des durées.")
    st.dataframe(df[[col_garde, col_inter]].head(10))
else:
    # 4) Corrélations
    pearson = sub["duree_garde_h"].corr(sub["duree_inter_h"], method="pearson")
    spearman = sub["duree_garde_h"].corr(sub["duree_inter_h"], method="spearman")
    st.markdown(
        f"**Corrélation Pearson :** {pearson:.2f} — **Spearman :** {spearman:.2f}"
    )

    # 5) Scatter + droite de régression
    st.subheader("Nuage de points + droite de tendance")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=sub, x="duree_garde_h", y="duree_inter_h", alpha=0.5)
    sns.regplot(
        data=sub,
        x="duree_garde_h",
        y="duree_inter_h",
        scatter=False,
        color="red",
        line_kws={"label": "Régression"},
    )
    ax.set_xlabel("Durée garde (heures)")
    ax.set_ylabel("Durée intervention (heures)")
    ax.legend()
    st.pyplot(fig)

    # 6) Densité (hexbin)
    st.subheader("Carte de densité (hexbin)")
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(
        sub["duree_garde_h"],
        sub["duree_inter_h"],
        gridsize=35,
        cmap="viridis",
        mincnt=1,
    )
    ax.set_xlabel("Durée garde (heures)")
    ax.set_ylabel("Durée intervention (heures)")
    fig.colorbar(hb, ax=ax, label="Comptes")
    st.pyplot(fig)

    # 7) Distributions
    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(sub["duree_garde_h"], bins=30, kde=True)
        ax.set_xlabel("Durée garde (heures)")
        ax.set_title("Garde")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(sub["duree_inter_h"], bins=30, kde=True)
        ax.set_xlabel("Durée intervention (heures)")
        ax.set_title("Intervention")
        st.pyplot(fig)

    # 8) Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.table(
        sub.describe(percentiles=[0.25, 0.5, 0.75])
        .T.loc[:, ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        .round(1)
        .rename(index={"duree_garde_h": "Garde (h)", "duree_inter_h": "Inter (h)"})
    )
