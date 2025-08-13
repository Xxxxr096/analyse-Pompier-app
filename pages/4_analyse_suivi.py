import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.image as mpimg


# --- Chargement des données ---
@st.cache_data()
@st.cache_data()
def load_data():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "cleannn.csv")
    )

    df = pd.read_csv(data_path, dtype={"matricule": str})

    if "matricule" in df.columns:
        df["matricule"] = df["matricule"].str.replace(".0", "", regex=False).str.strip()

    df.columns = df.columns.str.strip().str.lower()

    # Colonnes à convertir depuis 'xx,yy' → float
    cols_to_fix = [
        "poids",
        "taille",
        "imc",
        "resul ll",
        "resul pompes",
        "resul tractions",
        "systol",
        "diastol",
        "périmétre abdominal",
    ]

    for col in cols_to_fix:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([0-9.]+)")  # ne garde que les valeurs valides
                .astype(float)
            )

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
if "périmétre abdominal" in df.columns:
    df["périmétre abdominal"] = (
        df["périmétre abdominal"].astype(str).str.replace(",", ".").astype(float)
    )
df["taille"] = df["taille"].astype(str).str.replace(",", ".").astype(float)
df.loc[(df["taille"] <= 100) | (df["taille"] > 250), "taille"] = None
df["taille"] = df["taille"] / 100


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
# Convertir sexe_volontaire_volontaire_volontaire en numérique
df["sexe_num"] = df["sexe"].str.upper().map(lambda x: 1 if x == "M" else 0).fillna(0)

# Convertir palier resul ll en vitesse, 0 si NaN
df["vitesse"] = df["resul ll"].map(palier_to_vitesse).fillna(0)

# Calcul VO2max de la formule avec âge, sexe_volontaire_volontaire_volontaire et vitesse
df["vo2max"] = (
    31.025
    + 3.238 * df["vitesse"]
    - 3.248 * df["age"].fillna(0)
    + 6.318 * df["sexe_num"]
)
df["vo2max"] = df["vo2max"].clip(lower=0)

# Formule de Léger (1988)
df["vo2max_leger"] = (5.857 * df["vitesse"]).fillna(0) - 19.458
df["vo2max_leger"] = df["vo2max_leger"].clip(lower=0)


st.title("Analyse de la Condition Physique et de la Santé(spv)")
with st.expander("📘 Guide d'utilisation de l'application", expanded=False):
    st.markdown(
        """
### 🧭 Guide d'utilisation

Bienvenue dans l'application d'analyse de la condition physique et de la santé.

---

#### 🔍 1. Filtres dynamiques (colonne de gauche)
Utilisez les filtres pour explorer les données :

- **Cie / UT** : sélectionnez une ou plusieurs groupements ou unités territoriales.
- **sexe_volontaire_volontaire_volontaire** : filtrez par genre.
- **Aptitude générale** : explorez les performances selon l'aptitude.
- **Âge** : sélection par tranche d'âge (16–29, 30–39, etc.).
- **IMC (Indice de Masse Corporelle)** : sélection par catégorie OMS (normal, surpoids...).
- **Poids** : filtrez les individus selon leur poids (kg).
- **resul ll – Paliers** : filtrez par niveau d’endurance (1 à >6).
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
    - IMC par niveau de resul ll.
    - resul ll par catégorie d’IMC.
- **Histogrammes et boxplots croisés** :
    - resul ll par aptitude ou exposition à l'incendie.
    - systolique et diastolique (colorées selon les seuils OMS).
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
cie = st.sidebar.multiselect("Cie:", df["groupement"].dropna().unique())
ut = st.sidebar.multiselect("UT:", df["compagnie"].dropna().unique())
sexe_options = st.sidebar.multiselect(
    "sexe:",
    df["sexe"].dropna().unique(),
    default=df["sexe"].dropna().unique(),
)

# --- Filtre Grade ---
if "grade" in df.columns:
    grades_disponibles = df["grade"].dropna().unique()
    grade_selection = st.sidebar.multiselect("Grade :", sorted(grades_disponibles))
else:
    grade_selection = []


# --- Filtre par catégorie (Volontaire ou Professionnel) ---
if "catégorie" in df.columns:
    categories_disponibles = df["catégorie"].dropna().unique()
    selected_categories = st.sidebar.multiselect(
        "Catégorie (Volontaire / Professionnel) :", sorted(categories_disponibles)
    )
else:
    selected_categories = []

# --- Filtre des tests physiques à afficher ---
tests_physiques_disponibles = {
    "Gainage": "resul gain",
    "Killy": "resul killy",
    "Luc Léger": "resul ll",
    "Pompes": "resul pompes",
    "Souplesse": "resul souplesse",
    "Tractions": "resul tractions",
}


selected_tests = st.sidebar.multiselect(
    "Tests physiques à suivre (progression annuelle) :",
    options=list(tests_physiques_disponibles.keys()),
    default=list(
        tests_physiques_disponibles.keys()
    ),  # ou sélectionne seulement quelques-uns
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
    df_vo2max_leger = df["vo2max_leger"].dropna()
    if not df_vo2max_leger.empty and df_vo2max_leger.min() < df_vo2max_leger.max():
        st.sidebar.markdown("**VO2max Léger (Formule 1988)**")
        vo2l_min, vo2l_max = st.sidebar.slider(
            "Plage VO2max (Léger 1988) :",
            min_value=float(df_vo2max_leger.min()),
            max_value=float(df_vo2max_leger.max()),
            value=(float(df_vo2max_leger.min()), float(df_vo2max_leger.max())),
            step=1.0,
        )
    else:
        vo2l_min, vo2l_max = 0.0, 1000.0  # valeurs par défaut
        st.sidebar.warning("Pas de données valides pour VO2max Léger.")
else:
    vo2l_min, vo2l_max = 0.0, 1000.0  # valeurs fallback


# Slider pour systolique
# Nettoyage des colonnes de tension artérielle
if "systol" in df.columns:
    df["systol"] = df["systol"].astype(str).str.replace(",", ".").astype(float)
    # Correction des valeurs aberrantes : si > 250, on divise par 10
    df.loc[df["systol"] > 250, "systol"] /= 10

if "diastol" in df.columns:
    df["diastol"] = df["diastol"].astype(str).str.replace(",", ".").astype(float)
    # Correction des valeurs aberrantes : si > 150, on divise par 10
    df.loc[df["diastol"] > 150, "diastol"] /= 10

if "systol" in df.columns:
    st.sidebar.markdown("**systolique (mmHg)**")
    sys_min, sys_max = st.sidebar.slider(
        "Sélectionnez une plage pour la tension systolique :",
        min_value=float(df["systol"].min()),
        max_value=float(df["systol"].max()),
        value=(
            float(df["systol"].min()),
            float(df["systol"].max()),
        ),
    )

# Slider pour diastolique
if "diastol" in df.columns:
    st.sidebar.markdown("**diastolique (mmHg)**")
    dia_min, dia_max = st.sidebar.slider(
        "Sélectionnez une plage pour la tension diastolique :",
        min_value=float(df["diastol"].min()),
        max_value=float(df["diastol"].max()),
        value=(
            float(df["diastol"].min()),
            float(df["diastol"].max()),
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
# --- Filtre Tour de Taille (périmétre abdominal) ---

if "périmétre abdominal" in df.columns:
    tour_min, tour_max = st.sidebar.slider(
        "Tour de taille (cm) :",
        min_value=float(df["périmétre abdominal"].min()),
        max_value=float(df["périmétre abdominal"].max()),
        value=(
            float(df["périmétre abdominal"].min()),
            float(df["périmétre abdominal"].max()),
        ),
        step=1.0,
    )

poids_min, poids_max = st.sidebar.slider(
    "poids:", float(df["poids"].min()), float(df["poids"].max()), (0.0, 144.0)
)

# --- Application des filtres ---
df_filtered = df.copy()
# Filtrage par catégorie sélectionnée
if selected_categories:
    df_filtered = df_filtered[df_filtered["catégorie"].isin(selected_categories)]

if cie:
    df_filtered = df_filtered[df_filtered["groupement"].isin(cie)]
if ut:
    df_filtered = df_filtered[df_filtered["compagnie"].isin(ut)]


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
df_filtered["tranche_age"] = df_filtered["age"].apply(age_to_categorie)

if "périmétre abdominal" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["périmétre abdominal"] >= tour_min)
        & (df_filtered["périmétre abdominal"] <= tour_max)
    ]

# Application des filtres de tension artérielle
if "systol" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["systol"] >= sys_min) & (df_filtered["systol"] <= sys_max)
    ]

if "diastol" in df_filtered.columns:
    df_filtered = df_filtered[
        (df_filtered["diastol"] >= dia_min) & (df_filtered["diastol"] <= dia_max)
    ]


if age_category:
    filtres_age = []
    for cat in age_category:
        if cat == "16 à 29":
            filtres_age.append((df_filtered["age"] >= 16) & (df_filtered["age"] <= 29))
        elif cat == "30 à 39":
            filtres_age.append((df_filtered["age"] >= 30) & (df_filtered["age"] <= 39))
        elif cat == "40 à 49":
            filtres_age.append((df_filtered["age"] >= 40) & (df_filtered["age"] <= 49))
        elif cat == "50 à 57":
            filtres_age.append((df_filtered["age"] >= 50) & (df_filtered["age"] <= 57))
        elif cat == "Plus de 57":
            filtres_age.append(df_filtered["age"] > 57)

    if filtres_age:
        df_filtered = df_filtered[pd.concat(filtres_age, axis=1).any(axis=1)]


df_filtered = df_filtered[
    (df_filtered["poids"] >= poids_min) & (df_filtered["poids"] <= poids_max)
]

st.sidebar.markdown("**Luc Léger - Paliers**")
luc_leger_categories = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs catégories de palier Luc Léger :",
    ["0", "1", "2", "3", "4", "5", "plus de 6"],
)

if sexe_options:
    df_filtered = df_filtered[df_filtered["sexe"].isin(sexe_options)]


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

# Filtrage par grade sélectionné
if grade_selection:
    df_filtered = df_filtered[df_filtered["grade"].isin(grade_selection)]


# --- VISUALISATIONS ---
st.subheader("Statistiques Globales sur les Données Filtrées")
st.write(f"Nombre d'individus: {df_filtered.shape[0]}")


st.subheader("Distribution de l’imc empilée selon le niveau Luc léger")

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

    ax.set_title("Distribution empilée de l’imc par niveau Luc Léger")
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

    df_viz = df_filtered[["resul ll", "imc"]].dropna()

    if df_viz.empty:
        st.info("Aucune donnée disponible pour cette combinaison de filtres.")
    else:
        ordre_imc = [
            "Insuffisance pondérale",
            "Normal",
            "Surpoids",
            "Obésité modérée",
            "Obésité sévère",
            "Obésité massive",
            "Inconnu",
        ]

        df_viz["imc_cat"] = df_viz["imc"].apply(classify_imc)
        df_viz["imc_cat"] = pd.Categorical(
            df_viz["imc_cat"], categories=ordre_imc, ordered=True
        )

        palette = {
            "Insuffisance pondérale": "blue",
            "Normal": "green",
            "Surpoids": "orange",
            "Obésité modérée": "red",
            "Obésité sévère": "darkred",
            "Obésité massive": "black",
            "Inconnu": "gray",
        }

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
        labels = []
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
                labels.append(cat)
        ax.legend(handles=handles, title="Catégorie IMC")

        st.pyplot(fig)
else:
    st.warning("Les colonnes nécessaires 'resul ll' et 'imc' sont manquantes.")


st.subheader("Distribution du Tour de Taille selon le sexe et les Normes de Santé")

if "périmétre abdominal" in df_filtered.columns and "sexe" in df_filtered.columns:
    df_tour = df_filtered[["périmétre abdominal", "sexe"]].dropna()

    def couleur_tour(row):
        sexe = str(row["sexe"]).lower()
        tour = row["périmétre abdominal"]
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
                subset["périmétre abdominal"],
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
        "La colonne 'périmétre abdominal' ou 'sexe' est manquante dans les données."
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

if "vo2max" in df_filtered.columns and "age" in df_filtered.columns:
    df_vo2_age = df_filtered[["vo2max", "age"]].dropna()

    if not df_vo2_age.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_vo2_age, x="age", y="vo2max", alpha=0.6)
        sns.regplot(
            data=df_vo2_age,
            x="age",
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
    st.warning("Les colonnes nécessaires 'vo2max' et 'age' sont manquantes.")


st.subheader("Relation entre l'âge et la VO2max (Formule de Léger 1988)")

if "vo2max_leger" in df_filtered.columns and "age" in df_filtered.columns:
    df_vo2_leger_age = df_filtered[["vo2max_leger", "age"]].dropna()

    if not df_vo2_leger_age.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_vo2_leger_age, x="age", y="vo2max_leger", alpha=0.6)
        sns.regplot(
            data=df_vo2_leger_age,
            x="age",
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
    st.warning("Les colonnes 'vo2max_leger' et 'age' sont manquantes.")
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
        sns.boxplot(x="groupement", y=test, data=df_filtered, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.info(f"Aucune donnée disponible pour {test}.")


st.subheader("Relation entre l'âge et le palier Luc Léger")

if "age" in df_filtered.columns and "resul ll" in df_filtered.columns:
    df_age_luc = df_filtered[["age", "resul ll"]].dropna()
    if not df_age_luc.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(
            data=df_age_luc,
            x="age",
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

if "systol" in df_filtered.columns and "diastol" in df_filtered.columns:
    df_tension = df_filtered[["systol", "diastol"]].dropna()

    # Création des catégories selon les seuils OMS
    df_tension["sys_couleur"] = df_tension["systol"].apply(
        lambda x: "red" if x > 140 else "green"
    )
    df_tension["dia_couleur"] = df_tension["diastol"].apply(
        lambda x: "red" if x > 90 else "green"
    )

    # Histogramme tension systolique
    fig, ax = plt.subplots(figsize=(10, 6))
    for couleur in ["green", "red"]:
        subset = df_tension[df_tension["sys_couleur"] == couleur]
        if not subset.empty:
            ax.hist(
                subset["systol"],
                bins=15,
                alpha=0.7,
                label=f"Systolique ({couleur})",
                color=couleur,
                edgecolor="black",
            )
    ax.set_title("Distribution de la systolique")
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
                subset["diastol"],
                bins=15,
                alpha=0.7,
                label=f"Diastolique ({couleur})",
                color=couleur,
                edgecolor="black",
            )
    ax.set_title("Distribution de la diastolique")
    ax.set_xlabel("Tension Diastolique (mmHg)")
    ax.set_ylabel("Nombre d'individus")
    ax.legend(title="État (90 mmHg seuil)")
    st.pyplot(fig)

else:
    st.warning("Les colonnes de tension artérielle sont manquantes ou incomplètes.")


st.subheader("🔗 Corrélations avec le Palier luc léger")

# Sélection des colonnes numériques pertinentes
cols_corr = [
    "resul ll",
    "imc",
    "poids",
    "taille",
    "systol",
    "diastol",
    "resul pompes",
    "resul tractions",
    "niv ll",
    "niv pompes",
    "niv tractions",
    "périmétre abdominal",
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

st.subheader("Répartition des niveaux ICP (Filtres appliqués)")

cols_group = [
    "couleur_globale",
    "groupement",
    "compagnie",
    "sexe",
    "tranche_age",
]
for col in cols_group[1:]:
    if col in df_filtered.columns:
        st.markdown(f"#### Répartition par {col}")
        tab = df_filtered.groupby([col, "couleur_globale"]).size().unstack(fill_value=0)
        tab["Total"] = tab.sum(axis=1)
        for color in ["Vert", "Orange", "Rouge"]:
            if color in tab.columns:
                tab[f"% {color}"] = round(100 * tab[color] / tab["Total"], 1)
        st.dataframe(tab)

st.subheader(
    "📋 Liste des agents avec des résultats manquants ou égaux à 0 aux tests physiques"
)


# Sélection dynamique des tests à surveiller
colonnes_disponibles = [
    "resul ll",
    "niv ll",
    "resul pompes",
    "niv pompes",
    "resul tractions",
    "niv tractions",
]

colonnes_a_surveiller = st.multiselect(
    "🧪 Sélectionnez les tests à surveiller :",
    options=colonnes_disponibles,
    default=colonnes_disponibles,
)

if colonnes_a_surveiller:
    conditions = df_filtered[colonnes_a_surveiller].isna() | (
        df_filtered[colonnes_a_surveiller] == 0
    )
    agents_incomplets = df_filtered[conditions.any(axis=1)].copy()

    # Colonnes principales à afficher
    colonnes_info = [
        "matricule",
        "nom",
        "prenom",
        "sexe",
        "groupement",
        "compagnie",
    ]
    colonnes_presentes = [
        col for col in colonnes_info if col in agents_incomplets.columns
    ]

    # Final table = infos + tests sélectionnés
    colonnes_finales = colonnes_presentes + colonnes_a_surveiller
    table_finale = agents_incomplets[colonnes_finales].sort_values(by="matricule")
    table_finale["matricule"] = table_finale["matricule"].astype(str).str.strip()

    if table_finale.empty:
        st.success(
            "✅ Aucun agent avec des résultats manquants ou nuls dans les tests sélectionnés."
        )
    else:
        st.write(f"{len(table_finale)} agents concernés :")
        st.dataframe(table_finale.reset_index(drop=True), use_container_width=True)

        # Export CSV
        csv_incomplets = table_finale.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger le tableau propre au format CSV",
            data=csv_incomplets,
            file_name="agents_incomplets_structures.csv",
            mime="text/csv",
        )
else:
    st.info("Veuillez sélectionner au moins un test à surveiller.")


st.subheader("📈 Progression Annuelle Moyenne des Tests Physiques (2019–2024)")

# Définir les colonnes de résultats sportifs
# Filtrer dynamiquement les tests sélectionnés par l'utilisateur
test_columns = {
    label: tests_physiques_disponibles[label]
    for label in selected_tests
    if tests_physiques_disponibles[label] in df_filtered.columns
}

# Vérifier que la colonne 'année' est bien présente
if "année" in df_filtered.columns:
    # Calcul de la moyenne annuelle (tous agents filtrés)
    data_progression = (
        df_filtered.groupby("année")[
            [col for col in test_columns.values() if col in df_filtered.columns]
        ]
        .mean()
        .dropna(how="all")
    )

    if not data_progression.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for label, col in test_columns.items():
            if col in data_progression.columns:
                ax.plot(
                    data_progression.index,
                    data_progression[col],
                    marker="o",
                    label=label,
                )

        ax.set_title("Progression annuelle moyenne des tests physiques (2019–2024)")
        ax.set_xlabel("Année")
        ax.set_ylabel("Résultat moyen")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("Aucune donnée disponible pour les années concernées après filtrage.")
else:
    st.warning("La colonne 'année' est absente des données.")


st.markdown(
    """
**Utilisation suggérée :**
- Comparez les performances physiques entre différentes groupements
- Repérez les zones avec des tensions élevées ou imc critiques
- Analysez la progression ou la performance moyenne par région ou groupe d'âge
- Identifiez les corrélations entre les indicateurs (ex : poids vs imc, ou imc vs luc léger)
- Visualisez clairement la répartition des niveaux de luc léger par unité ou groupement
"""
)
