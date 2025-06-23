import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import chardet
import os
import matplotlib.image as mpimg

# Titre
st.title("Analyse de l'accidentologie")


# Fonction de chargement des données avec détection automatique de l'encodage
def load_data():
    data_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "accidentologie.csv")
    )

    # Détection de l'encodage
    with open(data_path, "rb") as f:
        result = chardet.detect(f.read())
    encoding_detected = result["encoding"]

    # Lecture du fichier avec l'encodage détecté et le séparateur correct
    data = pd.read_csv(data_path, sep=";", encoding=encoding_detected)
    return data


# Lecture des données
data = load_data()

# Nettoyage des données
data.columns = data.columns.str.replace("*", "", regex=False).str.strip()
data.drop(columns=["Agent"], inplace=True)
data["Date de l'accident"] = pd.to_datetime(
    data["Date de l'accident"], errors="coerce", dayfirst=True
)
data["Année"] = data["Date de l'accident"].dt.year
data["Mois"] = data["Date de l'accident"].dt.month
data["Jour"] = data["Date de l'accident"].dt.day
data["Jour_semaine"] = data["Date de l'accident"].dt.day_name()
data["Durée totale arrêt"] = pd.to_numeric(data["Durée totale arrêt"], errors="coerce")
data["Heure_accident"] = pd.to_datetime(
    data["Heure de l'accident"], errors="coerce"
).dt.hour
# Affichage du tableau
st.subheader("Aperçu des données")
st.dataframe(data.head())

# Graphique: accidents par année
st.subheader("Nombre d'accidents par année")
fig1, ax1 = plt.subplots()
data["Année"].value_counts().sort_index().plot(kind="bar", ax=ax1)
ax1.set_title("Nombre d'accidents par année")
ax1.set_xlabel("Année")
ax1.set_ylabel("Nombre d'accidents")
ax1.grid(True)
st.pyplot(fig1)


st.subheader("Nombre d'accidents par jour de la semaine")
fig2, ax2 = plt.subplots()
jours = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
data["Jour_semaine"] = pd.Categorical(
    data["Jour_semaine"], categories=jours, ordered=True
)
data["Jour_semaine"].value_counts().sort_index().plot(kind="bar", ax=ax2)
ax2.set_title("Accidents par jour de la semaine")
ax2.set_xlabel("Jour")
ax2.set_ylabel("Nombre d'accidents")
ax2.grid(True)
st.pyplot(fig2)


st.subheader("Top 10 des natures d'accidents")
fig3, ax3 = plt.subplots(figsize=(10, 6))
data["Nature de l'accident"].value_counts().head(10).plot(kind="barh", ax=ax3)
ax3.set_title("Top 10 des natures d'accidents")
ax3.set_xlabel("Nombre")
ax3.invert_yaxis()
ax3.grid(True)
fig3.tight_layout()
st.pyplot(fig3)


st.subheader("Top 10 - Durée moyenne d'arrêt par nature de lésion")
fig4, ax4 = plt.subplots(figsize=(10, 6))
(
    data.groupby("Nature lésion")["Durée totale arrêt"]
    .mean()
    .dropna()
    .sort_values(ascending=False)
    .head(10)
    .plot(kind="barh", ax=ax4)
)
ax4.set_title("Top 10 - Durée moyenne d'arrêt par nature de lésion")
ax4.set_xlabel("Durée moyenne (jours)")
ax4.invert_yaxis()
ax4.grid(True)
fig4.tight_layout()
st.pyplot(fig4)


from datetime import datetime

# Conversion de la date de naissance
data["Date de naissance"] = pd.to_datetime(
    data["Date de naissance"], errors="coerce", dayfirst=True
)

# Calcul de l'âge en années
aujourd_hui = pd.Timestamp("today")
data["Age_calculé"] = ((aujourd_hui - data["Date de naissance"]).dt.days // 365).astype(
    "Int64"
)

# --- Répartition par tranche d'âge ---
st.subheader("Nombre d'accidents par tranche d'âge")
age_distribution = data["Age_calculé"].value_counts().sort_index()

fig5, ax5 = plt.subplots(figsize=(8, 5))
age_distribution.plot(kind="bar", ax=ax5)
ax5.set_title("Nombre d'accidents par tranche d'âge")
ax5.set_xlabel("Âge")
ax5.set_ylabel("Nombre d'accidents")
ax5.grid(True)
fig5.tight_layout()
st.pyplot(fig5)

# --- Répartition selon le moment de l'accident ---
st.subheader("Répartition des accidents par moment de service")
moment_distribution = data["Moment de l'accident"].value_counts()

fig6, ax6 = plt.subplots(figsize=(8, 5))
moment_distribution.plot(kind="bar", ax=ax6)
ax6.set_title("Répartition des accidents par moment de service")
ax6.set_xlabel("Moment")
ax6.set_ylabel("Nombre d'accidents")
ax6.grid(True)
fig6.tight_layout()
st.pyplot(fig6)


# Statistiques durée arrêt
st.subheader("Statistiques sur la durée totale d'arrêt")
st.write(data["Durée totale arrêt"].describe())


st.subheader("📊 Blessures par type de sport")
sport_counts = data["Type de sport"].value_counts().dropna()
st.bar_chart(sport_counts)

# --- 2. Blessures par heure ---
st.subheader("🕒 Blessures par heure de la journée")
heures = data["Heure_accident"].value_counts().sort_index()
st.bar_chart(heures)


# Chargement image
data_img = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "human_map.png")
)
image = mpimg.imread(data_img)
# 🧠 Mapping de normalisation des sièges
mapping_siege_harmonisé = {
    # Tête et visage
    "tête": "Tête",
    "face (sauf nez et bouche)": "Tête",
    "yeux": "Tête",
    "nez": "Tête",
    "bouche": "Tête",
    "region cranienne": "Tête",
    "appareil auditif": "Tête",
    # Cou
    "cervicale": "Cou",
    "cou (sauf vertebres cervicales)": "Cou",
    # Haut du corps
    "epaule": "Épaule",
    "bras": "Bras",
    "avant-bras": "Avant-bras",
    "coude": "Coude",
    # Mains et poignets
    "poignet": "Poignet",
    "main": "Poignet",
    "paume et dos": "Poignet",
    "pouce": "Main",
    "index": "Main",
    "majeur": "Main",
    "annulaire": "Main",
    "auriculaire": "Main",
    "plusieurs doigts": "Main",
    "autre doigt": "Main",
    "pouce et index": "Main",
    # Dos
    "lombaire": "Dos",
    "region lombaire": "Dos",
    "dorsale": "Dos",
    # Tronc
    "thorax": "Tronc",
    "abdomen": "Tronc",
    # Membres inférieurs
    "hanche": "Hanche",
    "cuisse": "Cuisse",
    "genou": "Genou",
    "jambe": "Jambe",
    # Pieds et chevilles
    "cheville": "Cheville",
    "cheville, cou de pied": "Cheville",
    "plante et dessus": "Pied",
    "talon": "Pied",
    "orteils": "Pied",
    # Organes internes
    "organes genitaux": "Organes internes",
    "siege interne non precise": "Organes internes",
    # Non précisé
    "localisation multiple non precise": "Non précisé",
    "non precise": "Non précisé",
    "non precise - colonne vertebrale": "Dos",
    "non precise - mains": "Main",
    "non precise - membres inferieurs ( pieds exceptes)": "Jambe",
    "non precise - membres superieurs": "Bras",
    "non precise - pieds": "Pied",
    "non precise - tete (yeux exceptes)": "Tête",
}


# Nettoyage
data["Siège lésion"] = data["Siège lésion"].astype(str).str.strip().str.lower()

# Création de la colonne normalisée
data["Siège normalisé"] = data["Siège lésion"].map(mapping_siege_harmonisé)


# Mapping des coordonnées
siege_map = {
    # Tête et cou
    "Tête": (0.5, 0.10),
    "Cou": (0.5, 0.15),
    # Épaules
    "Épaule": (0.5, 0.22),
    "Épaule gauche": (0.30, 0.22),
    "Épaule droite": (0.70, 0.22),
    # Bras
    "Bras": (0.5, 0.30),
    # Avant-bras
    "Avant-bras": (0.5, 0.38),
    "Avant-bras gauche": (0.30, 0.40),
    "Avant-bras droit": (0.70, 0.40),
    # Coudes
    "Coude": (0.5, 0.42),
    "Coude gauche": (0.28, 0.45),
    "Coude droit": (0.72, 0.45),
    # Poignets
    "Poignet": (0.5, 0.48),
    "Poignet gauche": (0.20, 0.52),
    "Poignet droit": (0.80, 0.52),
    # Mains
    "Main": (0.5, 0.53),
    "Main gauche": (0.15, 0.58),
    "Main droite": (0.85, 0.58),
    # Tronc / Dos
    "Tronc": (0.5, 0.35),
    "Dos": (0.5, 0.27),
    "Organes internes": (0.5, 0.33),
    # Hanche
    "Hanche": (0.5, 0.58),
    # Cuisses
    "Cuisse": (0.5, 0.65),
    # Genoux
    "Genou": (0.5, 0.73),
    "Genou gauche": (0.42, 0.73),
    "Genou droit": (0.58, 0.73),
    # Jambes
    "Jambe": (0.5, 0.80),
    # Chevilles
    "Cheville": (0.5, 0.90),
    "Cheville gauche": (0.44, 0.90),
    "Cheville droite": (0.56, 0.90),
    # Pieds
    "Pied": (0.5, 0.95),
    # Siège non précisé
    "Non précisé": (0.5, 0.5),
}


# Exemple de données
# 🧍 Carte des blessures pour un agent
st.subheader("🧍 Carte des blessures pour un agent")

matricule_input_map = st.text_input(
    "Entrez un matricule à afficher sur la carte (ex: 38638):", key="map"
)

if matricule_input_map:
    blessure_agent = data[data["Mat."] == str(matricule_input_map)][
        [
            "Age",
            "Siège normalisé",
            "Nature lésion",
            "Durée totale arrêt",
            "Date début initial",
            "Date fin initial",
        ]
    ].dropna(subset=["Siège normalisé"])

    if not blessure_agent.empty:
        st.write(f"🔎 Blessures relevées pour l'agent {matricule_input_map}:")
        st.dataframe(blessure_agent)

        fig, ax = plt.subplots(figsize=(4, 7))
        ax.imshow(image)
        ax.axis("off")

        # Sièges par défaut (non précisés) → rediriger vers un seul côté (gauche ici)
        lateralisation_par_defaut = {
            "Avant-bras": "Avant-bras gauche",
            "Poignet": "Poignet gauche",
            "Main": "Main gauche",
            "Coude": "Coude gauche",
            "Épaule": "Épaule gauche",
            "Genou": "Genou gauche",
            "Cheville": "Cheville gauche",
        }

        for _, row in blessure_agent.iterrows():
            siege_base = row["Siège normalisé"]
            lesion = row["Nature lésion"]

            # Forcer côté gauche si siège non latéralisé
            siege = lateralisation_par_defaut.get(siege_base, siege_base)

            if siege in siege_map:
                x, y = siege_map[siege]
                ax.plot(x * image.shape[1], y * image.shape[0], "ro", markersize=10)
                ax.text(
                    x * image.shape[1],
                    y * image.shape[0] - 10,
                    siege_base,  # Affiche le texte d'origine (pas le siège redirigé)
                    color="white",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="black",
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round,pad=0.2",
                    ),
                )
            else:
                st.warning(f"❗️ Le siège « {siege_base} » n'est pas mappé.")

        st.pyplot(fig)
