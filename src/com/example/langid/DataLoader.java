package com.example.langid;

import java.io.*;
import java.util.*;

/**
 * Classe DataLoader
 *
 * Cette classe permet de charger des données textuelles annotées depuis un fichier CSV,
 * d'éliminer les doublons, et de générer un nouveau fichier nettoyé au format CSV.
 * Elle stocke en mémoire les textes, leurs labels associés, et la liste des labels uniques.
 */
public class DataLoader {

    // Liste des textes extraits du fichier d'entrée
    private List<String> texts = new ArrayList<>();
    // Liste des labels (langues) correspondant à chaque texte
    private List<String> labels = new ArrayList<>();
    // Liste des labels uniques rencontrés
    private List<String> uniqueLabels = new ArrayList<>();

    /**
     * Constructeur DataLoader
     *
     * Lit un fichier CSV en entrée, filtre et nettoie les données,
     * puis écrit les résultats dans un nouveau fichier CSV.
     *
     * @param in  chemin du fichier CSV en entrée (avec en-tête)
     * @param out chemin du fichier CSV de sortie nettoyé
     * @throws IOException en cas d'erreur de lecture ou d'écriture
     */
    public DataLoader(String in, String out) throws IOException {
        // Ensemble pour détecter et ignorer les doublons de texte
        Set<String> seen = new HashSet<>();
        BufferedReader br = new BufferedReader(new FileReader(in));
        BufferedWriter bw = new BufferedWriter(new FileWriter(out));

        // Lire et ignorer la première ligne (en-tête)
        String header = br.readLine();
        // Écrire un nouvel en-tête standardisé dans le fichier de sortie
        bw.write("Text,language\n");

        String line;
        // Parcours de chaque ligne du fichier d'entrée
        while ((line = br.readLine()) != null) {
            // Trouver la dernière virgule pour séparer texte et label
            int idx = line.lastIndexOf(',');
            if (idx < 0) continue; // Ligne mal formée, on l'ignore

            // Extraire et nettoyer le texte
            String txt = line.substring(0, idx)
                             .trim()
                             .replaceAll("^\"|\"$", "");
            // Extraire et nettoyer le label (langue)
            String lang = line.substring(idx + 1)
                              .trim()
                              .replaceAll("^\"|\"$", "")
                              .toLowerCase();

            // Filtrer : texte ou label vide, ou doublon détecté
            if (txt.isEmpty() || lang.isEmpty() || seen.contains(txt)) continue;

            // Marquer le texte comme vu pour éviter les doublons
            seen.add(txt);
            // Ajouter aux listes internes
            texts.add(txt);
            labels.add(lang);
            // Si nouveau label, l'ajouter à la liste des uniques
            if (!uniqueLabels.contains(lang)) {
                uniqueLabels.add(lang);
            }

            // Écrire la ligne nettoyée dans le fichier de sortie
            bw.write("\"" + txt.replace("\"", "'") + "\"," + lang + "\n");
        }

        // Fermer les flux de lecture et d'écriture
        br.close();
        bw.close();
    }

    /**
     * Retourne la liste des textes chargés
     *
     * @return liste de textes
     */
    public List<String> getTexts() {
        return texts;
    }

    /**
     * Retourne la liste des labels correspondants aux textes
     *
     * @return liste de labels
     */
    public List<String> getLabels() {
        return labels;
    }

    /**
     * Retourne la liste des labels uniques
     *
     * @return liste des labels uniques
     */
    public List<String> getUniqueLabels() {
        return uniqueLabels;
    }
}
