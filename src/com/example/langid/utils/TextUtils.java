package com.example.langid.utils;

import java.text.Normalizer;
import java.util.*;

/**
 * Classe TextUtils
 *
 * Fournit des utilitaires de tokenisation textuelle basés sur des n-grammes de caractères.
 * - Normalise le texte (minuscules, suppression des accents et signes de ponctuation).
 * - Génère pour chaque mot : le token mot et tous ses bigrammes de caractères.
 */
public class TextUtils {

    /**
     * Découpe une chaîne en tokens (mots et bigrammes de caractères).
     *
     * @param text chaîne d'entrée à tokeniser
     * @return liste ordonnée de tokens (mots puis bigrammes)
     */
    public static List<String> getTokens(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null) return tokens; // gérer le cas null

        // 1) Normalisation Unicode : minuscules, suppression des accents
        String s = Normalizer.normalize(
            text.toLowerCase(), Normalizer.Form.NFD
        ).replaceAll("\\p{InCombiningDiacriticalMarks}+", "")
         // 2) Remplacement des caractères non littéraux par espaces
         .replaceAll("[^\\p{L}\\s]", " ")
         // 3) Nettoyage des espaces (trim + fusion de plusieurs espaces)
         .trim()
         .replaceAll("\\s+", " ");

        // 4) Pour chaque mot, on ajoute :
        //    - le mot lui-même
        //    - tous les bigrammes de caractères glissants
        for (String w : s.split(" ")) {
            if (w.isEmpty()) continue;
            tokens.add(w);
            for (int i = 0; i + 1 < w.length(); i++) {
                tokens.add(w.substring(i, i + 2));
            }
        }
        return tokens;
    }
}
