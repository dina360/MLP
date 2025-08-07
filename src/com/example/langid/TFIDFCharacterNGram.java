package com.example.langid;

import com.example.langid.utils.TextUtils;
import java.text.Normalizer;
import java.util.*;

/**
 * Classe TFIDFCharacterNGram
 *
 * Calcule les vecteurs TF-IDF basés sur des n-grammes de caractères.
 * - Construit un vocabulaire limité par `maxVocab` à partir des documents fournis.
 * - Calcule les fréquences de termes (TF) et l'inverse document frequency (IDF).
 * - Fournit une représentation sparse TF-IDF pour de nouveaux textes.
 */
public class TFIDFCharacterNGram {
    // Map des tokens vers leur index dans le vocabulaire (ordonné)
    private Map<String,Integer> tokenToIndex = new LinkedHashMap<>();
    // Liste des maps de fréquence TF pour chaque document
    private List<Map<Integer,Integer>> tfPerDoc;
    // Tableau des IDF pour chaque token
    private double[] idf;
    // Taille effective du vocabulaire
    private int vocabSize;

    /**
     * Constructeur : initialise TF et IDF à partir des documents.
     *
     * @param docs     liste des documents (chaînes) à indexer
     * @param maxVocab nombre maximal de tokens à conserver dans le vocabulaire
     */
    public TFIDFCharacterNGram(List<String> docs, int maxVocab) {
        tfPerDoc = new ArrayList<>();

        // 1) Parcours des documents pour construire TF et vocabulaire
        for (String d : docs) {
            Map<Integer,Integer> freq = new HashMap<>();
            // Pour chaque token généré par TextUtils
            for (String tk : TextUtils.getTokens(d)) {
                // Récupère ou assigne un nouvel index au token
                int idx = tokenToIndex.computeIfAbsent(
                    tk, k -> tokenToIndex.size()
                );
                // Incrémente la fréquence pour ce token
                freq.put(idx, freq.getOrDefault(idx, 0) + 1);
            }
            tfPerDoc.add(freq);
            // Stopper si on atteint la taille max du vocabulaire
            if (tokenToIndex.size() > maxVocab) break;
        }
        vocabSize = tokenToIndex.size();

        // Nombre de documents réellement indexés
        int N = tfPerDoc.size();

        // 2) Calcul du document frequency (DF) pour chaque token
        int[] df = new int[vocabSize];
        for (var m : tfPerDoc) {
            for (int i : m.keySet()) {
                if (i < vocabSize) df[i]++;
            }
        }

        // 3) Calcul de l'IDF = log(N / (1 + DF))
        idf = new double[vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            idf[i] = Math.log((double) N / (1 + df[i]));
        }
    }

    /**
     * Retourne la taille du vocabulaire.
     *
     * @return nombre de tokens indexés
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Produit un vecteur TF-IDF sparse pour un nouveau texte.
     *
     * @param text le document à transformer
     * @return map index->poids TF-IDF
     */
    public Map<Integer,Double> getTfIdfSparseForNewDoc(String text) {
        Map<Integer,Integer> freq = new HashMap<>();
        // 1) Calcul des fréquences TF pour le nouveau texte
        for (String tk : TextUtils.getTokens(text)) {
            Integer idx = tokenToIndex.get(tk);
            if (idx != null) {
                freq.put(idx, freq.getOrDefault(idx, 0) + 1);
            }
        }
        // Somme des occurrences pour normalisation
        int sum = freq.values().stream().mapToInt(x -> x).sum();

        // Si aucun token reconnu, retourner vecteur vide
        Map<Integer,Double> tfidf = new HashMap<>();
        if (sum == 0) return tfidf;

        // 2) Calcul TF-IDF normalisé : (TF/sum) * IDF
        for (var e : freq.entrySet()) {
            int i = e.getKey();
            int f = e.getValue();
            tfidf.put(
                i,
                (f / (double) sum) * idf[i]
            );
        }
        return tfidf;
    }
}
