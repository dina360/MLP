package com.example.langid;

/**
 * Classe AdamOptimizer
 * 
 * Cette classe implémente l'algorithme d'optimisation Adam pour mettre à jour
 * les paramètres d'un modèle (vecteurs ou matrices) en fonction des gradients.
 * Elle maintient des moyennes mobiles du premier et second moment des gradients,
 * applique une correction de biais, puis ajuste les paramètres.
 */
public class AdamOptimizer {
    // Facteurs de décroissance pour les moyennes mobiles
    private final double beta1;   // Poids pour la moyenne du premier moment
    private final double beta2;   // Poids pour la moyenne du second moment
    private final double epsilon; // Petit terme pour la stabilité numérique
    private final double lr;      // Taux d'apprentissage
    
    // Variables pour stocker les moments pour les vecteurs
    private double[] m1, v1;
    // Variables pour stocker les moments pour les matrices
    private double[][] m2, v2;
    // Compteur d'itérations de mise à jour
    private int t = 0;

    /**
     * Constructeur pour optimiser des paramètres sous forme de vecteur
     *
     * @param lr       taux d'apprentissage
     * @param beta1    coefficient de décroissance pour le premier moment
     * @param beta2    coefficient de décroissance pour le second moment
     * @param epsilon  terme d'incrément pour éviter la division par zéro
     * @param dim      dimension du vecteur de paramètres
     */
    public AdamOptimizer(double lr, double beta1, double beta2, double epsilon, int dim) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        // Initialisation des tableaux de moments à zéro
        this.m1 = new double[dim];
        this.v1 = new double[dim];
    }

    /**
     * Constructeur pour optimiser des paramètres sous forme de matrice
     *
     * @param lr       taux d'apprentissage
     * @param beta1    coefficient de décroissance pour le premier moment
     * @param beta2    coefficient de décroissance pour le second moment
     * @param epsilon  terme d'incrément pour éviter la division par zéro
     * @param rows     nombre de lignes de la matrice de paramètres
     * @param cols     nombre de colonnes de la matrice de paramètres
     */
    public AdamOptimizer(double lr, double beta1, double beta2, double epsilon, int rows, int cols) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        // Initialisation des matrices de moments à zéro
        this.m2 = new double[rows][cols];
        this.v2 = new double[rows][cols];
    }

    /**
     * Met à jour un vecteur de paramètres (par exemple biais)
     *
     * @param param  tableau de paramètres à ajuster
     * @param grad   tableau des gradients correspondants
     */
    public void update(double[] param, double[] grad) {
        t++; // Incrémenter la date d'itération
        for (int i = 0; i < param.length; i++) {
            // Calcul des moyennes mobiles des moments
            m1[i] = beta1 * m1[i] + (1 - beta1) * grad[i];
            v1[i] = beta2 * v1[i] + (1 - beta2) * grad[i] * grad[i];
            // Correction de biais
            double mHat = m1[i] / (1 - Math.pow(beta1, t));
            double vHat = v1[i] / (1 - Math.pow(beta2, t));
            // Mise à jour du paramètre
            param[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }

    /**
     * Met à jour une matrice de paramètres (par exemple poids d'une couche)
     *
     * @param param  matrice de paramètres à ajuster
     * @param grad   matrice des gradients correspondants
     */
    public void update(double[][] param, double[][] grad) {
        t++; // Incrémenter la date d'itération
        int rows = param.length;
        int cols = param[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Calcul des moyennes mobiles des moments
                m2[i][j] = beta1 * m2[i][j] + (1 - beta1) * grad[i][j];
                v2[i][j] = beta2 * v2[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
                // Correction de biais
                double mHat = m2[i][j] / (1 - Math.pow(beta1, t));
                double vHat = v2[i][j] / (1 - Math.pow(beta2, t));
                // Mise à jour du paramètre
                param[i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }
}
