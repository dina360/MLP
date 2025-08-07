package com.example.langid;

import java.util.Map;
import java.util.Random;

/**
 * Classe NeuralNetwork
 *
 * Implémente un réseau de neurones à une couche cachée pour la classification.
 * - Initialise les poids et biais avec la méthode Xavier.
 * - Réalise le forward + backward pour un exemple et calcule le loss.
 * - Applique l'optimiseur Adam sur des mini-batchs.
 * - Fournit des méthodes de prédiction (classe et probabilités).
 */
public class NeuralNetwork {
    // Dimensions de l'entrée, couche cachée et sortie
    public final int inputDim, hiddenDim, outputDim;
    // Poids et biais
    private double[][] W1; // poids entre l'entrée et la couche cachée
    private double[][] W2; // poids entre la couche cachée et la sortie
    private double[] b1;   // biais de la couche cachée
    private double[] b2;   // biais de la couche de sortie
    // Optimiseurs Adam pour chaque paramètre
    private final AdamOptimizer optW1, optb1, optW2, optb2;
    // Générateur de nombres aléatoires pour l'initialisation
    private final Random rng = new Random(123);

    /**
     * Constructeur : initialise poids, biais et optimiseurs.
     * Utilise l'initialisation Xavier pour W1 et W2.
     *
     * @param inputDim  nombre de features en entrée
     * @param hiddenDim nombre de neurones dans la couche cachée
     * @param outputDim nombre de classes en sortie
     * @param lr        taux d'apprentissage pour Adam
     */
    public NeuralNetwork(int inputDim, int hiddenDim, int outputDim, double lr) {
        this.inputDim  = inputDim;
        this.hiddenDim = hiddenDim;
        this.outputDim = outputDim;

        // --- Initialisation Xavier des poids ---
        W1 = new double[hiddenDim][inputDim];
        double s1 = Math.sqrt(2.0/(inputDim + hiddenDim));
        for (int i = 0; i < hiddenDim; i++)
            for (int j = 0; j < inputDim; j++)
                W1[i][j] = rng.nextGaussian() * s1;

        b1 = new double[hiddenDim]; // biais initialisés à 0

        W2 = new double[outputDim][hiddenDim];
        double s2 = Math.sqrt(2.0/(hiddenDim + outputDim));
        for (int i = 0; i < outputDim; i++)
            for (int j = 0; j < hiddenDim; j++)
                W2[i][j] = rng.nextGaussian() * s2;

        b2 = new double[outputDim]; // biais initialisés à 0

        // --- Création des optimiseurs Adam ---
        double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        optW1 = new AdamOptimizer(lr, beta1, beta2, eps, hiddenDim, inputDim);
        optb1 = new AdamOptimizer(lr, beta1, beta2, eps, hiddenDim);
        optW2 = new AdamOptimizer(lr, beta1, beta2, eps, outputDim, hiddenDim);
        optb2 = new AdamOptimizer(lr, beta1, beta2, eps, outputDim);
    }

    /**
     * Conteneur des gradients, loss et prédiction pour une itération.
     */
    public static class Result {
        public final double[][] gW1;
        public final double[]   gb1;
        public final double[][] gW2;
        public final double[]   gb2;
        public final double     loss;
        public final int        pred;

        public Result(double[][] gW1, double[] gb1,
                      double[][] gW2, double[] gb2,
                      double loss, int pred) {
            this.gW1 = gW1; this.gb1 = gb1;
            this.gW2 = gW2; this.gb2 = gb2;
            this.loss = loss; this.pred = pred;
        }
    }

    /**
     * Forward + backward pour un exemple (sparse input Map).
     *
     * @param x     entrée sous forme d'indices sparsifiés
     * @param yTrue label réel
     * @return gradients, loss et prédiction
     */
    public Result forwardBackward(Map<Integer, Double> x, int yTrue) {
        // --- Forward ---
        // Couche cachée avec tanh
        double[] h = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double sum = b1[i];
            for (var e : x.entrySet())
                sum += W1[i][e.getKey()] * e.getValue();
            h[i] = Math.tanh(sum);
        }
        // Couche de sortie (logits)
        double[] o = new double[outputDim];
        for (int k = 0; k < outputDim; k++) {
            double sum = b2[k];
            for (int i = 0; i < hiddenDim; i++)
                sum += W2[k][i] * h[i];
            o[k] = sum;
        }
        // Softmax + calcul du loss cross-entropy
        double max = o[0];
        for (double v : o) if (v > max) max = v;
        double denom = 0;
        for (int k = 0; k < outputDim; k++) denom += Math.exp(o[k] - max);
        double[] p = new double[outputDim];
        for (int k = 0; k < outputDim; k++)
            p[k] = Math.exp(o[k] - max) / denom;
        double loss = -Math.log(p[yTrue] + 1e-15);
        // Prédiction de la classe
        int yPred = 0;
        for (int k = 1; k < outputDim; k++)
            if (p[k] > p[yPred]) yPred = k;

        // --- Backpropagation ---
        // Gradient sur la couche de sortie
        double[] delta2 = new double[outputDim];
        for (int k = 0; k < outputDim; k++)
            delta2[k] = p[k] - (k == yTrue ? 1.0 : 0.0);

        double[][] gW2 = new double[outputDim][hiddenDim];
        double[] gb2   = new double[outputDim];
        for (int k = 0; k < outputDim; k++) {
            gb2[k] = delta2[k];
            for (int i = 0; i < hiddenDim; i++)
                gW2[k][i] = delta2[k] * h[i];
        }

        // Gradient sur la couche cachée
        double[] delta1 = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double sum = 0;
            for (int k = 0; k < outputDim; k++)
                sum += W2[k][i] * delta2[k];
            delta1[i] = (1 - h[i] * h[i]) * sum;
        }

        double[][] gW1 = new double[hiddenDim][inputDim];
        double[] gb1   = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            gb1[i] = delta1[i];
            for (var e : x.entrySet())
                gW1[i][e.getKey()] = delta1[i] * e.getValue();
        }

        return new Result(gW1, gb1, gW2, gb2, loss, yPred);
    }

    /**
     * Applique Adam sur les sommes de gradients du mini-batch.
     *
     * @param sumW1 sommes des gradients pour W1
     * @param sumb1 sommes des gradients pour b1
     * @param sumW2 sommes des gradients pour W2
     * @param sumb2 sommes des gradients pour b2
     */
    public void applyAdam(double[][] sumW1, double[] sumb1,
                          double[][] sumW2, double[] sumb2) {
        optW1.update(W1, sumW1);
        optb1.update(b1, sumb1);
        optW2.update(W2, sumW2);
        optb2.update(b2, sumb2);
    }

    /**
     * Effectue un forward pour la prédiction seule.
     *
     * @param x entrée sparsifiée
     * @return classe prédite
     */
    public int predict(Map<Integer, Double> x) {
        double[] h = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double sum = b1[i];
            for (var e : x.entrySet()) sum += W1[i][e.getKey()] * e.getValue();
            h[i] = Math.tanh(sum);
        }
        double[] o = new double[outputDim];
        for (int c = 0; c < outputDim; c++) {
            double sum = b2[c];
            for (int i = 0; i < hiddenDim; i++) sum += W2[c][i] * h[i];
            o[c] = sum;
        }
        int pred = 0;
        for (int c = 1; c < outputDim; c++)
            if (o[c] > o[pred]) pred = c;
        return pred;
    }

    /**
     * Effectue un forward et retourne les probabilités (softmax).
     *
     * @param x entrée sparsifiée
     * @return vecteur de probabilités
     */
    public double[] predictProbs(Map<Integer, Double> x) {
        // Identique à predict(), mais avec softmax
        double[] h = new double[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) {
            double sum = b1[i];
            for (var e : x.entrySet()) sum += W1[i][e.getKey()] * e.getValue();
            h[i] = Math.tanh(sum);
        }
        double[] o = new double[outputDim];
        for (int k = 0; k < outputDim; k++) {
            double sum = b2[k];
            for (int i = 0; i < hiddenDim; i++) sum += W2[k][i] * h[i];
            o[k] = sum;
        }
        double max = o[0];
        for (double v : o) if (v > max) max = v;
        double denom = 0;
        for (double v : o) denom += Math.exp(v - max);
        double[] p = new double[outputDim];
        for (int k = 0; k < outputDim; k++)
            p[k] = Math.exp(o[k] - max) / denom;
        return p;
    }
}
