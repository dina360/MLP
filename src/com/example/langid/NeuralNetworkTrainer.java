     package com.example.langid;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.IntStream;

/**
 * Classe NeuralNetworkTrainer
 *
 * Cette application Java :
 * 1) Charge et nettoie un dataset CSV (DataLoader).
 * 2) Convertit les textes en vecteurs TF-IDF (TFIDFCharacterNGram).
 * 3) Initialise et entraîne un réseau de neurones simple (NeuralNetwork) en mini-batchs.
 * 4) Évalue la précision sur un jeu de test.
 * 5) Démarre un serveur HTTP pour exposer :
 *    - /stats : statistiques d'entraînement au format JSON
 *    - /classify : classification de nouveaux textes via POST
 *    - /      : fichiers statiques du frontend
 */
public class NeuralNetworkTrainer {

    // Réseau de neurones et extraction TF-IDF
    private static NeuralNetwork nn;
    private static TFIDFCharacterNGram tfidf;
    private static List<String> uniqueLabels; // classes détectées

    // Statistiques globales exposées par /stats
    private static int totalExamples;
    private static int trainSize;
    private static int testSize;
    private static int epochsRun;
    private static int inputDim;
    private static int hiddenDim;
    private static int outputDim;
    private static double trainAcc;
    private static double testAcc;

    public static void main(String[] args) throws Exception {
        // 1) Chargement et nettoyage des données
        DataLoader loader = new DataLoader(
            "src/dataset/dataset.csv",
            "src/dataset/cleaned.csv"
        );
        List<String> texts      = loader.getTexts();       // textes nettoyés
        List<String> labelsList = loader.getLabels();      // labels associés
        uniqueLabels            = loader.getUniqueLabels();
        int numClasses          = uniqueLabels.size();
        int n                   = texts.size();
        totalExamples = n;      // enregistre le nombre total

        // 2) Extraction TF-IDF sur caractères (top 5000 n-grammes)
        tfidf     = new TFIDFCharacterNGram(texts, 5000);
        inputDim  = tfidf.getVocabSize(); // taille du vecteur d'entrée
        hiddenDim = 64;                   // dimension cachée fixe
        outputDim = numClasses;

        // 3) Construction des features et encodage des labels
        List<Map<Integer, Double>> features = new ArrayList<>(n);
        List<Integer> labelsIdx = new ArrayList<>(n);
        Map<String,Integer> labelMap = new HashMap<>();
        for (int i = 0; i < numClasses; i++) {
            labelMap.put(uniqueLabels.get(i), i);
        }
        for (int i = 0; i < n; i++) {
            features.add(tfidf.getTfIdfSparseForNewDoc(texts.get(i)));
            labelsIdx.add(labelMap.get(labelsList.get(i)));
        }

        // 4) Mélange et découpage 80% entraînement / 20% test
        List<Integer> idxs = IntStream.range(0, n).collect(ArrayList::new, List::add, List::addAll);
        Collections.shuffle(idxs, new Random(12345));
        trainSize = (int)(0.8 * n);
        testSize  = n - trainSize;
        List<Map<Integer, Double>> trainF = new ArrayList<>(trainSize);
        List<Integer> trainL            = new ArrayList<>(trainSize);
        List<Map<Integer, Double>> testF  = new ArrayList<>(testSize);
        List<Integer> testL             = new ArrayList<>(testSize);
        for (int i = 0; i < n; i++) {
            int id = idxs.get(i);
            if (i < trainSize) {
                trainF.add(features.get(id));
                trainL.add(labelsIdx.get(id));
            } else {
                testF.add(features.get(id));
                testL.add(labelsIdx.get(id));
            }
        }
        System.out.printf("Exemples d entrainement : %d  |  Exemples de test : %d  |  Nombre de classes : %d%n",
            trainSize, testSize, numClasses
        );

        // 5) Création du réseau de neurones avec lr=0.001
        nn = new NeuralNetwork(inputDim, hiddenDim, outputDim, 0.001);

        // 6) Préparation de l'ordre de mini-batch
        List<Integer> trainOrder = IntStream.range(0, trainSize).collect(ArrayList::new, List::add, List::addAll);

        // 7) Boucle d'entraînement en mini-batchs
        int maxEpochs  = 2;
        int batchSize  = 32;
        int numBatches = (trainSize + batchSize - 1) / batchSize;
        trainAcc       = 0.0;
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            epochsRun = epoch;
            Collections.shuffle(trainOrder, new Random(12345 + epoch));
            double epochLoss    = 0;
            double epochCorrect = 0;

            for (int b = 0; b < numBatches; b++) {
                int start = b * batchSize;
                int end   = Math.min(start + batchSize, trainSize);

                // Accumulateurs de gradients
                double[][] sumW1 = new double[hiddenDim][inputDim];
                double[]   sumb1 = new double[hiddenDim];
                double[][] sumW2 = new double[outputDim][hiddenDim];
                double[]   sumb2 = new double[outputDim];

                double batchLoss    = 0;
                double batchCorrect = 0;

                // Calcul forward/backward pour chaque exemple du batch
                for (int i = start; i < end; i++) {
                    int localIdx = trainOrder.get(i);
                    var res = nn.forwardBackward(
                        trainF.get(localIdx), trainL.get(localIdx)
                    );
                    batchLoss    += res.loss;
                    if (res.pred == trainL.get(localIdx)) batchCorrect++;

                    // Ajout des gradients
                    for (int x = 0; x < hiddenDim; x++) {
                        sumb1[x] += res.gb1[x];
                        for (int y = 0; y < inputDim; y++) sumW1[x][y] += res.gW1[x][y];
                    }
                    for (int x = 0; x < outputDim; x++) {
                        sumb2[x] += res.gb2[x];
                        for (int y = 0; y < hiddenDim; y++) sumW2[x][y] += res.gW2[x][y];
                    }
                }

                // Moyennage des gradients
                int bs = end - start;
                for (int x = 0; x < hiddenDim; x++) {
                    sumb1[x] /= bs;
                    for (int y = 0; y < inputDim; y++) sumW1[x][y] /= bs;
                }
                for (int x = 0; x < outputDim; x++) {
                    sumb2[x] /= bs;
                    for (int y = 0; y < hiddenDim; y++) sumW2[x][y] /= bs;
                }

                // Mise à jour des paramètres par Adam
                nn.applyAdam(sumW1, sumb1, sumW2, sumb2);

                epochLoss    += batchLoss;
                epochCorrect += batchCorrect;
                double batchAcc = 100.0 * batchCorrect / bs;
                if (batchAcc < 70.0) {
                    System.out.printf(
                        "Epoque %d, lot %d/%d  |  Precision lot : %.2f%%  |  Perte lot : %.4f%n",
                        epoch, b+1, numBatches, batchAcc, batchLoss/bs
                    );
                }
            }

            // Statistiques par epoch
            trainAcc = epochCorrect / trainSize;
            double trainAccPct = 100.0 * trainAcc;
            double avgLoss     = epochLoss / trainSize;
            System.out.printf(
                "- Epoque %d/%d  |  Precision entrainement : %.2f%%  |  Perte moyenne : %.4f%n",
                epoch, maxEpochs, trainAccPct, avgLoss
            );
            if (trainAccPct >= 100.0) break; // arrêt anticipé si 100%
        }

        // 8) Évaluation sur le set de test
        double testCorr = 0;
        for (int i = 0; i < testF.size(); i++) {
            int pred = nn.predict(testF.get(i));
            if (pred == testL.get(i)) testCorr++;
        }
        testAcc = testCorr / testSize;
        System.out.printf("Precision sur les donnees de test : %.2f%%%n", testAcc * 100);

        // 9) Démarrage du serveur HTTP
        HttpServer server = HttpServer.create(new InetSocketAddress(4567), 0);
        server.createContext("/stats",    new StatsHandler());
        server.createContext("/classify", new ClassifyHandler());
        server.createContext("/",         new StaticFileHandler("src/frontend"));
        server.setExecutor(null);
        server.start();
        System.out.println("Serveur HTTP demarre sur http://localhost:4567");
    }

    /**
     * Handler pour /stats : renvoie les statistiques au format JSON
     */
    static class StatsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            if (!"GET".equalsIgnoreCase(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }
            // Construction du JSON avec toutes les stats et classes
            StringBuilder sb = new StringBuilder();
            sb.append("{")
              .append("\"totalExamples\":").append(totalExamples).append(",")
              .append("\"trainSize\":").append(trainSize).append(",")
              .append("\"testSize\":").append(testSize).append(",")
              .append("\"epochs\":").append(epochsRun).append(",")
              .append("\"inputDim\":").append(inputDim).append(",")
              .append("\"hiddenDim\":").append(hiddenDim).append(",")
              .append("\"outputDim\":").append(outputDim).append(",")
              .append("\"trainAcc\":").append(String.format(Locale.US, "%.4f", trainAcc)).append(",")
              .append("\"testAcc\":").append(String.format(Locale.US, "%.4f", testAcc)).append(",")
              .append("\"classes\":[");
            for (int i = 0; i < uniqueLabels.size(); i++) {
                sb.append("\"").append(uniqueLabels.get(i)).append("\"");
                if (i + 1 < uniqueLabels.size()) sb.append(",");
            }
            sb.append("]}");

            byte[] resp = sb.toString().getBytes("UTF-8");
            ex.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
            ex.sendResponseHeaders(200, resp.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(resp);
            }
        }
    }

    /**
     * Handler pour /classify : POST text -> retourne language+confidence
     */
    static class ClassifyHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                sendJson(ex, 405, "{\"error\":\"Method Not Allowed\"}");
                return;
            }
            String body = new String(ex.getRequestBody().readAllBytes(), "UTF-8");
            String text = parseForm(body, "text");
            if (text == null || text.isBlank()) {
                sendJson(ex, 400, "{\"error\":\"'text' manquant\"}");
                return;
            }
            text = URLDecoder.decode(text, "UTF-8").trim();
            if (text.length() > 500) text = text.substring(0, 500);

            Map<Integer, Double> vec = tfidf.getTfIdfSparseForNewDoc(text);
            double[] probs = nn.predictProbs(vec);
            int idx = 0;
            for (int i = 1; i < probs.length; i++) {
                if (probs[i] > probs[idx]) idx = i;
            }
            String lang = uniqueLabels.get(idx);
            double conf = probs[idx];

            String json = String.format(
                Locale.US,
                "{\"language\":\"%s\",\"confidence\":%.4f}",
                lang, conf
            );
            sendJson(ex, 200, json);
        }

        // Extrait la valeur d'un champ du formulaire x-www-form-urlencoded
        private String parseForm(String body, String key) {
            for (String pair : body.split("&")) {
                String[] kv = pair.split("=", 2);
                if (kv.length == 2 && kv[0].equals(key)) return kv[1];
            }
            return null;
        }
        // Envoie une réponse JSON standardisée
        private void sendJson(HttpExchange ex, int code, String json)
        throws IOException {
            byte[] bytes = json.getBytes("UTF-8");
            ex.getResponseHeaders().set(
              "Content-Type","application/json; charset=UTF-8"
            );
            ex.sendResponseHeaders(code, bytes.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(bytes);
            }
        }
    }

    /**
     * Handler pour servir les fichiers statiques du frontend
     */
    static class StaticFileHandler implements HttpHandler {
        private final String root; // répertoire racine
        public StaticFileHandler(String root) { this.root = root; }
        @Override
        public void handle(HttpExchange ex) throws IOException {
            String uri = ex.getRequestURI().getPath();
            if (uri.equals("/")) uri = "/index.html";
            if (uri.contains("..")) { send404(ex); return; }
            File f = new File(root + uri.replace("/", File.separator));
            if (!f.exists() || f.isDirectory()) { send404(ex); return; }
            String ct = guessContentType(f.getName());
            byte[] data = Files.readAllBytes(f.toPath());
            ex.getResponseHeaders().set("Content-Type", ct);
            ex.sendResponseHeaders(200, data.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(data); }
        }
        // Répond 404 simple
        private void send404(HttpExchange ex) throws IOException {
            byte[] msg = "404 Not Found".getBytes("UTF-8");
            ex.getResponseHeaders().set(
              "Content-Type","text/plain; charset=UTF-8"
            );
            ex.sendResponseHeaders(404, msg.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(msg);
            }
        }
        // Déduit le Content-Type selon l'extension
        private String guessContentType(String name) {
            if (name.endsWith(".html")) return "text/html; charset=UTF-8";
            if (name.endsWith(".css"))  return "text/css; charset=UTF-8";
            if (name.endsWith(".js"))   return "application/javascript; charset=UTF-8";
            if (name.endsWith(".json")) return "application/json; charset=UTF-8";
            if (name.endsWith(".png"))  return "image/png";
            if (name.endsWith(".jpg")||name.endsWith(".jpeg"))
                return "image/jpeg";
            return "application/octet-stream";
        }
    }
}
