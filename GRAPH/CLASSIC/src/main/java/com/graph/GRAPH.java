package com.graph;

import com.mxgraph.layout.mxCircleLayout;
import com.mxgraph.util.mxCellRenderer;
import com.mxgraph.view.mxGraph;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.stat.clustering.KMeansPlusPlusClusterer;
import java.util.stream.IntStream;

public class GRAPH {
    private boolean[][] adjacencyMatrix;
    private int vertices;
    private int[] nodeCluster; // Store cluster assignments
    private double[][] nodePositions; // Store node positions
    private static final String[] COLORS = {
        "#4CAF50", "#2196F3", "#F44336", "#FFC107", "#9C27B0",
        "#00BCD4", "#FF9800", "#795548", "#607D8B", "#E91E63"
    };

    public GRAPH(int vertices) {
        this.vertices = vertices;
        this.adjacencyMatrix = new boolean[vertices][vertices];
        this.nodePositions = new double[vertices][2];
    }

    public void generateSpanningTree() {
        Random rand = new Random();
        java.util.List<Integer> connected = new java.util.ArrayList<>();
        java.util.List<Integer> unconnected = new java.util.ArrayList<>();

        // Initialize unconnected vertices
        for (int i = 0; i < vertices; i++) {
            unconnected.add(i);
        }

        // Start with a random vertex
        connected.add(unconnected.remove(rand.nextInt(unconnected.size())));

        // Connect unconnected vertices to the spanning tree
        while (!unconnected.isEmpty()) {
            int source = connected.get(rand.nextInt(connected.size()));
            int target = unconnected.remove(rand.nextInt(unconnected.size()));

            adjacencyMatrix[source][target] = true;
            adjacencyMatrix[target][source] = true; // Undirected graph
            connected.add(target);
        }

        // Force-directed layout
        double k = 100.0;
        double gravity = 0.1;
        int iterations = 100;
        
        double[][] forces = new double[vertices][2];
        
        // Initialize random positions
        for (int i = 0; i < vertices; i++) {
            nodePositions[i][0] = rand.nextDouble() * 700 + 50;
            nodePositions[i][1] = rand.nextDouble() * 700 + 50;
        }
        
        // Force-directed layout iterations
        for (int iter = 0; iter < iterations; iter++) {
            // Reset forces
            for (int i = 0; i < vertices; i++) {
                forces[i][0] = forces[i][1] = 0;
            }
            
            // Calculate repulsive forces
            for (int i = 0; i < vertices; i++) {
                for (int j = i + 1; j < vertices; j++) {
                    double dx = nodePositions[j][0] - nodePositions[i][0];
                    double dy = nodePositions[j][1] - nodePositions[i][1];
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1) dist = 1;
                    
                    double force = k * k / dist;
                    forces[i][0] -= (dx / dist) * force;
                    forces[i][1] -= (dy / dist) * force;
                    forces[j][0] += (dx / dist) * force;
                    forces[j][1] += (dy / dist) * force;
                }
            }
            
            // Calculate attractive forces
            for (int i = 0; i < vertices; i++) {
                for (int j = i + 1; j < vertices; j++) {
                    if (adjacencyMatrix[i][j]) {
                        double dx = nodePositions[j][0] - nodePositions[i][0];
                        double dy = nodePositions[j][1] - nodePositions[i][1];
                        double dist = Math.sqrt(dx * dx + dy * dy);
                        if (dist < 1) dist = 1;
                        
                        double force = dist * dist / k;
                        forces[i][0] += (dx / dist) * force;
                        forces[i][1] += (dy / dist) * force;
                        forces[j][0] -= (dx / dist) * force;
                        forces[j][1] -= (dy / dist) * force;
                    }
                }
            }
            
            // Apply forces and gravity
            for (int i = 0; i < vertices; i++) {
                forces[i][0] += (400 - nodePositions[i][0]) * gravity;
                forces[i][1] += (400 - nodePositions[i][1]) * gravity;
                
                nodePositions[i][0] += Math.min(Math.max(forces[i][0], -10), 10);
                nodePositions[i][1] += Math.min(Math.max(forces[i][1], -10), 10);
            }
        }
    }

    public void printAdjacencyMatrix() {
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                System.out.print((adjacencyMatrix[i][j] ? 1 : 0) + " ");
            }
            System.out.println();
        }
    }

    public void saveGraphAsPNG(String filename) {
        mxGraph graph = new mxGraph();
        Object parent = graph.getDefaultParent();

        graph.getStylesheet().getDefaultEdgeStyle().put("endArrow", "none");
        
        graph.getModel().beginUpdate();
        try {
            // Add vertices with stored positions
            Object[] vertexObjects = new Object[vertices];
            for (int i = 0; i < vertices; i++) {
                vertexObjects[i] = graph.insertVertex(parent, null, String.valueOf(i), 
                    nodePositions[i][0], nodePositions[i][1], 25, 25, "VERTEX_STYLE");
            }

            // Add edges
            for (int i = 0; i < vertices; i++) {
                for (int j = i + 1; j < vertices; j++) {
                    if (adjacencyMatrix[i][j]) {
                        graph.insertEdge(parent, null, "", vertexObjects[i], vertexObjects[j]);
                    }
                }
            }
        } finally {
            graph.getModel().endUpdate();
        }

        BufferedImage image = mxCellRenderer.createBufferedImage(graph, null, 2, null, true, null);
        try {
            ImageIO.write(image, "PNG", new File(filename));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void spectralClustering(int k) {
        // Create Laplacian matrix
        RealMatrix adjacency = new Array2DRowRealMatrix(vertices, vertices);
        RealMatrix degree = new Array2DRowRealMatrix(vertices, vertices);
        
        // Fill adjacency matrix and calculate degrees
        for (int i = 0; i < vertices; i++) {
            int degreeSum = 0;
            for (int j = 0; j < vertices; j++) {
                double value = adjacencyMatrix[i][j] ? 1.0 : 0.0;
                adjacency.setEntry(i, j, value);
                degreeSum += value;
            }
            degree.setEntry(i, i, degreeSum);
        }
        
        // Calculate Laplacian matrix (L = D - A)
        RealMatrix laplacian = degree.subtract(adjacency);
        
        // Compute eigenvalues and eigenvectors
        EigenDecomposition eigenDecomposition = new EigenDecomposition(laplacian);
        
        // Get k smallest eigenvalues (excluding the first one)
        double[] eigenvalues = eigenDecomposition.getRealEigenvalues();
        int[] indices = IntStream.range(1, vertices)
                               .boxed()
                               .sorted((i, j) -> Double.compare(eigenvalues[i], eigenvalues[j]))
                               .mapToInt(i -> i)
                               .limit(k)
                               .toArray();
        
        // Create feature matrix using k eigenvectors
        double[][] features = new double[vertices][k];
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < k; j++) {
                features[i][j] = eigenDecomposition.getEigenvector(indices[j]).getEntry(i);
            }
        }
        
        // Apply k-means to the feature matrix
        nodeCluster = new int[vertices];
        double[][] normalizedFeatures = normalizeFeatures(features);
        kMeansOnFeatures(normalizedFeatures, k);
    }
    
    private double[][] normalizeFeatures(double[][] features) {
        double[][] normalized = new double[vertices][features[0].length];
        for (int i = 0; i < vertices; i++) {
            double norm = 0.0;
            for (int j = 0; j < features[0].length; j++) {
                norm += features[i][j] * features[i][j];
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < features[0].length; j++) {
                    normalized[i][j] = features[i][j] / norm;
                }
            }
        }
        return normalized;
    }
    
    private void kMeansOnFeatures(double[][] features, int k) {
        Random rand = new Random();
        int[] centroidIndices = new int[k];
        double[][] centroids = new double[k][features[0].length];
        
        // Initialize centroids randomly
        for (int i = 0; i < k; i++) {
            centroidIndices[i] = rand.nextInt(vertices);
            centroids[i] = features[centroidIndices[i]].clone();
        }
        
        int maxIterations = 50;
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            // Assign points to nearest centroid
            for (int i = 0; i < vertices; i++) {
                int bestCluster = getBestCluster(features[i], centroids);
                if (nodeCluster[i] != bestCluster) {
                    nodeCluster[i] = bestCluster;
                    changed = true;
                }
            }
            
            // Update centroids
            updateCentroids(features, centroids);
            
            if (!changed) break;
        }
    }
    
    private int getBestCluster(double[] point, double[][] centroids) {
        int best = 0;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < centroids.length; i++) {
            double dist = 0;
            for (int j = 0; j < point.length; j++) {
                double diff = point[j] - centroids[i][j];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                best = i;
            }
        }
        return best;
    }
    
    private void updateCentroids(double[][] features, double[][] centroids) {
        int[] counts = new int[centroids.length];
        double[][] newCentroids = new double[centroids.length][features[0].length];
        
        for (int i = 0; i < vertices; i++) {
            counts[nodeCluster[i]]++;
            for (int j = 0; j < features[0].length; j++) {
                newCentroids[nodeCluster[i]][j] += features[i][j];
            }
        }
        
        for (int i = 0; i < centroids.length; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < features[0].length; j++) {
                    centroids[i][j] = newCentroids[i][j] / counts[i];
                }
            }
        }
    }

    public void kMeansClustering(int k) {
        if (nodePositions == null) return;
        
        nodeCluster = new int[vertices];
        Random rand = new Random();
        
        // Initialize centroids at random node positions
        double[][] centroids = new double[k][2];
        for (int i = 0; i < k; i++) {
            int randomNode = rand.nextInt(vertices);
            centroids[i][0] = nodePositions[randomNode][0];
            centroids[i][1] = nodePositions[randomNode][1];
        }

        int maxIterations = 50;
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            // Assign nodes to nearest centroid
            for (int i = 0; i < vertices; i++) {
                double minDist = Double.MAX_VALUE;
                int bestCluster = 0;
                
                for (int c = 0; c < k; c++) {
                    double dx = nodePositions[i][0] - centroids[c][0];
                    double dy = nodePositions[i][1] - centroids[c][1];
                    double dist = dx * dx + dy * dy;
                    
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = c;
                    }
                }
                
                if (nodeCluster[i] != bestCluster) {
                    changed = true;
                    nodeCluster[i] = bestCluster;
                }
            }

            // Update centroids
            double[][] newCentroids = new double[k][2];
            int[] counts = new int[k];
            
            for (int i = 0; i < vertices; i++) {
                int cluster = nodeCluster[i];
                newCentroids[cluster][0] += nodePositions[i][0];
                newCentroids[cluster][1] += nodePositions[i][1];
                counts[cluster]++;
            }
            
            for (int i = 0; i < k; i++) {
                if (counts[i] > 0) {
                    centroids[i][0] = newCentroids[i][0] / counts[i];
                    centroids[i][1] = newCentroids[i][1] / counts[i];
                }
            }
            
            if (!changed) break;
        }
    }

    public void labelPropagation(int k) {
        nodeCluster = new int[vertices];
        Random rand = new Random();
        
        // Initialize random labels
        for (int i = 0; i < vertices; i++) {
            nodeCluster[i] = rand.nextInt(k);
        }

        int maxIterations = 50;
        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            for (int i = 0; i < vertices; i++) {
                int[] clusterCounts = new int[k];
                // Count neighbor labels
                for (int j = 0; j < vertices; j++) {
                    if (adjacencyMatrix[i][j]) {
                        clusterCounts[nodeCluster[j]]++;
                    }
                }
                // Find most frequent label
                int maxCount = -1;
                int bestCluster = nodeCluster[i];
                for (int c = 0; c < k; c++) {
                    if (clusterCounts[c] > maxCount) {
                        maxCount = clusterCounts[c];
                        bestCluster = c;
                    }
                }
                if (nodeCluster[i] != bestCluster) {
                    nodeCluster[i] = bestCluster;
                    changed = true;
                }
            }
            if (!changed) break;
        }
    }

    public void saveClusteredGraphAsPNG(String filename) {
        mxGraph graph = new mxGraph();
        Object parent = graph.getDefaultParent();

        graph.getStylesheet().getDefaultEdgeStyle().put("endArrow", "none");

        graph.getModel().beginUpdate();
        try {
            Object[] vertexObjects = new Object[vertices];
            for (int i = 0; i < vertices; i++) {
                // Use original node positions
                // Create style with cluster color
                java.util.Map<String, Object> vertexStyle = new HashMap<>();
                vertexStyle.put("shape", "ellipse");
                vertexStyle.put("perimeter", "ellipsePerimeter");
                vertexStyle.put("fillColor", COLORS[nodeCluster[i] % COLORS.length]);
                vertexStyle.put("strokeColor", "#000000");
                vertexStyle.put("strokeWidth", "1");
                vertexStyle.put("fontColor", "#FFFFFF");
                vertexStyle.put("fontSize", "10");
                vertexStyle.put("fontStyle", "1");
                vertexStyle.put("verticalAlign", "middle");
                
                String styleName = "VERTEX_STYLE_" + i;
                graph.getStylesheet().putCellStyle(styleName, vertexStyle);
                vertexObjects[i] = graph.insertVertex(parent, null, String.valueOf(i), 
                    nodePositions[i][0], nodePositions[i][1], 25, 25, styleName);
            }

            // Add edges
            for (int i = 0; i < vertices; i++) {
                for (int j = i + 1; j < vertices; j++) {
                    if (adjacencyMatrix[i][j]) {
                        graph.insertEdge(parent, null, "", vertexObjects[i], vertexObjects[j]);
                    }
                }
            }
        } finally {
            graph.getModel().endUpdate();
        }

        BufferedImage image = mxCellRenderer.createBufferedImage(graph, null, 2, null, true, null);
        try {
            ImageIO.write(image, "PNG", new File(filename));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void edgeBetweennessClustering(int k) {
        nodeCluster = new int[vertices];
        boolean[][] currentGraph = new boolean[vertices][vertices];
        
        // Copy original graph
        for (int i = 0; i < vertices; i++) {
            currentGraph[i] = adjacencyMatrix[i].clone();
        }

        // Remove edges until we have k components
        while (countComponents(currentGraph) < k) {
            int[] edge = findHighestBetweennessEdge(currentGraph);
            if (edge == null) break;
            currentGraph[edge[0]][edge[1]] = false;
            currentGraph[edge[1]][edge[0]] = false;
        }

        // Label components
        boolean[] visited = new boolean[vertices];
        int currentCluster = 0;
        
        for (int i = 0; i < vertices; i++) {
            if (!visited[i]) {
                labelComponent(i, currentCluster, visited, currentGraph);
                currentCluster++;
            }
        }
    }

    private int[] findHighestBetweennessEdge(boolean[][] graph) {
        double maxBetweenness = -1;
        int[] bestEdge = null;

        for (int i = 0; i < vertices; i++) {
            for (int j = i + 1; j < vertices; j++) {
                if (graph[i][j]) {
                    double betweenness = calculateEdgeBetweenness(i, j, graph);
                    if (betweenness > maxBetweenness) {
                        maxBetweenness = betweenness;
                        bestEdge = new int[]{i, j};
                    }
                }
            }
        }
        return bestEdge;
    }

    private double calculateEdgeBetweenness(int u, int v, boolean[][] graph) {
        double betweenness = 0;
        
        for (int source = 0; source < vertices; source++) {
            if (source == u || source == v) continue;
            
            // Run BFS from source
            int[] parent = new int[vertices];
            Arrays.fill(parent, -1);
            Queue<Integer> queue = new LinkedList<>();
            queue.add(source);
            parent[source] = source;
            
            while (!queue.isEmpty()) {
                int current = queue.poll();
                for (int next = 0; next < vertices; next++) {
                    if (graph[current][next] && parent[next] == -1) {
                        parent[next] = current;
                        queue.add(next);
                    }
                }
            }
            
            // Check if the edge lies on the shortest path
            for (int target = source + 1; target < vertices; target++) {
                if (target == u || target == v) continue;
                
                int current = target;
                while (current != source && current != -1) {
                    int prev = parent[current];
                    if ((prev == u && current == v) || (prev == v && current == u)) {
                        betweenness += 1.0;
                        break;
                    }
                    current = prev;
                }
            }
        }
        
        return betweenness;
    }

    private int countComponents(boolean[][] graph) {
        boolean[] visited = new boolean[vertices];
        int components = 0;
        
        for (int i = 0; i < vertices; i++) {
            if (!visited[i]) {
                dfs(i, visited, graph);
                components++;
            }
        }
        
        return components;
    }

    private void dfs(int vertex, boolean[] visited, boolean[][] graph) {
        visited[vertex] = true;
        for (int i = 0; i < vertices; i++) {
            if (graph[vertex][i] && !visited[i]) {
                dfs(i, visited, graph);
            }
        }
    }

    private void labelComponent(int vertex, int cluster, boolean[] visited, boolean[][] graph) {
        visited[vertex] = true;
        nodeCluster[vertex] = cluster;
        for (int i = 0; i < vertices; i++) {
            if (graph[vertex][i] && !visited[i]) {
                labelComponent(i, cluster, visited, graph);
            }
        }
    }

    public void walktrapClustering(int k) {
        nodeCluster = new int[vertices];
        double[][] distances = new double[vertices][vertices];
        
        // Calculate transition probabilities
        double[][] P = new double[vertices][vertices];
        int[] degrees = new int[vertices];
        
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                if (adjacencyMatrix[i][j]) {
                    degrees[i]++;
                }
            }
        }
        
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                if (adjacencyMatrix[i][j]) {
                    P[i][j] = 1.0 / degrees[i];
                }
            }
        }
        
        // Compute random walk distances (4 steps)
        int steps = 4;
        for (int i = 0; i < vertices; i++) {
            double[] probabilities = new double[vertices];
            probabilities[i] = 1.0;
            
            // Perform random walk
            for (int step = 0; step < steps; step++) {
                double[] newProb = new double[vertices];
                for (int v = 0; v < vertices; v++) {
                    for (int u = 0; u < vertices; u++) {
                        newProb[v] += probabilities[u] * P[u][v];
                    }
                }
                probabilities = newProb;
            }
            
            // Store distances
            for (int j = 0; j < vertices; j++) {
                distances[i][j] = Math.sqrt(Math.abs(probabilities[j] - probabilities[i]));
            }
        }
        
        // Hierarchical clustering based on distances
        List<Set<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < vertices; i++) {
            Set<Integer> cluster = new HashSet<>();
            cluster.add(i);
            clusters.add(cluster);
        }
        
        while (clusters.size() > k) {
            // Find closest clusters
            double minDist = Double.MAX_VALUE;
            int c1 = -1, c2 = -1;
            
            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double dist = clusterDistance(clusters.get(i), clusters.get(j), distances);
                    if (dist < minDist) {
                        minDist = dist;
                        c1 = i;
                        c2 = j;
                    }
                }
            }
            
            // Merge clusters
            clusters.get(c1).addAll(clusters.get(c2));
            clusters.remove(c2);
        }
        
        // Assign cluster labels
        for (int i = 0; i < clusters.size(); i++) {
            for (int vertex : clusters.get(i)) {
                nodeCluster[vertex] = i;
            }
        }
    }
    
    private double clusterDistance(Set<Integer> c1, Set<Integer> c2, double[][] distances) {
        double sum = 0;
        for (int i : c1) {
            for (int j : c2) {
                sum += distances[i][j];
            }
        }
        return sum / (c1.size() * c2.size());
    }

    public void randomWalkClustering(int k) {
        nodeCluster = new int[vertices];
        Random rand = new Random();
        int numWalks = 100;
        int walkLength = 3;
        
        // Store walk frequencies
        double[][] walkMatrix = new double[vertices][vertices];
        
        // Perform random walks
        for (int start = 0; start < vertices; start++) {
            for (int w = 0; w < numWalks; w++) {
                int current = start;
                Set<Integer> visited = new HashSet<>();
                visited.add(current);
                
                for (int step = 0; step < walkLength; step++) {
                    List<Integer> neighbors = new ArrayList<>();
                    for (int i = 0; i < vertices; i++) {
                        if (adjacencyMatrix[current][i]) {
                            neighbors.add(i);
                        }
                    }
                    
                    if (neighbors.isEmpty()) break;
                    
                    int next = neighbors.get(rand.nextInt(neighbors.size()));
                    walkMatrix[start][next]++;
                    current = next;
                }
            }
        }
        
        // Normalize walk matrix
        for (int i = 0; i < vertices; i++) {
            double sum = 0;
            for (int j = 0; j < vertices; j++) {
                sum += walkMatrix[i][j];
            }
            if (sum > 0) {
                for (int j = 0; j < vertices; j++) {
                    walkMatrix[i][j] /= sum;
                }
            }
        }
        
        // Apply k-means to walk profiles
        kMeansOnFeatures(walkMatrix, k);
    }

    private static class ClusteringScore {
        String algorithm;
        double score;
        int interClusterEdges;
        double sizeDifference;

        ClusteringScore(String algorithm, int interClusterEdges, double stdDev, double avgSize, int numVertices) {
            this.algorithm = algorithm;
            this.interClusterEdges = interClusterEdges;
            this.sizeDifference = stdDev / avgSize; // Normalized standard deviation
            // Compute score (lower is better)
            this.score = (interClusterEdges * 0.7) + (this.sizeDifference * numVertices * 0.3);
        }
    }

    private static List<ClusteringScore> scores = new ArrayList<>();

    private void printClusteringStats(String algorithmName) {
        // Count nodes in each cluster
        Map<Integer, Integer> clusterSizes = new HashMap<>();
        for (int i = 0; i < vertices; i++) {
            clusterSizes.merge(nodeCluster[i], 1, Integer::sum);
        }

        // Calculate average cluster size
        double avgSize = clusterSizes.values().stream()
            .mapToDouble(Integer::doubleValue)
            .average()
            .orElse(0.0);

        // Calculate standard deviation of cluster sizes
        double variance = clusterSizes.values().stream()
            .mapToDouble(size -> Math.pow(size - avgSize, 2))
            .average()
            .orElse(0.0);
        double stdDev = Math.sqrt(variance);

        // Count inter-cluster edges
        int interClusterEdges = 0;
        for (int i = 0; i < vertices; i++) {
            for (int j = i + 1; j < vertices; j++) {
                if (adjacencyMatrix[i][j] && nodeCluster[i] != nodeCluster[j]) {
                    interClusterEdges++;
                }
            }
        }

        // Create score object
        ClusteringScore score = new ClusteringScore(algorithmName, interClusterEdges, stdDev, avgSize, vertices);
        scores.add(score);

        // Print statistics
        System.out.println("\n=== " + algorithmName + " Statistics ===");
        System.out.println("Number of clusters: " + clusterSizes.size());
        System.out.println("Average cluster size: " + String.format("%.2f", avgSize));
        System.out.println("Standard deviation of cluster sizes: " + String.format("%.2f", stdDev));
        System.out.println("Number of inter-cluster edges: " + interClusterEdges);
        System.out.println("Normalized size difference: " + String.format("%.4f", score.sizeDifference));
        System.out.println("Overall score (lower is better): " + String.format("%.2f", score.score));
        
        // Print individual cluster sizes
        System.out.println("\nCluster sizes:");
        clusterSizes.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(e -> System.out.println("Cluster " + e.getKey() + ": " + e.getValue() + " nodes"));
    }

    public static void main(String[] args) {
        int vertices = 200;
        int k = 10;
        GRAPH graph = new GRAPH(vertices);

        graph.generateSpanningTree();
        graph.saveGraphAsPNG("spanning_tree.png");

        // K-means clustering
        graph.kMeansClustering(k);
        graph.saveClusteredGraphAsPNG("kmeans_clusters.png");
        graph.printClusteringStats("K-means");

        // Label propagation
        graph.labelPropagation(k);
        graph.saveClusteredGraphAsPNG("label_propagation_clusters.png");
        graph.printClusteringStats("Label Propagation");

        // Spectral clustering
        graph.spectralClustering(k);
        graph.saveClusteredGraphAsPNG("spectral_clusters.png");
        graph.printClusteringStats("Spectral");

        // Edge betweenness clustering
        graph.edgeBetweennessClustering(k);
        graph.saveClusteredGraphAsPNG("betweenness_clusters.png");
        graph.printClusteringStats("Edge Betweenness");

        // Walktrap clustering
        graph.walktrapClustering(k);
        graph.saveClusteredGraphAsPNG("walktrap_clusters.png");
        graph.printClusteringStats("Walktrap");

        // Random walk clustering
        graph.randomWalkClustering(k);
        graph.saveClusteredGraphAsPNG("randomwalk_clusters.png");
        graph.printClusteringStats("Random Walk");

        // Print final rankings
        System.out.println("\n=== Algorithm Rankings ===");
        System.out.println("Ranked by combined score (inter-cluster edges and size balance)");
        System.out.println("Lower scores are better\n");
        
        scores.sort(Comparator.comparingDouble(s -> s.score));
        
        for (int i = 0; i < scores.size(); i++) {
            ClusteringScore score = scores.get(i);
            System.out.printf("%d. %s%n", i + 1, score.algorithm);
            System.out.printf("   Inter-cluster edges: %d%n", score.interClusterEdges);
            System.out.printf("   Size difference: %.4f%n", score.sizeDifference);
            System.out.printf("   Overall score: %.2f%n%n", score.score);
        }

        System.out.println("\nAll graphs have been saved");
    }
}