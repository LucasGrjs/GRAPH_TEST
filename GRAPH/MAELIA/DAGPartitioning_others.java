import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class DAGPartitioning_others {

    public static List<List<Integer>> partitionGraphLayered(int[][] edges, int numClusters) {
        Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        Map<Integer, Integer> inDegree = new HashMap<>();
        Map<Integer, Integer> level = new HashMap<>();
        Set<Integer> allNodes = new HashSet<>();

        for (int[] edge : edges) {
            int parent = edge[1];
            int child = edge[0];
            adjacencyList.computeIfAbsent(parent, k -> new ArrayList<>()).add(child);
            inDegree.put(child, inDegree.getOrDefault(child, 0) + 1);
            inDegree.putIfAbsent(parent, 0);
            allNodes.add(parent);
            allNodes.add(child);
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int node : allNodes) {
            if (inDegree.get(node) == 0) {
                queue.add(node);
                level.put(node, 0);
            }
        }

        while (!queue.isEmpty()) {
            int node = queue.poll();
            int currentLevel = level.get(node);
            if (adjacencyList.containsKey(node)) {
                for (int neighbor : adjacencyList.get(node)) {
                    inDegree.put(neighbor, inDegree.get(neighbor) - 1);
                    if (inDegree.get(neighbor) == 0) {
                        queue.add(neighbor);
                        level.put(neighbor, currentLevel + 1);
                    }
                }
            }
        }

        List<List<Integer>> clusters = new ArrayList<>();
        for (int i = 0; i < numClusters; i++) {
            clusters.add(new ArrayList<>());
        }

        List<Integer> sortedNodes = new ArrayList<>(level.keySet());
        sortedNodes.sort(Comparator.comparingInt(level::get));

        int index = 0;
        for (int node : sortedNodes) {
            clusters.get(index % numClusters).add(node);
            index++;
        }

        return enforceAcyclicClusters(clusters, adjacencyList);
    }

    private static List<List<Integer>> enforceAcyclicClusters(List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList) {
        Map<Integer, Integer> nodeToCluster = new HashMap<>();
        for (int i = 0; i < clusters.size(); i++) {
            for (int node : clusters.get(i)) {
                nodeToCluster.put(node, i);
            }
        }

        if (detectClusterCycle(clusters, adjacencyList, nodeToCluster)) {
            System.out.println("Dependency Violation: Cycle detected between clusters! Resolving...");
            resolveClusterCycles(clusters, adjacencyList, nodeToCluster);
        }

        return clusters;
    }

    private static boolean detectClusterCycle(List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList, Map<Integer, Integer> nodeToCluster) {
        int numClusters = clusters.size();
        Map<Integer, List<Integer>> clusterGraph = new HashMap<>();
        int[] inDegree = new int[numClusters];

        for (int i = 0; i < numClusters; i++) {
            clusterGraph.put(i, new ArrayList<>());
        }

        for (int i = 0; i < numClusters; i++) {
            for (int node : clusters.get(i)) {
                if (adjacencyList.containsKey(node)) {
                    for (int neighbor : adjacencyList.get(node)) {
                        int neighborCluster = nodeToCluster.get(neighbor);
                        if (neighborCluster == i) {
                            System.out.println("Self-loop detected: " + node + " -> " + neighbor);
                            return true;
                        }
                        if (neighborCluster != i) {
                            clusterGraph.get(i).add(neighborCluster);
                            inDegree[neighborCluster]++;
                        }
                    }
                }
            }
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numClusters; i++) {
            if (inDegree[i] == 0) {
                queue.add(i);
            }
        }

        int visitedClusters = 0;
        while (!queue.isEmpty()) {
            int cluster = queue.poll();
            visitedClusters++;
            for (int neighbor : clusterGraph.get(cluster)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.add(neighbor);
                }
            }
        }

        return visitedClusters != numClusters;
    }

    private static void resolveClusterCycles(List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList, Map<Integer, Integer> nodeToCluster) {
        Set<Integer> movedNodes = new HashSet<>();
        for (int i = 0; i < clusters.size(); i++) {
            Iterator<Integer> it = clusters.get(i).iterator();
            while (it.hasNext()) {
                int node = it.next();
                if (movedNodes.contains(node)) continue;
                if (adjacencyList.containsKey(node)) {
                    for (int neighbor : adjacencyList.get(node)) {
                        int neighborCluster = nodeToCluster.get(neighbor);
                        if (neighborCluster == i) {
                            System.out.println("Breaking cycle by moving node: " + node);
                            it.remove();
                            int newCluster = findSafeCluster(node, i, clusters, adjacencyList, nodeToCluster);
                            clusters.get(newCluster).add(node);
                            nodeToCluster.put(node, newCluster);
                            movedNodes.add(node);
                            break;
                        }
                    }
                }
            }
        }
    }
    private static int findSafeCluster(int node, int currentCluster, List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList, Map<Integer, Integer> nodeToCluster) {
        for (int i = 0; i < clusters.size(); i++) {
            if (i != currentCluster && isSafeToMove(node, i, clusters, adjacencyList, nodeToCluster)) {
                return i;
            }
        }
        return (currentCluster + 1) % clusters.size();
    }

    private static boolean isSafeToMove(int node, int targetCluster, List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList, Map<Integer, Integer> nodeToCluster) {
        for (int neighbor : adjacencyList.getOrDefault(node, new ArrayList<>())) {
            if (nodeToCluster.get(neighbor) == targetCluster) {
                return false;
            }
        }
        return true;
    }

    public static int[][] generateRandomReverseTree(int numNodes) {
        Random random = new Random();
        List<int[]> edges = new ArrayList<>();
        for (int i = 1; i < numNodes; i++) {
            int parent = random.nextInt(i);
            edges.add(new int[]{i, parent});
        }
        return edges.toArray(new int[0][]);
    }

    public static void generateGraphPNG(int[][] edges, List<List<Integer>> clusters, String outputFileName) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName + ".dot"));
            writer.write("digraph G {\n");
            writer.write("  rankdir=TB;\n");

            String[] colors = {"red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta"};
            Map<Integer, String> nodeColors = new HashMap<>();

            for (int i = 0; i < clusters.size(); i++) {
                String color = colors[i % colors.length];
                for (int node : clusters.get(i)) {
                    nodeColors.put(node, color);
                }
            }

            for (int[] edge : edges) {
                int from = edge[1];
                int to = edge[0];
                writer.write("  " + from + " -> " + to + ";\n");
            }

            for (int node : nodeColors.keySet()) {
                writer.write("  " + node + " [style=filled, fillcolor=" + nodeColors.get(node) + "]\n");
            }

            writer.write("}\n");
            writer.close();

            Process process = Runtime.getRuntime().exec("dot -Tpng " + outputFileName + ".dot -o " + outputFileName + ".png");
            process.waitFor();

            System.out.println("Graph image generated: " + outputFileName + ".png");
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static List<List<Integer>> getClusterSizeDifferences(List<List<Integer>> clusters) {
        
        if (clusters == null || clusters.isEmpty()) {
        System.out.println("No clusters to analyze.");
        return null;
        }

        List<List<Integer>> exceedingClusters = new ArrayList<>(); // List to store exceeding clusters
        List<Integer> toRemove = new ArrayList<>();

        // Calculate median cluster size
        List<Integer> clusterSizes = new ArrayList<>();
        for (List<Integer> cluster : clusters) {
            clusterSizes.add(cluster.size());
        }
        Collections.sort(clusterSizes);
        double medianSize;
        if (clusterSizes.size() % 2 == 0) {
            int mid1 = clusterSizes.get(clusterSizes.size() / 2 - 1);
            int mid2 = clusterSizes.get(clusterSizes.size() / 2);
            medianSize = (mid1 + mid2) / 2.0;
        } else {
            medianSize = clusterSizes.get(clusterSizes.size() / 2);
        }

        // Analyze each cluster and calculate standard deviation.
        double sum = 0;
        double squaredDifferencesSum = 0;

        for (int i = 0; i < clusters.size(); i++) {
            List<Integer> currentCluster = clusters.get(i);
            double clusterSize = currentCluster.size();

            // Calculate sum for average
            sum += clusterSize;

            // Calculate difference and percentage difference
            double difference = clusterSize - medianSize;
            double percentageDifference = (medianSize != 0) ? (difference / medianSize) * 100 : (difference == 0 ? 0 : Double.POSITIVE_INFINITY); // Handle division by zero

            // Calculate squared differences for standard deviation
            double averageSize = (i == clusters.size() - 1 && clusters.size() != 0) ? sum / clusters.size() : 0; // Calculate average only on the last iteration
            if (clusters.size() != 0) {
                double differenceFromAverage = clusterSize - averageSize;
                squaredDifferencesSum += Math.pow(differenceFromAverage, 2);
            }

            System.out.printf("Cluster %d: Size=%.0f, Median=%.2f, Difference=%.2f (%.2f%%)\n", i, clusterSize, medianSize, difference, percentageDifference);

            // Calculate standard deviation if last element.
            if (i == clusters.size() - 1 && clusters.size() != 0) {
                double variance = squaredDifferencesSum / clusters.size();
                double standardDeviation = Math.sqrt(variance);
                System.out.println("Standard Deviation of Cluster Sizes: " + standardDeviation);
            }

            if(percentageDifference > 150.0)
            {
                exceedingClusters.add(clusters.get(i)); // Store cluster exceeding 50% difference
                toRemove.add(i);
            }
        }

        int tttt = toRemove.size()-1;
        for(int n = tttt; n >= 0; n--){
            clusters.remove(n);
        }

        return exceedingClusters; // Return the list of exceeding clusters
    }

    public static void main(String[] args) {
        
        int node_number = 60;
        int cluster_number = 6;
        String outputFileName = "other";
        int[][] edges = generateRandomReverseTree(node_number); // Générer des arêtes aléatoires

        /*int[][] edges = {
            {45,24}, {39,24}, 
            {44,45}, {46,45}, {50,39}, {38,39},
            {57,50}, {51,50}, {55,38}, {40,38},
            {47,55}, {56,55}, {52,40}, {42,40},
            {49,47}, {48,47}, {53,52}, {54,52},
            {63,42}, {62,42}, {41,42},
            {58,41},{59,58},{61,58},
            {37,41}, {64,37}, {60,64}, {65,64},
            {66,36}, {35,36}, {68,35}, {34,35},
            {69,68}, {17,34}, {32,34},
            {22,32}, {9,32}, {31,32},
            {23,22}, {21,22}, {11,9}, {15,9},
            {13,11}, {10,11}, {14,13}, {12,13},
            {36,37},
            {18,31}, {30,31}, {19,18}, {20,18},
            {16,30}, {29,30}, {2,29}, {4,29},
            {1,2}, {3,2}, {28,4},
            {5,28}, {26,28}, {7,5}, {6,5},
            {8,26}, {25,26}
        };*/

        System.out.println("edges");
        List<List<Integer>> clusters = partitionGraphLayered(edges, cluster_number);

        generateGraphPNG(edges, clusters, outputFileName);
        getClusterSizeDifferences(clusters);
    }
}
