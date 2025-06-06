import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

class TreeNode {
    int value;
    List<TreeNode> children;

    public TreeNode(int value) {
        this.value = value;
        this.children = new ArrayList<>();
    }
}

class DAGPartitioning_level
{

    public static List<List<Integer>> partitionGraphWithDependencies(int[][] edges, int numClusters) {
        // Step 1: Build adjacency list
        Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        Map<Integer, Integer> inDegree = new HashMap<>();
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

        if (allNodes.size() < numClusters) {
            numClusters = allNodes.size();
            System.out.println("Number of clusters cannot exceed the number of nodes in the graph. Updated number of cluster " + numClusters);
        }
    
        // Step 2: Topological Sorting (Ensures no cycles in dependencies)
        Queue<Integer> queue = new LinkedList<>();
        for (int node : allNodes) {
            if (inDegree.get(node) == 0) {
                queue.add(node);
            }
        }
    
        List<Integer> topoOrder = new ArrayList<>();
        while (!queue.isEmpty()) {
            int node = queue.poll();
            topoOrder.add(node);
            if (adjacencyList.containsKey(node)) {
                for (int neighbor : adjacencyList.get(node)) {
                    inDegree.put(neighbor, inDegree.get(neighbor) - 1);
                    if (inDegree.get(neighbor) == 0) {
                        queue.add(neighbor);
                    }
                }
            }
        }
    
        // Step 3: Partitioning (Balanced Clustering)
        //double clusterSize = (double) Math.round( allNodes.size() / numClusters);
        //double remainder = allNodes.size() - (clusterSize * numClusters);

        int clusterSize = allNodes.size() / numClusters;
        int remainder = allNodes.size() % numClusters;
        


        System.out.println("allNodes.size() "+ allNodes.size());
        System.out.println("remainder "+ remainder);
        System.out.println("numClusters "+ numClusters);
        System.out.println("clusterSize " + clusterSize);
        List<List<Integer>> clusters = new ArrayList<>();
        Map<Integer, Integer> nodeToCluster = new HashMap<>();
    
        for (int i = 0; i < numClusters; i++) {
            clusters.add(new ArrayList<>());
        }
    
        int clusterIndex = 0, count = 0;
        for (int node : topoOrder) {
            clusters.get(clusterIndex).add(node);
            nodeToCluster.put(node, clusterIndex);

            // Determine the expected size for this cluster
            int expectedSize = (clusterIndex < remainder) ? (clusterSize + 1) : clusterSize;
            
            count++;
            if (count == expectedSize) {
                clusterIndex++;
                count = 0;
            }
        }
            
        // Ensure every cluster has at least one node
        while (clusters.stream().anyMatch(List::isEmpty)) {
            List<Integer> largestCluster = clusters.stream().max(Comparator.comparing(List::size)).orElse(null);
            List<Integer> smallestCluster = clusters.stream().filter(List::isEmpty).findFirst().orElse(null);
            if (largestCluster != null && smallestCluster != null && !largestCluster.isEmpty()) {
                smallestCluster.add(largestCluster.remove(largestCluster.size() - 1));
            }
        }
    
        // Step 4: Detect Cyclic Dependencies Between Clusters
        Set<Integer> visitedClusters = new HashSet<>();
        Set<Integer> stack = new HashSet<>();
        boolean hasCycle = false;
    
        for (int i = 0; i < numClusters; i++) {
            if (!visitedClusters.contains(i)) {
                hasCycle = hasCycle || detectCycleInClusters(i, clusters, adjacencyList, nodeToCluster, visitedClusters, stack);
            }
        }
    
        if (hasCycle) {
            System.out.println("Dependency Violation: Cyclic dependency detected between clusters!");
        }

        int node = 0;
        for(var auto : clusters)
        {   
            node += auto.size();
            System.out.println(auto);
        }

        System.out.println("TOTAL SIZE " + node);
        
        return clusters;
    }

    private static boolean detectCycleInClusters(int cluster, List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList, Map<Integer, Integer> nodeToCluster, Set<Integer> visited, Set<Integer> stack) {
        if (stack.contains(cluster)) {
            return true; // Cycle detected
        }
        if (visited.contains(cluster)) {
            return false;
        }
        
        visited.add(cluster);
        stack.add(cluster);
        
        for (int node : clusters.get(cluster)) {
            if (adjacencyList.containsKey(node)) {
                for (int neighbor : adjacencyList.get(node)) {
                    int neighborCluster = nodeToCluster.get(neighbor);
                    if (neighborCluster != cluster && detectCycleInClusters(neighborCluster, clusters, adjacencyList, nodeToCluster, visited, stack)) {
                        return true;
                    }
                }
            }
        }
        
        stack.remove(cluster);
        return false;
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

            String[] colors = {"red", "blue", "green", "yellow", "orange", "purple", "cyan", "magenta", "pink", "brown", "gold", "silver"};
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
                writer.write("  " + to + " -> " + from + ";\n");
            }

            for (int node : nodeColors.keySet()) {
                //writer.write("  " + node + " [style=filled, fillcolor=" + nodeColors.get(node) + "]\n");
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

    // Add function to print topological order
    public static void printTopologicalSort(int[][] edges) {
        // Build adjacency list and inDegree map with reversed edges: treat edge[0] as parent, edge[1] as child
        Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        Map<Integer, Integer> inDegree = new HashMap<>();
        Set<Integer> allNodes = new HashSet<>();
        for (int[] edge : edges) {
             int parent = edge[0];
             int child = edge[1];
             adjacencyList.computeIfAbsent(parent, k -> new ArrayList<>()).add(child);
             inDegree.put(child, inDegree.getOrDefault(child, 0) + 1);
             inDegree.putIfAbsent(parent, 0);
             allNodes.add(parent);
             allNodes.add(child);
        }
        // Topological sort using a queue
        Queue<Integer> queue = new LinkedList<>();
        for (int node : allNodes) {
             if (inDegree.get(node) == 0) {
                  queue.add(node);
             }
        }
        List<Integer> topoOrder = new ArrayList<>();
        while (!queue.isEmpty()){
             int node = queue.poll();
             topoOrder.add(node);
             if (adjacencyList.containsKey(node)){
                  for (int neighbor : adjacencyList.get(node)) {
                       inDegree.put(neighbor, inDegree.get(neighbor) - 1);
                       if (inDegree.get(neighbor) == 0) {
                            queue.add(neighbor);
                       }
                  }
             }
        }
        System.out.println("Topological Order: " + topoOrder);
    }

    public static void main(String[] args) {
        
        int node_number = 60;
        int cluster_number = 4;
        String outputFileName = "level";
        //int[][] edges = generateRandomReverseTree(node_number); // Générer des arêtes aléatoires

        int[][] edges = {
            /*{45,24}, {39,24}, 
            {44,45}, {46,45}, {50,39}, {38,39},
            {57,50}, {51,50}, {55,38}, {40,38},
            {47,55}, {56,55}, {52,40}, {42,40},
            {49,47}, {48,47}, {53,52}, {54,52},
            {63,42}, {62,42}, {41,42},*/

            /*{58,41},{59,58},{61,58},
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
            {8,26}, {25,26}*/

            {2043, 2040},
            {2039, 2040},
            {2040, 3451},
            {3450, 3451},
            {3451, 2044},
            {2407, 2044},
            {2044, 2045},
            {2398, 2045},
            {2045, 1681},
            {4295, 1680},
            {4280, 4295},
            {4422, 4280},
            {4653, 4280},
            {1680, 183},
            {1681, 114},
            {183, 114},
            {114, 115},
            {1600, 115},
            {115, 192},
            {574, 575},
            {575, 193},
            {192, 193},
            {193, 2265},
            {2265, 2402},
            {2402, 4},
            {4709, 1857},
            {1857, 1444},
            {1444, 198},
            {4, 1444},
            {197, 198},
            {198, 3420},
            {4649, 3420},
            {3420, 110},
            {201, 110},
            {208, 110},
            {110, 111},
            {194, 195},
            {2072, 195},
            {2072, 4709},
            {549, 550},
            {4570, 550},
            {550, 837},
            {3890, 838},
            {837, 838},
            {2023, 2024},
            {838, 2025},
            {2024, 2025},
            {2025, 2032},
            {2031, 2032},
            {2032, 2036},
            {2036, 3},
            {23, 3},
            {2027, 23},
            {3, 4}

        };
        System.out.println("edges");
        List<List<Integer>> clusters = partitionGraphWithDependencies(edges, cluster_number);
        printTopologicalSort(edges);

        generateGraphPNG(edges,clusters,outputFileName);
        getClusterSizeDifferences(clusters);
    }
}


