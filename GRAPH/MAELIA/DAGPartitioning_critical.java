import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

class TreeNode {
    int value;
    List<TreeNode> children;

    public TreeNode(int value) {
        this.value = value;
        this.children = new ArrayList<>();
    }
}

public class DAGPartitioning_critical {

    private static Map<Integer, List<Integer>> buildAdjacencyListFromTree(TreeNode root) {
        Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        buildAdjacencyListRecursive(root, adjacencyList);
        return adjacencyList;
    }

    private static void buildAdjacencyListRecursive(TreeNode node, Map<Integer, List<Integer>> adjacencyList) {
        for (TreeNode child : node.children) {
            adjacencyList.computeIfAbsent(node.value, k -> new ArrayList<>()).add(child.value);
            buildAdjacencyListRecursive(child, adjacencyList);
        }
    }

    private static TreeNode buildTreeFromEdges(int[][] edges, int rootValue) {

        Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        for (int[] edge : edges) {
            int parent = edge[1];
            int child = edge[0];
            adjacencyList.computeIfAbsent(parent, k -> new ArrayList<>()).add(child);
        }
    
        TreeNode root = new TreeNode(rootValue);
    
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
    
        while (!queue.isEmpty()) {
            TreeNode currentNode = queue.poll();
            if (adjacencyList.containsKey(currentNode.value)) {
                for (int childValue : adjacencyList.get(currentNode.value)) {
                    TreeNode childNode = new TreeNode(childValue);
                    currentNode.children.add(childNode);
                    queue.add(childNode); 
                }
            }
        }
    
        return root;
    }

    public static TreeNode buildTreeFromEdges(int[][] edges) {
        Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
        Set<Integer> allNodes = new HashSet<>();
        Set<Integer> childNodes = new HashSet<>();

        for (int[] edge : edges) {
            int parent = edge[1];
            int child = edge[0];

            adjacencyList.computeIfAbsent(parent, k -> new ArrayList<>()).add(child);
            allNodes.add(parent);
            allNodes.add(child);
            childNodes.add(child);
        }
        int rootValue = -1;
        for (int node : allNodes) {
            if (!childNodes.contains(node)) {
                rootValue = node;
                break;
            }
        }

        if (rootValue == -1) {
            throw new IllegalStateException("No root found! Input edges may not form a valid tree.");
        }

        TreeNode root = new TreeNode(rootValue);
        Queue<TreeNode> queue = new LinkedList<>();
        Map<Integer, TreeNode> nodeMap = new HashMap<>();
        nodeMap.put(rootValue, root);
        queue.add(root);

        while (!queue.isEmpty()) {
            TreeNode currentNode = queue.poll();
            int currentValue = currentNode.value;

            if (adjacencyList.containsKey(currentValue)) {
                for (int childValue : adjacencyList.get(currentValue)) {
                    TreeNode childNode = new TreeNode(childValue);
                    currentNode.children.add(childNode);
                    queue.add(childNode);
                    nodeMap.put(childValue, childNode);
                }
            }
        }

        return root;
    }

    public static List<Integer> findRandomLongestPath(TreeNode root) {
        List<List<Integer>> longestPaths = findAllLongestPaths(root);
        if (longestPaths.isEmpty()) {
            return new ArrayList<>();
        }
        Random random = new Random();
        return longestPaths.get(random.nextInt(longestPaths.size()));
    }

    private static List<List<Integer>> findAllLongestPaths(TreeNode root) {
        List<List<Integer>> allLongestPaths = new ArrayList<>();
        int[] maxLength = {0};
        findAllLongestPathsRecursive(root, new ArrayList<>(), allLongestPaths, maxLength);
        return allLongestPaths;
    }

    private static void findAllLongestPathsRecursive(TreeNode node, List<Integer> currentPath, List<List<Integer>> allLongestPaths, int[] maxLength) {
        currentPath.add(node.value);
        if (node.children.isEmpty()) {
            if (currentPath.size() > maxLength[0]) {
                maxLength[0] = currentPath.size();
                allLongestPaths.clear();
                allLongestPaths.add(new ArrayList<>(currentPath));
            } else if (currentPath.size() == maxLength[0]) {
                allLongestPaths.add(new ArrayList<>(currentPath));
            }
        } else {
            for (TreeNode child : node.children) {
                findAllLongestPathsRecursive(child, new ArrayList<>(currentPath), allLongestPaths, maxLength);
            }
        }
    }

    public static List<List<Integer>> getSubtreesFromLongestPath(List<Integer> longestPath, Map<Integer, List<Integer>> adjacencyList) {
        List<List<Integer>> subtrees = new ArrayList<>();
        Set<Integer> longestPathSet = new HashSet<>(longestPath);
        for (int node : longestPath) {
            if (adjacencyList.containsKey(node)) {
                for (int child : adjacencyList.get(node)) {
                    if (!longestPathSet.contains(child)) {
                        List<Integer> subtree = new ArrayList<>();
                        getSubtreeNodes(child, adjacencyList, subtree, longestPathSet);
                        if (!subtree.isEmpty()) {
                            subtrees.add(subtree);
                        }
                    }
                }
            }
        }
        return subtrees;
    }

    private static void getSubtreeNodes(int node, Map<Integer, List<Integer>> adjacencyList, List<Integer> subtree, Set<Integer> longestPathSet) {
        if (!adjacencyList.containsKey(node)) { // If it's a leaf node
            subtree.add(node); // Add it to the subtree
            return;
        }
        subtree.add(node); // Add the current node
        for (int child : adjacencyList.get(node)) {
            if (!longestPathSet.contains(child)) {
                getSubtreeNodes(child, adjacencyList, subtree, longestPathSet);
            }
        }
    }

    // Reworked generatePngFromGraph method to generate the same graph as DAGPartitioning_level.java:
    public static void generatePngFromGraph(int[][] edges, List<Integer> longestPath, List<List<Integer>> clusters, String outputFileName, Map<Integer, List<Integer>> adjacencyList) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName + ".dot"));
            writer.write("digraph G {\n");
            writer.write("  rankdir=TB;\n");
            
            String[] colors = {
                "yellow", "blue", "green", "orange", "purple", "cyan", 
                "magenta", "pink", "brown", "gold", "silver", "red",
                "violet", "indigo", "coral", "crimson", "darkgreen", "darkcyan",
                "darkblue", "darkmagenta", "darkred", "darkorange", "olive", "teal",
                "navy", "maroon", "aquamarine", "turquoise", "salmon", "sienna",
                "chocolate", "firebrick", "steelblue", "lightsalmon", "palegreen", "orchid",
                "plum", "khaki", "powderblue", "paleturquoise", "rosybrown", "goldenrod",
                "mediumorchid", "darkkhaki", "indianred", "mediumseagreen", "darkseagreen", "lightcoral",
                "mediumpurple", "cadetblue"
            };
            
            Map<Integer, String> nodeColors = new HashMap<>();
            
            // Assign one color per cluster
            for (int i = 0; i < clusters.size(); i++) {
                String color = colors[i % colors.length];
                for (int node : clusters.get(i)) {
                    nodeColors.put(node, color);
                }
            }
            
            // Write edges
            for (int[] edge : edges) {
                int parent = edge[1];
                int child = edge[0];
                writer.write("  " + child + " -> " + parent + ";\n");
            }
            
            // Write nodes with their colors
            for (int node : nodeColors.keySet()) {
                String color = nodeColors.get(node);
                writer.write("  " + node + " [style=filled, fillcolor=" + color + "];\n");
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

    // NEW METHOD: Generate PNG with no node colors, outputting "normal.png"
    public static void generateNormalPngFromGraph(int[][] edges, String outputFileName) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName + ".dot"));
            writer.write("digraph G {\n");
            writer.write("  rankdir=TB;\n");
            // Write edges (using reverse order as in the level version)
            for (int[] edge : edges) {
                int parent = edge[1];
                int child = edge[0];
                writer.write("  " + child + " -> " + parent + ";\n");
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

    private static TreeNode buildTree(Map<Integer, List<Integer>> adjacencyList, int rootValue) {
        TreeNode root = new TreeNode(rootValue);
        buildTreeRecursive(root, adjacencyList);
        return root;
    }

    private static void buildTreeRecursive(TreeNode node, Map<Integer, List<Integer>> adjacencyList) {
        int nodeValue = node.value;
        if (adjacencyList.containsKey(nodeValue)) {
            for (int childValue : adjacencyList.get(nodeValue)) {
                TreeNode child = new TreeNode(childValue);
                node.children.add(child);
                buildTreeRecursive(child, adjacencyList);
            }
        }
    }

    public static List<List<Integer>> clusterSubtrees(List<List<Integer>> subtrees, int cluster_number) {
        List<List<Integer>> clusters = new ArrayList<>();
        
        // Keep splitting largest subtrees until we have enough clusters or can't split anymore
        while (subtrees.size() < cluster_number) {
            // Sort subtrees by size (largest first)
            subtrees.sort((a, b) -> Integer.compare(b.size(), a.size()));
            
            // Try to split the largest subtree
            boolean split = false;
            for (int i = 0; i < subtrees.size() && !split; i++) {
                List<Integer> largeSubtree = subtrees.get(i);
                if (largeSubtree.size() > 1) {  // Can only split if size > 1
                    // Split the subtree into two parts
                    int midPoint = largeSubtree.size() / 2;
                    List<Integer> part1 = new ArrayList<>(largeSubtree.subList(0, midPoint));
                    List<Integer> part2 = new ArrayList<>(largeSubtree.subList(midPoint, largeSubtree.size()));
                    
                    // Replace the original subtree with the two parts
                    subtrees.remove(i);
                    if (!part1.isEmpty()) subtrees.add(part1);
                    if (!part2.isEmpty()) subtrees.add(part2);
                    split = true;
                }
            }
            
            // If we couldn't split any subtree, we've reached maximum possible clusters
            if (!split) {
                System.out.println("Warning: Cannot create more clusters. Maximum possible: " + subtrees.size());
                break;
            }
        }

        // Now distribute the subtrees (which may be split) into clusters
        for (int i = 0; i < cluster_number; i++) {
            clusters.add(new ArrayList<>());
        }

        // Sort again for balanced distribution
        subtrees.sort((a, b) -> Integer.compare(b.size(), a.size()));
        
        // Distribute subtrees to clusters
        int[] clusterSizes = new int[cluster_number];
        for (List<Integer> subtree : subtrees) {
            int minIndex = 0;
            for (int i = 1; i < cluster_number; i++) {
                if (clusterSizes[i] < clusterSizes[minIndex]) {
                    minIndex = i;
                }
            }
            clusters.get(minIndex).addAll(subtree);
            clusterSizes[minIndex] += subtree.size();
        }

        return clusters;
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

    public static List<List<Integer>> optimizeNodeOnPath(List<List<Integer>> longestPaths, Map<Integer, List<Integer>> adjacencyList, List<List<Integer>> clusters) {
        var newClusters = new ArrayList<>(clusters);
        
        for (var longestPath : longestPaths) {
            Set<Integer> subtreeRootsMet = new HashSet<>();
            int firstSubtreeRoot = -1;
            int firstSubtreeClusterIndex = -1;
            List<Integer> nodesToAdd = new ArrayList<>();
    
            for (int i = longestPath.size() - 1; i >= 0; i--) {
                int currentNode = longestPath.get(i);
                
                if (adjacencyList.containsKey(currentNode)) {
                    List<Integer> children = adjacencyList.get(currentNode);
                    List<Integer> subtreeChildren = new ArrayList<>();
                    
                    for (int child : children) {
                        if (!longestPath.contains(child)) {
                            subtreeChildren.add(child);
                        }
                    }
    
                    if (!subtreeChildren.isEmpty()) { 
                        if (firstSubtreeRoot == -1) { 
                            firstSubtreeRoot = subtreeChildren.get(0);
    
                            // Find the first valid cluster
                            for (int j = 0; j < newClusters.size(); j++) {
                                if (newClusters.get(j).contains(firstSubtreeRoot)) {
                                    firstSubtreeClusterIndex = j;
                                    break; // Stop at the first match
                                }
                            }
    
                            if (firstSubtreeClusterIndex != -1) {
                                if (subtreeChildren.size() > 1) {
                                    nodesToAdd.addAll(longestPath.subList(i + 1, longestPath.size()));
                                } else {
                                    nodesToAdd.addAll(longestPath.subList(i, longestPath.size()));
                                }
    
                                // Prevent duplicate nodes before adding
                                for (int node : nodesToAdd) {
                                    newClusters.get(firstSubtreeClusterIndex).add(node);
                                }
    
                                // Remove nodes from other clusters
                                for (int node : nodesToAdd) {
                                    for (int j = 0; j < newClusters.size(); j++) {
                                        if (j != firstSubtreeClusterIndex) {
                                            newClusters.get(j).remove(Integer.valueOf(node));
                                        }
                                    }
                                }
    
                                nodesToAdd.clear();
                                subtreeRootsMet.add(firstSubtreeRoot);
    
                                if (subtreeChildren.size() > 1) {
                                    for (var auto : subtreeChildren) {
                                        if (!newClusters.get(firstSubtreeClusterIndex).contains(auto)) {
                                            longestPath.subList(i + 1, longestPath.size()).clear();
                                            break;
                                        }
                                    }
    
                                    // Prevent duplicate `currentNode`
                                    boolean isAlreadyAssigned = false;
                                    for (List<Integer> cluster : newClusters) {
                                        if (cluster.contains(currentNode)) {
                                            isAlreadyAssigned = true;
                                            break;
                                        }
                                    }
    
                                    if (!isAlreadyAssigned) {
                                        newClusters.get(firstSubtreeClusterIndex).add(currentNode);
                                    }
                                }
                            }
                        } else {
                            longestPath.subList(i + 1, longestPath.size()).clear();
                            break;
                        }
                    } else {
                        // Prevent duplicate `currentNode`
                        boolean isAlreadyAssigned = false;
                        for (List<Integer> cluster : newClusters) {
                            if (cluster.contains(currentNode)) {
                                isAlreadyAssigned = true;
                                break;
                            }
                        }
    
                        if (firstSubtreeClusterIndex != -1 && !isAlreadyAssigned) {
                            newClusters.get(firstSubtreeClusterIndex).add(currentNode);
                        }
                    }
                }
            }
        }
        return newClusters;
    }
    
    public static Map<Integer, Integer> computeClusterLevels(List<List<Integer>> clusters, List<Integer> longestPath, Map<Integer, List<Integer>> adjacencyList) {
        if (clusters == null || clusters.isEmpty() || longestPath == null || longestPath.isEmpty() || adjacencyList == null || adjacencyList.isEmpty()) {
            System.out.println("Invalid input data.");
            return null;
        }

        Set<Integer> longestPathNodes = new HashSet<>(longestPath);
        Map<Integer, Integer> clusterLevels = new HashMap<>();
        Map<Integer, Set<Integer>> nodeClusterMap = new HashMap<>(); // Maps nodes to their clusters

        // Initialize nodeClusterMap
        for (int i = 0; i < clusters.size(); i++) {
            for (int node : clusters.get(i)) {
                if (!nodeClusterMap.containsKey(node)) {
                    nodeClusterMap.put(node, new HashSet<>());
                }
                nodeClusterMap.get(node).add(i);
            }
        }

        // Assign level 0 to the cluster containing the longest path
        int longestPathCluster = -1;
        for (int i = 0; i < clusters.size(); i++) {
            for (int node : clusters.get(i)) {
                if (longestPathNodes.contains(node)) {
                    longestPathCluster = i;
                    break;
                }
            }
            if (longestPathCluster != -1) {
                clusterLevels.put(longestPathCluster, 0);
                break;
            }
        }

        // Calculate levels for other clusters
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(longestPathCluster);

        while (!queue.isEmpty()) {
            int currentCluster = queue.poll();
            int currentLevel = clusterLevels.get(currentCluster);

            for (int node : clusters.get(currentCluster)) {
                if (adjacencyList.containsKey(node)) {
                    for (int neighbor : adjacencyList.get(node)) {
                        if (nodeClusterMap.containsKey(neighbor)) {
                            for (int neighborCluster : nodeClusterMap.get(neighbor)) {
                                if (!clusterLevels.containsKey(neighborCluster)) {
                                    clusterLevels.put(neighborCluster, currentLevel + 1);
                                    queue.offer(neighborCluster);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Print cluster levels
        for (int i = 0; i < clusters.size(); i++) {
            System.out.println("Cluster " + i + ": Level " + (clusterLevels.containsKey(i) ? clusterLevels.get(i) : -1));
        }

        return clusterLevels;
    }

    public static List<List<Integer>> divideCluster(List<Integer> clusterToDivide, Map<Integer, List<Integer>> adjacencyList, int threshold, int maxClusters) {
        List<List<Integer>> dividedClusters = new ArrayList<>();
        
        if (clusterToDivide.size() <= threshold) {
            dividedClusters.add(clusterToDivide); // Cluster is small enough
            return dividedClusters;
        }
    
        // Find the root of the cluster
        int root = clusterToDivide.get(0);
        if (root != -1) {
            // Find the longest path in the cluster
            TreeNode rootNode = buildTree(adjacencyList, root);
            List<Integer> longestPath = findRandomLongestPath(rootNode);
    
            // Divide the cluster based on the longest path
            List<List<Integer>> subClusters = getSubtreesFromLongestPath(longestPath, adjacencyList);
    
            // Check if dividing the cluster will exceed maxClusters
            if (dividedClusters.size() + subClusters.size() <= maxClusters) {
                for (List<Integer> subCluster : subClusters) {
                    if (!subCluster.isEmpty()) {
                        dividedClusters.addAll(divideCluster(subCluster, adjacencyList, threshold, maxClusters)); // Recursively divide sub-clusters
                    }
                }
            } else {
                dividedClusters.add(clusterToDivide); // Cannot divide further without exceeding maxClusters
            }
        } else {
            dividedClusters.add(clusterToDivide); // No root found, return original cluster
        }
    
        return dividedClusters;
    }
    
    public static void saveDotStringToFile(String dotString, String filePath) {
            try {
                java.nio.file.Files.write(java.nio.file.Paths.get(filePath), dotString.getBytes());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    
    public static void generatePNGFromDOT(String dotFilePath, String pngFilePath) {
        try {
            ProcessBuilder pb = new ProcessBuilder("dot", "-Tpng", dotFilePath, "-o", pngFilePath);
            Process process = pb.start();
            process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static List<List<Integer>> getSubtreeSizeDifferences(List<List<Integer>> subtrees) {
        if (subtrees == null || subtrees.isEmpty()) {
            System.out.println("No subtrees to analyze.");
            return new ArrayList<>();
        }
    
        List<Integer> subtreeSizes = new ArrayList<>();
        for (List<Integer> subtree : subtrees) {
            subtreeSizes.add(subtree.size());
        }
    
        Collections.sort(subtreeSizes);
    
        // Compute average size
        double averageSize = subtreeSizes.stream().mapToInt(Integer::intValue).average().orElse(0.0);
    
        // Compute median size
        double medianSize;
        int n = subtreeSizes.size();
        if (n % 2 == 0) {
            medianSize = (subtreeSizes.get(n / 2 - 1) + subtreeSizes.get(n / 2)) / 2.0;
        } else {
            medianSize = subtreeSizes.get(n / 2);
        }
    
        // List to store subtrees exceeding 150% difference
        List<List<Integer>> exceedingSubtrees = new ArrayList<>();
    
        // Print statistics
        System.out.println("==================================");
        System.out.printf("Average subtree size: %.2f\n", averageSize);
        System.out.printf("Median subtree size: %.2f\n", medianSize);
        System.out.println("==================================");
    
        // Print size differences for each subtree
        System.out.println("Subtree Size Differences (relative to median):");
        for (int i = 0; i < subtrees.size(); i++) {
            int size = subtrees.get(i).size();
            double difference = size - medianSize;
            double percentageDifference = (medianSize != 0) ? (difference / medianSize) * 100 : 0;
    
            System.out.printf("Subtree %d: Size = %d, Difference = %.2f (%.2f%% from median)\n",
                    i, size, difference, percentageDifference);
    
            // Check if the subtree exceeds 150% of the median
            if (Math.abs(percentageDifference) > 150.0) {
                exceedingSubtrees.add(subtrees.get(i));
            }
        }
    
        System.out.println("==================================");
        System.out.println("Subtrees exceeding 150% size difference from median: " + exceedingSubtrees.size());
        System.out.println("==================================");
    
        return exceedingSubtrees;
    }
    
    public static List<List<Integer>> splitOversizedSubtrees(List<List<Integer>> subtrees, List<List<Integer>> oversizedSubtrees, 
    int[][] edges, Map<Integer, List<Integer>> adjacencyList, List<List<Integer>> longuestPaths) {
    List<List<Integer>> updatedSubtrees = new ArrayList<>(subtrees);

        // Process each oversized subtree
        for (List<Integer> oversizedSubtree : oversizedSubtrees) {
            System.out.println("Splitting large subtree...");

            // Build a tree for the oversized subtree
            TreeNode rootBig = buildTreeFromEdges(edges, oversizedSubtree.get(0));

            // Compute the longest path on this subtree
            List<Integer> longestPath = findRandomLongestPath(rootBig);
            longuestPaths.add(longestPath);

            // Extract new subtrees from the longest path
            List<List<Integer>> extractedSubtrees = getSubtreesFromLongestPath(longestPath, adjacencyList);

            // Remove the original large subtree
            updatedSubtrees.remove(oversizedSubtree);

            // Add the new smaller subtrees instead
            updatedSubtrees.addAll(extractedSubtrees);
            //updatedSubtrees.add(longestPath);
        }

        System.out.println("==================================");
        System.out.println("Final number of subtrees after splitting: " + updatedSubtrees.size());
        System.out.println("==================================");

        return updatedSubtrees;
    } 

    public static List<List<Integer>> mergeClusters(List<List<Integer>> clusters, int cluster_number, Map<Integer, List<Integer>> adjacencyList, List<Integer> longuestPath) {
        if (clusters.size() <= cluster_number) {
            return clusters; // Already at or below the required number
        }
        // Get cluster levels
        Map<Integer, Integer> clusterLevels = computeClusterLevels(clusters, longuestPath, adjacencyList);
        // Group clusters by level
        Map<Integer, List<List<Integer>>> levelClusters = new HashMap<>();
        for (int i = 0; i < clusters.size(); i++) {
            int level = clusterLevels.getOrDefault(i, -1);
            levelClusters.computeIfAbsent(level, k -> new ArrayList<>()).add(clusters.get(i));
        }
    
        // Sort levels
        List<Integer> sortedLevels = new ArrayList<>(levelClusters.keySet());
        Collections.sort(sortedLevels);

        clusters = mergeListsInMapKeepLevelsNoLast(levelClusters, cluster_number);

        return clusters;
    }

    public static List<List<Integer>> mergeListsInMapKeepLevelsNoLast(Map<Integer, List<List<Integer>>> dataMap,int targetTotalLists) {

        List<List<Integer>> allLists = new ArrayList<>();
        Map<Integer, Integer> levelLastIndices = new HashMap<>();
        int currentIndex = 0;

        for (Map.Entry<Integer, List<List<Integer>>> entry : dataMap.entrySet()) {
            int level = entry.getKey();
            List<List<Integer>> levelLists = entry.getValue();
            currentIndex += levelLists.size();
            levelLastIndices.put(level, currentIndex - 1);
            allLists.addAll(levelLists);
        }

        if (allLists.size() <= targetTotalLists) {
            if (allLists.size() < targetTotalLists){
                return allLists;
            }
            return allLists;
        }

        while (allLists.size() > targetTotalLists) {
            List<List<Integer>> mergeableLists = new ArrayList<>();
            List<Integer> mergeableIndices = new ArrayList<>();

            for (int i = 0; i < allLists.size(); i++) {
                boolean isLastList = false;
                for (int level : levelLastIndices.keySet()) {
                    if (levelLastIndices.get(level) == i) {
                        isLastList = true;
                        break;
                    }
                }
                if (!isLastList) {
                    mergeableLists.add(allLists.get(i));
                    mergeableIndices.add(i);
                }
            }

            if (mergeableLists.size() < 2) {
                return allLists;
            }

            // Find the two lists with the closest sizes
            int minSizeDiff = Integer.MAX_VALUE;
            int index1 = -1;
            int index2 = -1;

            for (int i = 0; i < mergeableLists.size(); i++) {
                for (int j = i + 1; j < mergeableLists.size(); j++) {
                    int sizeDiff = Math.abs(mergeableLists.get(i).size() - mergeableLists.get(j).size());
                    if (sizeDiff < minSizeDiff) {
                        minSizeDiff = sizeDiff;
                        index1 = allLists.indexOf(mergeableLists.get(i));
                        index2 = allLists.indexOf(mergeableLists.get(j));
                    }
                }
            }

            List<Integer> mergedList = new ArrayList<>(allLists.get(index1));
            mergedList.addAll(allLists.get(index2));

            if (index1 > index2) {
                int temp = index1;
                index1 = index2;
                index2 = temp;
            }

            allLists.remove(index2);
            allLists.remove(index1);
            allLists.add(index1, mergedList);

            // Update levelLastIndices if necessary
            for (int level : levelLastIndices.keySet()) {
                if (levelLastIndices.get(level) >= index2) {
                    levelLastIndices.put(level, levelLastIndices.get(level) - 1);
                }
            }
        }

        return allLists;
    }

    public static List<List<Integer>> partitionTreeIntoClusters(int[][] edges, TreeNode root, int cluster_number, String outputFileName) {
        // Construire la liste d'adjacence
        Map<Integer, List<Integer>> adjacencyList = buildAdjacencyListFromTree(root);
        List<List<Integer>> longestPaths = new ArrayList<>();
        
        // Trouver le plus long chemin
        List<Integer> longestPath = findRandomLongestPath(root);
        
        // Split the critical path into segments
        int criticalPathSplits = Math.min(cluster_number / 2, longestPath.size() / 3);
        List<List<Integer>> criticalPathSegments = splitCriticalPath(longestPath, criticalPathSplits);
        
        // Extraire les sous-arbres à partir du plus long chemin
        List<List<Integer>> subtrees = getSubtreesFromLongestPath(longestPath, adjacencyList);
        
        // Gérer les sous-arbres de grande taille
        List<List<Integer>> biggerSubTree = getSubtreeSizeDifferences(subtrees);
        subtrees = splitOversizedSubtrees(subtrees, biggerSubTree, edges, adjacencyList, longestPaths);
        
        // If we still don't have enough subtrees, split critical path segments further
        if (subtrees.size() + criticalPathSegments.size() < cluster_number) {
            List<List<Integer>> newCriticalPathSegments = new ArrayList<>();
            for (List<Integer> segment : criticalPathSegments) {
                if (segment.size() > 1) {
                    // Split segment in half if possible
                    int mid = segment.size() / 2;
                    newCriticalPathSegments.add(new ArrayList<>(segment.subList(0, mid)));
                    newCriticalPathSegments.add(new ArrayList<>(segment.subList(mid, segment.size())));
                } else {
                    newCriticalPathSegments.add(segment);
                }
            }
            criticalPathSegments = newCriticalPathSegments;
        }
        
        // Calculer le nombre de clusters restants après les divisions du chemin critique
        int remainingClusters = cluster_number - criticalPathSegments.size();
        
        // Regrouper les sous-arbres en clusters
        List<List<Integer>> clusters = clusterSubtrees(subtrees, remainingClusters);
        
        // Ajouter les segments du chemin critique aux clusters
        clusters.addAll(criticalPathSegments);
        
        // Optimiser l'affectation des nœuds
        clusters = optimizeNodeOnPath(longestPaths, adjacencyList, clusters);
        
        // Fusionner les clusters si nécessaire
        if (clusters.size() != cluster_number) {
            clusters = mergeClusters(clusters, cluster_number, adjacencyList, longestPath);
        }
        
        // Générer un fichier PNG représentant le graph
        generatePngFromGraph(edges, longestPath, clusters, outputFileName, adjacencyList);
        
        System.out.println("NEED CLUSTER OF CLUSTER : " + cluster_number);
        System.out.println("NUMBER OF CLUSTER : " + clusters.size());
        
        List<List<Integer>> allClusters = new ArrayList<>();
        
        // Keep splitting and merging until we have exactly cluster_number clusters
        while (true) {
            allClusters.clear();
            allClusters.addAll(subtrees);
            allClusters.addAll(criticalPathSegments);
            
            if (allClusters.size() == cluster_number) {
                break; // We have exactly the right number
            } else if (allClusters.size() < cluster_number) {
                // Need more clusters - split the largest cluster
                List<Integer> largestCluster = allClusters.stream()
                    .max((a, b) -> Integer.compare(a.size(), b.size()))
                    .orElse(null);
                    
                if (largestCluster != null && largestCluster.size() > 1) {
                    int mid = largestCluster.size() / 2;
                    List<Integer> part1 = new ArrayList<>(largestCluster.subList(0, mid));
                    List<Integer> part2 = new ArrayList<>(largestCluster.subList(mid, largestCluster.size()));
                    
                    if (subtrees.contains(largestCluster)) {
                        subtrees.remove(largestCluster);
                        subtrees.add(part1);
                        subtrees.add(part2);
                    } else {
                        criticalPathSegments.remove(largestCluster);
                        criticalPathSegments.add(part1);
                        criticalPathSegments.add(part2);
                    }
                } else {
                    System.out.println("Cannot create exactly " + cluster_number + " clusters. Maximum possible: " + allClusters.size());
                    break;
                }
            } else {
                // Too many clusters - merge the two smallest
                allClusters.sort((a, b) -> Integer.compare(a.size(), b.size()));
                List<Integer> merged = new ArrayList<>(allClusters.get(0));
                merged.addAll(allClusters.get(1));
                
                if (subtrees.contains(allClusters.get(0))) {
                    subtrees.remove(allClusters.get(0));
                    if (subtrees.contains(allClusters.get(1))) {
                        subtrees.remove(allClusters.get(1));
                        subtrees.add(merged);
                    }
                } else {
                    criticalPathSegments.remove(allClusters.get(0));
                    if (criticalPathSegments.contains(allClusters.get(1))) {
                        criticalPathSegments.remove(allClusters.get(1));
                        criticalPathSegments.add(merged);
                    }
                }
            }
        }

        // Ensure all nodes are assigned to a cluster
        Set<Integer> allNodes = new HashSet<>();
        Set<Integer> assignedNodes = new HashSet<>();
        
        // Collect all nodes from edges
        for (int[] edge : edges) {
            allNodes.add(edge[0]);
            allNodes.add(edge[1]);
        }
        
        // Collect all assigned nodes
        for (List<Integer> cluster : allClusters) {
            assignedNodes.addAll(cluster);
        }
        
        // Find unassigned nodes
        Set<Integer> unassignedNodes = new HashSet<>(allNodes);
        unassignedNodes.removeAll(assignedNodes);
        
        if (!unassignedNodes.isEmpty()) {
            // Sort unassigned nodes based on their dependencies
            List<Integer> sortedUnassigned = new ArrayList<>(unassignedNodes);
            sortedUnassigned.sort((a, b) -> {
                boolean aHasDeps = adjacencyList.containsKey(a);
                boolean bHasDeps = adjacencyList.containsKey(b);
                return Boolean.compare(bHasDeps, aHasDeps);
            });

            for (Integer node : sortedUnassigned) {
                int bestClusterIndex = findBestClusterForNode(node, allClusters, adjacencyList);
                if (bestClusterIndex >= 0) {
                    allClusters.get(bestClusterIndex).add(node);
                }
            }
        }
        
        // Generate visualization with all nodes assigned
        generatePngFromGraph(edges, longestPath, allClusters, outputFileName, adjacencyList);
        
        System.out.println("Created " + allClusters.size() + " clusters with all nodes assigned");
        return allClusters;
    }
    
    private static boolean isConnected(int node1, int node2, Map<Integer, List<Integer>> adjacencyList) {
        // Check direct connections in both directions
        if (adjacencyList.containsKey(node1) && adjacencyList.get(node1).contains(node2)) return true;
        if (adjacencyList.containsKey(node2) && adjacencyList.get(node2).contains(node1)) return true;
        
        // Check if nodes share a neighbor
        if (adjacencyList.containsKey(node1) && adjacencyList.containsKey(node2)) {
            Set<Integer> node1Neighbors = new HashSet<>(adjacencyList.get(node1));
            Set<Integer> node2Neighbors = new HashSet<>(adjacencyList.get(node2));
            node1Neighbors.retainAll(node2Neighbors);
            return !node1Neighbors.isEmpty();
        }
        return false;
    }

    private static int findBestClusterForNode(int node, List<List<Integer>> clusters, Map<Integer, List<Integer>> adjacencyList) {
        // First, try to find a cluster containing direct parent or child
        for (int i = 0; i < clusters.size(); i++) {
            List<Integer> cluster = clusters.get(i);
            for (int clusterNode : cluster) {
                // Check if the cluster contains a direct parent
                if (adjacencyList.containsKey(clusterNode) && 
                    adjacencyList.get(clusterNode).contains(node)) {
                    return i;
                }
                // Check if the cluster contains a direct child
                if (adjacencyList.containsKey(node) && 
                    adjacencyList.get(node).contains(clusterNode)) {
                    return i;
                }
            }
        }

        // If no direct connections found, fall back to connection count
        int bestCluster = -1;
        int maxConnections = -1;
        
        for (int i = 0; i < clusters.size(); i++) {
            List<Integer> cluster = clusters.get(i);
            int connections = 0;
            boolean violatesDependencies = false;
            
            for (int clusterNode : cluster) {
                if (isConnected(node, clusterNode, adjacencyList)) {
                    connections++;
                }
            }
            
            if (!violatesDependencies && connections > maxConnections) {
                maxConnections = connections;
                bestCluster = i;
            }
        }
        
        // If still no suitable cluster found, return the smallest cluster
        if (bestCluster == -1) {
            int minSize = Integer.MAX_VALUE;
            for (int i = 0; i < clusters.size(); i++) {
                if (clusters.get(i).size() < minSize) {
                    minSize = clusters.get(i).size();
                    bestCluster = i;
                }
            }
        }
        
        return bestCluster;
    }

    public static List<List<Integer>> splitCriticalPath(List<Integer> criticalPath, int desiredSplits) {
        List<List<Integer>> splits = new ArrayList<>();
        if (criticalPath == null || criticalPath.isEmpty() || desiredSplits <= 0) {
            return splits;
        }

        int segmentSize = criticalPath.size() / desiredSplits;
        int remainingNodes = criticalPath.size() % desiredSplits;

        int startIndex = 0;
        for (int i = 0; i < desiredSplits; i++) {
            int currentSegmentSize = segmentSize + (remainingNodes > 0 ? 1 : 0);
            if (remainingNodes > 0) remainingNodes--;

            int endIndex = Math.min(startIndex + currentSegmentSize, criticalPath.size());
            splits.add(new ArrayList<>(criticalPath.subList(startIndex, endIndex)));
            startIndex = endIndex;
        }
        return splits;
    }

    public static void main(String[] args) {
        int cluster_number = 5;
        //int[][] edges = generateRandomReverseTree(node_number); // Générer des arêtes aléatoires
        
        int[][] edges = {
            {45,24}, {39,24}, 
            {44,45}, {46,45}, {50,39}, {38,39},
            {57,50}, {51,50}, {55,38}, {40,38},
            {47,55}, {56,55}, {52,40}, {42,40},
            {49,47}, {48,47}, {53,52}, {54,52},
            {63,42}, {62,42} , {41,42},
            {58,41},{59,58},{61,58},
            {37,41}, {64,37}, {60,64}, {65,64},
            {36,37}, {66,36}, {35,36}, {68,35}, {34,35},
            {69,68}, {17,34}, {32,34},
            {22,32}, {9,32}, {31,32},
            {23,22}, {21,22}, {11,9}, {15,9},
            {13,11}, {10,11}, {14,13}, {12,13},
            {18,31}, {30,31}, {19,18}, {20,18},
            {16,30}, {29,30}, {2,29}, {4,29},
            {1,2}, {3,2}, {28,4},
            {5,28}, {26,28}, {7,5}, {6,5},
            {8,26}, {25,26}
        };

        // Generate PNG of graph without color, named "normal.png"
        generateNormalPngFromGraph(edges, "normal");

        // ...existing code that partitions graph and generates colored output...
        TreeNode root = buildTreeFromEdges(edges); // Construire l'arbre à partir des arêtes
        List<List<Integer>> clusters = partitionTreeIntoClusters(edges, root, cluster_number, "graph_output_critical"); // Partitionner l'arbre en clusters et générer l'image

        System.out.println("FINAL CLUSTER ");
        int node = 0;
        for(var auto : clusters)
        {   
            node += auto.size();
            System.out.println(auto);
        }

        System.out.println("TOTAL SIZE " + node);
        getClusterSizeDifferences(clusters);
    }
}