import java.util.*;

class TreeNode {
    int value;
    List<TreeNode> children;

    public TreeNode(int value) {
        this.value = value;
        this.children = new ArrayList<>();
    }
}

class DAGPartitioning_main 
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
        int clusterSize = (int) Math.ceil((double) topoOrder.size() / numClusters);
        List<List<Integer>> clusters = new ArrayList<>();
        Map<Integer, Integer> nodeToCluster = new HashMap<>();
    
        for (int i = 0; i < numClusters; i++) {
            clusters.add(new ArrayList<>());
        }
    
        int clusterIndex = 0;
        for (int node : topoOrder) {
            clusters.get(clusterIndex).add(node);
            nodeToCluster.put(node, clusterIndex);
            if (clusters.get(clusterIndex).size() >= clusterSize) {
                clusterIndex = Math.min(clusterIndex + 1, numClusters - 1);
            }
        }
    
        // Step 4: Enforce Dependency Constraints (No Circular Dependencies)
        for (int i = 0; i < numClusters; i++) {
            for (int node : clusters.get(i)) {
                if (adjacencyList.containsKey(node)) {
                    for (int neighbor : adjacencyList.get(node)) {
                        int neighborCluster = nodeToCluster.get(neighbor);
                        if (neighborCluster < i) { 
                            System.out.println("Dependency Violation: Cluster " + i + " depends on Cluster " + neighborCluster);
                        }
                    }
                }
            }
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
    public static void main(String[] args) {
        
        int node_number = 200;
        int[][] edges = generateRandomReverseTree(node_number); // Générer des arêtes aléatoires

        System.out.println("edges");
        List<List<Integer>> clusters = partitionGraphWithDependencies(edges, 2);
        
        int clusterNumber = 1;
        for (List<Integer> cluster : clusters) {
            System.out.print("Cluster " + clusterNumber + ": ");
            for (Integer node : cluster) {
                System.out.print(node + " ");
            }
            System.out.println();
            clusterNumber++;
        }
    }
}


