import java.io.*;
import java.awt.image.BufferedImage;
import java.awt.Color;
import javax.imageio.ImageIO;
import java.util.*;

public class CAMMISOL {
    private Integer[][] clusters; // Stores cluster assignments
    private int[][] agentMask; // 1 = agent present, 0 = no agent
    private int rows;
    private int cols;
    private int numClusters;

    public CAMMISOL(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    public void setAgentMask(int[][] mask) {
        if (mask.length != rows || mask[0].length != cols) {
            throw new IllegalArgumentException("Agent mask dimensions must match grid dimensions");
        }
        this.agentMask = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.agentMask[i][j] = mask[i][j];
            }
        }
    }

    public void generateRandomAgentMask(double agentRatio) {
        Random rand = new Random();
        this.agentMask = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.agentMask[i][j] = rand.nextDouble() < agentRatio ? 1 : 0;
            }
        }
    }

    public void partitionAgents(int numClusters) {
        if (agentMask == null) {
            throw new IllegalStateException("Agent mask must be set before partitioning");
        }
        
        this.numClusters = numClusters;
        this.clusters = new Integer[rows][cols];
        
        // Initialize all cells to -1 (no cluster assigned)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clusters[i][j] = -1;
            }
        }
        
        // Collect all agent cells
        List<int[]> agentCells = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (agentMask[i][j] == 1) {
                    agentCells.add(new int[]{i, j});
                }
            }
        }
        
        if (agentCells.isEmpty()) {
            System.out.println("No agents found in mask");
            return;
        }
        
        int totalAgents = agentCells.size();
        int agentsPerCluster = totalAgents / numClusters;
        int remainingAgents = totalAgents % numClusters;
        
        System.out.println("Total agents: " + totalAgents);
        System.out.println("Agents per cluster: " + agentsPerCluster + 
                          (remainingAgents > 0 ? " (+" + remainingAgents + " extra)" : ""));
        
        // Create seeds for clusters distributed across the grid for maximum spatial separation
        List<int[]> seeds = new ArrayList<>();
        Random rand = new Random(System.currentTimeMillis());
        
        // Ensure we create exactly numClusters seeds with maximum spatial separation
        if (numClusters <= agentCells.size()) {
            // Method 1: Use recursive spatial subdivision for better distribution
            List<int[]> regions = new ArrayList<>();
            regions.add(new int[]{0, 0, rows, cols}); // [startRow, startCol, endRow, endCol]
            
            while (regions.size() < numClusters && !regions.isEmpty()) {
                List<int[]> newRegions = new ArrayList<>();
                for (int[] region : regions) {
                    int startRow = region[0], startCol = region[1];
                    int endRow = region[2], endCol = region[3];
                    int midRow = (startRow + endRow) / 2;
                    int midCol = (startCol + endCol) / 2;
                    
                    // Split the largest dimension
                    if (endRow - startRow > endCol - startCol) {
                        // Split horizontally
                        newRegions.add(new int[]{startRow, startCol, midRow, endCol});
                        newRegions.add(new int[]{midRow, startCol, endRow, endCol});
                    } else {
                        // Split vertically
                        newRegions.add(new int[]{startRow, startCol, endRow, midCol});
                        newRegions.add(new int[]{startRow, midCol, endRow, endCol});
                    }
                }
                regions = newRegions;
                if (regions.size() >= numClusters) break;
            }
            
            // Take the first numClusters regions and find seed in each
            for (int i = 0; i < numClusters && i < regions.size(); i++) {
                int[] region = regions.get(i);
                int startRow = region[0], startCol = region[1];
                int endRow = region[2], endCol = region[3];
                int centerRow = (startRow + endRow) / 2;
                int centerCol = (startCol + endCol) / 2;
                
                // Find closest agent to center of this region
                int[] bestAgent = null;
                double minDist = Double.MAX_VALUE;
                for (int[] agent : agentCells) {
                    if (clusters[agent[0]][agent[1]] != -1) continue; // Already assigned
                    // Prefer agents within the region
                    boolean inRegion = agent[0] >= startRow && agent[0] < endRow && 
                                     agent[1] >= startCol && agent[1] < endCol;
                    double dist = Math.sqrt(Math.pow(agent[0] - centerRow, 2) + 
                                          Math.pow(agent[1] - centerCol, 2));
                    if (inRegion) dist *= 0.1; // Strong preference for agents in region
                    
                    if (dist < minDist) {
                        minDist = dist;
                        bestAgent = agent;
                    }
                }
                if (bestAgent != null) {
                    seeds.add(bestAgent);
                    clusters[bestAgent[0]][bestAgent[1]] = seeds.size() - 1;
                }
            }
            
            // Fill remaining seeds if needed
            while (seeds.size() < numClusters) {
                boolean foundSeed = false;
                for (int[] agent : agentCells) {
                    if (clusters[agent[0]][agent[1]] == -1) {
                        seeds.add(agent);
                        clusters[agent[0]][agent[1]] = seeds.size() - 1;
                        foundSeed = true;
                        break;
                    }
                }
                if (!foundSeed) break;
            }
        }
        
        System.out.println("Created " + seeds.size() + " seed clusters");

        // Assign remaining agents to maintain even distribution and minimize frontiers
        Collections.shuffle(agentCells, rand);
        Map<Integer, Integer> clusterCounts = new HashMap<>();
        for (int i = 0; i < numClusters; i++) {
            clusterCounts.put(i, 0); // Initialize all clusters to 0
        }
        
        // Count seeds already assigned
        for (int[] seed : seeds) {
            if (clusters[seed[0]][seed[1]] >= 0) {
                int cluster = clusters[seed[0]][seed[1]];
                clusterCounts.put(cluster, clusterCounts.get(cluster) + 1);
            }
        }

        for (int[] agent : agentCells) {
            int i = agent[0], j = agent[1];
            if (clusters[i][j] != -1) continue; // Already assigned
            
            // Find cluster with fewest agents that's spatially close and minimizes frontiers
            int bestCluster = -1;
            double bestScore = Double.MAX_VALUE;
            
            for (int cluster = 0; cluster < numClusters; cluster++) {
                int currentCount = clusterCounts.get(cluster);
                int targetCount = agentsPerCluster + (cluster < remainingAgents ? 1 : 0);
                
                if (currentCount >= targetCount) continue; // Cluster is full
                
                // Calculate spatial distance to cluster
                double minDistToCluster = Double.MAX_VALUE;
                for (int[] seed : seeds) {
                    if (clusters[seed[0]][seed[1]] == cluster) {
                        double dist = Math.sqrt(Math.pow(i - seed[0], 2) + Math.pow(j - seed[1], 2));
                        minDistToCluster = Math.min(minDistToCluster, dist);
                    }
                }
                
                // Find nearest agent in this cluster
                for (int ii = 0; ii < rows; ii++) {
                    for (int jj = 0; jj < cols; jj++) {
                        if (clusters[ii][jj] == cluster) {
                            double dist = Math.sqrt(Math.pow(i - ii, 2) + Math.pow(j - jj, 2));
                            minDistToCluster = Math.min(minDistToCluster, dist);
                        }
                    }
                }
                
                // Calculate frontier penalty - how many neighbors would be in different clusters
                double frontierPenalty = calculateFrontierPenalty(i, j, cluster);
                
                // Score combines spatial distance, load balancing, and frontier minimization
                double loadFactor = (double)currentCount / targetCount;
                double score = minDistToCluster * (1.0 + loadFactor) + frontierPenalty * 10.0;
                
                if (score < bestScore) {
                    bestScore = score;
                    bestCluster = cluster;
                }
            }
            
            if (bestCluster != -1) {
                clusters[i][j] = bestCluster;
                clusterCounts.put(bestCluster, clusterCounts.get(bestCluster) + 1);
            }
        }
        
        // Enhanced refinement phase to prioritize spatial coherence and minimize frontiers
        for (int iteration = 0; iteration < 200; iteration++) {
            boolean improved = false;
            Collections.shuffle(agentCells, rand);
            
            for (int[] agent : agentCells) {
                int i = agent[0], j = agent[1];
                int currentCluster = clusters[i][j];
                
                // Find neighbor clusters
                Set<Integer> neighborClusters = getNeighborAgentClusters(i, j);
                
                for (int newCluster : neighborClusters) {
                    if (newCluster == currentCluster) continue;
                    
                    int currentCount = clusterCounts.getOrDefault(currentCluster, 0);
                    int targetCount = clusterCounts.getOrDefault(newCluster, 0);
                    
                    // Calculate improvement in frontier reduction (higher weight)
                    double currentFrontierCost = calculateFrontierPenalty(i, j, currentCluster);
                    double newFrontierCost = calculateFrontierPenalty(i, j, newCluster);
                    double frontierImprovement = (currentFrontierCost - newFrontierCost) * 5.0;
                    
                    // Calculate spatial coherence improvement (higher weight)
                    double spatialImprovement = (calculateAgentSpatialCoherence(i, j, newCluster) - 
                                               calculateAgentSpatialCoherence(i, j, currentCluster)) * 3.0;
                    
                    // Load balancing factor (lower weight for more contiguous regions)
                    double balanceFactor = 0;
                    if (currentCount > targetCount + 2) {
                        balanceFactor = 1.0; // Encourage moves from overpopulated clusters
                    } else if (currentCount < targetCount - 2) {
                        balanceFactor = -1.0; // Discourage moves from underpopulated clusters
                    }
                    
                    // Combined score prioritizing spatial coherence
                    double totalImprovement = frontierImprovement + spatialImprovement + balanceFactor;
                    
                    if (totalImprovement > 0.05) {
                        clusters[i][j] = newCluster;
                        clusterCounts.put(currentCluster, currentCount - 1);
                        clusterCounts.put(newCluster, targetCount + 1);
                        improved = true;
                        break;
                    }
                }
            }
            
            if (!improved) break;
        }
        
        // Calculate and print final statistics
        int totalFrontierCells = calculateTotalFrontierCells();
        int totalCommunicationEdges = calculateCommunicationEdges();
        
        System.out.println("\nFinal agent distribution:");
        for (int i = 0; i < numClusters; i++) {
            int count = clusterCounts.getOrDefault(i, 0);
            System.out.println("Cluster " + i + ": " + count + " agents");
        }
        System.out.println("\nCommunication metrics:");
        System.out.println("Total frontier cells: " + totalFrontierCells);
        System.out.println("Total communication edges: " + totalCommunicationEdges);
    }

    private Set<Integer> getNeighborAgentClusters(int i, int j) {
        Set<Integer> neighbors = new HashSet<>();
        int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] dir : dirs) {
            int ni = i + dir[0];
            int nj = j + dir[1];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && 
                agentMask[ni][nj] == 1 && clusters[ni][nj] >= 0) {
                neighbors.add(clusters[ni][nj]);
            }
        }
        return neighbors;
    }

    private double calculateAgentSpatialCoherence(int i, int j, int targetCluster) {
        int sameClusterNeighbors = 0;
        int totalNeighbors = 0;
        for (int di = -2; di <= 2; di++) {
            for (int dj = -2; dj <= 2; dj++) {
                if (di == 0 && dj == 0) continue;
                
                int ni = i + di;
                int nj = j + dj;
                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && 
                    agentMask[ni][nj] == 1 && clusters[ni][nj] >= 0) {
                    totalNeighbors++;
                    if (clusters[ni][nj] == targetCluster) {
                        sameClusterNeighbors++;
                    }
                }
            }
        }
        return totalNeighbors > 0 ? (double)sameClusterNeighbors / totalNeighbors : 0;
    }

    public void saveAgentVisualization(String filename, int pngWidth, int pngHeight) throws IOException {
        BufferedImage image = new BufferedImage(pngWidth * 2, pngHeight, BufferedImage.TYPE_INT_RGB);
        double scaleX = pngWidth / (double)cols;
        double scaleY = pngHeight / (double)rows;
        
        // Use fixed color palette for consistent visualization
        Color[] clusterColors = {
            Color.RED,
            Color.BLUE,
            Color.YELLOW,
            Color.GREEN,
            Color.ORANGE,
            Color.MAGENTA,  // Purple/Pink
            Color.PINK,
            Color.GRAY
        };

        for (int y = 0; y < pngHeight; y++) {
            for (int x = 0; x < pngWidth * 2; x++) {
                int i = (int)(y / scaleY);
                int j = (int)((x % pngWidth) / scaleX);
                
                if (x < pngWidth) {
                    // Left: agent mask (white = agent, black = no agent)
                    Color color = agentMask[i][j] == 1 ? Color.WHITE : Color.BLACK;
                    image.setRGB(x, y, color.getRGB());
                } else {
                    // Right: cluster assignments (only for agent cells)
                    if (agentMask[i][j] == 1 && clusters[i][j] >= 0) {
                        Color clusterColor = clusterColors[clusters[i][j] % clusterColors.length];
                        image.setRGB(x, y, clusterColor.getRGB());
                    } else {
                        image.setRGB(x, y, Color.BLACK.getRGB());
                    }
                }
            }
        }

        File outputFile = new File(filename);
        ImageIO.write(image, "png", outputFile);
    }

    public Map<Integer, Integer> getAgentCounts() {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (agentMask[i][j] == 1 && clusters[i][j] >= 0) {
                    counts.merge(clusters[i][j], 1, Integer::sum);
                }
            }
        }
        return counts;
    }

    public int[][] getPartition() {
        int[][] partition = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                partition[i][j] = clusters[i][j];
            }
        }
        return partition;
    }

    private double calculateFrontierPenalty(int i, int j, int targetCluster) {
        int differentClusterNeighbors = 0;
        int totalAgentNeighbors = 0;
        
        int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] dir : dirs) {
            int ni = i + dir[0];
            int nj = j + dir[1];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && agentMask[ni][nj] == 1) {
                totalAgentNeighbors++;
                if (clusters[ni][nj] >= 0 && clusters[ni][nj] != targetCluster) {
                    differentClusterNeighbors++;
                }
            }
        }
        
        return totalAgentNeighbors > 0 ? (double)differentClusterNeighbors / totalAgentNeighbors : 0;
    }

    private int calculateTotalFrontierCells() {
        Set<Integer> frontierCells = new HashSet<>();
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (agentMask[i][j] == 1 && clusters[i][j] >= 0) {
                    // Check if this agent has neighbors in different clusters
                    int currentCluster = clusters[i][j];
                    boolean isFrontier = false;
                    
                    int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
                    for (int[] dir : dirs) {
                        int ni = i + dir[0];
                        int nj = j + dir[1];
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && 
                            agentMask[ni][nj] == 1 && clusters[ni][nj] >= 0 && 
                            clusters[ni][nj] != currentCluster) {
                            isFrontier = true;
                            break;
                        }
                    }
                    
                    if (isFrontier) {
                        frontierCells.add(i * cols + j);
                    }
                }
            }
        }
        
        return frontierCells.size();
    }

    private int calculateCommunicationEdges() {
        int communicationEdges = 0;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (agentMask[i][j] == 1 && clusters[i][j] >= 0) {
                    int currentCluster = clusters[i][j];
                    
                    // Check right and down neighbors to avoid double counting
                    int[][] dirs = {{0,1}, {1,0}};
                    for (int[] dir : dirs) {
                        int ni = i + dir[0];
                        int nj = j + dir[1];
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && 
                            agentMask[ni][nj] == 1 && clusters[ni][nj] >= 0 && 
                            clusters[ni][nj] != currentCluster) {
                            communicationEdges++;
                        }
                    }
                }
            }
        }
        
        return communicationEdges;
    }

    public static void main(String[] args) {
        try {
            // Configuration variables - adjust these parameters as needed
            int GRID_SIZE = 150;           // Grid dimensions (rows x cols)
            double AGENT_RATIO = 0.3;      // Percentage of cells with agents (0.0 to 1.0)
            int NUM_CLUSTERS = 128;          // Number of partitions to create
            int IMAGE_SIZE = 1000;          // PNG output size in pixels
            String OUTPUT_DIR = "output";   // Directory for output files
            
            // Example usage
            CAMMISOL grid = new CAMMISOL(GRID_SIZE, GRID_SIZE);
            
            // Create custom agent pattern
            int[][] agentMask = new int[GRID_SIZE][GRID_SIZE];
            Random rand = new Random(System.currentTimeMillis());
            
            // Create scattered agents based on ratio
            for (int i = 0; i < GRID_SIZE; i++) {
                for (int j = 0; j < GRID_SIZE; j++) {
                    if (rand.nextDouble() < AGENT_RATIO) {
                        agentMask[i][j] = 1;
                    }
                }
            }
            
            grid.setAgentMask(agentMask);
            grid.partitionAgents(NUM_CLUSTERS);
            
            // Save visualization
            new File(OUTPUT_DIR).mkdirs();
            grid.saveAgentVisualization(OUTPUT_DIR + "/agent_partition.png", IMAGE_SIZE, IMAGE_SIZE);
            
            // Print results
            Map<Integer, Integer> agentCounts = grid.getAgentCounts();
            System.out.println("\nFinal agent distribution:");
            agentCounts.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .forEach(entry -> {
                    System.out.println("Cluster " + entry.getKey() + ": " + entry.getValue() + " agents");
                });
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}