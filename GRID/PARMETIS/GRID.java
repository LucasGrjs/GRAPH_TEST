import java.io.*;
import java.awt.image.BufferedImage;
import java.awt.Color;
import javax.imageio.ImageIO;
import java.util.*;
import java.util.stream.Collectors;

public class GRID {
    private Integer[][] grid; 
    private Integer[][] clusters; // Stores cluster assignments
    private int[][] computationMask; // 1 = agent present, 0 = no agent
    private int rows;
    private int cols;
    private int numClusters;

    public GRID(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.grid = new Integer[rows][cols];
    }

    public void generateRandomGrid(int maxValue) {
        Random rand = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grid[i][j] = rand.nextInt(maxValue);
            }
        }
    }

    public void saveGridAsPNG(String filename, int pngWidth, int pngHeight) throws IOException {
        BufferedImage image = new BufferedImage(pngWidth, pngHeight, BufferedImage.TYPE_INT_RGB);
        double scaleX = pngWidth / (double)cols;
        double scaleY = pngHeight / (double)rows;
        
        // Find max value for color scaling
        int maxVal = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                maxVal = Math.max(maxVal, grid[i][j]);
            }
        }

        // Create scaled image
        for (int y = 0; y < pngHeight; y++) {
            for (int x = 0; x < pngWidth; x++) {
                int i = (int)(y / scaleY);
                int j = (int)(x / scaleX);
                float intensity = grid[i][j] / (float)maxVal;
                Color color = new Color(intensity, intensity, intensity);
                image.setRGB(x, y, color.getRGB());
            }
        }

        File outputFile = new File(filename);
        ImageIO.write(image, "png", outputFile);
    }

    public void saveCombinedImage(String filename, int pngWidth, int pngHeight) throws IOException {
        // Create a combined image twice as wide to hold both visualizations side by side
        BufferedImage image = new BufferedImage(pngWidth * 2, pngHeight, BufferedImage.TYPE_INT_RGB);
        double scaleX = pngWidth / (double)cols;
        double scaleY = pngHeight / (double)rows;
        
        // Find max value for color scaling
        int maxVal = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                maxVal = Math.max(maxVal, grid[i][j]);
            }
        }

        // Generate distinct colors for clusters
        Color[] clusterColors = new Color[numClusters];
        Random rand = new Random(42); // Fixed seed for consistent colors
        for (int i = 0; i < numClusters; i++) {
            clusterColors[i] = new Color(rand.nextFloat(), rand.nextFloat(), rand.nextFloat());
        }

        // Draw both visualizations side by side
        for (int y = 0; y < pngHeight; y++) {
            for (int x = 0; x < pngWidth * 2; x++) {
                int i = (int)(y / scaleY);
                int j = (int)((x % pngWidth) / scaleX);
                
                if (x < pngWidth) {
                    // Left side: grid visualization
                    float intensity = grid[i][j] / (float)maxVal;
                    Color color = new Color(intensity, intensity, intensity);
                    image.setRGB(x, y, color.getRGB());
                } else {
                    // Right side: cluster visualization
                    image.setRGB(x, y, clusterColors[clusters[i][j]].getRGB());
                }
            }
        }

        File outputFile = new File(filename);
        ImageIO.write(image, "png", outputFile);
    }

    private double calculateImbalance(Map<Integer, Long> weights) {
        long maxWeight = Collections.max(weights.values());
        long minWeight = Collections.min(weights.values());
        return maxWeight - minWeight;  // Return absolute difference instead of ratio
    }

    private boolean tryMoveCell(int i, int j, Map<Integer, Long> clusterWeights) {
        int currentCluster = clusters[i][j];
        long currentValue = grid[i][j];
        Set<Integer> neighborClusters = getNeighborClusters(i, j);
        
        // Check if moving would leave cluster with no weight
        if (clusterWeights.get(currentCluster) <= currentValue) {
            return false;
        }

        long currentWeight = clusterWeights.get(currentCluster);
        long avgWeight = getAverageWeight(clusterWeights);
        
        for (int newCluster : neighborClusters) {
            if (newCluster == currentCluster) continue;
            
            long targetWeight = clusterWeights.get(newCluster);
            
            // Calculate weight balance improvement
            long oldDiff = Math.abs(currentWeight - avgWeight) + Math.abs(targetWeight - avgWeight);
            long newDiff = Math.abs(currentWeight - currentValue - avgWeight) + 
                          Math.abs(targetWeight + currentValue - avgWeight);
            double balanceGain = oldDiff - newDiff;
            
            // Calculate spatial coherence score
            double spatialCoherence = calculateSpatialCoherence(i, j, newCluster);
            
            // Combined score with higher weight on spatial coherence
            double improvement = balanceGain + (spatialCoherence * 2.0);
            
            if (improvement > 0) {
                clusters[i][j] = newCluster;
                clusterWeights.put(currentCluster, currentWeight - currentValue);
                clusterWeights.put(newCluster, targetWeight + currentValue);
                return true;
            }
        }
        return false;
    }

    private double calculateSpatialCoherence(int i, int j, int targetCluster) {
        int sameClusterNeighbors = 0;
        int totalNeighbors = 0;
        // Include more neighbors for better coherence
        for (int di = -2; di <= 2; di++) {
            for (int dj = -2; dj <= 2; dj++) {
                if (di == 0 && dj == 0) continue;
                
                int ni = i + di;
                int nj = j + dj;
                if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                    totalNeighbors++;
                    if (clusters[ni][nj] == targetCluster) {
                        sameClusterNeighbors++;
                    }
                }
            }
        }
        return totalNeighbors > 0 ? (double)sameClusterNeighbors / totalNeighbors : 0;
    }

    private boolean isLastCellInCluster(int ci, int cj) {
        int targetCluster = clusters[ci][cj];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == ci && j == cj) continue;
                if (clusters[i][j] == targetCluster) {
                    return false;
                }
            }
        }
        return true;
    }

    private void validateClusterCount() {
        Set<Integer> uniqueClusters = new HashSet<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                uniqueClusters.add(clusters[i][j]);
            }
        }
        if (uniqueClusters.size() != numClusters) {
            throw new IllegalStateException(
                String.format("Invalid cluster count: %d (expected %d)", 
                uniqueClusters.size(), numClusters));
        }
    }

    public void partitionGrid(int numClusters) {
        this.numClusters = numClusters;
        this.clusters = new Integer[rows][cols];
        
        // Initialize seeds for clusters at evenly spaced positions
        List<int[]> seeds = new ArrayList<>();
        int seedSpacingRow = rows / (int)Math.sqrt(numClusters);
        int seedSpacingCol = cols / (int)Math.sqrt(numClusters);
        
        for (int i = seedSpacingRow/2; i < rows; i += seedSpacingRow) {
            for (int j = seedSpacingCol/2; j < cols; j += seedSpacingCol) {
                if (seeds.size() < numClusters) {
                    seeds.add(new int[]{i, j});
                }
            }
        }
        
        // Initialize all cells to nearest seed, weighted by grid value
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double minCost = Double.MAX_VALUE;
                int bestCluster = 0;
                
                for (int k = 0; k < seeds.size(); k++) {
                    int[] seed = seeds.get(k);
                    double dist = Math.sqrt(Math.pow(i - seed[0], 2) + Math.pow(j - seed[1], 2));
                    double cost = dist * (1.0 + grid[i][j] / 1000.0); // Weight distance by cell value
                    
                    if (cost < minCost) {
                        minCost = cost;
                        bestCluster = k;
                    }
                }
                clusters[i][j] = bestCluster;
            }
        }

        // Refinement phase with stronger spatial coherence
        final int MAX_ITERATIONS = 100;
        Random rand = new Random();
        List<int[]> cellPositions = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cellPositions.add(new int[]{i, j});
            }
        }

        Map<Integer, Long> weights = calculateClusterWeights();
        long targetWeight = weights.values().stream().mapToLong(Long::longValue).sum() / numClusters;
        
        for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
            Collections.shuffle(cellPositions, rand);
            boolean improved = false;
            
            for (int[] pos : cellPositions) {
                int i = pos[0], j = pos[1];
                int currentCluster = clusters[i][j];
                Set<Integer> neighbors = getNeighborClusters(i, j);
                
                for (int newCluster : neighbors) {
                    if (newCluster == currentCluster) continue;
                    
                    // Calculate spatial coherence score (weighted more heavily)
                    double spatialScore = calculateSpatialCoherence(i, j, newCluster) * 3.0;
                    
                    // Calculate balance score
                    long currentWeight = weights.get(currentCluster);
                    long neighborWeight = weights.get(newCluster);
                    long cellWeight = grid[i][j];
                    
                    double balanceScore = Math.abs(currentWeight - targetWeight) + 
                                        Math.abs(neighborWeight - targetWeight) -
                                        (Math.abs(currentWeight - cellWeight - targetWeight) + 
                                         Math.abs(neighborWeight + cellWeight - targetWeight));
                    
                    // Combined score with higher weight on spatial coherence
                    if (spatialScore + balanceScore > 0) {
                        clusters[i][j] = newCluster;
                        weights.put(currentCluster, currentWeight - cellWeight);
                        weights.put(newCluster, neighborWeight + cellWeight);
                        improved = true;
                        break;
                    }
                }
            }
            
            if (!improved) break;
        }
    }

    private Map<Integer, Long> calculateClusterWeights() {
        Map<Integer, Long> weights = new HashMap<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int cluster = clusters[i][j];
                weights.merge(cluster, (long)grid[i][j], Long::sum);
            }
        }
        return weights;
    }

    private Set<Integer> getNeighborClusters(int i, int j) {
        Set<Integer> neighbors = new HashSet<>();
        int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] dir : dirs) {
            int ni = i + dir[0];
            int nj = j + dir[1];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                neighbors.add(clusters[ni][nj]);
            }
        }
        return neighbors;
    }

    private long getAverageWeight(Map<Integer, Long> weights) {
        return weights.values().stream().mapToLong(Long::longValue).sum() / numClusters;
    }

    public void saveClustersPNG(String filename, int pngWidth, int pngHeight) throws IOException {
        BufferedImage image = new BufferedImage(pngWidth, pngHeight, BufferedImage.TYPE_INT_RGB);
        double scaleX = pngWidth / (double)cols;
        double scaleY = pngHeight / (double)rows;
        
        // Generate distinct colors for clusters
        Color[] clusterColors = new Color[numClusters];
        Random rand = new Random(42);
        for (int i = 0; i < numClusters; i++) {
            clusterColors[i] = new Color(rand.nextFloat(), rand.nextFloat(), rand.nextFloat());
        }

        // Create scaled image
        for (int y = 0; y < pngHeight; y++) {
            for (int x = 0; x < pngWidth; x++) {
                int i = (int)(y / scaleY);
                int j = (int)(x / scaleX);
                image.setRGB(x, y, clusterColors[clusters[i][j]].getRGB());
            }
        }

        File outputFile = new File(filename);
        ImageIO.write(image, "png", outputFile);
    }

    public void generateGradientGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grid[i][j] = (i + j) * 255 / (rows + cols);
            }
        }
    }

    public void generateBlockGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if ((i < rows/2 && j < cols/2) || (i >= rows/2 && j >= cols/2)) {
                    grid[i][j] = 255;
                } else {
                    grid[i][j] = 0;
                }
            }
        }
    }

    public void generateCircleGrid() {
        int centerX = rows/2;
        int centerY = cols/2;
        int radius = Math.min(rows, cols)/4;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double distance = Math.sqrt(Math.pow(i-centerX, 2) + Math.pow(j-centerY, 2));
                grid[i][j] = distance <= radius ? 255 : 0;
            }
        }
    }

    public void generateSpiralGrid() {
        int centerX = rows/2, centerY = cols/2;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double angle = Math.atan2(i - centerX, j - centerY);
                double distance = Math.sqrt(Math.pow(i-centerX, 2) + Math.pow(j-centerY, 2));
                grid[i][j] = (int)(((angle + Math.PI) / (2 * Math.PI) + distance/Math.max(rows, cols)) * 127);
            }
        }
    }

    public void generateHotspotGrid() {
        Random rand = new Random();
        int numHotspots = 5;
        int[][] hotspots = new int[numHotspots][2];
        
        // Generate random hotspot locations
        for (int i = 0; i < numHotspots; i++) {
            hotspots[i] = new int[]{rand.nextInt(rows), rand.nextInt(cols)};
        }

        // Generate values based on distance to nearest hotspot
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double minDistance = Double.MAX_VALUE;
                for (int[] hotspot : hotspots) {
                    double distance = Math.sqrt(Math.pow(i-hotspot[0], 2) + Math.pow(j-hotspot[1], 2));
                    minDistance = Math.min(minDistance, distance);
                }
                grid[i][j] = (int)(255 * Math.exp(-minDistance/20));
            }
        }
    }

    public void generateCheckerboardGrid() {
        int blockSize = 50;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                boolean isEvenBlock = ((i/blockSize) + (j/blockSize)) % 2 == 0;
                grid[i][j] = isEvenBlock ? 255 : 0;
            }
        }
    }

    public void generateDiagonalStripesGrid() {
        int stripeWidth = 30;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grid[i][j] = ((i + j) / stripeWidth) % 2 == 0 ? 255 : 0;
            }
        }
    }

    public void generateGaussianGrid() {
        Random rand = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = rand.nextGaussian() * 64 + 128;
                grid[i][j] = (int)Math.max(0, Math.min(255, value));
            }
        }
    }

    public void generateWaveGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = Math.sin(i * 0.1) * Math.cos(j * 0.1) * 127 + 128;
                grid[i][j] = (int)value;
            }
        }
    }

    public void generatePowerLawGrid() {
        Random rand = new Random();
        double alpha = 2.0; // Power law exponent
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double u = rand.nextDouble();
                int value = (int)(255 * Math.pow(u, -1/alpha));
                grid[i][j] = Math.min(255, value);
            }
        }
    }

    public void generateCornerHeavyGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // High values in corners, low in center
                int distToCorner = Math.min(
                    Math.min(i + j, (rows-i-1) + (cols-j-1)),
                    Math.min(i + (cols-j-1), (rows-i-1) + j)
                );
                grid[i][j] = 255 - (distToCorner * 255 / (rows + cols));
            }
        }
    }

    public void generateExponentialGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = i / (double)rows;
                double y = j / (double)cols;
                grid[i][j] = (int)(255 * Math.exp(-(x*x + y*y) * 3));
            }
        }
    }

    public void generateBimodalGrid() {
        Random rand = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // 70% low values, 30% high values
                if (rand.nextDouble() < 0.7) {
                    grid[i][j] = rand.nextInt(50);
                } else {
                    grid[i][j] = 200 + rand.nextInt(55);
                }
            }
        }
    }

    public void generateStepGrid() {
        int numSteps = 5;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int step = (i * numSteps / rows);
                grid[i][j] = step * 255 / (numSteps - 1);
            }
        }
    }

    public void generateConcentricsGrid() {
        int centerX = rows/2, centerY = cols/2;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double distance = Math.sqrt(Math.pow(i-centerX, 2) + Math.pow(j-centerY, 2));
                grid[i][j] = (int)(127 * (1 + Math.cos(distance * 0.3)));
            }
        }
    }

    public void generateMountainsGrid() {
        Random rand = new Random(42);
        double[][] noise = new double[rows][cols];
        // Generate Perlin-like noise
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                noise[i][j] = rand.nextDouble();
            }
        }
        // Smooth the noise
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                int count = 0;
                for (int di = -2; di <= 2; di++) {
                    for (int dj = -2; dj <= 2; dj++) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                            sum += noise[ni][nj];
                            count++;
                        }
                    }
                }
                grid[i][j] = (int)(255 * (sum / count));
            }
        }
    }

    public void generateQuadrantsGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i < rows/2) {
                    grid[i][j] = j < cols/2 ? 255 : 128;
                } else {
                    grid[i][j] = j < cols/2 ? 64 : 32;
                }
            }
        }
    }

    public void generateSparsePeaksGrid() {
        // Initialize with low values
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grid[i][j] = 10;
            }
        }
        
        // Add random high-value peaks
        Random rand = new Random(42);
        int numPeaks = rows * cols / 100;
        for (int k = 0; k < numPeaks; k++) {
            int i = rand.nextInt(rows);
            int j = rand.nextInt(cols);
            grid[i][j] = 255;
        }
    }

    public void generateFractalGrid() {
        Random rand = new Random(42);
        subdivide(0, 0, rows, cols, 255, 0.5, rand);
    }

    private void subdivide(int x1, int y1, int x2, int y2, int value, double roughness, Random rand) {
        if (x2 - x1 < 2 || y2 - y1 < 2) return;
        
        int midX = (x1 + x2) / 2;
        int midY = (y1 + y2) / 2;
        int offset = (int)(value * roughness * (rand.nextDouble() - 0.5));
        
        grid[midX][midY] = Math.max(0, Math.min(255, value + offset));
        
        subdivide(x1, y1, midX, midY, grid[midX][midY], roughness * 0.7, rand);
        subdivide(midX, y1, x2, midY, grid[midX][midY], roughness * 0.7, rand);
        subdivide(x1, midY, midX, y2, grid[midX][midY], roughness * 0.7, rand);
        subdivide(midX, midY, x2, y2, grid[midX][midY], roughness * 0.7, rand);
    }

    public Map<String, Double> analyzeClusterMetrics() {
        Map<String, Double> metrics = new HashMap<>();
        Map<Integer, Long> weights = calculateClusterWeights();
        
        // Calculate weight metrics
        double avgWeight = getAverageWeight(weights);
        double maxDeviation = weights.values().stream()
            .mapToDouble(w -> Math.abs(w - avgWeight) / avgWeight)
            .max()
            .orElse(0.0);
        
        // Calculate connectivity metric (average number of different neighbor clusters)
        double avgConnectivity = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                avgConnectivity += getNeighborClusters(i, j).size();
            }
        }
        avgConnectivity /= (rows * cols);
        
        metrics.put("maxWeightDeviation", maxDeviation);
        metrics.put("avgConnectivity", avgConnectivity);
        return metrics;
    }

    public Map<Integer, Double> analyzeClusterDistribution() {
        Map<Integer, Long> weights = calculateClusterWeights();
        long totalWeight = weights.values().stream().mapToLong(Long::longValue).sum();
        
        Map<Integer, Double> distribution = new HashMap<>();
        for (Map.Entry<Integer, Long> entry : weights.entrySet()) {
            distribution.put(entry.getKey(), entry.getValue() / (double)totalWeight);
        }
        return distribution;
    }

    public Map<Integer, Integer> getClusterSizes() {
        Map<Integer, Integer> sizes = new HashMap<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sizes.merge(clusters[i][j], 1, Integer::sum);
            }
        }
        return sizes;
    }

    public void setAgentMask(int[][] agentMask) {
        if (agentMask.length != rows || agentMask[0].length != cols) {
            throw new IllegalArgumentException("Agent mask dimensions must match grid dimensions");
        }
        this.computationMask = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.computationMask[i][j] = agentMask[i][j];
            }
        }
    }

    public void generateRandomAgentMask(double agentRatio) {
        Random rand = new Random();
        this.computationMask = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.computationMask[i][j] = rand.nextDouble() < agentRatio ? 1 : 0;
            }
        }
    }

    public void partitionAgents(int numClusters) {
        if (computationMask == null) {
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
                if (computationMask[i][j] == 1) {
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
        
        // Create seeds for clusters distributed across the grid
        List<int[]> seeds = new ArrayList<>();
        Random rand = new Random(42);
        Collections.shuffle(agentCells, rand);
        
        // Select seeds from different regions of the grid
        int seedSpacing = (int)Math.sqrt(numClusters);
        for (int i = 0; i < seedSpacing && seeds.size() < numClusters; i++) {
            for (int j = 0; j < seedSpacing && seeds.size() < numClusters; j++) {
                int targetRow = i * rows / seedSpacing + rows / (seedSpacing * 2);
                int targetCol = j * cols / seedSpacing + cols / (seedSpacing * 2);
                
                // Find closest agent to this target position
                int[] bestAgent = null;
                double minDist = Double.MAX_VALUE;
                for (int[] agent : agentCells) {
                    if (clusters[agent[0]][agent[1]] != -1) continue; // Already assigned
                    double dist = Math.sqrt(Math.pow(agent[0] - targetRow, 2) + 
                                          Math.pow(agent[1] - targetCol, 2));
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
        }
        
        // Assign remaining agents to maintain even distribution
        Collections.shuffle(agentCells, rand);
        Map<Integer, Integer> clusterCounts = new HashMap<>();
        for (int i = 0; i < numClusters; i++) {
            clusterCounts.put(i, 1); // Seeds already assigned
        }
        
        for (int[] agent : agentCells) {
            int i = agent[0], j = agent[1];
            if (clusters[i][j] != -1) continue; // Already assigned
            
            // Find cluster with fewest agents that's spatially close
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
                
                // Score combines spatial distance and load balancing
                double loadFactor = (double)currentCount / targetCount;
                double score = minDistToCluster * (1.0 + loadFactor);
                
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
        
        // Refinement phase to improve spatial coherence while maintaining balance
        for (int iteration = 0; iteration < 50; iteration++) {
            boolean improved = false;
            Collections.shuffle(agentCells, rand);
            
            for (int[] agent : agentCells) {
                int i = agent[0], j = agent[1];
                int currentCluster = clusters[i][j];
                
                // Find neighbor clusters
                Set<Integer> neighborClusters = getNeighborAgentClusters(i, j);
                
                for (int newCluster : neighborClusters) {
                    if (newCluster == currentCluster) continue;
                    
                    int currentCount = clusterCounts.get(currentCluster);
                    int targetCount = clusterCounts.get(newCluster);
                    
                    // Only move if it improves balance or maintains it while improving spatial coherence
                    if (currentCount > targetCount + 1) {
                        double spatialImprovement = calculateAgentSpatialCoherence(i, j, newCluster) - 
                                                   calculateAgentSpatialCoherence(i, j, currentCluster);
                        
                        if (spatialImprovement > 0.1) {
                            clusters[i][j] = newCluster;
                            clusterCounts.put(currentCluster, currentCount - 1);
                            clusterCounts.put(newCluster, targetCount + 1);
                            improved = true;
                            break;
                        }
                    }
                }
            }
            
            if (!improved) break;
        }
        
        // Print final distribution
        System.out.println("\nFinal agent distribution:");
        for (int i = 0; i < numClusters; i++) {
            System.out.println("Cluster " + i + ": " + clusterCounts.get(i) + " agents");
        }
    }

    private Set<Integer> getNeighborAgentClusters(int i, int j) {
        Set<Integer> neighbors = new HashSet<>();
        int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] dir : dirs) {
            int ni = i + dir[0];
            int nj = j + dir[1];
            if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && 
                computationMask[ni][nj] == 1 && clusters[ni][nj] >= 0) {
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
                    computationMask[ni][nj] == 1 && clusters[ni][nj] >= 0) {
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
        BufferedImage image = new BufferedImage(pngWidth * 3, pngHeight, BufferedImage.TYPE_INT_RGB);
        double scaleX = pngWidth / (double)cols;
        double scaleY = pngHeight / (double)rows;
        
        // Generate distinct colors for clusters
        Color[] clusterColors = new Color[numClusters];
        Random rand = new Random(42);
        for (int i = 0; i < numClusters; i++) {
            clusterColors[i] = new Color(rand.nextFloat(), rand.nextFloat(), rand.nextFloat());
        }

        for (int y = 0; y < pngHeight; y++) {
            for (int x = 0; x < pngWidth * 3; x++) {
                int i = (int)(y / scaleY);
                int j = (int)((x % pngWidth) / scaleX);
                
                if (x < pngWidth) {
                    // Left: grid values (if available)
                    if (grid != null && grid[i][j] != null) {
                        float intensity = grid[i][j] / 255.0f;
                        Color color = new Color(intensity, intensity, intensity);
                        image.setRGB(x, y, color.getRGB());
                    } else {
                        image.setRGB(x, y, Color.BLACK.getRGB());
                    }
                } else if (x < pngWidth * 2) {
                    // Middle: agent mask (white = agent, black = no agent)
                    Color color = computationMask[i][j] == 1 ? Color.WHITE : Color.BLACK;
                    image.setRGB(x, y, color.getRGB());
                } else {
                    // Right: cluster assignments (only for agent cells)
                    if (computationMask[i][j] == 1 && clusters[i][j] >= 0) {
                        image.setRGB(x, y, clusterColors[clusters[i][j]].getRGB());
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
                if (computationMask[i][j] == 1 && clusters[i][j] >= 0) {
                    counts.merge(clusters[i][j], 1, Integer::sum);
                }
            }
        }
        return counts;
    }

    private static void runAgentPartitioningTest(GridConfig config) throws IOException {
        new File("agents").mkdirs();
        
        // Create grid with random agent distribution
        GRID grid = new GRID(config.rows, config.cols);
        grid.generateRandomGrid(config.randomMaxValue);
        grid.generateRandomAgentMask(0.3); // 30% of cells have agents
        
        System.out.println("=== Agent Partitioning Test ===");
        long startTime = System.currentTimeMillis();
        grid.partitionAgents(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        grid.saveAgentVisualization("agents/partition.png", config.pngWidth, config.pngHeight);
        
        System.out.println("\nProcessing time: " + (endTime - startTime) + "ms");
        
        Map<Integer, Integer> agentCounts = grid.getAgentCounts();
        System.out.println("\nAgent distribution verification:");
        agentCounts.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                System.out.println("Cluster " + entry.getKey() + ": " + entry.getValue() + " agents");
            });
    }

    private static void runCustomAgentTest(GridConfig config) throws IOException {
        new File("custom_agents").mkdirs();
        
        GRID grid = new GRID(config.rows, config.cols);
        grid.generateRandomGrid(config.randomMaxValue);
        
        // Create custom agent pattern - clusters of agents
        int[][] agentMask = new int[config.rows][config.cols];
        Random rand = new Random(42);
        
        // Create 5 clusters of agents
        for (int cluster = 0; cluster < 5; cluster++) {
            int centerX = rand.nextInt(config.rows);
            int centerY = rand.nextInt(config.cols);
            int radius = 15 + rand.nextInt(10);
            
            for (int i = 0; i < config.rows; i++) {
                for (int j = 0; j < config.cols; j++) {
                    double dist = Math.sqrt(Math.pow(i - centerX, 2) + Math.pow(j - centerY, 2));
                    if (dist <= radius && rand.nextDouble() < 0.7) {
                        agentMask[i][j] = 1;
                    }
                }
            }
        }
        
        grid.setAgentMask(agentMask);
        
        System.out.println("=== Custom Agent Pattern Test ===");
        long startTime = System.currentTimeMillis();
        grid.partitionAgents(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        grid.saveAgentVisualization("custom_agents/partition.png", config.pngWidth, config.pngHeight);
        
        System.out.println("\nProcessing time: " + (endTime - startTime) + "ms");
        
        Map<Integer, Integer> agentCounts = grid.getAgentCounts();
        System.out.println("\nAgent distribution verification:");
        agentCounts.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                System.out.println("Cluster " + entry.getKey() + ": " + entry.getValue() + " agents");
            });
    }

    public static void main(String[] args) {
        GridConfig config = new GridConfig();
        config.rows = 100;
        config.cols = 100;
        config.numClusters = 4;
        config.maxIterations = 200;
        config.convergenceThreshold = 100;
        config.randomMaxValue = 32;
        config.blockSize = 40;
        config.stripeWidth = 25;
        config.numHotspots = 8;
        config.numSteps = 8;
        config.powerLawAlpha = 1.5;
        config.pngWidth = 800;
        config.pngHeight = 800;

        try {
            // Test agent partitioning
            runAgentPartitioningTest(config);
            runCustomAgentTest(config);
            
            // Original tests (commented out)
            //runAdaptationTestWithFigureEight(config, 100);
            //runAdaptationTestWithSpiral(config, 100);
            //runAdaptationTestWithChaos(config, 100);
            //runAdaptationTestWithWaves(config, 100);
            //runAdaptationTestWithCellularAutomata(config, 100);
            //runAdaptationTestWithDiffusion(config, 100);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}