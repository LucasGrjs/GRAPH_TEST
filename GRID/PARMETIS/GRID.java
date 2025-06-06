import java.io.*;
import java.awt.image.BufferedImage;
import java.awt.Color;
import javax.imageio.ImageIO;
import java.util.*;
import java.util.stream.Collectors;

public class GRID {
    private Integer[][] grid; 
    private Integer[][] clusters; // Stores cluster assignments
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

    public static class GridConfig {
        public int rows = 100;
        public int cols = 100;
        public int numClusters = 8;
        public int maxIterations = 100;
        public double convergenceThreshold = 50;
        public int randomMaxValue = 1024;
        public int blockSize = 50;
        public int stripeWidth = 30;
        public int numHotspots = 5;
        public int numSteps = 5;
        public double powerLawAlpha = 2.0;
        public int pngWidth = 800;
        public int pngHeight = 800;
    }

    private static void runTestCase(GRID grid, String testName, GridConfig config) {
        try {
            grid.saveGridAsPNG(testName + "_input.png", config.pngWidth, config.pngHeight);
            long startTime = System.currentTimeMillis();
            grid.partitionGrid(config.numClusters);
            long endTime = System.currentTimeMillis();
            grid.saveClustersPNG(testName + "_clusters.png", config.pngWidth, config.pngHeight);

            // Analyze results
            Map<String, Double> metrics = grid.analyzeClusterMetrics();
            Map<Integer, Integer> clusterSizes = grid.getClusterSizes();
            Map<Integer, Long> clusterWeights = grid.calculateClusterWeights();
            
            printAnalysis(startTime, endTime, metrics, clusterSizes, clusterWeights);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void testCases(GridConfig config) {
        // Create parent directories
        new File("test_cases").mkdirs();
        
        GRID grid = new GRID(config.rows, config.cols);
        
        Map<String, Runnable> testCases = new LinkedHashMap<>();
        testCases.put("random", () -> grid.generateRandomGrid(config.randomMaxValue));
        testCases.put("gradient", () -> grid.generateGradientGrid());
        testCases.put("block", () -> grid.generateBlockGrid());
        testCases.put("circle", () -> grid.generateCircleGrid());
        testCases.put("spiral", () -> grid.generateSpiralGrid());
        testCases.put("hotspot", () -> grid.generateHotspotGrid());
        testCases.put("checkerboard", () -> grid.generateCheckerboardGrid());
        testCases.put("diagonal", () -> grid.generateDiagonalStripesGrid());
        testCases.put("gaussian", () -> grid.generateGaussianGrid());
        testCases.put("wave", () -> grid.generateWaveGrid());
        testCases.put("powerlaw", () -> grid.generatePowerLawGrid());
        testCases.put("cornerheavy", () -> grid.generateCornerHeavyGrid());
        testCases.put("exponential", () -> grid.generateExponentialGrid());
        testCases.put("bimodal", () -> grid.generateBimodalGrid());
        testCases.put("step", () -> grid.generateStepGrid());
        testCases.put("concentrics", () -> grid.generateConcentricsGrid());
        testCases.put("mountains", () -> grid.generateMountainsGrid());
        testCases.put("quadrants", () -> grid.generateQuadrantsGrid());
        testCases.put("sparsepeaks", () -> grid.generateSparsePeaksGrid());
        testCases.put("fractal", () -> grid.generateFractalGrid());

        for (Map.Entry<String, Runnable> testCase : testCases.entrySet()) {
            String testDir = "test_cases/" + testCase.getKey();
            System.out.println("\nTest Case: " + testCase.getKey().toUpperCase());
            testCase.getValue().run();
            try {
                grid.saveGridAsPNG(testDir + "input.png", config.pngWidth, config.pngHeight);
                grid.partitionGrid(config.numClusters);
                grid.saveClustersPNG(testDir + "clusters.png", config.pngWidth, config.pngHeight);

                // Print analysis results
                Map<String, Double> metrics = grid.analyzeClusterMetrics();
                Map<Integer, Integer> clusterSizes = grid.getClusterSizes();
                Map<Integer, Long> clusterWeights = grid.calculateClusterWeights();
                
                System.out.println("\nCluster Analysis:");
                System.out.println("----------------------------------------");
                long totalWeight = clusterWeights.values().stream().mapToLong(Long::longValue).sum();
                clusterSizes.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(entry -> {
                        int clusterId = entry.getKey();
                        long weight = clusterWeights.get(clusterId);
                        double weightPercent = (weight * 100.0) / totalWeight;
                        System.out.printf("Cluster %d: Size=%d, Weight=%d (%.2f%%)\n", 
                            clusterId, entry.getValue(), weight, weightPercent);
                    });
                System.out.println("----------------------------------------");
                System.out.printf("Max Weight Deviation: %.2f%%\n", metrics.get("maxWeightDeviation") * 100);
                System.out.printf("Average Connectivity: %.2f\n", metrics.get("avgConnectivity"));
                
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
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

    private int[][] convertToIntArray(Integer[][] array) {
        int[][] result = new int[array.length][array[0].length];
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[0].length; j++) {
                result[i][j] = array[i][j];
            }
        }
        return result;
    }

    public int[][] getGrid() {
        return convertToIntArray(grid);
    }

    private int getCellId(int row, int col) {
        return row * cols + col;
    }

    public void printClusterMigrations(int[][] originalPartition) {
        Map<Integer, Map<Integer, List<Integer>>> migrations = new HashMap<>();
        
        // Initialize the migration tracking map
        for (int i = 0; i < numClusters; i++) {
            migrations.put(i, new HashMap<>());
        }
        
        // Track all cell movements
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int sourceCluster = originalPartition[i][j];
                int targetCluster = clusters[i][j];
                
                if (sourceCluster != targetCluster) {
                    migrations.get(sourceCluster)
                        .computeIfAbsent(targetCluster, k -> new ArrayList<>())
                        .add(getCellId(i, j));
                }
            }
        }
        
        // Print migrations for each source cluster
        for (int sourceCluster = 0; sourceCluster < numClusters; sourceCluster++) {
            Map<Integer, List<Integer>> clusterMigrations = migrations.get(sourceCluster);
            
            if (!clusterMigrations.isEmpty()) {
                System.out.println("Moved from Cluster " + sourceCluster + ":");
                clusterMigrations.forEach((targetCluster, cells) -> {
                    System.out.print("<" + targetCluster + " :: [");
                    System.out.print(cells.stream()
                        .map(String::valueOf)
                        .collect(Collectors.joining(", ")));
                    System.out.println("]>");
                });
                System.out.println();
            }
        }
    }

    public void adaptPartition(int[][] sourcePartition, int[][] sourceGrid) {
        // Store original partition for comparison
        int[][] originalPartition = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                originalPartition[i][j] = sourcePartition[i][j];
            }
        }
        
        this.numClusters = Arrays.stream(sourcePartition)
            .flatMapToInt(Arrays::stream)
            .max()
            .orElse(0) + 1;
        
        this.clusters = new Integer[rows][cols];
        
        // Step 1: Create mapping matrix to track partition membership
        double[][][] partitionMembership = new double[rows][cols][numClusters];
        
        // Calculate membership weights based on source partition overlap
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Calculate corresponding region in source grid
                int startI = (int)(i * sourceGrid.length / (double)rows);
                int endI = (int)((i + 1) * sourceGrid.length / (double)rows);
                int startJ = (int)(j * sourceGrid[0].length / (double)cols);
                int endJ = (int)((j + 1) * sourceGrid[0].length / (double)cols);
                
                // Count partition memberships in overlapping region
                Map<Integer, Double> membershipCounts = new HashMap<>();
                double totalWeight = 0;
                
                for (int si = startI; si < endI; si++) {
                    for (int sj = startJ; sj < endJ; sj++) {
                        int partition = sourcePartition[si][sj];
                        double weight = sourceGrid[si][sj];
                        membershipCounts.merge(partition, weight, Double::sum);
                        totalWeight += weight;
                    }
                }
                
                // Convert counts to normalized weights
                if (totalWeight > 0) {
                    for (Map.Entry<Integer, Double> entry : membershipCounts.entrySet()) {
                        partitionMembership[i][j][entry.getKey()] = entry.getValue() / totalWeight;
                    }
                }
            }
        }
        
        // Step 2: Initial assignment based on strongest membership
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int bestCluster = 0;
                double bestWeight = partitionMembership[i][j][0];
                
                for (int k = 1; k < numClusters; k++) {
                    if (partitionMembership[i][j][k] > bestWeight) {
                        bestWeight = partitionMembership[i][j][k];
                        bestCluster = k;
                    }
                }
                clusters[i][j] = bestCluster;
            }
        }

        // Step 3: Iterative refinement considering both balance and membership
        Map<Integer, Long> weights = calculateClusterWeights();
        long targetWeight = weights.values().stream().mapToLong(Long::longValue).sum() / numClusters;
        double membershipWeight = 0.001; // Weight factor for membership vs balance
        
        // Create list of all cell positions
        List<int[]> cellPositions = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cellPositions.add(new int[]{i, j});
            }
        }
        Random rand = new Random();
        
        boolean improved;
        do {
            improved = false;
            // Shuffle cell positions for random processing order
            Collections.shuffle(cellPositions, rand);
            
            for (int[] pos : cellPositions) {
                int i = pos[0], j = pos[1];
                int currentCluster = clusters[i][j];
                Set<Integer> neighbors = getNeighborClusters(i, j);
                
                for (int newCluster : neighbors) {
                    if (newCluster == currentCluster) continue;
                    
                    // Calculate balance improvement
                    long currentWeight = weights.get(currentCluster);
                    long neighborWeight = weights.get(newCluster);
                    long cellWeight = grid[i][j];
                    
                    double currentImbalance = Math.abs(currentWeight - targetWeight) + 
                                            Math.abs(neighborWeight - targetWeight);
                    double newImbalance = Math.abs(currentWeight - cellWeight - targetWeight) + 
                                        Math.abs(neighborWeight + cellWeight - targetWeight);
                    double balanceGain = currentImbalance - newImbalance;
                    
                    // Calculate membership change cost
                    double membershipCost = partitionMembership[i][j][currentCluster] - 
                                          partitionMembership[i][j][newCluster];
                    
                    // Combined metric
                    double improvement = balanceGain - membershipWeight * membershipCost;
                    
                    if (improvement > 0) {
                        clusters[i][j] = newCluster;
                        weights.put(currentCluster, currentWeight - cellWeight);
                        weights.put(newCluster, neighborWeight + cellWeight);
                        improved = true;
                        break;
                    }
                }
            }
        } while (improved);

        // Print migration information at the end
        System.out.println("\nCell Migration Analysis:");
        printClusterMigrations(originalPartition);
    }

    public double calculatePartitionChange(int[][] sourcePartition) {
        int changes = 0;
        int total = 0;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Map current position to source partition position
                int sourceI = (int)(i * sourcePartition.length / (double)rows);
                int sourceJ = (int)(j * sourcePartition[0].length / (double)cols);
                
                if (clusters[i][j] != sourcePartition[sourceI][sourceJ]) {
                    changes++;
                }
                total++;
            }
        }
        
        return (double)changes / total;
    }

    public double calculatePartitionSimilarity(int[][] otherPartition) {
        if (rows != otherPartition.length || cols != otherPartition[0].length) {
            throw new IllegalArgumentException("Partitions must have the same dimensions");
        }
        
        int matches = 0;
        int total = rows * cols;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (clusters[i][j] == otherPartition[i][j]) {
                    matches++;
                }
            }
        }
        
        return (double) matches / total;
    }
    
    private static void runAdaptationTestWithPoints(GridConfig config, int run) throws IOException {
        new File("grid").mkdirs();
        new File("clusters").mkdirs();

        // Initialize interest points with positions and velocities
        List<double[]> points = new ArrayList<>();
        Random rand = new Random();
        int numPoints = 5;  // Number of interest points
        
        // Create initial points with position and velocity
        for (int i = 0; i < numPoints; i++) {
            points.add(new double[]{
                rand.nextDouble() * config.rows,  // x position
                rand.nextDouble() * config.cols,  // y position
                rand.nextDouble() * 2 - 1,        // x velocity
                rand.nextDouble() * 2 - 1         // y velocity
            });
        }

        // Create and analyze original grid
        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = new GRID(config.rows, config.cols);
        generateGridFromPoints(originalGrid, points, config.randomMaxValue);
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        originalGrid.saveGridAsPNG("grid/grid_0_input.png", config.pngWidth, config.pngHeight);
        originalGrid.saveClustersPNG("clusters/grid_0_clusters.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;
        double[] clusterChanges = new double[config.numClusters];
        int[] clusterChangeCount = new int[config.numClusters];
        double totalChanges = 0;

        // Perform transitions with moving points
        for (int i = 1; i <= run; i++) {
            // Update point positions
            for (double[] point : points) {
                // Update position
                point[0] = Math.floorMod((int)(point[0] + point[2] + config.rows), config.rows);
                point[1] = Math.floorMod((int)(point[1] + point[3] + config.cols), config.cols);
                
                // Randomly adjust velocity
                if (rand.nextDouble() < 0.1) {  // 10% chance to change direction
                    point[2] += (rand.nextDouble() * 0.4 - 0.2);  // Adjust velocity by ±0.2
                    point[3] += (rand.nextDouble() * 0.4 - 0.2);
                    // Clamp velocities
                    point[2] = Math.max(-1, Math.min(1, point[2]));
                    point[3] = Math.max(-1, Math.min(1, point[3]));
                }
            }

            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = new GRID(config.rows, config.cols);
            generateGridFromPoints(transitionGrid, points, config.randomMaxValue);
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            
            transitionGrid.saveGridAsPNG("grid/grid_" + i + "_input.png", config.pngWidth, config.pngHeight);
            transitionGrid.saveClustersPNG("clusters/grid_" + i + "_clusters.png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            // Track changes
            double similarity = transitionGrid.calculatePartitionChange(previousGrid.getPartition());
            int totalCellsMoved = (int)(similarity * config.rows * config.cols);
            System.out.println("\nPartition Change Analysis (vs previous):");
            System.out.println("Percentage of cells that changed partition: " + 
                String.format("%.2f%%", similarity * 100));
            System.out.println("Total number of cells moved: " + totalCellsMoved);

            totalChanges += similarity;
            previousGrid = transitionGrid;
        }

        System.out.println("\nAverage Partition Changes Analysis:");
        System.out.println("Average percentage of cells that changed partition per transition: " + 
            String.format("%.2f%%", (totalChanges / run) * 100));

        System.out.println("\nPer-Cluster Movement Analysis:");
        System.out.println("Cluster\tAvg Cells Moved/Transition");
        System.out.println("----------------------------------------");
        for (int i = 0; i < config.numClusters; i++) {
            double avgMoves = clusterChangeCount[i] > 0 ? 
                (clusterChanges[i] / run) : 0;
            System.out.printf("%d\t%.0f%n", i, avgMoves);
        }
        System.out.println("----------------------------------------");
    }

    private static void generateGridFromPoints(GRID grid, List<double[]> points, int maxValue) {
        // Clear grid
        for (int i = 0; i < grid.rows; i++) {
            for (int j = 0; j < grid.cols; j++) {
                grid.grid[i][j] = 0;
            }
        }

        // Generate values based on distance to interest points
        for (int i = 0; i < grid.rows; i++) {
            for (int j = 0; j < grid.cols; j++) {
                double maxInfluence = 0;
                for (double[] point : points) {
                    double distance = Math.sqrt(
                        Math.pow(i - point[0], 2) + 
                        Math.pow(j - point[1], 2)
                    );
                    double influence = maxValue * Math.exp(-distance / 20.0);
                    maxInfluence = Math.max(maxInfluence, influence);
                }
                grid.grid[i][j] = (int)maxInfluence;
            }
        }
    }

    private static void printAnalysis(long startTime, long endTime, 
                                    Map<String, Double> metrics,
                                    Map<Integer, Integer> clusterSizes,
                                    Map<Integer, Long> clusterWeights) {
        System.out.println("\nProcessing time: " + (endTime - startTime) + "ms");
        
        System.out.println("\nCluster Analysis:");
        System.out.println("Cluster\tSize\tWeight\tWeight%");
        System.out.println("----------------------------------------");
        
        long totalWeight = clusterWeights.values().stream().mapToLong(Long::longValue).sum();
        
        clusterSizes.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(entry -> {
                int clusterId = entry.getKey();
                long weight = clusterWeights.get(clusterId);
                double weightPercent = (weight * 100.0) / totalWeight;
                System.out.printf("%d\t%d\t%d\t%.2f%%%n", 
                    clusterId, 
                    entry.getValue(),
                    weight,
                    weightPercent
                );
            });
        System.out.println("----------------------------------------");
    }

    private static void runAdaptationTest(GridConfig config, int run) throws IOException {
        // Create directories
        new File("grid").mkdirs();
        new File("clusters").mkdirs();

        // Create and analyze original grid
        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = createAndPartitionGrid(config);
        long endTime = System.currentTimeMillis();
        
        // Save original grid results
        originalGrid.saveGridAsPNG("grid/grid_0_input.png", config.pngWidth, config.pngHeight);
        originalGrid.saveClustersPNG("clusters/grid_0_clusters.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        // Previous grid starts as original
        GRID previousGrid = originalGrid;

        // Initialize cluster change tracking
        double[] clusterChanges = new double[config.numClusters];
        int[] clusterChangeCount = new int[config.numClusters];
        double totalChanges = 0;

        // Perform transitions
        for (int i = 1; i <= run; i++) {
            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = createAndAdaptGrid(config, previousGrid);
            endTime = System.currentTimeMillis();
            
            // Save transition grid results
            transitionGrid.saveGridAsPNG("grid/grid_" + i + "_input.png", config.pngWidth, config.pngHeight);
            transitionGrid.saveClustersPNG("grid/grid_" + i + "_clusters.png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            // Calculate changes per cluster
            for (int r = 0; r < config.rows; r++) {
                for (int c = 0; c < config.cols; c++) {
                    int oldCluster = previousGrid.clusters[r][c];
                    if (oldCluster != transitionGrid.clusters[r][c]) {
                        clusterChanges[oldCluster]++;
                        clusterChangeCount[oldCluster]++;
                    }
                }
            }

            // Track overall changes
            double similarity = transitionGrid.calculatePartitionChange(previousGrid.getPartition());
            int totalCellsMoved = (int)(similarity * config.rows * config.cols);
            System.out.println("\nPartition Change Analysis (vs previous):");
            System.out.println("Percentage of cells that changed partition: " + 
                String.format("%.2f%%", similarity * 100));
            System.out.println("Total number of cells moved: " + totalCellsMoved);

            totalChanges += similarity;
            previousGrid = transitionGrid;
        }

        // Print average changes across all transitions
        System.out.println("\nAverage Partition Changes Analysis:");
        System.out.println("Average percentage of cells that changed partition per transition: " + 
            String.format("%.2f%%", (totalChanges / run) * 100));

        // At end of all transitions
        System.out.println("\nPer-Cluster Movement Analysis:");
        System.out.println("Cluster\tAvg Cells Moved/Transition");
        System.out.println("----------------------------------------");
        for (int i = 0; i < config.numClusters; i++) {
            double avgMoves = clusterChangeCount[i] > 0 ? 
                (clusterChanges[i] / run) : 0;
            System.out.printf("%d\t%.0f%n", i, avgMoves);
        }
        System.out.println("----------------------------------------");
    }

    private static GRID createAndPartitionGrid(GridConfig config) {
        GRID grid = new GRID(config.rows, config.cols);
        grid.generateRandomGrid(config.randomMaxValue);
        grid.partitionGrid(config.numClusters);
        return grid;
    }

    private static GRID createAndAdaptGrid(GridConfig config, GRID originalGrid) {
        GRID grid = new GRID(config.rows, config.cols);
        grid.generateRandomGrid(config.randomMaxValue);
        grid.adaptPartition(originalGrid.getPartition(), originalGrid.getGrid());
        return grid;
    }

    private static void printGridAnalysis(GRID grid, long startTime, long endTime) {
        Map<String, Double> metrics = grid.analyzeClusterMetrics();
        Map<Integer, Integer> sizes = grid.getClusterSizes();
        Map<Integer, Long> weights = grid.calculateClusterWeights();
        printAnalysis(startTime, endTime, metrics, sizes, weights);
    }

    private static void generateGridFromInterestPoints(GRID grid, int numPoints, int maxValue) {
        Random rand = new Random();
        // Generate random interest points
        List<int[]> points = new ArrayList<>();
        for (int i = 0; i < numPoints; i++) {
            points.add(new int[]{
                rand.nextInt(grid.rows),
                rand.nextInt(grid.cols)
            });
        }

        // Clear grid
        for (int i = 0; i < grid.rows; i++) {
            for (int j = 0; j < grid.cols; j++) {
                grid.grid[i][j] = 0;
            }
        }

        // Generate values based on distance to interest points
        for (int i = 0; i < grid.rows; i++) {
            for (int j = 0; j < grid.cols; j++) {
                double maxInfluence = 0;
                for (int[] point : points) {
                    double distance = Math.sqrt(
                        Math.pow(i - point[0], 2) + 
                        Math.pow(j - point[1], 2)
                    );
                    double influence = maxValue * Math.exp(-distance / 20.0);
                    maxInfluence = Math.max(maxInfluence, influence);
                }
                grid.grid[i][j] = (int)maxInfluence;
            }
        }
    }

    private static void runAdaptationTestWithInterestPoints(GridConfig config, int run) throws IOException {
        new File("grid").mkdirs();
        new File("clusters").mkdirs();

        // Initialize interest points with positions and velocities
        List<double[]> points = new ArrayList<>();
        Random rand = new Random();
        int numPoints = 5;  // Number of interest points
        
        // Create initial points with position and velocity
        for (int i = 0; i < numPoints; i++) {
            points.add(new double[]{
                rand.nextDouble() * config.rows,   // x position
                rand.nextDouble() * config.cols,   // y position
                rand.nextDouble() * 2 - 1,         // x velocity
                rand.nextDouble() * 2 - 1          // y velocity
            });
        }

        // Create and analyze original grid
        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = new GRID(config.rows, config.cols);
        generateGridFromPoints(originalGrid, points, config.randomMaxValue);
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        originalGrid.saveGridAsPNG("grid/grid_0_input.png", config.pngWidth, config.pngHeight);
        originalGrid.saveClustersPNG("clusters/grid_0_clusters.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;
        double[] clusterChanges = new double[config.numClusters];
        int[] clusterChangeCount = new int[config.numClusters];
        double totalChanges = 0;

        // Perform transitions with moving points
        for (int i = 1; i <= run; i++) {
            // Update point positions
            for (double[] point : points) {
                // Update position with wrap-around
                point[0] = (point[0] + point[2] + config.rows) % config.rows;
                point[1] = (point[1] + point[3] + config.cols) % config.cols;
                
                // Randomly adjust velocity occasionally (10% chance per point per iteration)
                if (rand.nextDouble() < 0.1) {
                    point[2] += (rand.nextDouble() * 0.2 - 0.1); // Small velocity adjustment
                    point[3] += (rand.nextDouble() * 0.2 - 0.1);
                    // Keep velocity within bounds
                    point[2] = Math.max(-1.5, Math.min(1.5, point[2]));
                    point[3] = Math.max(-1.5, Math.min(1.5, point[3]));
                }
                
                // Add slight attraction to center to keep points from staying at edges
                double centerForceX = (config.rows/2 - point[0]) * 0.001;
                double centerForceY = (config.cols/2 - point[1]) * 0.001;
                point[2] += centerForceX;
                point[3] += centerForceY;
            }

            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = new GRID(config.rows, config.cols);
            generateGridFromPoints(transitionGrid, points, config.randomMaxValue);
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            
            transitionGrid.saveGridAsPNG("grid/grid_" + i + "_input.png", config.pngWidth, config.pngHeight);
            transitionGrid.saveClustersPNG("clusters/grid_" + i + "_clusters.png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            // Track changes
            double similarity = transitionGrid.calculatePartitionChange(previousGrid.getPartition());
            int totalCellsMoved = (int)(similarity * config.rows * config.cols);
            System.out.println("\nPartition Change Analysis (vs previous):");
            System.out.println("Percentage of cells that changed partition: " + 
                String.format("%.2f%%", similarity * 100));
            System.out.println("Total number of cells moved: " + totalCellsMoved);

            totalChanges += similarity;
            previousGrid = transitionGrid;
        }

        System.out.println("\nAverage Partition Changes Analysis:");
        System.out.println("Average percentage of cells that changed partition per transition: " + 
            String.format("%.2f%%", (totalChanges / run) * 100));

        System.out.println("\nPer-Cluster Movement Analysis:");
        System.out.println("Cluster\tAvg Cells Moved/Transition");
        System.out.println("----------------------------------------");
        for (int i = 0; i < config.numClusters; i++) {
            double avgMoves = clusterChangeCount[i] > 0 ? 
                (clusterChanges[i] / run) : 0;
            System.out.printf("%d\t%.0f%n", i, avgMoves);
        }
        System.out.println("----------------------------------------");
    }

    private static void runAdaptationTestWithCirclingPoints(GridConfig config, int run) throws IOException {
        new File("grid").mkdirs();
        new File("clusters").mkdirs();

        // Initialize points in a circle formation
        List<double[]> points = new ArrayList<>();
        int numPoints = 4;  // Using 4 points for a more stable pattern
        double radius = Math.min(config.rows, config.cols) / 4.0;
        double centerX = config.rows / 2.0;
        double centerY = config.cols / 2.0;
        
        // Create points with circular motion
        for (int i = 0; i < numPoints; i++) {
            double angle = (2 * Math.PI * i) / numPoints;
            points.add(new double[]{
                centerX + radius * Math.cos(angle),  // x position
                centerY + radius * Math.sin(angle),  // y position
                -Math.sin(angle) * 0.5,              // x velocity (tangential)
                Math.cos(angle) * 0.5                // y velocity (tangential)
            });
        }

        // Create and analyze original grid
        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = new GRID(config.rows, config.cols);
        generateGridFromPoints(originalGrid, points, config.randomMaxValue);
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        originalGrid.saveGridAsPNG("grid/grid_0_input.png", config.pngWidth, config.pngHeight);
        originalGrid.saveClustersPNG("grid/grid_0_clusters.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;
        double[] clusterChanges = new double[config.numClusters];
        int[] clusterChangeCount = new int[config.numClusters];
        double totalChanges = 0;

        // Perform transitions with rotating points
        for (int i = 1; i <= run; i++) {
            // Update point positions to maintain circular motion
            for (double[] point : points) {
                // Calculate current angle from center
                double dx = point[0] - centerX;
                double dy = point[1] - centerY;
                double currentAngle = Math.atan2(dy, dx);
                
                // Update angle (rotate points)
                double rotationSpeed = 0.05;  // Controls rotation speed
                double newAngle = currentAngle + rotationSpeed;
                
                // Update position maintaining radius
                point[0] = centerX + radius * Math.cos(newAngle);
                point[1] = centerY + radius * Math.sin(newAngle);
                
                // Update velocities to maintain tangential motion
                point[2] = -Math.sin(newAngle) * 0.5;
                point[3] = Math.cos(newAngle) * 0.5;
            }

            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = new GRID(config.rows, config.cols);
            generateGridFromPoints(transitionGrid, points, config.randomMaxValue);
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            
            transitionGrid.saveGridAsPNG("grid/grid_" + i + "_input.png", config.pngWidth, config.pngHeight);
            transitionGrid.saveClustersPNG("clusters/grid_" + i + "_clusters.png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            // Track changes
            double similarity = transitionGrid.calculatePartitionChange(previousGrid.getPartition());
            int totalCellsMoved = (int)(similarity * config.rows * config.cols);
            System.out.println("\nPartition Change Analysis (vs previous):");
            System.out.println("Percentage of cells that changed partition: " + 
                String.format("%.2f%%", similarity * 100));
            System.out.println("Total number of cells moved: " + totalCellsMoved);

            totalChanges += similarity;
            previousGrid = transitionGrid;
        }

        System.out.println("\nAverage Partition Changes Analysis:");
        System.out.println("Average percentage of cells that changed partition per transition: " + 
            String.format("%.2f%%", (totalChanges / run) * 100));

        System.out.println("\nPer-Cluster Movement Analysis:");
        System.out.println("Cluster\tAvg Cells Moved/Transition");
        System.out.println("----------------------------------------");
        for (int i = 0; i < config.numClusters; i++) {
            double avgMoves = clusterChangeCount[i] > 0 ? 
                (clusterChanges[i] / run) : 0;
            System.out.printf("%d\t%.0f%n", i, avgMoves);
        }
        System.out.println("----------------------------------------");
    }

    private static void runAdaptationTestWithFigureEight(GridConfig config, int run) throws IOException {
        new File("figure8").mkdirs();
        List<double[]> points = new ArrayList<>();
        int numPoints = 4;
        double centerX = config.rows / 2.0;
        double centerY = config.cols / 2.0;
        double radius = Math.min(config.rows, config.cols) / 5.0;
        
        for (int i = 0; i < numPoints; i++) {
            double phase = (2 * Math.PI * i) / numPoints;
            points.add(new double[]{centerX, centerY, 0, 0});
        }

        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = new GRID(config.rows, config.cols);
        generateGridFromPoints(originalGrid, points, config.randomMaxValue);
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        originalGrid.saveCombinedImage("figure8/grid_0.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;

        for (int i = 1; i <= run; i++) {
            double t = i * 0.05;  // Time parameter
            for (int p = 0; p < points.size(); p++) {
                double[] point = points.get(p);
                double phase = (2 * Math.PI * p) / numPoints + t;
                // Figure-eight pattern
                point[0] = centerX + radius * Math.sin(phase);
                point[1] = centerY + radius * Math.sin(phase) * Math.cos(phase);
            }

            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = new GRID(config.rows, config.cols);
            generateGridFromPoints(transitionGrid, points, config.randomMaxValue);
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            
            transitionGrid.saveCombinedImage("figure8/grid_" + i + ".png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            previousGrid = transitionGrid;
        }
    }

    private static void runAdaptationTestWithSpiral(GridConfig config, int run) throws IOException {
        new File("spiral").mkdirs();
        List<double[]> points = new ArrayList<>();
        int numPoints = 3;
        double centerX = config.rows / 2.0;
        double centerY = config.cols / 2.0;
        
        for (int i = 0; i < numPoints; i++) {
            points.add(new double[]{centerX, centerY, 0, 0});
        }

        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = new GRID(config.rows, config.cols);
        generateGridFromPoints(originalGrid, points, config.randomMaxValue);
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        originalGrid.saveCombinedImage("spiral/grid_0.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;

        for (int i = 1; i <= run; i++) {
            double t = i * 0.02;
            for (int p = 0; p < points.size(); p++) {
                double[] point = points.get(p);
                double phase = (2 * Math.PI * p) / numPoints + t;
                // Expanding/contracting spiral
                double r = 20 * (1 + Math.sin(t * 0.5)) * (p + 1) / numPoints;
                point[0] = centerX + r * Math.cos(phase * 3);
                point[1] = centerY + r * Math.sin(phase * 3);
            }

            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = new GRID(config.rows, config.cols);
            generateGridFromPoints(transitionGrid, points, config.randomMaxValue);
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            
            transitionGrid.saveCombinedImage("spiral/grid_" + i + ".png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            previousGrid = transitionGrid;
        }
    }

    private static void runAdaptationTestWithChaos(GridConfig config, int run) throws IOException {
        new File("chaos").mkdirs();
        List<double[]> points = new ArrayList<>();
        int numPoints = 5;
        Random rand = new Random(42);
        
        // Initialize with Lorenz attractor parameters
        for (int i = 0; i < numPoints; i++) {
            points.add(new double[]{
                config.rows/2.0 + rand.nextDouble() * 10,  // x
                config.cols/2.0 + rand.nextDouble() * 10,  // y
                0, 0  // velocities
            });
        }

        System.out.println("\n=== Original Grid Analysis ===");
        long startTime = System.currentTimeMillis();
        GRID originalGrid = new GRID(config.rows, config.cols);
        generateGridFromPoints(originalGrid, points, config.randomMaxValue);
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        
        originalGrid.saveCombinedImage("chaos/grid_0.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;

        double dt = 0.01;
        for (int i = 1; i <= run; i++) {
            for (double[] point : points) {
                // Simplified chaotic motion
                double dx = point[2];
                double dy = point[3];
                
                // Update velocities with chaotic terms
                point[2] += (-0.1 * point[2] + Math.sin(point[1]/20.0)) * dt;
                point[3] += (-0.1 * point[3] + Math.cos(point[0]/20.0)) * dt;
                
                // Update positions with boundary reflection
                point[0] = Math.max(0, Math.min(config.rows-1, point[0] + dx));
                point[1] = Math.max(0, Math.min(config.cols-1, point[1] + dy));
                
                // Reflect velocities at boundaries
                if (point[0] <= 0 || point[0] >= config.rows-1) point[2] *= -0.8;
                if (point[1] <= 0 || point[1] >= config.cols-1) point[3] *= -0.8;
            }

            System.out.println("\n=== Transition " + i + " Analysis ===");
            startTime = System.currentTimeMillis();
            GRID transitionGrid = new GRID(config.rows, config.cols);
            generateGridFromPoints(transitionGrid, points, config.randomMaxValue);
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            
            transitionGrid.saveCombinedImage("chaos/grid_" + i + ".png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);

            previousGrid = transitionGrid;
        }
    }

    private static void runAdaptationTestWithWaves(GridConfig config, int run) throws IOException {
        new File("waves").mkdirs();
        GRID originalGrid = new GRID(config.rows, config.cols);
        
        // Initialize with wave pattern
        for (int i = 0; i < config.rows; i++) {
            for (int j = 0; j < config.cols; j++) {
                originalGrid.grid[i][j] = (int)(128 + 127 * Math.sin(i * 0.1) * Math.cos(j * 0.1));
            }
        }

        long startTime = System.currentTimeMillis();
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        originalGrid.saveCombinedImage("waves/grid_0.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;
        double timeStep = 0.1;

        for (int i = 1; i <= run; i++) {
            GRID transitionGrid = new GRID(config.rows, config.cols);
            // Create moving wave patterns
            for (int r = 0; r < config.rows; r++) {
                for (int c = 0; c < config.cols; c++) {
                    double wave1 = Math.sin(r * 0.1 + i * timeStep) * Math.cos(c * 0.1);
                    double wave2 = Math.cos(r * 0.15 - i * timeStep) * Math.sin(c * 0.15);
                    transitionGrid.grid[r][c] = (int)(128 + 63 * (wave1 + wave2));
                }
            }
            
            startTime = System.currentTimeMillis();
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            transitionGrid.saveCombinedImage("waves/grid_" + i + ".png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);
            previousGrid = transitionGrid;
        }
    }

    private static void runAdaptationTestWithCellularAutomata(GridConfig config, int run) throws IOException {
        new File("cellular").mkdirs();
        GRID originalGrid = new GRID(config.rows, config.cols);
        Random rand = new Random(42);
        
        // Initialize with random values
        for (int i = 0; i < config.rows; i++) {
            for (int j = 0; j < config.cols; j++) {
                originalGrid.grid[i][j] = rand.nextInt(2) * 255;
            }
        }

        long startTime = System.currentTimeMillis();
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        originalGrid.saveCombinedImage("cellular/grid_0.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;

        for (int i = 1; i <= run; i++) {
            GRID transitionGrid = new GRID(config.rows, config.cols);
            // Apply Game of Life-like rules
            for (int r = 0; r < config.rows; r++) {
                for (int c = 0; c < config.cols; c++) {
                    int sum = 0;
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            int nr = (r + dr + config.rows) % config.rows;
                            int nc = (c + dc + config.cols) % config.cols;
                            sum += previousGrid.grid[nr][nc] > 127 ? 1 : 0;
                        }
                    }
                    sum -= previousGrid.grid[r][c] > 127 ? 1 : 0;
                    
                    if (previousGrid.grid[r][c] > 127) {
                        transitionGrid.grid[r][c] = (sum == 2 || sum == 3) ? 255 : 0;
                    } else {
                        transitionGrid.grid[r][c] = (sum == 3) ? 255 : 0;
                    }
                }
            }
            
            startTime = System.currentTimeMillis();
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            transitionGrid.saveCombinedImage("cellular/grid_" + i + ".png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);
            previousGrid = transitionGrid;
        }
    }

    private static void runAdaptationTestWithDiffusion(GridConfig config, int run) throws IOException {
        new File("diffusion").mkdirs();
        GRID originalGrid = new GRID(config.rows, config.cols);
        Random rand = new Random(42);
        
        // Initialize with random high-value spots
        for (int i = 0; i < config.rows; i++) {
            for (int j = 0; j < config.cols; j++) {
                originalGrid.grid[i][j] = rand.nextDouble() < 0.1 ? 255 : 0;
            }
        }

        long startTime = System.currentTimeMillis();
        originalGrid.partitionGrid(config.numClusters);
        long endTime = System.currentTimeMillis();
        originalGrid.saveCombinedImage("diffusion/grid_0.png", config.pngWidth, config.pngHeight);
        printGridAnalysis(originalGrid, startTime, endTime);

        GRID previousGrid = originalGrid;
        double diffusionRate = 0.2;

        for (int i = 1; i <= run; i++) {
            GRID transitionGrid = new GRID(config.rows, config.cols);
            // Apply diffusion
            for (int r = 0; r < config.rows; r++) {
                for (int c = 0; c < config.cols; c++) {
                    double sum = 0;
                    int count = 0;
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            if (r + dr >= 0 && r + dr < config.rows && 
                                c + dc >= 0 && c + dc < config.cols) {
                                sum += previousGrid.grid[r + dr][c + dc];
                                count++;
                            }
                        }
                    }
                    double avgNeighbor = sum / count;
                    double currentValue = previousGrid.grid[r][c];
                    transitionGrid.grid[r][c] = (int)(currentValue + 
                        (avgNeighbor - currentValue) * diffusionRate);
                }
            }
            
            startTime = System.currentTimeMillis();
            transitionGrid.adaptPartition(previousGrid.getPartition(), previousGrid.getGrid());
            endTime = System.currentTimeMillis();
            transitionGrid.saveCombinedImage("diffusion/grid_" + i + ".png", config.pngWidth, config.pngHeight);
            printGridAnalysis(transitionGrid, startTime, endTime);
            previousGrid = transitionGrid;
        }
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
            // Choose one of the movement patterns:

            //runAdaptationTestWithFigureEight(config, 100);
            //runAdaptationTestWithSpiral(config, 100);
            //runAdaptationTestWithChaos(config, 100);
            //runAdaptationTestWithWaves(config, 100);
            runAdaptationTestWithCellularAutomata(config, 100);
            //runAdaptationTestWithDiffusion(config, 100);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}