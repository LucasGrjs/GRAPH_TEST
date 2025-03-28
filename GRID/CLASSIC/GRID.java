import java.io.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;
import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.imageio.stream.ImageOutputStream;
import javax.imageio.stream.FileImageOutputStream;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.metadata.IIOMetadataNode;
import javax.imageio.ImageWriter;
import javax.imageio.ImageTypeSpecifier;
import javax.imageio.ImageWriteParam;
import javax.imageio.IIOImage;
import java.awt.AlphaComposite;
import java.awt.image.RenderedImage;

// Remove CellType enum

class ClusterStats {
    final int minClusterSize;
    final int maxClusterSize;
    final double avgClusterSize;
    final double stdDevClusterSize;
    final int borderCellCount;
    final Map<Integer, Integer> borderCellsPerCluster;
    final Map<Integer, Integer> sizesPerCluster;
    final int totalCells;
    final double borderPct;
    final int numClusters;
    final double sizeVariationCoeff;  // Coefficient of variation (stdDev/mean)

    ClusterStats(int min, int max, double avg, double stdDev, int borders, int adj, 
                 Map<Integer, Integer> borderPerCluster,
                 Map<Integer, Integer> sizesPerCluster,
                 int totalCells, double borderPct) {
        this.minClusterSize = min;
        this.maxClusterSize = max;
        this.avgClusterSize = avg;
        this.stdDevClusterSize = stdDev;
        this.borderCellCount = adj;
        this.borderCellsPerCluster = borderPerCluster;
        this.sizesPerCluster = sizesPerCluster;
        this.totalCells = totalCells;
        this.borderPct = borderPct;
        this.numClusters = sizesPerCluster.size();
        this.sizeVariationCoeff = avg != 0 ? (stdDev/avg) : 0;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "Cluster Stats:\n" +
            "Total cells: %d\n" +
            "Number of clusters: %d\n" +
            "Min cells per cluster: %d\n" +
            "Max cells per cluster: %d\n" +
            "Avg cells per cluster: %.2f\n" +
            "StdDev of cluster sizes: %.2f\n" +
            "Size variation coefficient: %.2f\n" +
            "Border Cells: %d (%.2f%%)\n\n",
            totalCells, numClusters, minClusterSize, maxClusterSize, 
            avgClusterSize, stdDevClusterSize, sizeVariationCoeff,
            borderCellCount, borderPct));

        // Add detailed size distribution
        sb.append("Size distribution:\n");
        sizesPerCluster.entrySet().stream()
            .sorted(Map.Entry.comparingByKey())
            .forEach(e -> sb.append(String.format("Cluster %d: %d cells (%.1f%% of total)\n", 
                e.getKey(), e.getValue(), (100.0 * e.getValue() / totalCells))));
        
        return sb.toString();
    }
}

public class GRID {
    private Integer[][] grid; 
    private int rows;
    private int cols;
    private Integer[][] clusterGrid;
    private List<Point> borderCells = new ArrayList<>();
    private List<Point> adjCell = new ArrayList<>();
    private Map<Point, Set<Integer>> borderCellClusters = new HashMap<>();
    
    // Add global color array
    private static final Color[] clusterColors = {
        Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, 
        Color.CYAN, Color.MAGENTA, Color.ORANGE, Color.PINK,
        Color.LIGHT_GRAY, Color.DARK_GRAY,
        new Color(128, 0, 0),     // Maroon
        new Color(0, 128, 0),     // Dark Green
        new Color(0, 0, 128),     // Navy
        new Color(128, 128, 0),   // Olive
        new Color(128, 0, 128),   // Purple
        new Color(0, 128, 128),   // Teal
        new Color(255, 160, 122), // Light Salmon
        new Color(32, 178, 170),  // Light Sea Green
        new Color(255, 69, 0),    // Orange Red
        new Color(218, 112, 214), // Orchid
        new Color(60, 179, 113),  // Medium Sea Green
        new Color(186, 85, 211),  // Medium Orchid
        new Color(255, 99, 71),   // Tomato
        new Color(64, 224, 208),  // Turquoise
        new Color(218, 165, 32),  // Golden Rod
        new Color(147, 112, 219), // Medium Purple
        new Color(0, 250, 154),   // Medium Spring Green
        new Color(255, 127, 80),  // Coral
        new Color(100, 149, 237), // Cornflower Blue
        new Color(189, 183, 107), // Dark Khaki
        new Color(153, 50, 204),  // Dark Orchid
        new Color(139, 69, 19),   // Saddle Brown
        new Color(143, 188, 143), // Dark Sea Green
        new Color(72, 61, 139),   // Dark Slate Blue
        new Color(47, 79, 79),    // Dark Slate Gray
        new Color(148, 0, 211),   // Dark Violet
        new Color(255, 140, 0),   // Dark Orange
        new Color(153, 153, 0),   // Olive Drab
        new Color(139, 0, 139),   // Dark Magenta
        new Color(233, 150, 122), // Dark Salmon
        new Color(143, 188, 143), // Dark Sea Green
        new Color(85, 107, 47),   // Dark Olive Green
        new Color(139, 0, 0),     // Dark Red
        new Color(233, 150, 122), // Dark Salmon
        new Color(184, 134, 11)   // Dark Goldenrod
    };

    public GRID(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.grid = new Integer[rows][cols];
        initializeRandomGrid();
    }
    private static class Point {
        int row, col;
        Point(int row, int col) {
            this.row = row;
            this.col = col;
        }
        double distanceTo(Point other) {
            return Math.sqrt(Math.pow(row - other.row, 2) + Math.pow(col - other.col, 2));
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Point point = (Point) o;
            return row == point.row && col == point.col;
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(row, col);
        }
    }

    private void initializeRandomGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grid[i][j] = 0;
            }
        }
    }

    public void saveGridAsImage(String filePath) throws IOException {
        
        clearClusterData();
        int cellSize = 10;
        BufferedImage image = new BufferedImage(cols * cellSize, rows * cellSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Draw cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                g2d.setColor(Color.WHITE);
                g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                g2d.setColor(Color.BLACK);
                g2d.drawRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }

        g2d.dispose();
        ImageIO.write(image, "png", new File(filePath));
    }

    private void initializeClusters(int K) {
        // Initialize clusterGrid with -1
        clusterGrid = new Integer[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clusterGrid[i][j] = -1;
            }
        }
    }

    private void clearClusterData() {
        borderCells.clear();
        adjCell.clear();
        borderCellClusters.clear();
        clusterGrid = null;
    }

    public Map<Integer, Integer> getClusterSizes() {
        Map<Integer, Integer> sizes = new HashMap<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 0 && clusterGrid[i][j] != -1) {
                    sizes.merge(clusterGrid[i][j], 1, Integer::sum);
                }
            }
        }
        return sizes;
    }

    private Map<Integer, Integer> getBorderCellsPerCluster() {
        Map<Integer, Integer> counts = new HashMap<>();
        for (Point p : borderCells) {
            Set<Integer> clusters = getBorderCellClusters(p);
            for (Integer cluster : clusters) {
                counts.merge(cluster, 1, Integer::sum);
            }
        }
        return counts;
    }
    int getTotalCells()
    {
        return rows * cols;
    }

    public ClusterStats getClusterStats() {
        Map<Integer, Integer> sizes = getClusterSizes();
        int totalCells = getTotalCells();
        int minSize = sizes.values().stream().mapToInt(i -> i).min().orElse(0);
        int maxSize = sizes.values().stream().mapToInt(i -> i).max().orElse(0);
        double avgSize = sizes.values().stream().mapToInt(i -> i).average().orElse(0);
        
        // Calculate standard deviation
        double variance = sizes.values().stream()
            .mapToDouble(size -> {
                double diff = size - avgSize;
                return diff * diff;
            })
            .average()
            .orElse(0);
        double stdDev = Math.sqrt(variance);
        
        Map<Integer, Integer> borderPerCluster = getBorderCellsPerCluster();
        double borderPct = (double)adjCell.size() / totalCells * 100;

        return new ClusterStats(
            minSize, maxSize, avgSize, stdDev,
            borderCells.size(), adjCell.size(),
            borderPerCluster, sizes,
            totalCells, borderPct
        );
    }

    public void divideGridToClusters(int K) {
        clearClusterData();
        List<Point> allCells = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                allCells.add(new Point(i, j));
            }
        }

        if (allCells.isEmpty()) return;

        // Initialize cluster grid
        initializeClusters(K);

        // Initialize K random centroids from all cells
        Random random = new Random();
        List<Point> centroids = new ArrayList<>();
        Set<Integer> usedIndices = new HashSet<>();
        for (int i = 0; i < K && i < allCells.size(); i++) {
            int index;
            do {
                index = random.nextInt(allCells.size());
            } while (!usedIndices.add(index));
            centroids.add(allCells.get(index));
        }

        // K-means iteration
        boolean changed;
        int maxIterations = 100;
        do {
            changed = false;
            // Assign each cell to nearest centroid
            Map<Integer, List<Point>> clusterPoints = new HashMap<>();
            for (Point cell : allCells) {
                int nearestCluster = 0;
                double minDistance = Double.MAX_VALUE;
                
                for (int i = 0; i < centroids.size(); i++) {
                    double distance = cell.distanceTo(centroids.get(i));
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCluster = i;
                    }
                }

                clusterPoints.computeIfAbsent(nearestCluster, k -> new ArrayList<>()).add(cell);
                int oldCluster = clusterGrid[cell.row][cell.col];
                if (oldCluster != nearestCluster) {
                    clusterGrid[cell.row][cell.col] = nearestCluster;
                    changed = true;
                }
            }

            // Update centroids
            for (int i = 0; i < K; i++) {
                List<Point> clusterCells = clusterPoints.getOrDefault(i, new ArrayList<>());
                if (!clusterCells.isEmpty()) {
                    double avgRow = clusterCells.stream().mapToInt(p -> p.row).average().orElse(0);
                    double avgCol = clusterCells.stream().mapToInt(p -> p.col).average().orElse(0);
                    centroids.set(i, new Point((int)avgRow, (int)avgCol));
                }
            }

            maxIterations--;
        } while (changed && maxIterations > 0);

        // Count active clusters and remap indices
        Set<Integer> activeClusters = new HashSet<>();
        for (Point p : allCells) {
            if (clusterGrid[p.row][p.col] != -1) {
                activeClusters.add(clusterGrid[p.row][p.col]);
            }
        }
        
        // Create index mapping
        Map<Integer, Integer> indexMap = new HashMap<>();
        int newIndex = 0;
        for (int oldIndex : activeClusters) {
            indexMap.put(oldIndex, newIndex++);
        }

        // Remap cluster indices
        for (Point p : allCells) {
            if (clusterGrid[p.row][p.col] != -1) {
                clusterGrid[p.row][p.col] = indexMap.get(clusterGrid[p.row][p.col]);
            }
        }

        // After remapping indices, calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 0 && clusterGrid[i][j] != -1) {  // Only consider cells in clusters
                    if (isBorderCell(i, j, false)) {
                        Point p = new Point(i, j);
                        adjCell.add(p);
                    }
                }
            }
        }
    }

    private boolean isBorderCell(int row, int col, boolean includeDiagonals) {
        Set<Integer> adjacentClusters = new HashSet<>();
        int currentCluster = clusterGrid[row][col];
        
        // Check direct neighbors (up, down, left, right)
        int[][] directNeighbors = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] neighbor : directNeighbors) {
            int newRow = row + neighbor[0];
            int newCol = col + neighbor[1];
            
            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                && clusterGrid[newRow][newCol] != -1
                && clusterGrid[newRow][newCol] != currentCluster) {

                adjacentClusters.add(clusterGrid[newRow][newCol]);
            }
        }
        
        // Check diagonal neighbors if requested
        if (includeDiagonals) {
            int[][] diagonalNeighbors = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
            for (int[] neighbor : diagonalNeighbors) {
                int newRow = row + neighbor[0];
                int newCol = col + neighbor[1];
                
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                    && clusterGrid[newRow][newCol] != -1
                    && clusterGrid[newRow][newCol] != currentCluster) {
                    adjacentClusters.add(clusterGrid[newRow][newCol]);
                }
            }
        }

        if(!adjacentClusters.isEmpty()) {  // Changed condition: any different adjacent cluster makes it a border cell
            Point p = new Point(row, col);
            borderCells.add(p);
            borderCellClusters.put(p, adjacentClusters);
        }
        return !adjacentClusters.isEmpty();  // Return true if there are any adjacent clusters
    }

    private void drawClusterImage(BufferedImage image, boolean includeDiagonals, boolean isKMean, boolean drawBorders) {
        int cellSize = 10;
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Draw cells with cluster colors and compute border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Handle null or invalid cluster values
                if (clusterGrid[i][j] == null || clusterGrid[i][j] < 0) {
                    g2d.setColor(Color.WHITE);
                    g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                    continue;
                }

                // Check if cell is a border cell
                boolean isBorder = isBorderCell(i, j, includeDiagonals);

                // Fill cell with appropriate color
                if (drawBorders && isBorder) {
                    g2d.setColor(Color.BLACK);
                } else {
                    int colorIndex = Math.abs(clusterGrid[i][j] % clusterColors.length);
                    g2d.setColor(clusterColors[colorIndex]);
                }
                g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

                // Draw cell borders
                g2d.setColor(Color.BLACK);
                g2d.setStroke(new BasicStroke(1.0f));
                g2d.drawRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }
        g2d.dispose();
    }

    public void saveGridAsKMEANClusterImage(String filePath, int K, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToClusters(K);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, true, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsGridClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToGridClusters(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsHorizontalStripImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToHorizontalStrips(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsVerticalStripImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToVerticalStrips(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsBSPClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToBSPClusters(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsCircularClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToCircularClusters(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsSpiralClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToSpiralClusters(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void saveGridAsVoronoiClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToVoronoiClusters(numClusters);
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public void divideGridToDiagonalClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Calculate diagonal width to achieve desired number of clusters
        int diagonalWidth = (int) Math.ceil((rows + cols) / (double) numClusters);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Assign cluster based on position along main diagonal
                int clusterId = Math.min((i + j) / diagonalWidth, numClusters - 1);
                clusterGrid[i][j] = clusterId;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToCheckerboardClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Calculate checkerboard dimensions
        int boardSize = (int) Math.ceil(Math.sqrt(numClusters));
        int cellSize = Math.max(rows, cols) / boardSize;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int blockRow = i / cellSize;
                int blockCol = j / cellSize;
                int clusterId = (blockRow * boardSize + blockCol) % numClusters;
                clusterGrid[i][j] = clusterId;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToFractalClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Use Hilbert curve for fractal-like partitioning
        int order = (int) Math.ceil(Math.log(numClusters) / Math.log(4));
        int totalPoints = 1 << (2 * order);
        double pointsPerCluster = totalPoints / (double) numClusters;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Map 2D coordinates to Hilbert curve position
                int hilbertPos = getHilbertCurvePosition(i * (1 << order) / rows, 
                                                       j * (1 << order) / cols, 
                                                       order);
                int clusterId = Math.min((int)(hilbertPos / pointsPerCluster), numClusters - 1);
                clusterGrid[i][j] = clusterId;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToHoneycombClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Calculate hexagon size to achieve desired number of clusters
        double hexSize = Math.sqrt((rows * cols) / (numClusters * 2.598)); // 2.598 is hex area ratio
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Convert to hex coordinates
                double q = (2.0/3 * j) / hexSize;
                double r = (-1.0/3 * j + Math.sqrt(3)/3 * i) / hexSize;
                
                // Get nearest hex center
                int qGrid = (int) Math.round(q);
                int rGrid = (int) Math.round(r);
                
                // Convert hex coordinate to cluster ID
                int clusterId = Math.abs((qGrid * 31 + rGrid * 37)) % numClusters;
                clusterGrid[i][j] = clusterId;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToWaveClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Use sine waves with different phases for interesting patterns
        double frequency = 2 * Math.PI * 3 / Math.max(rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double wave1 = Math.sin(frequency * i);
                double wave2 = Math.cos(frequency * j);
                double combined = (wave1 + wave2 + 2) / 4; // Normalize to 0-1
                
                int clusterId = Math.min((int)(combined * numClusters), numClusters - 1);
                clusterGrid[i][j] = clusterId;
            }
        }
        
        calculateBorderCells();
    }

    private int getHilbertCurvePosition(int x, int y, int order) {
        int position = 0;
        for (int i = 0; i < order; i++) {
            int xi = (x >> i) & 1;
            int yi = (y >> i) & 1;
            position += ((3 * xi) ^ yi) << (2 * i);
        }
        return position;
    }

    private void calculateBorderCells() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    // Add save methods for new clustering approaches
    public void saveGridAsDiagonalClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToDiagonalClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsCheckerboardClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToCheckerboardClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsFractalClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToFractalClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsHoneycombClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToHoneycombClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsWaveClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToWaveClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    private void saveClusterImage(String filePath, boolean includeDiagonals, boolean drawBorders) throws IOException {
        BufferedImage image = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
        drawClusterImage(image, includeDiagonals, false, drawBorders);
        ImageIO.write(image, "png", new File(filePath));
    }

    public Set<Integer> getBorderCellClusters(Point p) {
        return borderCellClusters.getOrDefault(p, new HashSet<>());
    }

    private int getCellId(int row, int col) {
        return row * cols + col;
    }

    public List<List<Integer>> getClusteringAsLists() {
        if (clusterGrid == null) {
            return new ArrayList<>();
        }

        Map<Integer, List<Integer>> clusterMap = new HashMap<>();
        
        // Go through all cells and group them by cluster
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 0 && clusterGrid[i][j] != -1) {
                    int clusterId = clusterGrid[i][j];
                    int cellId = getCellId(i, j);
                    clusterMap.computeIfAbsent(clusterId, k -> new ArrayList<>()).add(cellId);
                }
            }
        }
        System.out.println("clusterMap " + clusterMap);
        System.out.println("ArrayList<>(clusterMap.values() " + new ArrayList<>(clusterMap.values()));
        // Convert map to list of lists
        return new ArrayList<>(clusterMap.values());
    }

    public void divideGridToGridClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);

        // Find the best grid dimensions that are close to square
        int numRows = (int) Math.sqrt(numClusters);
        while (numClusters % numRows != 0 && numRows > 1) {
            numRows--;
        }
        int numCols = numClusters / numRows;

        // Calculate the size of each grid cell
        int baseHeight = rows / numRows;
        int baseWidth = cols / numCols;
        
        // Calculate remainders
        int remainderRows = rows % numRows;
        int remainderCols = cols % numCols;

        // Create arrays to store the actual size of each row and column
        int[] rowHeights = new int[numRows];
        int[] colWidths = new int[numCols];

        // Distribute the remainder pixels evenly
        for (int i = 0; i < numRows; i++) {
            rowHeights[i] = baseHeight + (i < remainderRows ? 1 : 0);
        }
        for (int i = 0; i < numCols; i++) {
            colWidths[i] = baseWidth + (i < remainderCols ? 1 : 0);
        }

        // Assign clusters based on position
        int currentRow = 0;
        int currentY = 0;
        
        for (int i = 0; i < rows; i++) {
            if (i >= currentY + rowHeights[currentRow]) {
                currentY += rowHeights[currentRow];
                currentRow++;
            }
            
            int currentCol = 0;
            int currentX = 0;
            
            for (int j = 0; j < cols; j++) {
                if (j >= currentX + colWidths[currentCol]) {
                    currentX += colWidths[currentCol];
                    currentCol++;
                }
                
                clusterGrid[i][j] = currentRow * numCols + currentCol;
            }
        }

        // Calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    public void divideGridToHorizontalStrips(int numClusters) {
        clearClusterData();
        initializeClusters(0);

        // Calculate the height of each strip
        int stripHeight = (int) Math.ceil(rows / (double) numClusters);

        // Assign each cell to its strip cluster
        for (int i = 0; i < rows; i++) {
            int clusterId = Math.min(i / stripHeight, numClusters - 1);
            for (int j = 0; j < cols; j++) {
                clusterGrid[i][j] = clusterId;
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    public void divideGridToVerticalStrips(int numClusters) {
        clearClusterData();
        initializeClusters(0);

        // Calculate the width of each strip
        int stripWidth = (int) Math.ceil(cols / (double) numClusters);

        // Assign each cell to its strip cluster
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int clusterId = Math.min(j / stripWidth, numClusters - 1);
                clusterGrid[i][j] = clusterId;
            }
        }

        // Calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    private void recursiveSplit(int startRow, int startCol, int endRow, int endCol, 
                              int currentCluster, int remainingClusters, boolean splitVertical) {
        // Base case: if we only need one cluster, assign all cells to it
        if (remainingClusters == 1) {
            for (int i = startRow; i < endRow; i++) {
                for (int j = startCol; j < endCol; j++) {
                    clusterGrid[i][j] = currentCluster;
                }
            }
            return;
        }

        // Calculate how to split remaining clusters
        int firstPartClusters = remainingClusters / 2;
        int secondPartClusters = remainingClusters - firstPartClusters;

        if (splitVertical) {
            // Split vertically
            int midCol = startCol + (endCol - startCol) * firstPartClusters / remainingClusters;
            recursiveSplit(startRow, startCol, endRow, midCol, currentCluster, firstPartClusters, false);
            recursiveSplit(startRow, midCol, endRow, endCol, currentCluster + firstPartClusters, secondPartClusters, false);
        } else {
            // Split horizontally
            int midRow = startRow + (endRow - startRow) * firstPartClusters / remainingClusters;
            recursiveSplit(startRow, startCol, midRow, endCol, currentCluster, firstPartClusters, true);
            recursiveSplit(midRow, startCol, endRow, endCol, currentCluster + firstPartClusters, secondPartClusters, true);
        }
    }

    public void divideGridToBSPClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);

        // Start recursive splitting
        recursiveSplit(0, 0, rows, cols, 0, numClusters, true);

        // Calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    public void divideGridToCircularClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Calculate center point
        int centerX = cols / 2;
        int centerY = rows / 2;
        
        // Calculate max radius and radius step
        double maxRadius = Math.sqrt(Math.pow(Math.max(rows, cols), 2) / 2);
        double radiusStep = maxRadius / numClusters;
        
        // Assign clusters based on distance from center
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double distance = Math.sqrt(Math.pow(i - centerY, 2) + Math.pow(j - centerX, 2));
                int clusterId = Math.min((int)(distance / radiusStep), numClusters - 1);
                clusterGrid[i][j] = clusterId;
            }
        }

        // Calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    public void divideGridToSpiralClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        int currentCluster = 0;
        int left = 0, right = cols-1, top = 0, bottom = rows-1;
        int cellsPerCluster = (rows * cols) / numClusters;
        int cellsInCurrentCluster = 0;
        
        while (left <= right && top <= bottom) {
            // Move right
            for (int j = left; j <= right && currentCluster < numClusters; j++) {
                clusterGrid[top][j] = currentCluster;
                cellsInCurrentCluster++;
                if (cellsInCurrentCluster >= cellsPerCluster && currentCluster < numClusters - 1) {
                    currentCluster++;
                    cellsInCurrentCluster = 0;
                }
            }
            top++;
            
            // Move down
            for (int i = top; i <= bottom && currentCluster < numClusters; i++) {
                clusterGrid[i][right] = currentCluster;
                cellsInCurrentCluster++;
                if (cellsInCurrentCluster >= cellsPerCluster && currentCluster < numClusters - 1) {
                    currentCluster++;
                    cellsInCurrentCluster = 0;
                }
            }
            right--;
            
            // Move left
            for (int j = right; j >= left && currentCluster < numClusters; j--) {
                clusterGrid[bottom][j] = currentCluster;
                cellsInCurrentCluster++;
                if (cellsInCurrentCluster >= cellsPerCluster && currentCluster < numClusters - 1) {
                    currentCluster++;
                    cellsInCurrentCluster = 0;
                }
            }
            bottom--;
            
            // Move up
            for (int i = bottom; i >= top && currentCluster < numClusters; i--) {
                clusterGrid[i][left] = currentCluster;
                cellsInCurrentCluster++;
                if (cellsInCurrentCluster >= cellsPerCluster && currentCluster < numClusters - 1) {
                    currentCluster++;
                    cellsInCurrentCluster = 0;
                }
            }
            left++;
        }

        // Fill any remaining cells with the last cluster
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (clusterGrid[i][j] == -1) {
                    clusterGrid[i][j] = numClusters - 1;
                }
            }
        }

        // Calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    private int findNearestNonEmptyCluster(int row, int col) {
        int radius = 1;
        while (radius < Math.max(rows, cols)) {
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    if (Math.abs(i) == radius || Math.abs(j) == radius) {
                        int ni = row + i;
                        int nj = col + j;
                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && clusterGrid[ni][nj] != -1) {
                            return clusterGrid[ni][nj];
                        }
                    }
                }
            }
            radius++;
        }
        return 0;
    }

    public void divideGridToVoronoiClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        Random random = new Random();
        
        int targetSize = (rows * cols) / numClusters;
        Map<Integer, Integer> clusterSizes = new HashMap<>();
        PriorityQueue<Point> growthFrontier = new PriorityQueue<>((a, b) -> {
            int clusterA = clusterGrid[a.row][a.col];
            int clusterB = clusterGrid[b.row][b.col];
            return Integer.compare(clusterSizes.get(clusterA), clusterSizes.get(clusterB));
        });

        // Place seeds in a roughly uniform distribution
        double seedSpacing = Math.sqrt((rows * cols) / (double) numClusters);
        for (int k = 0; k < numClusters; k++) {
            int row = (int) ((k / Math.sqrt(numClusters)) * seedSpacing + random.nextDouble() * seedSpacing);
            int col = (int) ((k % Math.sqrt(numClusters)) * seedSpacing + random.nextDouble() * seedSpacing);
            
            row = Math.min(Math.max(row, 0), rows - 1);
            col = Math.min(Math.max(col, 0), cols - 1);
            
            clusterGrid[row][col] = k;
            clusterSizes.put(k, 1);
            growthFrontier.add(new Point(row, col));
        }

        // Grow regions while maintaining size balance
        int[][] directions = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        boolean[][] visited = new boolean[rows][cols];
        
        while (!growthFrontier.isEmpty()) {
            Point current = growthFrontier.poll();
            int currentCluster = clusterGrid[current.row][current.col];
            
            // Skip if this cluster is already at target size
            if (clusterSizes.get(currentCluster) >= targetSize) {
                continue;
            }

            // Try to grow in each direction
            for (int[] dir : directions) {
                int newRow = current.row + dir[0];
                int newCol = current.col + dir[1];
                
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                    && !visited[newRow][newCol] 
                    && clusterGrid[newRow][newCol] == -1) {
                    
                    clusterGrid[newRow][newCol] = currentCluster;
                    clusterSizes.merge(currentCluster, 1, Integer::sum);
                    visited[newRow][newCol] = true;
                    growthFrontier.add(new Point(newRow, newCol));
                }
            }
        }

        // Assign any remaining unassigned cells to nearest cluster
        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (clusterGrid[i][j] == -1) {
                        int nearestCluster = findNearestNonEmptyCluster(i, j);
                        clusterGrid[i][j] = nearestCluster;
                        changed = true;
                    }
                }
            }
        } while (changed);

        // Calculate border cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (isBorderCell(i, j, false)) {
                    Point p = new Point(i, j);
                    adjCell.add(p);
                }
            }
        }
    }

    private static class ClusteringResult implements Comparable<ClusteringResult> {
        String name;
        double borderPct;
        double sizeVariation;
        
        ClusteringResult(String name, double borderPct, double sizeVariation) {
            this.name = name;
            this.borderPct = borderPct;
            this.sizeVariation = sizeVariation;
        }
        
        @Override
        public int compareTo(ClusteringResult other) {
            // First compare by border percentage
            int borderCompare = Double.compare(this.borderPct, other.borderPct);
            if (borderCompare != 0) {
                return borderCompare;
            }
            // If border percentages are equal, compare by size variation
            return Double.compare(this.sizeVariation, other.sizeVariation);
        }
        
        @Override
        public String toString() {
            return String.format("%-20s Border: %6.2f%%  Size Variation: %6.2f%%", 
                               name, borderPct, sizeVariation * 100);
        }
    }

    public void analyzeAllClusteringMethods(final int numClusters, final boolean drawBorders) throws IOException {
        List<ClusteringResult> results = new ArrayList<>();
        
        // Create a method to handle running a clustering method and collecting results
        interface ClusteringMethod {
            void execute(int clusters, boolean borders) throws IOException;
        }
        
        // Map of method names to their implementations
        Map<String, ClusteringMethod> methods = new LinkedHashMap<>();
        methods.put("K-Means", (clusters, borders) -> saveGridAsKMEANClusterImage("kmeans_clusters.png", clusters, false, borders));
        methods.put("Grid", (clusters, borders) -> saveGridAsGridClusterImage("grid_clusters.png", clusters, false, borders));
        methods.put("Horizontal Strips", (clusters, borders) -> saveGridAsHorizontalStripImage("horizontal_strips.png", clusters, false, borders));
        methods.put("Vertical Strips", (clusters, borders) -> saveGridAsVerticalStripImage("vertical_strips.png", clusters, false, borders));
        methods.put("BSP", (clusters, borders) -> saveGridAsBSPClusterImage("bsp_clusters.png", clusters, false, borders));
        methods.put("Circular", (clusters, borders) -> saveGridAsCircularClusterImage("circular_clusters.png", clusters, false, borders));
        methods.put("Spiral", (clusters, borders) -> saveGridAsSpiralClusterImage("spiral_clusters.png", clusters, false, borders));
        methods.put("Voronoi", (clusters, borders) -> saveGridAsVoronoiClusterImage("voronoi_clusters.png", clusters, false, borders));
        methods.put("Diagonal", (clusters, borders) -> saveGridAsDiagonalClusterImage("diagonal_clusters.png", clusters, false, borders));
        methods.put("Checkerboard", (clusters, borders) -> saveGridAsCheckerboardClusterImage("checkerboard_clusters.png", clusters, false, borders));
        methods.put("Fractal", (clusters, borders) -> saveGridAsFractalClusterImage("fractal_clusters.png", clusters, false, borders));
        methods.put("Honeycomb", (clusters, borders) -> saveGridAsHoneycombClusterImage("honeycomb_clusters.png", clusters, false, borders));
        methods.put("Wave", (clusters, borders) -> saveGridAsWaveClusterImage("wave_clusters.png", clusters, false, borders));
        methods.put("Cross", (clusters, borders) -> saveGridAsCrossClusterImage("cross_clusters.png", clusters, false, borders));
        methods.put("Triangular", (clusters, borders) -> saveGridAsTriangularClusterImage("triangular_clusters.png", clusters, false, borders));
        methods.put("Mosaic", (clusters, borders) -> saveGridAsMosaicClusterImage("mosaic_clusters.png", clusters, false, borders));
        methods.put("Mandelbrot", (clusters, borders) -> saveGridAsMandelbrotClusterImage("mandelbrot_clusters.png", clusters, false, borders));
        methods.put("Concentric", (clusters, borders) -> saveGridAsConcentricClusterImage("concentric_clusters.png", clusters, false, borders));
        methods.put("Maze", (clusters, borders) -> saveGridAsMazeClusterImage("maze_clusters.png", clusters, false, borders));
        methods.put("Rotating Squares", (clusters, borders) -> saveGridAsRotatingSquaresClusterImage("rotating_squares_clusters.png", clusters, false, borders));
        methods.put("Sierpinski", (clusters, borders) -> saveGridAsSierpinskiClusterImage("sierpinski_clusters.png", clusters, false, borders));
        methods.put("Phyllotaxis", (clusters, borders) -> saveGridAsPhyllotaxisClusterImage("phyllotaxis_clusters.png", clusters, false, borders));
        methods.put("Interference Pattern", (clusters, borders) -> saveGridAsInterferencePatternClusterImage("interference_clusters.png", clusters, false, borders));
        methods.put("Weighted Voronoi", (clusters, borders) -> saveGridAsVoronoiWeightedClusterImage("weighted_voronoi_clusters.png", clusters, false, borders));
        methods.put("Perlin Noise", (clusters, borders) -> saveGridAsPerlinNoiseClusterImage("perlin_noise_clusters.png", clusters, false, borders));
        methods.put("Galaxy", (clusters, borders) -> saveGridAsGalaxyClusterImage("galaxy_clusters.png", clusters, false, borders));
        methods.put("Minimal Border", (clusters, borders) -> 
            saveGridAsMinimalBorderClusterImage("minimal_border_clusters.png", clusters, false, borders));

        // Execute each method and collect results
        for (Map.Entry<String, ClusteringMethod> method : methods.entrySet()) {
            try {
                method.getValue().execute(numClusters, drawBorders);
                ClusterStats stats = getClusterStats();
                results.add(new ClusteringResult(
                    method.getKey(),
                    stats.borderPct,
                    stats.sizeVariationCoeff
                ));
                System.out.println(method.getKey() + " : " + stats);
            } catch (IOException e) {
                System.err.println("Error processing " + method.getKey() + ": " + e.getMessage());
            }
        }
        
        // Sort and print rankings
        Collections.sort(results);
        
        System.out.println("\nClustering Methods Ranked by Border Cells and Size Balance:");
        System.out.println("====================================================");
        System.out.println("Method               Border %    Size Variation %");
        System.out.println("----------------------------------------------------");
        for (int i = 0; i < results.size(); i++) {
            System.out.printf("#%2d  %s%n", i + 1, results.get(i));
        }
    }

    public void divideGridToCrossClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        int centerX = cols / 2;
        int centerY = rows / 2;
        int armWidth = Math.min(rows, cols) / (2 * numClusters);
        
        // Start with everything in cluster 0
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clusterGrid[i][j] = 0;
            }
        }
        
        // Create cross pattern
        for (int k = 0; k < numClusters; k++) {
            int startOffset = k * armWidth;
            // Horizontal arm
            for (int j = 0; j < cols; j++) {
                for (int i = centerY + startOffset; i < centerY + startOffset + armWidth && i < rows; i++) {
                    clusterGrid[i][j] = k;
                }
            }
            // Vertical arm
            for (int i = 0; i < rows; i++) {
                for (int j = centerX + startOffset; j < centerX + startOffset + armWidth && j < cols; j++) {
                    clusterGrid[i][j] = k;
                }
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToTriangularClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        double centerX = cols / 2.0;
        double centerY = rows / 2.0;
        double angleStep = 2 * Math.PI / numClusters;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double angle = Math.atan2(i - centerY, j - centerX);
                if (angle < 0) angle += 2 * Math.PI;
                int clusterId = (int) (angle / angleStep);
                clusterGrid[i][j] = clusterId;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToMosaicClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        int tileSize = (int) Math.sqrt((rows * cols) / (double) numClusters);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double noise = Math.sin(i * 0.1) * Math.cos(j * 0.1) * tileSize * 0.5;
                int clusterId = ((i + (int)noise) / tileSize + (j + (int)noise) / tileSize) % numClusters;
                clusterGrid[i][j] = Math.abs(clusterId);
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToMandelbrotClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x0 = (j - cols/2.0) * 4.0/cols;
                double y0 = (i - rows/2.0) * 4.0/rows;
                double x = 0, y = 0;
                int iteration = 0;
                while (x*x + y*y < 4 && iteration < numClusters) {
                    double xtemp = x*x - y*y + x0;
                    y = 2*x*y + y0;
                    x = xtemp;
                    iteration++;
                }
                clusterGrid[i][j] = iteration % numClusters;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToConcentricClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        double centerX = cols / 2.0;
        double centerY = rows / 2.0;
        double maxDist = Math.sqrt(Math.pow(rows/2.0, 2) + Math.pow(cols/2.0, 2));
        double step = maxDist / numClusters;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double dist = Math.sqrt(Math.pow(i - centerY, 2) + Math.pow(j - centerX, 2));
                clusterGrid[i][j] = Math.min((int)(dist / step), numClusters-1);
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToMazeClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        Random random = new Random();
        
        // Generate maze using recursive backtracking
        int[][] maze = new int[rows][cols];
        Stack<Point> stack = new Stack<>();
        Point start = new Point(0, 0);
        stack.push(start);
        
        while (!stack.empty()) {
            Point current = stack.peek();
            List<Point> neighbors = getUnvisitedNeighbors(current, maze);
            
            if (neighbors.isEmpty()) {
                stack.pop();
            } else {
                Point next = neighbors.get(random.nextInt(neighbors.size()));
                maze[next.row][next.col] = maze[current.row][current.col] + 1;
                stack.push(next);
            }
        }
        
        // Normalize maze values to cluster numbers
        int maxValue = Arrays.stream(maze)
            .flatMapToInt(Arrays::stream)
            .max()
            .orElse(0);
            
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clusterGrid[i][j] = (maze[i][j] * numClusters) / (maxValue + 1);
            }
        }
        
        calculateBorderCells();
    }

    private List<Point> getUnvisitedNeighbors(Point p, int[][] maze) {
        List<Point> neighbors = new ArrayList<>();
        int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        
        for (int[] dir : dirs) {
            int newRow = p.row + dir[0];
            int newCol = p.col + dir[1];
            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols && maze[newRow][newCol] == 0) {
                neighbors.add(new Point(newRow, newCol));
            }
        }
        return neighbors;
    }

    public void divideGridToLSystemClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Generate L-System pattern with fixed number of iterations for complexity
        int iterations = 3; // Fixed number of iterations for pattern generation
        String pattern = "F";
        for (int i = 0; i < iterations; i++) {
            pattern = pattern.replace("F", "F+F-F+F");
        }
        
        // Convert pattern to grid with cluster assignment
        int x = 0, y = rows/2;
        int direction = 0; // 0=right, 1=down, 2=left, 3=up
        int pathLength = pattern.length();
        int segmentSize = Math.max(1, pathLength / numClusters);
        int currentStep = 0;
        
        // Initialize all cells to -1
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clusterGrid[i][j] = -1;
            }
        }
        
        // Draw L-System pattern and assign clusters
        for (char c : pattern.toCharArray()) {
            int clusterId = Math.min(currentStep / segmentSize, numClusters - 1);
            
            if (c == 'F') {
                int newX = x, newY = y;
                switch (direction) {
                    case 0: newX++; break;
                    case 1: newY++; break;
                    case 2: newX--; break;
                    case 3: newY--; break;
                }
                if (newX >= 0 && newX < cols && newY >= 0 && newY < rows) {
                    clusterGrid[y][x] = clusterId;
                    x = newX;
                    y = newY;
                }
                currentStep++;
            } else if (c == '+') {
                direction = (direction + 1) % 4;
            } else if (c == '-') {
                direction = (direction + 3) % 4;
            }
        }
        
        // Fill remaining cells using flood fill from existing clusters
        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (clusterGrid[i][j] == -1) {
                        int nearest = findNearestNonEmptyCluster(i, j);
                        if (nearest != -1) {
                            clusterGrid[i][j] = nearest % numClusters;
                            changed = true;
                        }
                    }
                }
            }
        } while (changed);
        
        // Ensure all cells are assigned to valid clusters
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (clusterGrid[i][j] == -1 || clusterGrid[i][j] >= numClusters) {
                    clusterGrid[i][j] = 0;
                }
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToFlowFieldClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Generate flow field using Perlin noise
        double[][] angleField = new double[rows][cols];
        Random random = new Random();
        double scale = 0.1;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = i * scale;
                double y = j * scale;
                angleField[i][j] = Math.sin(x) * Math.cos(y) * Math.PI;
            }
        }
        
        // Trace particles through flow field
        for (int k = 0; k < numClusters; k++) {
            double x = random.nextDouble() * cols;
            double y = random.nextDouble() * rows;
            
            for (int step = 0; step < rows * cols / numClusters; step++) {
                int ix = (int)x;
                int iy = (int)y;
                
                if (ix >= 0 && ix < cols && iy >= 0 && iy < rows) {
                    clusterGrid[iy][ix] = k;
                }
                
                double angle = angleField[Math.min(iy, rows-1)][Math.min(ix, cols-1)];
                x += Math.cos(angle);
                y += Math.sin(angle);
            }
        }
        
        // Fill remaining cells
        boolean changed;
        do {
            changed = false;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (clusterGrid[i][j] == -1) {
                        int nearest = findNearestNonEmptyCluster(i, j);
                        if (nearest != -1) {
                            clusterGrid[i][j] = nearest;
                            changed = true;
                        }
                    }
                }
            }
        } while (changed);
        
        calculateBorderCells();
    }

    public void saveGridAsCrossClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToCrossClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsTriangularClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToTriangularClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsMosaicClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToMosaicClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsMandelbrotClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToMandelbrotClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsConcentricClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToConcentricClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsMazeClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToMazeClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void divideGridToRotatingSquaresClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        double centerX = cols / 2.0;
        double centerY = rows / 2.0;
        double maxRadius = Math.sqrt(Math.pow(rows/2.0, 2) + Math.pow(cols/2.0, 2));
        double radiusStep = maxRadius / Math.sqrt(numClusters);
        double angleStep = Math.PI / (2 * Math.sqrt(numClusters));
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double dx = j - centerX;
                double dy = i - centerY;
                double radius = Math.sqrt(dx*dx + dy*dy);
                double angle = Math.atan2(dy, dx);
                if (angle < 0) angle += 2 * Math.PI;
                
                int radiusCluster = (int)(radius / radiusStep);
                int angleCluster = (int)(angle / angleStep);
                clusterGrid[i][j] = (radiusCluster + angleCluster) % numClusters;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToSierpinskiClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        int level = (int)Math.ceil(Math.log(numClusters) / Math.log(3));
        int[][] sierpinski = new int[rows][cols];
        
        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int px = (x * (1 << level)) / cols;
                int py = (y * (1 << level)) / rows;
                int result = 0;
                
                while (px > 0 && py > 0) {
                    if ((px & py) != 0) {
                        result++;
                    }
                    px >>= 1;
                    py >>= 1;
                }
                
                clusterGrid[y][x] = result % numClusters;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToPhyllotaxisClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        double goldenAngle = Math.PI * (3 - Math.sqrt(5));
        double centerX = cols / 2.0;
        double centerY = rows / 2.0;
        
        // Create seed points using phyllotaxis pattern
        List<Point> seeds = new ArrayList<>();
        for (int i = 0; i < numClusters; i++) {
            double distance = Math.sqrt(i) * Math.sqrt(rows * cols / (Math.PI * numClusters));
            double angle = i * goldenAngle;
            int x = (int)(centerX + distance * Math.cos(angle));
            int y = (int)(centerY + distance * Math.sin(angle));
            
            if (x >= 0 && x < cols && y >= 0 && y < rows) {
                seeds.add(new Point(y, x));
            }
        }
        
        // Assign each cell to nearest seed
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double minDist = Double.MAX_VALUE;
                int nearestSeed = 0;
                
                for (int k = 0; k < seeds.size(); k++) {
                    Point seed = seeds.get(k);
                    double dist = Math.sqrt(Math.pow(i - seed.row, 2) + Math.pow(j - seed.col, 2));
                    if (dist < minDist) {
                        minDist = dist;
                        nearestSeed = k;
                    }
                }
                
                clusterGrid[i][j] = nearestSeed;
            }
        }
        
        calculateBorderCells();
    }

    public void saveGridAsRotatingSquaresClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToRotatingSquaresClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsSierpinskiClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToSierpinskiClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsPhyllotaxisClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToPhyllotaxisClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void divideGridToInterferencePatternClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Create interference pattern using multiple wave sources
        List<Point> sources = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < Math.min(5, numClusters); i++) {
            sources.add(new Point(
                random.nextInt(rows),
                random.nextInt(cols)
            ));
        }
        
        double frequency = 2 * Math.PI * numClusters / Math.max(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double interference = 0;
                for (Point source : sources) {
                    double distance = Math.sqrt(Math.pow(i - source.row, 2) + Math.pow(j - source.col, 2));
                    interference += Math.sin(distance * frequency);
                }
                clusterGrid[i][j] = Math.abs((int)((interference + sources.size()) * numClusters / (2 * sources.size()))) % numClusters;
            }
        }
        
        calculateBorderCells();
    }

    public void divideGridToRecursiveHClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Generate H-tree like pattern recursively
        generateHPattern(0, 0, rows, cols, 0, numClusters);
        calculateBorderCells();
    }
    
    private void generateHPattern(int startRow, int startCol, int height, int width, int depth, int maxClusters) {
        if (depth >= Math.log(maxClusters) / Math.log(2)) {
            return;
        }
        
        int midRow = startRow + height/2;
        int midCol = startCol + width/2;
        int clusterId = depth;
        
        // Draw H pattern
        // Vertical bars
        for (int i = startRow; i < startRow + height; i++) {
            if (midCol >= 0 && midCol < cols) {
                clusterGrid[i][midCol] = clusterId % maxClusters;
            }
        }
        // Horizontal bar
        for (int j = startCol; j < startCol + width; j++) {
            if (midRow >= 0 && midRow < rows) {
                clusterGrid[midRow][j] = clusterId % maxClusters;
            }
        }
        
        // Recursively generate H patterns in quadrants
        int newHeight = height/2;
        int newWidth = width/2;
        generateHPattern(startRow, startCol, newHeight, newWidth, depth + 1, maxClusters);
        generateHPattern(startRow, midCol, newHeight, newWidth, depth + 1, maxClusters);
        generateHPattern(midRow, startCol, newHeight, newWidth, depth + 1, maxClusters);
        generateHPattern(midRow, midCol, newHeight, newWidth, depth + 1, maxClusters);
    }

    public void divideGridToVoronoiWeightedClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Generate weighted Voronoi centers with different influence ranges
        List<Point> centers = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        Random random = new Random();
        
        for (int i = 0; i < numClusters; i++) {
            centers.add(new Point(random.nextInt(rows), random.nextInt(cols)));
            weights.add(0.5 + random.nextDouble()); // Random weights between 0.5 and 1.5
        }
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double minWeightedDist = Double.MAX_VALUE;
                int nearestCenter = 0;
                
                for (int k = 0; k < centers.size(); k++) {
                    Point center = centers.get(k);
                    double dist = Math.sqrt(Math.pow(i - center.row, 2) + Math.pow(j - center.col, 2));
                    double weightedDist = dist / weights.get(k);
                    
                    if (weightedDist < minWeightedDist) {
                        minWeightedDist = weightedDist;
                        nearestCenter = k;
                    }
                }
                
                clusterGrid[i][j] = nearestCenter;
            }
        }
        
        calculateBorderCells();
    }

    public void saveGridAsInterferencePatternClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToInterferencePatternClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsVoronoiWeightedClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToVoronoiWeightedClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void divideGridToPerlinNoiseClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Implement simple Perlin noise
        Random random = new Random();
        double scale = 0.05;  // Adjust for different noise scales
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double nx = i * scale;
                double ny = j * scale;
                double noise = improvedNoise(nx, ny, 0.8);  // Using 0.8 as arbitrary z-coordinate
                int clusterId = (int)((noise + 1) * numClusters / 2) % numClusters;
                clusterGrid[i][j] = Math.abs(clusterId);
            }
        }
        
        calculateBorderCells();
    }

    // Improved Perlin noise implementation
    private double improvedNoise(double x, double y, double z) {
        int X = (int)Math.floor(x) & 255;
        int Y = (int)Math.floor(y) & 255;
        int Z = (int)Math.floor(z) & 255;
        
        x -= Math.floor(x);
        y -= Math.floor(y);
        z -= Math.floor(z);
        
        double u = fade(x);
        double v = fade(y);
        double w = fade(z);
        
        int A = p[X]+Y;
        int AA = p[A]+Z;
        int AB = p[A+1]+Z;
        int B = p[X+1]+Y;
        int BA = p[B]+Z;
        int BB = p[B+1]+Z;
        
        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                     grad(p[BA], x-1, y, z)),
                             lerp(u, grad(p[AB], x, y-1, z),
                                     grad(p[BB], x-1, y-1, z))),
                     lerp(v, lerp(u, grad(p[AA+1], x, y, z-1),
                                     grad(p[BA+1], x-1, y, z-1)),
                             lerp(u, grad(p[AB+1], x, y-1, z-1),
                                     grad(p[BB+1], x-1, y-1, z-1))));
    }

    private double fade(double t) { return t * t * t * (t * (t * 6 - 15) + 10); }
    
    private double lerp(double t, double a, double b) { return a + t * (b - a); }
    
    private double grad(int hash, double x, double y, double z) {
        int h = hash & 15;
        double u = h < 8 ? x : y;
        double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }
    
    private final int[] p = new int[512];
    {
        // Initialize permutation array
        Random random = new Random();
        for (int i = 0; i < 256; i++) p[i] = i;
        for (int i = 0; i < 256; i++) {
            int j = random.nextInt(256);
            int temp = p[i];
            p[i] = p[j];
            p[j] = temp;
        }
        for (int i = 0; i < 256; i++) p[256+i] = p[i];
    }

    public void divideGridToGalaxyClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        List<Point> centers = new ArrayList<>();
        List<Double> rotations = new ArrayList<>();
        List<Double> spiralTightness = new ArrayList<>();
        Random random = new Random();
        
        // Create galaxy centers with random parameters
        for (int i = 0; i < numClusters; i++) {
            centers.add(new Point(
                random.nextInt(rows),
                random.nextInt(cols)
            ));
            rotations.add(random.nextDouble() * Math.PI * 2);
            spiralTightness.add(0.1 + random.nextDouble() * 0.4);
        }
        
        // Assign cells based on spiral galaxy patterns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int nearestGalaxy = 0;
                double minValue = Double.MAX_VALUE;
                
                for (int k = 0; k < centers.size(); k++) {
                    Point center = centers.get(k);
                    double dx = j - center.col;
                    double dy = i - center.row;
                    double distance = Math.sqrt(dx*dx + dy*dy);
                    double angle = Math.atan2(dy, dx) - rotations.get(k);
                    
                    // Spiral function
                    double spiralValue = distance - 
                        spiralTightness.get(k) * angle * Math.sqrt(distance);
                    
                    if (spiralValue < minValue) {
                        minValue = spiralValue;
                        nearestGalaxy = k;
                    }
                }
                
                clusterGrid[i][j] = nearestGalaxy;
            }
        }
        
        calculateBorderCells();
    }

    public void saveGridAsPerlinNoiseClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToPerlinNoiseClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void saveGridAsGalaxyClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToGalaxyClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public void divideGridToMinimalBorderClusters(int numClusters) {
        clearClusterData();
        initializeClusters(0);
        
        // Create graph representation with defensive initialization
        Map<Point, Set<Point>> graph = new HashMap<>();
        Point[][] points = new Point[rows][cols];
        
        // First create all points and store them in array for quick lookup
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                points[i][j] = new Point(i, j);
                graph.put(points[i][j], new HashSet<>());
            }
        }
        
        // Then add neighbors using the stored points
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                Point p = points[i][j];
                Set<Point> neighbors = graph.get(p);
                
                // Add 4-connected neighbors
                int[][] dirs = {{-1,0}, {1,0}, {0,-1}, {0,1}};
                for (int[] dir : dirs) {
                    int ni = i + dir[0];
                    int nj = j + dir[1];
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        neighbors.add(points[ni][nj]);
                    }
                }
            }
        }
        
        // Initialize seeds in a more uniform distribution
        List<Point> seeds = new ArrayList<>();
        int gridSize = (int)Math.ceil(Math.sqrt(numClusters));
        int stepI = rows / gridSize;
        int stepJ = cols / gridSize;
        
        for (int i = stepI/2; i < rows && seeds.size() < numClusters; i += stepI) {
            for (int j = stepJ/2; j < cols && seeds.size() < numClusters; j += stepJ) {
                seeds.add(points[i][j]);
            }
        }

        // Rest of the method remains the same
        // Initialize regions with seeds
        for (int k = 0; k < seeds.size(); k++) {
            Point seed = seeds.get(k);
            clusterGrid[seed.row][seed.col] = k;
        }
        
        // Use priority queue for region growing
        PriorityQueue<PointWeight> queue = new PriorityQueue<>();
        Set<Point> visited = new HashSet<>(seeds);
        
        // Add initial boundaries to queue
        for (Point seed : seeds) {
            for (Point neighbor : graph.get(seed)) {
                if (!visited.contains(neighbor)) {
                    double weight = calculateWeight(seed, neighbor, graph);
                    queue.offer(new PointWeight(neighbor, weight, clusterGrid[seed.row][seed.col]));
                }
            }
        }
        
        // Grow regions
        while (!queue.isEmpty()) {
            PointWeight pw = queue.poll();
            Point current = pw.point;
            
            if (visited.contains(current)) continue;
            
            clusterGrid[current.row][current.col] = pw.sourceCluster;
            visited.add(current);
            
            // Add unvisited neighbors to queue
            for (Point neighbor : graph.get(current)) {
                if (!visited.contains(neighbor)) {
                    double weight = calculateWeight(current, neighbor, graph);
                    queue.offer(new PointWeight(neighbor, weight, pw.sourceCluster));
                }
            }
        }
        
        calculateBorderCells();
    }
    
    private static class PointWeight implements Comparable<PointWeight> {
        Point point;
        double weight;
        int sourceCluster;
        
        PointWeight(Point point, double weight, int sourceCluster) {
            this.point = point;
            this.weight = weight;
            this.sourceCluster = sourceCluster;
        }
        
        @Override
        public int compareTo(PointWeight other) {
            return Double.compare(this.weight, other.weight);
        }
    }
    
    private double calculateWeight(Point p1, Point p2, Map<Point, Set<Point>> graph) {
        // Calculate weight based on connectivity and local density
        double connectivity = graph.getOrDefault(p1, new HashSet<>()).size() + 
                            graph.getOrDefault(p2, new HashSet<>()).size();
        double distance = Math.sqrt(Math.pow(p1.row - p2.row, 2) + Math.pow(p1.col - p2.col, 2));
        // Add small epsilon to avoid division by zero
        return distance / (connectivity + 0.1);
    }

    public void saveGridAsMinimalBorderClusterImage(String filePath, int numClusters, boolean includeDiagonals, boolean drawBorders) throws IOException {
        divideGridToMinimalBorderClusters(numClusters);
        saveClusterImage(filePath, includeDiagonals, drawBorders);
    }

    public static void main(String[] args) {
        int num_clusters = 8;
        int rows = 500;
        int cols = 500;
        boolean drawBorders = true;

        GRID grid = new GRID(rows, cols);
        try {
            grid.analyzeAllClusteringMethods(num_clusters, drawBorders);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}