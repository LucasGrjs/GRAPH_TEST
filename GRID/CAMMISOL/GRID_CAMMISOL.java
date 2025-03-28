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

import java.awt.image.RenderedImage;

enum CellType {
    PORE, MINERAL, ORGANIC
}

class Nematode {
    private int row;
    private int col;

    public Nematode(int row, int col) {
        this.row = row;
        this.col = col;
    }

    public int getRow() { return row; }
    public int getCol() { return col; }
    public void setPosition(int row, int col) {
        this.row = row;
        this.col = col;
    }
}
class ClusterStats {
    final int minClusterSize;
    final int maxClusterSize;
    final double avgClusterSize;
    final int borderCellCount;
    final int adjPoreCount;
    final Map<Integer, Integer> borderCellsPerCluster;
    final Map<Integer, Integer> adjPoresPerCluster;
    final int totalPores;
    final double borderPct;
    final double adjPorePct;
    

    ClusterStats(int min, int max, double avg, int borders, int adj, 
                 Map<Integer, Integer> borderPerCluster, Map<Integer, Integer> adjPerCluster,
                 int totalPores, double borderPct, double adjPorePct) {
        minClusterSize = min;
        maxClusterSize = max;
        avgClusterSize = avg;
        borderCellCount = borders;
        adjPoreCount = adj;
        borderCellsPerCluster = borderPerCluster;
        adjPoresPerCluster = adjPerCluster;
        this.totalPores = totalPores;
        this.borderPct = borderPct;
        this.adjPorePct = adjPorePct;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "Cluster Stats:\n" +
            "Total Pores: %d\n" +
            "Min cells per Cluster: %d\n" +
            "Max cells per Cluster: %d\n" +
            "Avg cells per Cluster: %.2f\n" +
            "Border Cells: %d (%.2f%%)\n" +
            "Adjacent Pores: %d (%.2f%%)\n\n",
            totalPores, minClusterSize, maxClusterSize, avgClusterSize, 
            borderCellCount, borderPct, adjPoreCount, adjPorePct));

        sb.append("Per Cluster Stats:\n");
        borderCellsPerCluster.forEach((k, v) -> 
            sb.append(String.format("Cluster %d: %d pores, %d border cells, %d adjacent pores\n", 
                k, 
                borderCellsPerCluster.containsKey(k) ? borderCellsPerCluster.get(k) : 0,
                v, 
                adjPoresPerCluster.getOrDefault(k, 0))));

        return sb.toString();
    }
}

public class GRID_CAMMISOL {
    private CellType[][] grid;
    private int rows;
    private int cols;
    private List<Nematode> nematodes;
    private Integer[][] clusterGrid;
    private List<Point> borderCells = new ArrayList<>();
    private List<Point> adjPores = new ArrayList<>();
    private Map<Point, Set<Integer>> borderCellClusters = new HashMap<>();
    private Map<Point, Set<Integer>> adjPoreClusters = new HashMap<>();  // Add this field
    
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

    public GRID_CAMMISOL(int rows, int cols, int nematodes) {
        this.rows = rows;
        this.cols = cols;
        this.grid = new CellType[rows][cols];
        this.nematodes = new ArrayList<>();
        initializeRandomGrid();
        placeNematodes(nematodes);
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
    }

    private void initializeRandomGrid() {
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int randomNum = random.nextInt(3);
                switch (randomNum) {
                    case 0:
                        grid[i][j] = CellType.PORE;
                        break;
                    case 1:
                        grid[i][j] = CellType.MINERAL;
                        break;
                    case 2:
                        grid[i][j] = CellType.ORGANIC;
                        break;
                }
            }
        }
    }

    private void placeNematodes(int count) {
        Random random = new Random();
        int placed = 0;
        System.out.println(count);
        while (placed < count) {
            int row = random.nextInt(rows);
            int col = random.nextInt(cols);
            if (grid[row][col] == CellType.PORE) {
                nematodes.add(new Nematode(row, col));
                placed++;
            }
        }
    }

    public void saveGridAsImage(String filePath) throws IOException {
        int cellSize = 10;
        BufferedImage image = new BufferedImage(cols * cellSize, rows * cellSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Draw cells
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                switch (grid[i][j]) {
                    case PORE:
                        g2d.setColor(Color.BLACK);
                        break;
                    case MINERAL:
                        g2d.setColor(Color.YELLOW);
                        break;
                    case ORGANIC:
                        g2d.setColor(Color.GREEN);
                        break;
                }
                g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                g2d.setColor(Color.BLACK);
                g2d.drawRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }

        // Draw nematodes as red dots
        g2d.setColor(Color.RED);
        for (Nematode nematode : nematodes) {
            int x = nematode.getCol() * cellSize + cellSize/2;
            int y = nematode.getRow() * cellSize + cellSize/2;
            g2d.fillOval(x - 5, y - 5, 10, 10);
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
        adjPores.clear();
        borderCellClusters.clear();
        adjPoreClusters.clear();
        clusterGrid = null;
    }

    public Map<Integer, Integer> getClusterSizes() {
        Map<Integer, Integer> sizes = new HashMap<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == CellType.PORE && clusterGrid[i][j] != -1) {
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

    private Map<Integer, Integer> getAdjPoresPerCluster() {
        Map<Integer, Integer> counts = new HashMap<>();
        for (Point p : adjPores) {
            Set<Integer> clusters = getAdjPoreClusters(p);
            for (Integer cluster : clusters) {
                counts.merge(cluster, 1, Integer::sum);
            }
        }
        return counts;
    }

    private int getTotalPores() {
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == CellType.PORE) {
                    count++;
                }
            }
        }
        return count;
    }

    public ClusterStats getClusterStats() {
        Map<Integer, Integer> sizes = getClusterSizes();
        int totalPores = getTotalPores();
        int minSize = sizes.values().stream().mapToInt(i -> i).min().orElse(0);
        int maxSize = sizes.values().stream().mapToInt(i -> i).max().orElse(0);
        double avgSize = sizes.values().stream().mapToInt(i -> i).average().orElse(0);
        
        Map<Integer, Integer> borderPerCluster = getBorderCellsPerCluster();
        Map<Integer, Integer> adjPerCluster = getAdjPoresPerCluster();

        // Calculate percentage of pores that are border cells or adjacent pores
        double borderPct = (double)borderCells.size() / totalPores * 100;
        double adjPct = (double)adjPores.size() / totalPores * 100;

        return new ClusterStats(
            minSize, maxSize, avgSize,
            borderCells.size(), adjPores.size(),
            borderPerCluster, adjPerCluster,
            totalPores, borderPct, adjPct
        );
    }

    public void dividePoresToClusters(int K) {
        clearClusterData();
        // Get all pore cells
        List<Point> poreCells = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == CellType.PORE) {
                    poreCells.add(new Point(i, j));
                }
            }
        }

        if (poreCells.isEmpty()) return;

        // Initialize cluster grid
        initializeClusters(K);

        // Initialize K random centroids from pore cells
        Random random = new Random();
        List<Point> centroids = new ArrayList<>();
        Set<Integer> usedIndices = new HashSet<>();
        for (int i = 0; i < K && i < poreCells.size(); i++) {
            int index;
            do {
                index = random.nextInt(poreCells.size());
            } while (!usedIndices.add(index));
            centroids.add(poreCells.get(index));
        }

        // K-means iteration
        boolean changed;
        int maxIterations = 100;
        do {
            changed = false;
            // Assign each pore cell to nearest centroid
            Map<Integer, List<Point>> clusterPoints = new HashMap<>();
            for (Point pore : poreCells) {
                int nearestCluster = 0;
                double minDistance = Double.MAX_VALUE;
                
                for (int i = 0; i < centroids.size(); i++) {
                    double distance = pore.distanceTo(centroids.get(i));
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCluster = i;
                    }
                }

                clusterPoints.computeIfAbsent(nearestCluster, k -> new ArrayList<>()).add(pore);
                int oldCluster = clusterGrid[pore.row][pore.col];
                if (oldCluster != nearestCluster) {
                    clusterGrid[pore.row][pore.col] = nearestCluster;
                    changed = true;
                }
            }

            // Update centroids
            for (int i = 0; i < K; i++) {
                List<Point> clusterPores = clusterPoints.getOrDefault(i, new ArrayList<>());
                if (!clusterPores.isEmpty()) {
                    double avgRow = clusterPores.stream().mapToInt(p -> p.row).average().orElse(0);
                    double avgCol = clusterPores.stream().mapToInt(p -> p.col).average().orElse(0);
                    centroids.set(i, new Point((int)avgRow, (int)avgCol));
                }
            }

            maxIterations--;
        } while (changed && maxIterations > 0);

        // Count active clusters and remap indices
        Set<Integer> activeClusters = new HashSet<>();
        for (Point p : poreCells) {
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
        for (Point p : poreCells) {
            if (clusterGrid[p.row][p.col] != -1) {
                clusterGrid[p.row][p.col] = indexMap.get(clusterGrid[p.row][p.col]);
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
                && grid[newRow][newCol] == CellType.PORE 
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
                    && grid[newRow][newCol] == CellType.PORE 
                    && clusterGrid[newRow][newCol] != -1
                    && clusterGrid[newRow][newCol] != currentCluster) {
                    adjacentClusters.add(clusterGrid[newRow][newCol]);
                }
            }
        }

        if(adjacentClusters.size() >= 2) {
            Point p = new Point(row, col);
            borderCells.add(p);
            borderCellClusters.put(p, adjacentClusters);
        }
        return adjacentClusters.size() >= 2;
    }

    private boolean isPoreWithDifferentClusterNeighbor(int row, int col, boolean includeDiagonals) {
        if (grid[row][col] != CellType.PORE) {
            return false;
        }
        
        int currentCluster = clusterGrid[row][col];
        Set<Integer> neighborClusters = new HashSet<>();
        neighborClusters.add(currentCluster);
        
        // Check direct neighbors (up, down, left, right)
        int[][] directNeighbors = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] neighbor : directNeighbors) {
            int newRow = row + neighbor[0];
            int newCol = col + neighbor[1];
            
            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                && grid[newRow][newCol] == CellType.PORE 
                && clusterGrid[newRow][newCol] != -1
                && !clusterGrid[newRow][newCol].equals(currentCluster)) {
                neighborClusters.add(clusterGrid[newRow][newCol]);
                Point p = new Point(row, col);
                adjPores.add(p);
                adjPoreClusters.put(p, neighborClusters);
                return true;
            }
        }
        
        // Check diagonal neighbors if requested
        if (includeDiagonals) {
            int[][] diagonalNeighbors = {{-1,-1}, {-1,1}, {1,-1}, {1,1}};
            for (int[] neighbor : diagonalNeighbors) {
                int newRow = row + neighbor[0];
                int newCol = col + neighbor[1];
                
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                    && grid[newRow][newCol] == CellType.PORE 
                    && clusterGrid[newRow][newCol] != -1
                    && clusterGrid[newRow][newCol] != currentCluster) {
                    neighborClusters.add(clusterGrid[newRow][newCol]);
                    Point p = new Point(row, col);
                    adjPores.add(p);
                    adjPoreClusters.put(p, neighborClusters);
                    return true;
                }
            }
        }
        return false;
    }

    public void saveGridAsPoreClusterImage(String filePath, int K, boolean includeDiagonals) throws IOException {

        dividePoresToClusters(K);
        int cellSize = 10;

        BufferedImage image = new BufferedImage(cols * cellSize, rows * cellSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Remove local color array and use global one
        // Draw base grid
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == CellType.PORE) {
                    g2d.setColor(clusterColors[clusterGrid[i][j] % clusterColors.length]);
                } else {
                    if (isBorderCell(i, j, includeDiagonals)) {
                        g2d.setColor(Color.BLACK);
                    }else{
                        g2d.setColor(Color.WHITE);
                    }
                }
                g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

                // Draw cell border
                if (grid[i][j] == CellType.PORE && isPoreWithDifferentClusterNeighbor(i, j, includeDiagonals)) {
                    g2d.setColor(Color.RED);
                    g2d.setStroke(new BasicStroke(2.0f));
                } else {
                    g2d.setColor(Color.BLACK);
                    g2d.setStroke(new BasicStroke(1.0f));
                }
                g2d.drawRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }

        g2d.setColor(Color.RED);
        for (Nematode nematode : nematodes) {
            int x = nematode.getCol() * cellSize + cellSize/2;
            int y = nematode.getRow() * cellSize + cellSize/2;
            g2d.fillOval(x - 5, y - 5, 10, 10);
        }
        g2d.dispose();
        ImageIO.write(image, "png", new File(filePath));
    }

    // Add method to get clusters for a border cell
    public Set<Integer> getBorderCellClusters(Point p) {
        return borderCellClusters.getOrDefault(p, new HashSet<>());
    }

    // Add method to get clusters for adjacent pores
    public Set<Integer> getAdjPoreClusters(Point p) {
        return adjPoreClusters.getOrDefault(p, new HashSet<>());
    }

    private void createTransitionGif(String outputPath, BufferedImage img1, BufferedImage img2, int frames) throws IOException {
        ImageOutputStream output = new FileImageOutputStream(new File(outputPath));
        GifSequenceWriter writer = new GifSequenceWriter(output, img1.getType(), 100, true);

        // Generate transition frames
        for (int i = 0; i <= frames; i++) {
            float alpha = (float) i / frames;
            BufferedImage blended = new BufferedImage(cols * 10, rows * 10, BufferedImage.TYPE_INT_RGB);
            Graphics2D g = blended.createGraphics();
            
            g.drawImage(img1, 0, 0, null);
            g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
            g.drawImage(img2, 0, 0, null);
            g.dispose();
            
            writer.writeToSequence(blended);
        }
        
        // Add 20 extra frames of the final state to hold it longer
        for (int i = 0; i < 10; i++) {
            writer.writeToSequence(img2);
        }
        
        writer.close();
        output.close();
    }

    private static class GifSequenceWriter {
        protected ImageWriter writer;
        protected ImageWriteParam params;
        protected IIOMetadata metadata;

        public GifSequenceWriter(ImageOutputStream out, int imageType, int delay, boolean loop) throws IOException {
            writer = ImageIO.getImageWritersBySuffix("gif").next();
            params = writer.getDefaultWriteParam();

            ImageTypeSpecifier imageTypeSpecifier = ImageTypeSpecifier.createFromBufferedImageType(imageType);
            metadata = writer.getDefaultImageMetadata(imageTypeSpecifier, params);

            configureRootMetadata(delay, loop);

            writer.setOutput(out);
            writer.prepareWriteSequence(null);
        }

        private void configureRootMetadata(int delay, boolean loop) throws IOException {
            String metaFormatName = metadata.getNativeMetadataFormatName();
            IIOMetadataNode root = (IIOMetadataNode) metadata.getAsTree(metaFormatName);

            IIOMetadataNode graphicsControlExtensionNode = getNode(root, "GraphicControlExtension");
            graphicsControlExtensionNode.setAttribute("disposalMethod", "none");
            graphicsControlExtensionNode.setAttribute("userInputFlag", "FALSE");
            graphicsControlExtensionNode.setAttribute("transparentColorFlag", "FALSE");
            graphicsControlExtensionNode.setAttribute("delayTime", Integer.toString(delay / 10));
            graphicsControlExtensionNode.setAttribute("transparentColorIndex", "0");

            IIOMetadataNode appExtensionsNode = getNode(root, "ApplicationExtensions");
            IIOMetadataNode child = new IIOMetadataNode("ApplicationExtension");
            child.setAttribute("applicationID", "NETSCAPE");
            child.setAttribute("authenticationCode", "2.0");
            child.setUserObject(new byte[]{1, (byte) (loop ? 0 : 1), 0});
            appExtensionsNode.appendChild(child);

            metadata.setFromTree(metaFormatName, root);
        }

        private static IIOMetadataNode getNode(IIOMetadataNode rootNode, String nodeName) {
            int nNodes = rootNode.getLength();
            for (int i = 0; i < nNodes; i++) {
                if (rootNode.item(i).getNodeName().equalsIgnoreCase(nodeName)) {
                    return (IIOMetadataNode) rootNode.item(i);
                }
            }
            IIOMetadataNode node = new IIOMetadataNode(nodeName);
            rootNode.appendChild(node);
            return node;
        }

        public void writeToSequence(RenderedImage img) throws IOException {
            writer.writeToSequence(new IIOImage(img, null, metadata), params);
        }

        public void close() throws IOException {
            writer.endWriteSequence();
        }
    }

    // Add these new methods
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
                if (grid[i][j] == CellType.PORE && clusterGrid[i][j] != -1) {
                    int clusterId = clusterGrid[i][j];
                    int cellId = getCellId(i, j);
                    clusterMap.computeIfAbsent(clusterId, k -> new ArrayList<>()).add(cellId);
                }
            }
        }
        
        // Convert map to list of lists
        return new ArrayList<>(clusterMap.values());
    }

    // Example usage in main method:
    public static void main(String[] args) {
        int cluster_number = 16;
        int nematodes = 0;
        int rows = 100;
        int cols = 100;

        GRID_CAMMISOL grid = new GRID_CAMMISOL(rows, cols, nematodes);
        try {
            grid.saveGridAsImage("grid.png");
            grid.saveGridAsPoreClusterImage("grid_pore_clusters.png", cluster_number, false);
            
            // Get clustering as lists of cell IDs
            List<List<Integer>> clusters = grid.getClusteringAsLists();
            System.out.println("Clusters:");
            for (int i = 0; i < clusters.size(); i++) {
                System.out.printf("Cluster %d: %s%n", i, clusters.get(i));
            }
            
            System.out.println(grid.getClusterStats());
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}