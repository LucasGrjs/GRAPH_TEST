import java.io.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;
import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

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

public class GRID {
    private CellType[][] grid;
    private int rows;
    private int cols;
    private int[][] clusters;
    private List<Nematode> nematodes;
    
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
        this.grid = new CellType[rows][cols];
        this.nematodes = new ArrayList<>();
        initializeRandomGrid();
        placeNematodes(20);
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
        while (placed < count) {
            int row = random.nextInt(rows);
            int col = random.nextInt(cols);
            if (grid[row][col] == CellType.PORE) {
                nematodes.add(new Nematode(row, col));
                placed++;
            }
        }
    }

    public void printGrid() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(grid[i][j].toString().charAt(0) + " ");
            }
            System.out.println();
        }
    }

    public void saveGridAsImage(String filePath) throws IOException {
        int cellSize = 20;
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

    public void divideToClusters(int N) {
        clusters = new int[rows][cols];
        int totalCells = rows * cols;
        int baseSize = totalCells / N;
        int remainder = totalCells % N;
        
        int clusterNumber = 1;
        int cellsAssigned = 0;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                clusters[i][j] = clusterNumber;
                cellsAssigned++;
                
                if (cellsAssigned == baseSize + (remainder > 0 ? 1 : 0)) {
                    clusterNumber++;
                    cellsAssigned = 0;
                    if (remainder > 0) {
                        remainder--;
                    }
                }
            }
        }
    }

    public void saveGridAsClusterImage(String filePath, int N) throws IOException {
        divideToClusters(N);
        int cellSize = 20;
        BufferedImage image = new BufferedImage(cols * cellSize, rows * cellSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Remove local color array and use global one
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Fill cell with cluster color
                g2d.setColor(clusterColors[(clusters[i][j] - 1) % clusterColors.length]);
                g2d.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                
                // Draw black border
                g2d.setColor(Color.BLACK);
                g2d.drawRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }

        g2d.dispose();
        ImageIO.write(image, "png", new File(filePath));
    }

    public void dividePoresToClusters(int K) {
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

        // Initialize clusters array if not already initialized
        clusters = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            Arrays.fill(clusters[i], -1);
        }

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
                if (clusters[pore.row][pore.col] != nearestCluster) {
                    clusters[pore.row][pore.col] = nearestCluster;
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
    }

    private boolean isBorderCell(int row, int col, boolean includeDiagonals) {
        Set<Integer> adjacentClusters = new HashSet<>();
        int currentCluster = clusters[row][col];
        
        // Check direct neighbors (up, down, left, right)
        int[][] directNeighbors = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] neighbor : directNeighbors) {
            int newRow = row + neighbor[0];
            int newCol = col + neighbor[1];
            
            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                && grid[newRow][newCol] == CellType.PORE 
                && clusters[newRow][newCol] != -1
                && clusters[newRow][newCol] != currentCluster) {
                adjacentClusters.add(clusters[newRow][newCol]);
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
                    && clusters[newRow][newCol] != -1
                    && clusters[newRow][newCol] != currentCluster) {
                    adjacentClusters.add(clusters[newRow][newCol]);
                }
            }
        }

        return adjacentClusters.size() >= 2;
    }

    private boolean isPoreWithDifferentClusterNeighbor(int row, int col, boolean includeDiagonals) {
        if (grid[row][col] != CellType.PORE) {
            return false;
        }
        
        int currentCluster = clusters[row][col];
        
        // Check direct neighbors (up, down, left, right)
        int[][] directNeighbors = {{-1,0}, {1,0}, {0,-1}, {0,1}};
        for (int[] neighbor : directNeighbors) {
            int newRow = row + neighbor[0];
            int newCol = col + neighbor[1];
            
            if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols 
                && grid[newRow][newCol] == CellType.PORE 
                && clusters[newRow][newCol] != -1
                && clusters[newRow][newCol] != currentCluster) {
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
                    && clusters[newRow][newCol] != -1
                    && clusters[newRow][newCol] != currentCluster) {
                    return true;
                }
            }
        }
        
        return false;
    }

    public void saveGridAsPoreClusterImage(String filePath, int K, boolean includeDiagonals) throws IOException {
        dividePoresToClusters(K);
        int cellSize = 20;
        BufferedImage image = new BufferedImage(cols * cellSize, rows * cellSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // Remove local color array and use global one
        // Draw base grid
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == CellType.PORE) {
                    g2d.setColor(clusterColors[clusters[i][j] % clusterColors.length]);
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

        g2d.dispose();
        ImageIO.write(image, "png", new File(filePath));
    }

    // Update main method to test pore clustering
    public static void main(String[] args) {

        int cluster_number = 8;
        GRID grid = new GRID(90, 90);
        grid.printGrid();
        try {
            grid.saveGridAsImage("grid.png");
            // Generate two versions: one with 4-neighbors and one with 8-neighbors
            grid.saveGridAsPoreClusterImage("grid_pore_clusters.png", cluster_number, false);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}