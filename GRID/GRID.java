import java.io.*;
import java.util.List;
import java.util.ArrayList;
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

    public GRID(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.grid = new CellType[rows][cols];
        this.nematodes = new ArrayList<>();
        initializeRandomGrid();
        placeNematodes(20);
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

        // Define colors for clusters
        Color[] clusterColors = {
            Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, 
            Color.CYAN, Color.MAGENTA, Color.ORANGE, Color.PINK,
            Color.LIGHT_GRAY, Color.DARK_GRAY
        };

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

    public static void main(String[] args) {
        GRID grid = new GRID(30, 30);
        grid.printGrid();
        try {
            grid.saveGridAsImage("grid.png");
            grid.saveGridAsClusterImage("grid_clusters.png", 8); // Divide into 8 clusters
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}