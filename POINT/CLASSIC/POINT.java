import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.Graphics2D;
import javax.imageio.ImageIO;
import java.util.List;
import java.util.Arrays;
import java.awt.BasicStroke;
import java.awt.RenderingHints;
import java.util.Stack;
import java.util.PriorityQueue;
import java.util.Collections;
import java.util.Comparator;

public class POINT {
    private double x;
    private double y;
    private int index;
    private static final Random random = new Random();
    
    public POINT(double x, double y, int index) {
        this.x = x;
        this.y = y;
        this.index = index;
    }
    
    public double getX() { return x; }
    public double getY() { return y; }
    public int getIndex() { return index; }
    public void setX(double x) { this.x = x; }
    public void setY(double y) { this.y = y; }
    
    // Generate K random points within N x M space
    public static ArrayList<POINT> generateRandomPoints(int K, double N, double M) {
        ArrayList<POINT> points = new ArrayList<>();
        for (int i = 0; i < K; i++) {
            double x = random.nextDouble() * N;
            double y = random.nextDouble() * M;
            points.add(new POINT(x, y, i));
        }
        return points;
    }
    
    public static void generateImage(ArrayList<POINT> points, double width, double height, String filename) {
        int imgWidth = 800;  // output image width
        int imgHeight = 800; // output image height
        
        BufferedImage image = new BufferedImage(imgWidth, imgHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        
        // Set white background
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, imgWidth, imgHeight);
        
        // Draw points in blue
        g2d.setColor(Color.BLUE);
        for (POINT p : points) {
            int x = (int)((p.getX() / width) * imgWidth);
            int y = (int)((p.getY() / height) * imgHeight);
            g2d.fillOval(x-2, y-2, 4, 4);  // Draw 4x4 pixel point
        }
        
        g2d.dispose();
        
        try {
            File output = new File(filename);
            ImageIO.write(image, "PNG", output);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    // Inner class to represent a cluster
    private static class Cluster {
        POINT centroid;
        ArrayList<Integer> pointIndices;
        Color color;
        
        public Cluster(POINT centroid) {
            this.centroid = centroid;
            this.pointIndices = new ArrayList<>();
            this.color = new Color(random.nextFloat(), random.nextFloat(), random.nextFloat());
        }
    }
    
    public static Cluster[] kMeansClustering(ArrayList<POINT> points, int k, int maxIterations) {
        // Initialize k clusters using k-means++ initialization
        Cluster[] clusters = new Cluster[k];
        
        // Choose first centroid randomly
        clusters[0] = new Cluster(points.get(random.nextInt(points.size())));
        
        // Choose remaining centroids
        for (int i = 1; i < k; i++) {
            double[] distances = new double[points.size()];
            double totalDistance = 0;
            
            // Calculate minimum distance to existing centroids for each point
            for (int j = 0; j < points.size(); j++) {
                double minDist = Double.MAX_VALUE;
                for (int c = 0; c < i; c++) {
                    double dist = distanceTo(points.get(j), clusters[c].centroid);
                    minDist = Math.min(minDist, dist);
                }
                distances[j] = minDist * minDist; // Square distances for weighting
                totalDistance += distances[j];
            }
            
            // Choose next centroid with probability proportional to squared distance
            double rand = random.nextDouble() * totalDistance;
            double sum = 0;
            for (int j = 0; j < points.size(); j++) {
                sum += distances[j];
                if (sum >= rand) {
                    clusters[i] = new Cluster(points.get(j));
                    break;
                }
            }
        }
        
        // Continue with regular k-means algorithm
        boolean changed = true;
        int iteration = 0;
        
        while (changed && iteration < maxIterations) {
            changed = false;
            iteration++;
            
            // Clear current clusters
            for (Cluster cluster : clusters) {
                cluster.pointIndices.clear();
            }
            
            // Assign points to nearest centroid
            for (POINT point : points) {
                double minDistance = Double.MAX_VALUE;
                Cluster nearestCluster = null;
                
                for (Cluster cluster : clusters) {
                    double distance = distanceTo(point, cluster.centroid);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCluster = cluster;
                    }
                }
                
                nearestCluster.pointIndices.add(point.getIndex());
            }
            
            // Recalculate centroids
            for (Cluster cluster : clusters) {
                if (cluster.pointIndices.isEmpty()) continue;
                
                ArrayList<POINT> clusterPoints = new ArrayList<>();
                for (int index : cluster.pointIndices) {
                    clusterPoints.add(points.get(index));
                }
                POINT newCentroid = calculateCentroid(clusterPoints);
                if (!newCentroid.equals(cluster.centroid)) {
                    cluster.centroid = newCentroid;
                    changed = true;
                }
            }
        }
        
        return clusters;
    }
    
    private static double distanceTo(POINT p1, POINT p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    private static POINT calculateCentroid(ArrayList<POINT> points) {
        double sumX = 0, sumY = 0;
        for (POINT p : points) {
            sumX += p.x;
            sumY += p.y;
        }
        return new POINT(sumX / points.size(), sumY / points.size(), -1);
    }

    public static void generateClusteredImage(Cluster[] clusters, ArrayList<POINT> allPoints, double width, double height, String filename) {
        int imgWidth = 800;
        int imgHeight = 800;
        
        BufferedImage image = new BufferedImage(imgWidth, imgHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, imgWidth, imgHeight);
        
        // Generate visually distinct colors for clusters
        Color[] distinctColors = generateDistinctColors(clusters.length);
        for (int i = 0; i < clusters.length; i++) {
            clusters[i].color = distinctColors[i];
        }
        
        // Draw points with cluster colors
        for (Cluster cluster : clusters) {
            g2d.setColor(cluster.color);
            for (int index : cluster.pointIndices) {
                POINT p = allPoints.get(index);
                int x = (int)((p.getX() / width) * imgWidth);
                int y = (int)((p.getY() / height) * imgHeight);
                g2d.fillOval(x-4, y-4, 8, 8); // Larger points for better visibility
            }
            
            // Draw centroid as a larger point
            int cx = (int)((cluster.centroid.getX() / width) * imgWidth);
            int cy = (int)((cluster.centroid.getY() / height) * imgHeight);
            g2d.setColor(Color.BLACK);
            g2d.fillOval(cx-6, cy-6, 12, 12);
            g2d.setColor(cluster.color);
            g2d.fillOval(cx-4, cy-4, 8, 8);
        }
        
        g2d.dispose();
        
        try {
            // Create algorithm-specific directory if it doesn't exist
            File directory = new File(new File(filename).getParent());
            if (!directory.exists()) {
                directory.mkdirs();
            }
            ImageIO.write(image, "PNG", new File(filename));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Color[] generateDistinctColors(int n) {
        Color[] colors = new Color[n];
        for (int i = 0; i < n; i++) {
            float hue = i * (1.0f / n);
            colors[i] = Color.getHSBColor(hue, 0.8f, 1.0f);
        }
        return colors;
    }

    @Override
    public String toString() {
        return String.format("Point(%.2f, %.2f)", x, y);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (obj instanceof POINT) {
            POINT other = (POINT) obj;
            return this.x == other.x && this.y == other.y;
        }
        return false;
    }
    
    private static ArrayList<POINT> generateCircle(double centerX, double centerY, double radius, int points, double noise, int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        for (int i = 0; i < points; i++) {
            double angle = 2.0 * Math.PI * i / points;
            double r = radius + random.nextGaussian() * noise;
            double x = centerX + r * Math.cos(angle);
            double y = centerY + r * Math.sin(angle);
            result.add(new POINT(x, y, startIndex + i));
        }
        return result;
    }

    private static ArrayList<POINT> generateHalfMoon(double centerX, double centerY, double radius, 
                                                   double angle, int points, double width, int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        for (int i = 0; i < points; i++) {
            double theta = Math.PI * i / points + angle;
            double r = radius + random.nextGaussian() * width;
            double x = centerX + r * Math.cos(theta);
            double y = centerY + r * Math.sin(theta);
            result.add(new POINT(x, y, startIndex + i));
        }
        return result;
    }

    private static ArrayList<POINT> generateBlob(double centerX, double centerY, double radius, int points, int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        for (int i = 0; i < points; i++) {
            double angle = 2.0 * Math.PI * random.nextDouble();
            double r = radius * Math.sqrt(random.nextDouble());
            double x = centerX + r * Math.cos(angle);
            double y = centerY + r * Math.sin(angle);
            result.add(new POINT(x, y, startIndex + i));
        }
        return result;
    }

    private static ArrayList<POINT> generateChain(double startX, double startY, 
                                                double endX, double endY, 
                                                int points, double width, int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        double dx = endX - startX;
        double dy = endY - startY;
        for (int i = 0; i < points; i++) {
            double t = (double) i / points;
            double x = startX + dx * t + random.nextGaussian() * width;
            double y = startY + dy * t + random.nextGaussian() * width;
            result.add(new POINT(x, y, startIndex + i));
        }
        return result;
    }

    private static ArrayList<POINT> generateCheckerboard(double minX, double maxX, 
                                                       double minY, double maxY, 
                                                       int rows, int cols, 
                                                       int pointsPerCell, int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        double cellWidth = (maxX - minX) / cols;
        double cellHeight = (maxY - minY) / rows;
        int index = startIndex;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if ((i + j) % 2 == 0) {
                    double cellX = minX + j * cellWidth;
                    double cellY = minY + i * cellHeight;
                    for (int k = 0; k < pointsPerCell; k++) {
                        double x = cellX + random.nextDouble() * cellWidth;
                        double y = cellY + random.nextDouble() * cellHeight;
                        result.add(new POINT(x, y, index++));
                    }
                }
            }
        }
        return result;
    }

    private static ArrayList<POINT> generateSpiral(double centerX, double centerY, 
                                                 double maxRadius, int points, 
                                                 double startAngle, int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        for (int i = 0; i < points; i++) {
            double t = (double) i / points;
            double angle = 4 * Math.PI * t + startAngle;
            double radius = maxRadius * t;
            double x = centerX + radius * Math.cos(angle);
            double y = centerY + radius * Math.sin(angle);
            result.add(new POINT(x + random.nextGaussian() * 5, 
                               y + random.nextGaussian() * 5, 
                               startIndex + i));
        }
        return result;
    }

    private static ArrayList<POINT> generateUniformCoverage(double minX, double maxX, 
                                                          double minY, double maxY, 
                                                          double minDist, int startIndex) {
        ArrayList<POINT> points = new ArrayList<>();
        ArrayList<POINT> active = new ArrayList<>();
        double cellSize = minDist / Math.sqrt(2);
        
        int cols = (int) Math.ceil((maxX - minX) / cellSize);
        int rows = (int) Math.ceil((maxY - minY) / cellSize);
        POINT[][] grid = new POINT[cols][rows];
        
        // Generate first point
        POINT first = new POINT(
            minX + random.nextDouble() * (maxX - minX),
            minY + random.nextDouble() * (maxY - minY),
            startIndex
        );
        points.add(first);
        active.add(first);
        grid[(int)((first.x - minX) / cellSize)][(int)((first.y - minY) / cellSize)] = first;
        
        // Generate other points
        while (!active.isEmpty()) {
            int index = random.nextInt(active.size());
            POINT point = active.get(index);
            boolean found = false;
            
            for (int i = 0; i < 30; i++) { // Try 30 candidates
                double angle = random.nextDouble() * Math.PI * 2;
                double r = minDist + random.nextDouble() * minDist;
                double newX = point.x + r * Math.cos(angle);
                double newY = point.y + r * Math.sin(angle);
                
                if (newX < minX || newX >= maxX || newY < minY || newY >= maxY) continue;
                
                int gridX = (int)((newX - minX) / cellSize);
                int gridY = (int)((newY - minY) / cellSize);
                
                if (isValidPoint(newX, newY, minDist, grid, cellSize, minX, minY, cols, rows)) {
                    POINT newPoint = new POINT(newX, newY, startIndex + points.size());
                    points.add(newPoint);
                    active.add(newPoint);
                    grid[gridX][gridY] = newPoint;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                active.remove(index);
            }
        }
        
        return points;
    }
    
    private static boolean isValidPoint(double x, double y, double minDist, 
                                      POINT[][] grid, double cellSize,
                                      double minX, double minY, int cols, int rows) {
        int gridX = (int)((x - minX) / cellSize);
        int gridY = (int)((y - minY) / cellSize);
        
        for (int i = Math.max(0, gridX - 2); i < Math.min(cols, gridX + 3); i++) {
            for (int j = Math.max(0, gridY - 2); j < Math.min(rows, gridY + 3); j++) {
                if (grid[i][j] != null) {
                    double dx = grid[i][j].x - x;
                    double dy = grid[i][j].y - y;
                    if (dx * dx + dy * dy < minDist * minDist) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    private static ArrayList<POINT> generateFractal(double centerX, double centerY, 
                                                  double size, int depth, int points, 
                                                  int startIndex) {
        ArrayList<POINT> result = new ArrayList<>();
        if (depth == 0) {
            result.addAll(generateBlob(centerX, centerY, size/4, points, startIndex));
            return result;
        }
        
        double newSize = size / 3;
        int newPoints = points / 5;
        
        // Center
        result.addAll(generateFractal(centerX, centerY, newSize, depth-1, newPoints, startIndex + result.size()));
        
        // Corners
        double offset = size / 2;
        result.addAll(generateFractal(centerX - offset, centerY - offset, newSize, depth-1, newPoints, startIndex + result.size()));
        result.addAll(generateFractal(centerX + offset, centerY - offset, newSize, depth-1, newPoints, startIndex + result.size()));
        result.addAll(generateFractal(centerX - offset, centerY + offset, newSize, depth-1, newPoints, startIndex + result.size()));
        result.addAll(generateFractal(centerX + offset, centerY + offset, newSize, depth-1, newPoints, startIndex + result.size()));
        
        return result;
    }

    public static void main(String[] args) {
        int width = 1000;
        int height = 1000;
        
        String[] testCases = {
            "concentric_circles",
            "half_moons",
            "three_blobs",
            "chains",
            "separated_blobs",
            "checkerboard",
            "nested_squares",
            "spiral_triple",
            "diamond_pattern",
            "cross_pattern",
            "gaussian_clusters",
            "ring_blobs",
            "star_pattern",
            "double_spiral",
            "sine_wave",
            "nested_triangles",
            "grid_pattern",
            "random_density",
            "concentric_squares",
            "x_pattern",
            "uniform_coverage",
            "fractal_clusters",
            "text_cluster",
            "fibonacci_spiral",
            "olympic_rings",
            "heart_shape",
            "butterfly",
            "galaxy_spiral",
            "voronoi_centers",
            "noise_gradient",
            "mandala"
        };
        
        for (String testCase : testCases) {
            ArrayList<POINT> points = new ArrayList<>();
            
            switch (testCase) {
                case "concentric_circles":
                    points.addAll(generateCircle(500, 500, 100, 200, 10, points.size()));
                    points.addAll(generateCircle(500, 500, 200, 300, 10, points.size()));
                    points.addAll(generateCircle(500, 500, 300, 400, 10, points.size()));
                    break;
                    
                case "half_moons":
                    points.addAll(generateHalfMoon(500, 400, 200, 0, 300, 20, points.size()));
                    points.addAll(generateHalfMoon(500, 600, 200, Math.PI, 300, 20, points.size()));
                    break;
                    
                case "three_blobs":
                    points.addAll(generateBlob(300, 300, 100, 200, points.size()));
                    points.addAll(generateBlob(700, 300, 100, 200, points.size()));
                    points.addAll(generateBlob(500, 700, 100, 200, points.size()));
                    break;
                    
                case "chains":
                    points.addAll(generateChain(200, 200, 800, 400, 200, 20, points.size()));
                    points.addAll(generateChain(200, 600, 800, 800, 200, 20, points.size()));
                    break;
                    
                case "separated_blobs":
                    points.addAll(generateBlob(300, 500, 150, 300, points.size()));
                    points.addAll(generateBlob(700, 500, 150, 300, points.size()));
                    break;
                    
                case "checkerboard":
                    points.addAll(generateCheckerboard(100, 900, 100, 900, 4, 4, 50, points.size()));
                    break;
                    
                case "nested_squares":
                    for (int i = 0; i < 3; i++) {
                        double size = 150 + i * 150;
                        points.addAll(generateChain(500-size, 500-size, 500+size, 500-size, 100, 10, points.size()));
                        points.addAll(generateChain(500+size, 500-size, 500+size, 500+size, 100, 10, points.size()));
                        points.addAll(generateChain(500+size, 500+size, 500-size, 500+size, 100, 10, points.size()));
                        points.addAll(generateChain(500-size, 500+size, 500-size, 500-size, 100, 10, points.size()));
                    }
                    break;
                    
                case "spiral_triple":
                    points.addAll(generateSpiral(500, 300, 200, 200, Math.PI/2, points.size()));
                    points.addAll(generateSpiral(300, 700, 200, 200, Math.PI, points.size()));
                    points.addAll(generateSpiral(700, 700, 200, 200, 0, points.size()));
                    break;
                    
                case "diamond_pattern":
                    points.addAll(generateBlob(500, 300, 80, 100, points.size())); // top
                    points.addAll(generateBlob(300, 500, 80, 100, points.size())); // left
                    points.addAll(generateBlob(700, 500, 80, 100, points.size())); // right
                    points.addAll(generateBlob(500, 700, 80, 100, points.size())); // bottom
                    break;
                    
                case "cross_pattern":
                    points.addAll(generateChain(200, 500, 800, 500, 300, 20, points.size())); // horizontal
                    points.addAll(generateChain(500, 200, 500, 800, 300, 20, points.size())); // vertical
                    break;
                    
                case "gaussian_clusters":
                    double[][] centers = {{300,300}, {700,300}, {500,700}, {300,700}, {700,700}};
                    for (double[] center : centers) {
                        for (int i = 0; i < 100; i++) {
                            double x = center[0] + random.nextGaussian() * 50;
                            double y = center[1] + random.nextGaussian() * 50;
                            points.add(new POINT(x, y, points.size()));
                        }
                    }
                    break;
                    
                case "ring_blobs":
                    int numBlobs = 6;
                    for (int i = 0; i < numBlobs; i++) {
                        double angle = 2 * Math.PI * i / numBlobs;
                        double x = 500 + Math.cos(angle) * 300;
                        double y = 500 + Math.sin(angle) * 300;
                        points.addAll(generateBlob(x, y, 50, 100, points.size()));
                    }
                    break;

                case "star_pattern":
                    int arms = 5;
                    for (int i = 0; i < arms; i++) {
                        double angle = 2 * Math.PI * i / arms;
                        points.addAll(generateChain(500, 500, 
                            500 + Math.cos(angle) * 400,
                            500 + Math.sin(angle) * 400,
                            150, 15, points.size()));
                    }
                    break;

                case "double_spiral":
                    points.addAll(generateSpiral(400, 500, 300, 300, 0, points.size()));
                    points.addAll(generateSpiral(600, 500, 300, 300, Math.PI, points.size()));
                    break;

                case "sine_wave":
                    for (int i = 0; i < 800; i++) {
                        double x = 100 + i;
                        double y = 500 + Math.sin(i * 0.02) * 200;
                        points.add(new POINT(x, y + random.nextGaussian() * 20, points.size()));
                    }
                    break;

                case "nested_triangles":
                    for (int i = 0; i < 3; i++) {
                        double size = 150 + i * 100;
                        for (int j = 0; j < 3; j++) {
                            double angle = 2 * Math.PI * j / 3;
                            double nextAngle = 2 * Math.PI * ((j + 1) % 3) / 3;
                            points.addAll(generateChain(
                                500 + Math.cos(angle) * size,
                                500 + Math.sin(angle) * size,
                                500 + Math.cos(nextAngle) * size,
                                500 + Math.sin(nextAngle) * size,
                                100, 10, points.size()));
                        }
                    }
                    break;

                case "grid_pattern":
                    int gridSize = 4;
                    for (int i = 0; i < gridSize; i++) {
                        for (int j = 0; j < gridSize; j++) {
                            points.addAll(generateBlob(
                                200 + (600 * i / (gridSize-1)),
                                200 + (600 * j / (gridSize-1)),
                                30, 50, points.size()));
                        }
                    }
                    break;

                case "random_density":
                    for (int i = 0; i < 5; i++) {
                        points.addAll(generateBlob(
                            200 + random.nextDouble() * 600,
                            200 + random.nextDouble() * 600,
                            50 + random.nextDouble() * 100,
                            50 + random.nextInt(150),
                            points.size()));
                    }
                    break;

                case "concentric_squares":
                    for (int size = 100; size <= 300; size += 100) {
                        for (double t = 0; t < 1; t += 0.01) {
                            if (t < 0.25) {
                                points.add(new POINT(500 - size + t * 4 * size, 500 - size, points.size()));
                            } else if (t < 0.5) {
                                points.add(new POINT(500 + size, 500 - size + (t - 0.25) * 4 * size, points.size()));
                            } else if (t < 0.75) {
                                points.add(new POINT(500 + size - (t - 0.5) * 4 * size, 500 + size, points.size()));
                            } else {
                                points.add(new POINT(500 - size, 500 + size - (t - 0.75) * 4 * size, points.size()));
                            }
                        }
                    }
                    break;

                case "x_pattern":
                    points.addAll(generateChain(100, 100, 900, 900, 300, 20, points.size()));
                    points.addAll(generateChain(900, 100, 100, 900, 300, 20, points.size()));
                    break;

                case "uniform_coverage":
                    // Generate points with Poisson disk sampling
                    points = generateUniformCoverage(100, 900, 100, 900, 30, 0);
                    // Fill gaps with smaller points
                    points.addAll(generateUniformCoverage(100, 900, 100, 900, 15, points.size()));
                    break;

                case "fractal_clusters":
                    points.addAll(generateFractal(500, 500, 600, 3, 200, points.size()));
                    break;
                    
                case "fibonacci_spiral":
                    double phi = (1 + Math.sqrt(5)) / 2;
                    for (int i = 0; i < 500; i++) {
                        double theta = i * 2 * Math.PI / phi;
                        double r = Math.sqrt(i) * 15;
                        double x = 500 + r * Math.cos(theta);
                        double y = 500 + r * Math.sin(theta);
                        points.add(new POINT(x, y, points.size()));
                    }
                    break;
                    
                case "olympic_rings":
                    double[][] ringCenters = {
                        {300, 400}, {500, 400}, {700, 400},
                        {400, 500}, {600, 500}
                    };
                    for (double[] center : ringCenters) {
                        points.addAll(generateCircle(center[0], center[1], 80, 100, 5, points.size()));
                    }
                    break;
                    
                case "heart_shape":
                    for (int i = 0; i < 1000; i++) {
                        double t = 2 * Math.PI * i / 1000;
                        double x = 500 + 200 * Math.pow(Math.sin(t), 3);
                        double y = 450 - 200 * (Math.cos(t) - Math.cos(2*t)/2 - Math.cos(3*t)/3 - Math.cos(4*t)/4);
                        points.add(new POINT(x, y, points.size()));
                    }
                    break;
                    
                // Add more creative test cases...
                
                default:
                    continue;
            }

            int k = 2;
            while (k <= 64) {
                Cluster[] clusters = kMeansClustering(points, k, 100);
                
                String outputPath = String.format("output/%s/%s_k%d.png",  "kmean", testCase, k);
                generateClusteredImage(clusters, points, width, height, outputPath);
                k = k * 2;
            }
        }
        
        System.out.println("All clustering algorithms and test cases complete.");
    }
}