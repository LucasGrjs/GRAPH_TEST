import java.util.List;
import java.util.ArrayList;

public class GridPartitionerWrapper {
    
    private static Integer[][] convertToGrid(ArrayList<ArrayList<Integer>> input) {
        int rows = input.size();
        int cols = input.get(0).size();
        Integer[][] grid = new Integer[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                grid[i][j] = input.get(i).get(j);
            }
        }
        return grid;
    }
    
    private static ArrayList<ArrayList<Integer>> convertToLists(List<List<Integer>> clusters, int rows, int cols) {
        ArrayList<ArrayList<Integer>> result = new ArrayList<>();
        
        for (List<Integer> cluster : clusters) {
            ArrayList<Integer> convertedCluster = new ArrayList<>();
            for (int i = 0; i < cluster.size(); i++) {
                int cellId = cluster.get(i);
                convertedCluster.add(cellId); // Keep the original cell IDs instead of converting to coordinates
            }
            result.add(convertedCluster);
        }
        return result;
    }

    public static ArrayList<ArrayList<Integer>> grid_partitioning(ArrayList<ArrayList<Integer>> grid_to_divide, int cluster_number, int number_neighbors) {
        // Convert input grid to required format
        Integer[][] gridArray = convertToGrid(grid_to_divide);
        int rows = gridArray.length;
        int cols = gridArray[0].length;
        
        // Create GRID instance and apply partitioning
        GRID grid = new GRID(rows, cols);
        
        // Choose partitioning method based on analysis results
        try {
            grid.divideGridToVoronoiClusters(cluster_number);
            List<List<Integer>> clusters = grid.getClusteringAsLists();
            return convertToLists(clusters, rows, cols);
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    public static ArrayList<ArrayList<Integer>> grid_partitioning(int rows, int cols, int cluster_number, int number_neighbors) {
        // Create GRID instance and apply partitioning
        GRID grid = new GRID(rows, cols);
        
        // Choose partitioning method based on analysis results
        try {
            grid.divideGridToClusters(cluster_number);
            List<List<Integer>> clusters = grid.getClusteringAsLists();
            return convertToLists(clusters, rows, cols);
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Voronoi technique
     * 
     * @param rows : number of rows in the grid
     * @param columns : number of columns in the grid
     * @param cluster_number : number of clusters required
     * @param number_neighbors : number of neighbors to consider
     * 
     * @return List<List<Integer>> clustered graph
     */
    @SuppressWarnings("rawtypes")
    @operator(
            value = "grid_voronoi_partitioning",
            type = IType.LIST,
            category = { IOperatorCategory.GRID },
            concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Voronoi technique")
    public static List<List<Integer>> grid_voronoi_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToVoronoiClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Kmeans technique
     * 
     * @param rows : number of rows in the grid
     * @param columns : number of columns in the grid
     * @param cluster_number : number of clusters required
     * @param number_neighbors : number of neighbors to consider
     * 
     * @return List<List<Integer>> clustered graph
     */
    @SuppressWarnings("rawtypes")
    @operator(
            value = "grid_kmeans_partitioning",
            type = IType.LIST,
            category = { IOperatorCategory.GRID },
            concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Kmeans technique")
    public static List<List<Integer>> grid_kmeans_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToKmeansClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Spectral technique
     * 
     * @param rows : number of rows in the grid
     * @param columns : number of columns in the grid
     * @param cluster_number : number of clusters required
     * @param number_neighbors : number of neighbors to consider
     * 
     * @return List<List<Integer>> clustered graph
     */
    @SuppressWarnings("rawtypes")
    @operator(
            value = "grid_spectral_partitioning",
            type = IType.LIST,
            category = { IOperatorCategory.GRID },
            concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Spectral technique")
    public static List<List<Integer>> grid_spectral_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToSpectralClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Grid technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_grid_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Grid technique")
    public static List<List<Integer>> grid_grid_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToGridClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Horizontal Strips technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_horizontal_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Horizontal Strips technique")
    public static List<List<Integer>> grid_horizontal_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToHorizontalStrips(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Vertical Strips technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_vertical_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Vertical Strips technique")
    public static List<List<Integer>> grid_vertical_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToVerticalStrips(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using BSP technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_bsp_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using BSP technique")
    public static List<List<Integer>> grid_bsp_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToBSPClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Circular technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_circular_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Circular technique")
    public static List<List<Integer>> grid_circular_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToCircularClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Spiral technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_spiral_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Spiral technique")
    public static List<List<Integer>> grid_spiral_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToSpiralClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Diagonal technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_diagonal_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Diagonal technique")
    public static List<List<Integer>> grid_diagonal_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToDiagonalClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Checkerboard technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_checkerboard_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Checkerboard technique")
    public static List<List<Integer>> grid_checkerboard_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToCheckerboardClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Fractal technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_fractal_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Fractal technique")
    public static List<List<Integer>> grid_fractal_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToFractalClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Honeycomb technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_honeycomb_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Honeycomb technique")
    public static List<List<Integer>> grid_honeycomb_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToHoneycombClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    /**
     * Grid partitioning using Wave technique
     */
    @SuppressWarnings("rawtypes")
    @operator(value = "grid_wave_partitioning", type = IType.LIST, 
            category = { IOperatorCategory.GRID }, concept = { IConcept.GRID })
    @doc(value = "grid partitioning using Wave technique")
    public static List<List<Integer>> grid_wave_partitioning(int rows, int columns, int cluster_number, int number_neighbors) {
        GRID grid = new GRID(rows, columns);
        try {
            grid.divideGridToWaveClusters(cluster_number);
            return grid.getClusteringAsLists();
        } catch (Exception e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }

    public static void main(String[] args) {
        var clusters = grid_partitioning(100,100,10,4);

        for(var auto : clusters)
        {
            System.out.println(auto);
        }
    }
}
