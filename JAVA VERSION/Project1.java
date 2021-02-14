import java.util.ArrayList;

public class Project1 {
    public static void main(String[] args){
        int dimension = 50;
        Maze maze = new Maze(dimension, 0.2);
        maze.printMaze();

        ArrayList<Point> path = maze.astar(new Point(0,0), new Point(dimension-1,dimension-1));

        if(path != null) {
            System.out.println("Path found.");
            maze.printSolvedMaze(path);
        } else {
            System.out.println("Path not found.");
        }

        // problem 2

        // for (double density = 0; density <= 1.0; density += 0.10){
        //     for (int triesPerDensity = 0; triesPerDensity < 30; triesPerDensity++){
        //         System.out.println("Density: " + density + ", run: " + triesPerDensity);
        //         maze = new Maze(dimension, density);
        //         maze.dfs(new Point(0,0), new Point(dimension-1,dimension-1));
        //     }
        // }




    }
}
