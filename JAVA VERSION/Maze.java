import java.util.Random;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.ArrayDeque;

public class Maze{
    char[][] maze;
    int size;   

    public Maze(int n, double p){
        Random random = new Random();
        this.size = n;
        this.maze = new char[n][n];
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                if (random.nextDouble() <= p){
                    this.maze[i][j] = 'X';
                } else this.maze[i][j] = ' ';
            }
        }
        this.maze[0][0] = 'S';
        this.maze[n-1][n-1] = 'G';
    }

    public void printMaze(){
        for (int i = 0; i < maze.length; i++){
            for (int j = 0; j < maze.length; j++){
                System.out.print(this.maze[i][j]);
            }
            System.out.println();
        }
    }

    public void printSolvedMaze(ArrayList<Point> path){
        for (Point point : path){
            this.maze[point.x][point.y] = '*';
        }

        this.printMaze();

        for (Point point : path){
            this.maze[point.x][point.y] = ' ';
        }
    }

    public ArrayList<Point> getNeighbours(Point point){
        int x = point.x;
        int y = point.y;

        ArrayList<Point> neighbours = new ArrayList<Point>();

        if (x-1 >= this.size - 1 && this.maze[x-1][y] != 'X'){
            neighbours.add(new Point(x-1, y));
        }
        if (y-1 >= this.size - 1 && this.maze[x][y-1] != 'X'){
            neighbours.add(new Point(x, y-1));
        }
        if (y+1 <= this.size - 1 && this.maze[x][y+1] != 'X'){
            neighbours.add(new Point(x, y+1));
        }
        if (x+1 <= this.size - 1 && this.maze[x+1][y] != 'X'){
            neighbours.add(new Point(x+1, y));
        }       

        return neighbours;

    }

    public ArrayList<Point> dfs(Point start, Point goal) {
        Deque<Point> fringe = new ArrayDeque<Point>();
        boolean[][] visited = new boolean[size][size];
        Map<Point, Point> prev = new HashMap<Point, Point>();
        ArrayList<Point> path = null;

        fringe.add(start);
        prev.put(start, null);

        Point curr;

        while (fringe.size() > 0) {
            curr = fringe.pop();
            if (curr.equals(goal)){
                path = new ArrayList<Point>();
                while (true) {
                    if (prev.get(curr) == null) break;
                    else{
                        path.add(prev.get(curr));
                        curr = prev.get(curr);
                    }
                }
                return path;
            }
            for (Point neighbour : getNeighbours(curr)){
                if (!visited[neighbour.x][neighbour.y]){
                    fringe.add(neighbour);
                    prev.put(neighbour, curr);
                }
            }
            visited[curr.x][curr.y] = false;
        }

        return path;
    }

    public ArrayList<Point> bfs(Point start, Point goal) {
        Queue<Point> fringe = new LinkedList<Point>();
        boolean[][] visited = new boolean[size][size];
        Map<Point, Point> prev = new HashMap<Point, Point>();
        ArrayList<Point> path = null;

        fringe.add(start);
        prev.put(start, null);

        Point curr;

        while (fringe.size() > 0) {
            curr = fringe.remove();
            if (curr.equals(goal)){
                path = new ArrayList<Point>();
                while (true) {
                    if (prev.get(curr) == null) break;
                    else{
                        path.add(prev.get(curr));
                        curr = prev.get(curr);
                    }
                }
                return path;
            }
            for (Point neighbour : getNeighbours(curr)){
                if (!visited[neighbour.x][neighbour.y]){
                    fringe.add(neighbour);
                    prev.put(neighbour, curr);
                }
            }
            visited[curr.x][curr.y] = false;
        }

        return path;
    }

    public double euclideanDistance(Point a, Point b){
        return Math.sqrt((b.y - a.y) * (b.y - a.y) + (b.x - a.x) * (b.x - a.x));
    }

    public ArrayList<Point> astar(Point start, Point goal) {
        PriorityQueue<Point> fringe = new PriorityQueue<Point>();
        boolean[][] visited = new boolean[size][size];
        Map<Point, Point> prev = new HashMap<Point, Point>();
        Map<Point, Double> cost = new HashMap<Point, Double>();
        ArrayList<Point> path = null;

        fringe.add(start);
        prev.put(start, null);
        cost.put(start, 0.0);

        Point curr;

        while (fringe.size() > 0) {
            curr = fringe.remove();
            if (curr.equals(goal)){
                path = new ArrayList<Point>();
                while (true) {
                    if (prev.get(curr) == null) break;
                    else{
                        path.add(prev.get(curr));
                        curr = prev.get(curr);
                    }
                }
                return path;
            }
            for (Point neighbour : getNeighbours(curr)){
                double newCost = cost.get(curr) + 1;
                if (!cost.containsKey(neighbour) || newCost < cost.get(neighbour)){
                    cost.put(neighbour, newCost);
                    neighbour.cost = newCost + euclideanDistance(goal, neighbour);
                    fringe.add(neighbour);
                    prev.put(neighbour, curr);
                }
            }
            visited[curr.x][curr.y] = false;
        }

        return path;
    }
    
}