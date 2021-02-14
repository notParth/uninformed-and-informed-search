public class Point implements Comparable<Point>{
    int x;
    int y;
    double cost;

    public Point(int x, int y){
        this.x = x;
        this.y = y;
        this.cost = 0.0;
    }

    public String toString(){
        return "X: " + x + ", Y: " + y;
    }

    public boolean equals(Point e){
        return e.x == this.x && e.y == this.y;
    }

    @Override
    public int compareTo(Point o) {
        return o.cost > this.cost ? 1 : -1;
    }
}