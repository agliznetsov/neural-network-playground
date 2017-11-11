package agi.nn.problem.points;

import agi.nn.problem.Sample;
import lombok.AllArgsConstructor;
import lombok.Getter;

public class Point implements Sample{
    public final double x, y, value;

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    @Override
    public double getValue() {
        return value;
    }

    public Point(double x, double y, double value) {
        this.x = x;
        this.y = y;
        this.value = value;
    }
}
