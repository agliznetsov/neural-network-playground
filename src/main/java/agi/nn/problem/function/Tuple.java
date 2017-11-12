package agi.nn.problem.function;

import agi.nn.problem.Sample;

public class Tuple implements Sample{
    public final double x, y;

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    @Override
    public double getValue() {
        return y;
    }

    public Tuple(double x, double y) {
        this.x = x;
        this.y = y;
    }
}
