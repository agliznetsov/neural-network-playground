package agi.nn.problem.points;

import agi.nn.problem.Sample;

public class Point implements Sample {
    final double[] inputs;
    final double[] targets;

    public double getX() {
        return inputs[0];
    }

    public double getY() {
        return inputs[1];
    }

    public double getValue() {
        return targets[0];
    }

    @Override
    public double[] getInputs() {
        return inputs;
    }

    @Override
    public double[] getTargets() {
        return targets;
    }

    public Point(double x, double y, double value) {
        inputs = new double[]{x, y};
        targets = new double[]{value};
    }
}
