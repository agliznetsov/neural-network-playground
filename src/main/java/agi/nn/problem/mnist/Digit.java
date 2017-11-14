package agi.nn.problem.mnist;

import agi.nn.problem.Sample;

public class Digit implements Sample {
    private final int label;
    private final double[] targets;
    private double[] data;

    public Digit(int label) {
        this.label = label;
        targets = new double[10];
        targets[label] = 1;
    }

    public double getLabel() {
        return label;
    }

    public double[] getData() {
        return data;
    }

    public void setData(double[] data) {
        this.data = data;
    }

    @Override
    public double[] getInputs() {
        return data;
    }

    @Override
    public double[] getTargets() {
        return targets;
    }
}
