package agi.nn.ui;

import java.util.LinkedList;

public class LoopArray {
    private int maxSize;
    private LinkedList<Double> values;

    public LoopArray(int maxSize) {
        this.maxSize = maxSize;
        this.values = new LinkedList<Double>();
    }

    public int getSize() {
        return values.size();
    }

    public double get(int i) {
        return values.get(i);
    }

    public void add(double value) {
        values.add(value);
        if (values.size() > maxSize) {
            values.remove(0);
        }
    }

    public void clear() {
        this.values.clear();
    }
}
