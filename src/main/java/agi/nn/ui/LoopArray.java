package agi.nn.ui;

public class LoopArray {
    private int maxSize;
    private int end;
    private int start;
    private double[] values;

    public LoopArray(int maxSize) {
        this.maxSize = maxSize;
        this.values = new double[maxSize];
    }

    public int getSize() {
        int size = end - start;
        if (size > 0)
            return size;
        else if (size < 0)
            return maxSize;
        else
            return 0;
    }

    public double get(int i) {
        int index = start + i;
        if (index >= maxSize) {
            index -= maxSize;
        }
        return values[index];
    }

    public void add(double value) {
        end = increment(end);
        values[end] = value;
        if (end == start) {
            start = increment(start);
        }
    }

    private int increment(int i) {
        int res = i + 1;
        if (res == maxSize) {
            res = 0;
        }
        return res;
    }

    public void clear() {
        start = 0;
        end = 0;
    }
}
