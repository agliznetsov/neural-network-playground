package agi.nn.problem.points;

import java.util.ArrayList;
import java.util.List;

public class SinusProblem extends PointProblem {

    public SinusProblem() {
        super(8);
    }

    @Override
    public String getName() {
        return "Sinus";
    }

    @Override
    public List<Point> createSamples() {
        int count = SAMPLES_COUNT;
        List<Point> points = new ArrayList<>(count);
        double w = radius * 2 / count;
        double d = radius / 4;
        for (int i = 0; i < count; i++) {
            double x = (i - count / 2) * w;
            double y = Math.sin(i * w) * radius / 2;
            double value = i % 2 == 0 ? 1 : -1;
            y += value * d;
            points.add(new Point(x, y, value));
        }
        return points;
    }

}
