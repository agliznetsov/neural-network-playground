package agi.nn.problem.points;

import java.util.ArrayList;
import java.util.List;

import static agi.nn.ui.ChartUtils.randUniform;

public class SpiralProblem extends PointProblem {

    public SpiralProblem() {
        super(5);
    }

    @Override
    public String getName() {
        return "Spiral";
    }

    @Override
    public List<Point> createSamples() {
        List<Point> points = new ArrayList<>(SAMPLES_COUNT);
        int n = SAMPLES_COUNT / 2;
        double noise = 0;
        genSpiral(points, n, noise, 0, 1); // Positive examples.
        genSpiral(points, n, noise, Math.PI, -1); // Negative examples.
        return points;
    }

    void genSpiral(List<Point> points, int n, double noise, double deltaT, double value) {
        for (int i = 0; i < n; i++) {
            double r = i * radius / n;
            double t = 1.75 * i / n * 2 * Math.PI + deltaT;
            double x = r * Math.sin(t) + randUniform(-1, 1) * noise;
            double y = r * Math.cos(t) + randUniform(-1, 1) * noise;
            points.add(new Point(x, y, value));
        }
    }

}
