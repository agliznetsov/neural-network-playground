package agi.nn.problem.points;

import agi.nn.problem.Problem;

import java.util.ArrayList;
import java.util.List;

import static agi.nn.ui.ChartUtils.randUniform;

public class CircleProblem extends PointProblem {

    public CircleProblem() {
        super(5);
    }

    @Override
    public String getName() {
        return "Circle";
    }

    @Override
    public List<Point> createSamples(int count) {
        List<Point> points = new ArrayList<>(count);

        // Generate positive points inside the circle.
        for (int i = 0; i < count / 2; i++) {
            double r = randUniform(0, radius * 0.5);
            double angle = randUniform(0, 2 * Math.PI);
            double x = r * Math.sin(angle);
            double y = r * Math.cos(angle);
            points.add(new Point(x, y, 1));
        }

        // Generate negative points outside the circle.
        for (int i = 0; i < count / 2; i++) {
            double r = randUniform(radius * 0.7, radius);
            double angle = randUniform(0, 2 * Math.PI);
            double x = r * Math.sin(angle);
            double y = r * Math.cos(angle);
            points.add(new Point(x, y, -1));
        }

        return points;
    }
}
