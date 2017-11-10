package agi.nn.problem.points;

import agi.nn.problem.Sample;

import java.util.ArrayList;
import java.util.List;

import static agi.nn.util.Utils.randUniform;

public class CircleProblem extends PointProblem {
    @Override
    public String getName() {
        return "Circle";
    }

    @Override
    public List<Sample> createSamples(int count) {
        List<Sample> points = new ArrayList<>(count);

        // Generate positive points inside the circle.
        for (int i = 0; i < count / 2; i++) {
            double r = randUniform(0, RADIUS * 0.5);
            double angle = randUniform(0, 2 * Math.PI);
            double x = r * Math.sin(angle);
            double y = r * Math.cos(angle);
            points.add(new Point(x, y, 1));
        }

        // Generate negative points outside the circle.
        for (int i = 0; i < count / 2; i++) {
            double r = randUniform(RADIUS * 0.7, RADIUS);
            double angle = randUniform(0, 2 * Math.PI);
            double x = r * Math.sin(angle);
            double y = r * Math.cos(angle);
            points.add(new Point(x, y, -1));
        }

        return points;
    }
}
