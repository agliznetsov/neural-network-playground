package agi.nn.problem.points;

import agi.nn.problem.Feature;
import agi.nn.problem.Problem;
import agi.nn.problem.Sample;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public abstract class PointProblem implements Problem {
    public static final double RADIUS = 5.0;

    @Override
    public List<Feature> getFeatures() {
        return Arrays.asList(
                createFeature("X", it -> it.x),
                createFeature("Y", it -> it.y)
        );
    }

    Feature createFeature(String name, Function<Point, Double> function) {
        return new Feature<Point>() {
            @Override
            public String getName() {
                return name;
            }

            @Override
            public double getInput(Point sample) {
                return function.apply(sample);
            }
        };
    }

}
