package agi.nn.problem;

import java.util.List;

public interface Problem {
    String getName();

    List<Feature> getFeatures();

    List<Sample> createSamples(int count);
}
