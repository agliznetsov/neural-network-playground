package agi.nn.problem;

public interface Feature<T> {
    String getName();
    double getInput(T sample);
}
