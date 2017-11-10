package agi.nn.problem.points;

import agi.nn.problem.Sample;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class Point implements Sample{
    public final double x, y, value;
}
