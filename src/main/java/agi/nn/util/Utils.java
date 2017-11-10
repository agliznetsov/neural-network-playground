package agi.nn.util;

import javafx.scene.paint.Color;

public class Utils {
    public static Color[] linearGradient(int length, Color... colors) {
        Color[] res = new Color[length];
        double segmentLength = Math.ceil(length / (colors.length - 1));
        for (int i = 0; i < colors.length - 1; i++) {
            for (int n = 0; n < segmentLength; n++) {
                int index = (int)(i * segmentLength + n);
                if (index < length) {
                    double w = n * 1.0 / segmentLength;
                    res[index] = colorGradient(colors[i], colors[i + 1], w);
                }
            }
        }
        return res;
    }

    public static double randUniform(double a, double b) {
        return Math.random() * (b - a) + a;
    }

    private static Color colorGradient(Color from, Color to, double w) {
        return new Color(
                linearGradient(from.getRed(), to.getRed(), w),
                linearGradient(from.getGreen(), to.getGreen(), w),
                linearGradient(from.getBlue(), to.getBlue(), w),
                linearGradient(from.getOpacity(), to.getOpacity(), w)
        );
    }

    private static double linearGradient(double start, double end, double w) {
        return start + (end - start) * w;
    }

}
