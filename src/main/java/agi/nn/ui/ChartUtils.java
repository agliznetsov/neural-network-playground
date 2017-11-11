package agi.nn.ui;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class ChartUtils {
    public static Color[] COLOR_MAP = linearGradient(100, Color.BLUE, Color.WHITE, Color.ORANGE);

    public static void drawLineChart(Canvas canvas, LoopArray data) {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.strokeRect(0, 0, canvas.getWidth(), canvas.getHeight());

        gc.setStroke(Color.BLACK);
        gc.setLineWidth(2);

        double[] xPoints = new double[data.getSize()];
        double[] yPoints = new double[data.getSize()];
        double dx = canvas.getWidth() * 1.0 / data.getSize();
        for (int i = 0; i < xPoints.length; i++) {
            xPoints[i] = dx * i;
            yPoints[i] = (1 - data.get(i)) * canvas.getHeight();
        }
        gc.strokePolyline(xPoints, yPoints, xPoints.length);
    }
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
