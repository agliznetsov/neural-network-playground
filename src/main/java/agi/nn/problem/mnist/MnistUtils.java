package agi.nn.problem.mnist;

import lombok.SneakyThrows;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MnistUtils {
    public static final int IMAGE_SIZE = 28;
    private static final String TEST_LABELS = "t10k-labels.idx1-ubyte";
    private static final String TEST_IMAGES = "t10k-images.idx3-ubyte";

    @SneakyThrows
    static List<Digit> loadSamples(File dir, int count) {
        List<Digit> digits = new ArrayList<>();
        byte[] buffer = new byte[1000];

        File labelsFile = new File(dir, TEST_LABELS);
        if (!labelsFile.exists()) {
            throw new RuntimeException(labelsFile + " not found");
        }
        try (InputStream is = new FileInputStream(labelsFile)) {
            is.read(buffer, 0, 2 * 4);
            count = is.read(buffer, 0, count);
        }
        for (int i = 0; i < count; i++) {
            digits.add(new Digit(buffer[i]));
        }

        File imagesFile = new File(dir, TEST_IMAGES);
        if (!imagesFile.exists()) {
            throw new RuntimeException(imagesFile + " not found");
        }
        try (InputStream is = new FileInputStream(imagesFile)) {
            is.read(buffer, 0, 4 * 4);
            for (int i = 0; i < count; i++) {
                is.read(buffer, 0, IMAGE_SIZE * IMAGE_SIZE);
                double[] data = new double[IMAGE_SIZE * IMAGE_SIZE];
                for (int n = 0; n < IMAGE_SIZE * IMAGE_SIZE; n++) {
                    int v = buffer[n] & 0xFF; //to unsigned
                    data[n] = v / 255.0;
                }
                digits.get(i).setData(data);
            }
        }

        return digits;
    }
}
