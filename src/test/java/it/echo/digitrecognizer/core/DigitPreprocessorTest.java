package it.echo.digitrecognizer.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class DigitPreprocessorTest {

    @Test
    void emptyGridStaysEmpty() {
        float[] result = DigitPreprocessor.prepare(new float[28 * 28]);
        assertTrue(allZero(result));
    }

    @Test
    void outputIsAlwaysTwentyPixelsOrLess() {
        float[] grid = new float[28 * 28];
        for (int y = 4; y < 24; y++) {
            for (int x = 10; x < 14; x++) {
                grid[y * 28 + x] = 1f;
            }
        }

        float[] result = DigitPreprocessor.prepare(grid);
        int minX = 28;
        int maxX = -1;
        int minY = 28;
        int maxY = -1;

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                if (result[y * 28 + x] > 0.01f) {
                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;
        assertTrue(width <= 20, "width=" + width);
        assertTrue(height <= 20, "height=" + height);
        assertTrue(Math.max(width, height) >= 19, "the digit should use most of the 20px box");
    }

    @Test
    void centeringMovesTheInkTowardsTheCenter() {
        float[] grid = new float[28 * 28];
        for (int y = 2; y < 12; y++) {
            for (int x = 2; x < 12; x++) {
                grid[y * 28 + x] = 1f;
            }
        }

        float[] result = DigitPreprocessor.prepare(grid);
        double mass = 0;
        double massX = 0;
        double massY = 0;

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float value = result[y * 28 + x];
                mass += value;
                massX += x * value;
                massY += y * value;
            }
        }

        assertEquals(14.0, massX / mass, 1.5);
        assertEquals(14.0, massY / mass, 1.5);
        assertTrue(Arrays.stream(toDoubles(result)).max().orElse(0) <= 1.0);
    }

    private static boolean allZero(float[] values) {
        for (float value : values) {
            if (value != 0f) {
                return false;
            }
        }
        return true;
    }

    private static double[] toDoubles(float[] values) {
        double[] result = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            result[i] = values[i];
        }
        return result;
    }
}
