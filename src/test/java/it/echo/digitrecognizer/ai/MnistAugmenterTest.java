package it.echo.digitrecognizer.ai;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class MnistAugmenterTest {

    @Test
    void emptyImageStaysEmpty() {
        MnistAugmenter augmenter = new MnistAugmenter(1);
        float[] result = augmenter.augment(new float[28 * 28]);

        assertEquals(28 * 28, result.length);
        assertTrue(allZero(result));
    }

    @Test
    void digitStaysInRangeAndKeepsInk() {
        MnistAugmenter augmenter = new MnistAugmenter(2);
        float[] source = new float[28 * 28];

        for (int y = 8; y < 20; y++) {
            for (int x = 10; x < 18; x++) {
                source[y * 28 + x] = 1f;
            }
        }

        float[] result = augmenter.augment(source);

        assertTrue(hasInk(result));

        for (float value : result) {
            assertTrue(value >= 0f && value <= 1f, "value=" + value);
        }
    }

    @Test
    void sameSeedProducesSameDeformation() {
        float[] source = new float[28 * 28];
        source[14 * 28 + 14] = 1f;

        float[] first = new MnistAugmenter(7).augment(source);
        float[] second = new MnistAugmenter(7).augment(source);

        assertArrayEquals(first, second);
    }

    private static boolean allZero(float[] values) {
        for (float value : values) {
            if (value != 0f) {
                return false;
            }
        }

        return true;
    }

    private static boolean hasInk(float[] values) {
        for (float value : values) {
            if (value > 0.1f) {
                return true;
            }
        }

        return false;
    }
}
