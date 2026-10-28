package it.echo.digitrecognizer.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DigitRecognizerTest {

    @Test
    void bundledModelLoadsAndProducesAProbabilityDistribution() throws Exception {
        try (DigitRecognizer recognizer = DigitRecognizer.loadBundled()) {
            assertTrue(recognizer.description().contains("225,034 parameters"));

            float[] probabilities = recognizer.predict(new float[28 * 28]);

            assertEquals(DigitRecognizer.CLASSES, probabilities.length);

            double sum = 0;
            for (float probability : probabilities) {
                assertTrue(probability >= 0 && probability <= 1);
                sum += probability;
            }
            assertEquals(1.0, sum, 1e-3);
        }
    }
}
