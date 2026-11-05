package it.echo.digitrecognizer.ai;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ar.com.hjg.pngj.ImageLineInt;
import ar.com.hjg.pngj.PngReader;
import dev.hardwood.reader.RowReader;
import dev.hardwood.row.PqStruct;
import org.brain4j.datasets.Datasets;
import org.brain4j.datasets.core.dataset.DatasetFile;
import org.brain4j.datasets.core.dataset.HFDataset;
import org.brain4j.datasets.format.impl.ParquetFormat;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;

/**
 * Loads the official MNIST train and test splits without mixing them.
 * The bundled dataset helper splits a single pool, so this loader keeps the
 * reported accuracy comparable with published results.
 */
public final class MnistLoader {

    /**
     * Holds the two dataset splits.
     *
     * @param train the 60,000 training samples
     * @param test the 10,000 test samples
     */
    public record Splits(List<Sample> train, List<Sample> test) {
    }

    private MnistLoader() {
    }

    /**
     * Loads both official MNIST splits.
     *
     * @return the train and test samples
     * @throws IOException if the dataset cannot be read
     */
    public static Splits load() throws IOException {
        HFDataset dataset;

        try {
            dataset = Datasets.loadDataset("ylecun/mnist");
        } catch (Exception exception) {
            throw new IOException("Unable to load MNIST", exception);
        }

        List<Sample> train = new ArrayList<>();
        List<Sample> test = new ArrayList<>();
        ParquetFormat format = new ParquetFormat();

        for (DatasetFile file : dataset.filesByFormat(format.format())) {
            String fileName = file.path().getFileName().toString();
            List<Sample> target;

            if (fileName.startsWith("train")) {
                target = train;
            } else if (fileName.startsWith("test")) {
                target = test;
            } else {
                continue;
            }

            for (RowReader record : format.read(file.path().toFile())) {
                target.add(parse(record));
            }
        }

        if (train.isEmpty() || test.isEmpty()) {
            throw new IOException("MNIST splits are empty");
        }

        return new Splits(train, test);
    }

    private static Sample parse(RowReader record) throws IOException {
        int label = (int) record.getLong("label");
        PqStruct imageGroup = record.getStruct("image");
        byte[] pngBytes = imageGroup.getBinary("bytes");

        PngReader reader = new PngReader(new ByteArrayInputStream(pngBytes));

        try {
            int width = reader.imgInfo.cols;
            int height = reader.imgInfo.rows;
            float[] pixels = new float[width * height];

            for (int row = 0; row < height; row++) {
                ImageLineInt line = (ImageLineInt) reader.readRow(row);
                int[] scanline = line.getScanline();

                for (int column = 0; column < width; column++) {
                    pixels[row * width + column] = (scanline[column] & 0xFF) / 255.0f;
                }
            }

            Tensor input = Tensors.create(
                new int[] {1, MnistCnn.IMAGE_SIZE, MnistCnn.IMAGE_SIZE}, pixels);
            Tensor output = Tensors.zeros(MnistCnn.CLASSES);
            output.set(1, label);

            return new Sample(input, output);
        } finally {
            reader.end();
        }
    }
}
