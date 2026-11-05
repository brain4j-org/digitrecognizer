package it.echo.digitrecognizer.ai;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.SplittableRandom;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;

/**
 * Generates a freshly augmented batch every time a training epoch starts.
 * Reusing the source between epochs avoids storing a second copy of the dataset.
 */
public final class AugmentedDataSource extends ListDataSource {

    private final List<Sample> sourceSamples;
    private final MnistAugmenter augmenter;
    private final int sourceBatchSize;
    private final double augmentProbability;
    private final SplittableRandom random;

    /**
     * Wraps the given samples with per-epoch augmentation.
     *
     * @param samples the clean training samples
     * @param batchSize the number of samples per batch
     * @param augmenter the deformation generator
     * @param augmentProbability fraction of samples to deform, in [0, 1]
     * @param seed the seed used to choose which samples to deform
     */
    public AugmentedDataSource(
        List<Sample> samples,
        int batchSize,
        MnistAugmenter augmenter,
        double augmentProbability,
        long seed
    ) {
        super(List.of(), false, batchSize);
        this.sourceSamples = samples;
        this.augmenter = augmenter;
        this.sourceBatchSize = batchSize;
        this.augmentProbability = Math.clamp(augmentProbability, 0.0, 1.0);
        this.random = new SplittableRandom(seed);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return sourceSamples == null ? 0 : sourceSamples.size();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int batches() {
        return (sourceSamples.size() + sourceBatchSize - 1) / sourceBatchSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int batchSize() {
        return sourceBatchSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Sample> samples() {
        return sourceSamples;
    }

    /**
     * Returns a new iterator that augments samples on the fly.
     *
     * @return a fresh iterable of augmented batches
     */
    @Override
    public Iterable<Batch> batchIterator() {
        return new Iterable<>() {

            @Override
            public Iterator<Batch> iterator() {
                return new BatchIterator();
            }

            @Override
            public Spliterator<Batch> spliterator() {
                return Spliterators.spliterator(new BatchIterator(), batches(), Spliterator.ORDERED);
            }
        };
    }

    private final class BatchIterator implements Iterator<Batch> {

        private int index;

        @Override
        public boolean hasNext() {
            return index < sourceSamples.size();
        }

        @Override
        public Batch next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }

            int end = Math.min(index + sourceBatchSize, sourceSamples.size());
            List<Tensor> inputs = new ArrayList<>(end - index);
            List<Tensor> labels = new ArrayList<>(end - index);

            for (int i = index; i < end; i++) {
                Sample sample = sourceSamples.get(i);
                float[] pixels = sample.input(0).data();
                float[] augmented = random.nextDouble() < augmentProbability
                    ? augmenter.augment(pixels)
                    : pixels;

                inputs.add(Tensors.create(
                    new int[] {1, MnistCnn.IMAGE_SIZE, MnistCnn.IMAGE_SIZE}, augmented));
                labels.add(sample.label(0));
            }

            index = end;

            return new Batch(
                new Tensor[] {Tensors.mergeTensors(inputs)},
                new Tensor[] {Tensors.mergeTensors(labels)});
        }
    }
}
