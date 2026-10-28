package it.echo.digitrecognizer.ui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;

import javax.swing.JComponent;

public final class InputPreview extends JComponent {

    private static final int SIZE = 28;
    private final BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);

    public InputPreview() {
        setOpaque(false);
        setPreferredSize(new java.awt.Dimension(64, 64));
        setMinimumSize(new java.awt.Dimension(48, 48));
        clear();
    }

    public void setInput(float[] pixels) {
        Color background = Theme.CARD_INNER;
        Color ink = Theme.INK;

        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                float value = Math.clamp(pixels[y * SIZE + x], 0f, 1f);
                int red = Math.round(background.getRed() + (ink.getRed() - background.getRed()) * value);
                int green = Math.round(background.getGreen() + (ink.getGreen() - background.getGreen()) * value);
                int blue = Math.round(background.getBlue() + (ink.getBlue() - background.getBlue()) * value);
                image.setRGB(x, y, new Color(red, green, blue).getRGB());
            }
        }
        repaint();
    }

    public void clear() {
        Graphics2D g = image.createGraphics();
        g.setColor(Theme.CARD_INNER);
        g.fillRect(0, 0, SIZE, SIZE);
        g.dispose();
        repaint();
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        Graphics2D g = (Graphics2D) graphics.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);

        int size = Math.min(getWidth(), getHeight()) - 2;
        int x = (getWidth() - size) / 2;
        int y = (getHeight() - size) / 2;
        RoundRectangle2D shape = new RoundRectangle2D.Float(x, y, size, size, 3, 3);

        g.setColor(Theme.CARD_INNER);
        g.fill(shape);

        g.setClip(shape);
        g.drawImage(image, x, y, size, size, null);
        g.setClip(null);

        g.setColor(Theme.LINE);
        g.draw(shape);

        g.dispose();
    }
}
