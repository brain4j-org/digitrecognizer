package it.echo.digitrecognizer.ui;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.RoundRectangle2D;

import javax.swing.JPanel;
import javax.swing.border.EmptyBorder;

public class Card extends JPanel {

    private final int radius;
    private final Color fill;
    private final Color border;

    public Card() {
        this(6);
    }

    public Card(int radius) {
        this(Theme.CARD_TOP, Theme.BORDER, radius, 0);
    }

    public Card(Color fill, Color border, int radius, int padding) {
        this.radius = radius;
        this.fill = fill;
        this.border = border;

        setOpaque(false);
        setBorder(new EmptyBorder(padding, padding, padding, padding));
    }

    @Override
    protected void paintComponent(Graphics graphics) {
        Graphics2D g = (Graphics2D) graphics.create();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int width = getWidth();
        int height = getHeight();
        RoundRectangle2D shape = new RoundRectangle2D.Float(
            0.5f, 0.5f, width - 1f, height - 1f, radius, radius);

        g.setColor(fill);
        g.fill(shape);

        g.setColor(border);
        g.draw(shape);

        g.dispose();
        super.paintComponent(graphics);
    }
}
