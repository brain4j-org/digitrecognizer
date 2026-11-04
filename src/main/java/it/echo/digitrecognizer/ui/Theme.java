package it.echo.digitrecognizer.ui;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.awt.Insets;
import java.util.Arrays;

import javax.swing.UIManager;

import com.formdev.flatlaf.FlatDarkLaf;

public final class Theme {

    public static final Color NIGHT = new Color(0x131311);
    public static final Color SURFACE = new Color(0x1B1A17);
    public static final Color SURFACE_RAISED = new Color(0x25231E);
    public static final Color CANVAS = new Color(0x0F0F0D);
    public static final Color BONE = new Color(0xF1ECE1);
    public static final Color BONE_SOFT = new Color(0xD6D0C3);
    public static final Color MUTED = new Color(0xB8B1A4);
    public static final Color FAINT = new Color(0x8F887A);
    public static final Color LINE = new Color(0x35322B);
    public static final Color LINE_SOFT = new Color(0x292721);
    public static final Color ACCENT = new Color(0xE25833);
    public static final Color ACCENT_SOFT = new Color(0x5B2A1D);
    public static final Color GREEN = new Color(0x7FAE8D);
    public static final Color WARNING = new Color(0xD0A35A);
    public static final Color DANGER = new Color(0xE07A6A);

    // Compatibility aliases used by the existing components.
    public static final Color BG = NIGHT;
    public static final Color PAPER = NIGHT;
    public static final Color PAPER_LIGHT = SURFACE;
    public static final Color PAPER_DEEP = SURFACE_RAISED;
    public static final Color INK = BONE;
    public static final Color INK_SOFT = BONE_SOFT;
    public static final Color CARD_TOP = SURFACE;
    public static final Color CARD_BOTTOM = SURFACE;
    public static final Color CARD_INNER = CANVAS;
    public static final Color BORDER = LINE;
    public static final Color BORDER_SOFT = LINE_SOFT;
    public static final Color TEXT = BONE;
    public static final Color MINT = GREEN;
    public static final Color SUCCESS = GREEN;

    private static final String BODY_FAMILY = pickFamily(
        "Cantarell", "Noto Sans", "Adwaita Sans", "Segoe UI", "SansSerif");
    private static final String MONO_FAMILY = pickFamily(
        "JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "Monospaced");

    public static final Font FONT = new Font(BODY_FAMILY, Font.PLAIN, 15);
    public static final Font MONO = new Font(MONO_FAMILY, Font.PLAIN, 13);
    private static final float TYPE_STEP = 1f;

    private Theme() {
    }

    public static void install() {
        FlatDarkLaf.setup();

        UIManager.put("defaultFont", FONT);
        UIManager.put("Panel.background", BG);
        UIManager.put("Label.foreground", TEXT);
        UIManager.put("Button.arc", 4);
        UIManager.put("Component.arc", 4);
        UIManager.put("TextComponent.arc", 4);
        UIManager.put("Button.background", SURFACE);
        UIManager.put("Button.foreground", BONE);
        UIManager.put("Button.hoverBackground", SURFACE_RAISED);
        UIManager.put("Button.pressedBackground", new Color(0x302D27));
        UIManager.put("Button.borderColor", LINE);
        UIManager.put("Button.disabledBackground", NIGHT);
        UIManager.put("Button.disabledText", FAINT);
        UIManager.put("Button.focusedBorderColor", BONE);
        UIManager.put("Component.focusColor", BONE);
        UIManager.put("Component.borderColor", LINE);
        UIManager.put("Component.disabledBorderColor", LINE_SOFT);
        UIManager.put("Slider.thumbColor", BONE);
        UIManager.put("Slider.trackColor", LINE);
        UIManager.put("Slider.thumbWidth", 12);
        UIManager.put("Slider.trackWidth", 2);
        UIManager.put("Slider.focusColor", ACCENT);
        UIManager.put("ToolTip.background", SURFACE_RAISED);
        UIManager.put("ToolTip.foreground", BONE);
        UIManager.put("ToolTip.borderColor", LINE);

        UIManager.put("TitlePane.background", SURFACE);
        UIManager.put("TitlePane.inactiveBackground", SURFACE);
        UIManager.put("TitlePane.foreground", BONE_SOFT);
        UIManager.put("TitlePane.inactiveForeground", FAINT);
        UIManager.put("TitlePane.borderColor", LINE);
        UIManager.put("TitlePane.font", font(Font.BOLD, 12.5f));
        UIManager.put("TitlePane.centerTitle", true);
        UIManager.put("TitlePane.showIcon", false);
        UIManager.put("TitlePane.showIconInDialogs", false);
        UIManager.put("TitlePane.unifiedBackground", false);
        UIManager.put("TitlePane.titleMargins", new Insets(0, 12, 0, 12));
        UIManager.put("TitlePane.buttonSize", new Dimension(38, 34));
        UIManager.put("TitlePane.buttonArc", 999);
        UIManager.put("TitlePane.buttonsGap", 6);
        UIManager.put("TitlePane.buttonsMargins", new Insets(0, 10, 0, 10));
        UIManager.put("TitlePane.buttonHoverBackground", SURFACE_RAISED);
        UIManager.put("TitlePane.buttonPressedBackground", LINE_SOFT);
        UIManager.put("TitlePane.closeHoverBackground", DANGER);
        UIManager.put("TitlePane.closePressedBackground", ACCENT_SOFT);
        UIManager.put("TitlePane.closeHoverForeground", BONE);
        UIManager.put("TitlePane.closePressedForeground", BONE);
    }

    public static Font font(int style, float size) {
        return FONT.deriveFont(style, size + TYPE_STEP);
    }

    public static Font mono(int style, float size) {
        return MONO.deriveFont(style, size + TYPE_STEP);
    }

    public static Color alpha(Color color, int alpha) {
        return new Color(color.getRed(), color.getGreen(), color.getBlue(), alpha);
    }

    public static Color blend(Color from, Color to, float amount) {
        float clamped = Math.clamp(amount, 0f, 1f);
        return new Color(
            Math.round(from.getRed() + (to.getRed() - from.getRed()) * clamped),
            Math.round(from.getGreen() + (to.getGreen() - from.getGreen()) * clamped),
            Math.round(from.getBlue() + (to.getBlue() - from.getBlue()) * clamped),
            Math.round(from.getAlpha() + (to.getAlpha() - from.getAlpha()) * clamped)
        );
    }

    private static String pickFamily(String... preferred) {
        String[] available = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();

        for (String candidate : preferred) {
            if (Arrays.asList(available).contains(candidate)) {
                return candidate;
            }
        }

        return preferred[preferred.length - 1];
    }
}
