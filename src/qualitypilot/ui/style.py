"""Typography only: no generated-content replacements, layout tricks or JavaScript."""

CSS = """
<style>
[data-testid="stMainBlockContainer"] { padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stImageCaption"], figcaption {
    color: #334c5e;
    opacity: 1;
    font-size: 0.875rem;
    line-height: 1.5;
}
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p {
    line-height: 1.55;
}
[data-testid="stMetricLabel"] p { color: #334c5e; }
[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
"""
