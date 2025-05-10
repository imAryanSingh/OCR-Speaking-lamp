import matplotlib.font_manager

def print_fonts():
    # Get all available fonts
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

    print("Available Fonts:")
    for font_path in fonts:
        prop = matplotlib.font_manager.FontProperties(fname=font_path)

        # Extract font properties
        font_family = prop.get_family()[0]
        font_style = prop.get_style()
        font_variant = prop.get_variant()
        font_weight = prop.get_weight()
        font_stretch = prop.get_stretch()
        font_size = prop.get_size_in_points()

        print("Font Family:", font_family)
        print("Font Style:", font_style)
        print("Font Variant:", font_variant)
        print("Font Weight:", font_weight)
        print("Font Stretch:", font_stretch)
        print("Font Size (in points):", font_size)
        print("="*50)

if __name__ == "__main__":
    print_fonts()
