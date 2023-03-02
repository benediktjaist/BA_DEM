def get_colour_tuple(colour_str):
    colours = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'olive': (128, 128, 0),
        'teal': (0, 128, 128),
        'navy': (0, 0, 128),
        'maroon': (128, 0, 0),
        'gold': (255, 215, 0),
        'silver': (192, 192, 192),
        'skyblue': (135, 206, 235)
    }
    return colours.get(colour_str.lower(), None)

'''
\definecolor{TUM_blue}{RGB}{0,101,189}
\definecolor{TUM_blue1}{RGB}{0,82,147}
\definecolor{TUM_blue2}{RGB}{0,51,89}
\definecolor{TUM_gray1}{RGB}{88,88,90}
\definecolor{TUM_gray2}{RGB}{156,157,159}
\definecolor{TUM_grey3}{RGB}{217,218,219}
\definecolor{TUM_ivory}{RGB}{218,215,203}
\definecolor{TUM_orange}{RGB}{227,114,34}
\definecolor{TUM_green}{RGB}{162,173,0}
\definecolor{TUM_bluel1}{RGB}{152,198,234}
\definecolor{TUM_bluel2}{RGB}{100,160,200}
'''


