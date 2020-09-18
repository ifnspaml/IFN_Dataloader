# this routine creates a standalone tikz image from the code of a single tikz image

def create_standalone(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    f.close()
    with open(filename, "w") as f:
        f.write("\\documentclass[convert={convertexe={magick.exe}}]{standalone}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\usepackage{siunitx}\n")
        f.write("\\usetikzlibrary{calc}\n")
        f.write("\\usepackage{pgfplots}\n")
        f.write("\\pgfplotsset{compat=newest}\n")
        f.write("\\usepgfplotslibrary{groupplots}\n")
        f.write("\\usepgfplotslibrary{dateplot}\n")
        f.write("\\begin{document}\n")
        for line in lines:
            f.write(line)
    f.close()
    with open(filename, "a") as f:
        f.write("\n" + "\\end{document}")
    f.close()