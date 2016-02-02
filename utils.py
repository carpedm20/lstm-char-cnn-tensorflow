import sys
import pprint

try:
    xrange
except NameError:
    xrange = range

pp = pprint.PrettyPrinter()

def progress(progress, status=""):
    barLength = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Finished.\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [%s] %.2f%% | %s" % ("#"*block + " "*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

