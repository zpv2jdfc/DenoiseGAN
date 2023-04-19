from sys import argv
from gan import app
if __name__ == '__main__':
    plot = False
    if(len(argv)>=2):
        plot = argv[1]
    app.run(plot)