def clear_lines(n):
    # Clears n lines of output; doesn't work for notebooks
    for i in range(n):
        print('\033[F', end='') # Go to start of the above line
        print('\033[K', end='') # Clear line

def clear_line(n):
    # Clears n characters of output
    print('\r', end='')    # Go to start of line
    print(n*' ', end='\r') # Delete the line and return to the start of it

def prefix_to_micro(prefix):
    # Conversion ratio for a given prefix to microseconds
    if prefix.startswith('n'):
        return 1e-6
    elif prefix.startswith('µ'):
        return 1e-3
    elif prefix.startswith('m'):
        return 1
    elif prefix.startswith('s'): # no prefix, just seconds
        return 1e3