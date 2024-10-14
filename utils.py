def href(url):
    return f"{ConsoleColor.BLUE}{ConsoleColor.UNDERLINE}{url}{ConsoleColor.END}"

def warning(text):
    print(f"{ConsoleColor.YELLOW}{ConsoleColor.BOLD}WARNING:{ConsoleColor.END} {text}")

class ConsoleColor:
    '''color strings for console output'''
    # Color
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[97m'
    # Style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # BackgroundColor
    BgBLACK = '\033[40m'
    BgRED = '\033[41m'
    BgGREEN = '\033[42m'
    BgORANGE = '\033[43m'
    BgBLUE = '\033[44m'
    BgPURPLE = '\033[45m'
    BgCYAN = '\033[46m'
    BgGRAY = '\033[47m'
    # End
    END = '\033[0m'


def dict_str(d, indent=0):
    def get_str(s):
        # if key is number, do not add quotes
        if isinstance(s, int) or isinstance(s, float):
            return str(s)
        elif isinstance(s, list):
            return str(s)
        elif isinstance(s, str):
            return f"\'{s}\'"
        else:
            return str(s)

    s = ''
    for key, value in d.items():
        key_str = get_str(key)
        
        # add to string
        if not isinstance(value, dict):
            value_str = get_str(value)
            s += '\t' * indent + f'{key_str}: {value_str},\n'
        else:
            s += '\t' * indent + f'{key_str}: {{\n'
            s += dict_str(value, indent+1)
            s += '\t' * indent + '},\n'
    return s


def print_dict(d, indent=0):
    print(dict_str(d, indent))