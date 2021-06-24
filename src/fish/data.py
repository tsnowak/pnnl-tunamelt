from pathlib import Path

def get_file_path(  f: str,
                    data_paths: list,
                    return_first: bool = True,
                    absolute: bool = False) -> str:

    def d_func(x):
        if absolute:
            return Path(x).absolute()
        else:
            return Path(x)

    data_paths = [d_func(d) if isinstance(d, str) else exit(1) for d in data_paths]
    f_paths = []
    for d in data_paths:
        f_paths += d.glob('**/*')

    out = []
    for f_path in f_paths:
        if f_path.name == f:
            if return_first:
                return f_path
            else:
                out.append(f_path)
        
    return out