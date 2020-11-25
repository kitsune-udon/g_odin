import inspect
from argparse import Namespace


def extract_kwargs_from_argparse_args(cls, args, **kwargs):
    assert isinstance(args, Namespace)
    params = vars(args)

    tmp_kwargs = {}

    for c in inspect.getmro(cls):
        valid_kwargs = inspect.signature(c.__init__).parameters

        kv = dict((name, params[name])
                  for name in valid_kwargs if name in params)
        tmp_kwargs.update(kv)

    tmp_kwargs.update(**kwargs)

    return tmp_kwargs


def from_argparse_args(cls, args, **kwargs):
    return cls(**extract_kwargs_from_argparse_args(cls, args, **kwargs))
