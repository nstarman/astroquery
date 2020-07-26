# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Astroquery Grouped Tables.
"""

##############################################################################
# IMPORTS

# BUILT-IN

import pathlib
import typing as T
import warnings

from collections import OrderedDict
import collections.abc as cabc


# THIRD PARTY

import asdf

from astropy.table import Table, QTable
from astropy.utils.introspection import resolve_name
from astropy.utils.metadata import MetaData
from astropy.utils.collections import HomogeneousList

import numpy as np

# PROJECT-SPECIFIC

from ..exceptions import InputWarning


##############################################################################
# CODE


class TablesList(HomogeneousList):
    """Grouped Tables.

    A subclass of list that contains only elements of a given type or
    types.  If an item that is not of the specified type is added to
    the list, a `TypeError` is raised. Also includes some pretty printing
    methods for an OrderedDict of :class:`~astropy.table.Table` objects.

    """

    _types = None

    meta = MetaData(copy=False)

    def __init__(
        self,
        inp=[],
        *,
        name: T.Optional[str] = None,
        reference: T.Optional[T.Any] = None,
        **metadata,
    ):
        """Astroquery-style table list.

        Parameters
        ----------
        inp : sequence, optional
            An initial set of tables.
        name : str, optional
            name of the list of tables.
        reference : citation, optional
            citation.
        **metadata : Any
            arguments into meta

        """
        # meta
        self.meta["name"] = name
        self.meta["reference"] = reference
        for k, v in metadata.items():
            self.meta[k] = v

        inp = self._validate(inp)  # ODict, & ensure can assign values

        # convert input to correct to type
        if self._types is not None:
            converter = self._types
            for k, val in inp.items():
                inp[k] = converter(val)

        # Add the input
        # need to bypass HomogeneousList init, which uses ``extend``
        list.__init__(self, inp.values())
        # and the ordering of the names
        self._names = list(inp.keys())

    # -----------------

    def _assert(self, x):
        """Check `x` is correct type (set by _type)."""
        if self._types is None:  # allow any type
            return
        super()._assert(x)

    def _validate(self, value):
        """Validate `value` compatible with table."""
        if isinstance(value, TablesList):  # tablelist or subclass
            pass
        elif not isinstance(value, OrderedDict):
            try:
                value = OrderedDict(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "Input to TableList must be an OrderedDict "
                    "or list of (k,v) pairs"
                )

        return value

    # -----------------
    # Properties

    @property
    def name(self) -> str:
        """Name."""
        return self.meta["name"]

    @property
    def __reference__(self):
        """Get reference from metadata, if exists."""
        return self.meta.get("reference", None)

    # -----------------
    # Dictionary methods

    def keys(self):
        return cabc.KeysView(self._names)

    def values(self):
        """Set-like object providing a view on tables."""
        return cabc.Iterator.__iter__(self)

    def items(self):
        """Generator providing iterator over name, table."""
        return cabc.ItemsView(zip(self._names, self))

    # -----------------
    # Get / Set

    def index(self, key: str) -> int:
        """Index of `key`.

        Parameters
        ----------
        key : str

        Returns
        -------
        int

        """
        return self._names.find

    def __getslice__(self, slicer: T.Union[slice, T.Any]):
        """Slice TableList.

        slice(star, stop, step)
        supports string as slice start or stop

        Examples
        --------
        First creating tables
            >>> vizier = Table([[1], [2]], names=["a", "b"])
            >>> simbad = Table([[3], [5]], names=["d", "e"])
            >>> od = TableList([("vizier", vizier), ("simbad", simbad)],
            ...                name="test")

        Example of standard slicing

            >>> od[1:]  # doctest: +SKIP
            Table([[3], [5]], names=["d", "e"])

        Now slicing with string

            >>> od["vizier":]  # doctest: +SKIP
            Table([[3], [5]], names=["d", "e"])

        """
        # check if slice, if it is, then check for string inputs
        # it it isn't a slice, then still try to apply to list of values
        if isinstance(slicer, slice):

            start, stop = slicer.start, slicer.stop
            # string replacement for start, stop values
            # replace by int
            if isinstance(start, str):
                start = self.index(start)
            if isinstance(stop, str):
                stop = self.index(stop)
            slicer = slice(start, stop, slicer.step)

        return self[slicer]

    def __getitem__(self, key: T.Union[int, slice, str]):
        """Get item or slice.

        Parameters
        ----------
        key : str or int or slice
            if str, the dictionary key.
            if int, the dictionary index
            if slice, slices dictionary.values
            supports string as slice start or stop

        Returns
        -------
        Table

        Raises
        ------
        TypeError
            if key is not int or key

        """
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, slice):
            return self.__getslice__(key)
        # else:
        return super().__getitem__(self.index(key))

    def __setitem__(self, key: str, value):
        """Set item, but only if right type (managed by super)."""
        if not isinstance(key, str):
            raise TypeError

        # first try if exists
        if key in self._dict:
            ind = self._dict[key]
            return super().__setitem__(ind, value)  # (super _assert)

        # else append to end
        else:
            ind = len(self)
            self._dict[key] = ind
            return super().append(value)  # (super _assert)

    def __delitem__(self, key):
        """Delete Item. Forbidden."""
        raise NotImplementedError("Forbidden.")

    def update(self, other):
        """Update TableList using OrderedDict update method."""
        values = self._validate(other)  # first make sure adding a key-val pair
        for k, v in values.items():  # TODO better
            self[k] = v  # setitem manages _dict

    def extend(self, other):
        """Extend TableList. Unlike update, cannot have duplicate keys."""
        values = self._validate(other)  # first make sure adding a key-val pair

        if any((k in self.keys() for k in values.keys())):
            raise ValueError("cannot have duplicate keys")

        self.update(values)

    def __iadd__(self, other):
        """Add in-place."""
        return super().__iadd__(other)

    def append(self, key: str, value):
        """Append, if unique key and right type (managed by super)."""
        if key in self._dict:
            raise ValueError("cannot append duplicate key.")

        self._dict[key] = len(self)
        return super().append(value)

    def pop(self):
        """Pop. Forbidden."""
        raise NotImplementedError("Forbidden.")

    def insert(self, value):
        """Insert. Forbidden."""
        raise NotImplementedError("Forbidden.")

    # -----------------
    # string representation

    def __repr__(self):
        """String representation.

        Overrides the `OrderedDict.__repr__` method to return a simple summary
        of the `TableList` object.

        Returns
        -------
        str

        """
        return self.format_table_list()

    def format_table_list(self) -> str:
        """String Representation of list of Tables.

        Prints the names of all :class:`~astropy.table.Table` objects, with
        their respective number of row and columns, contained in the
        `TableList` instance.

        Returns
        -------
        str

        """
        ntables = len(list(self.keys()))
        if ntables == 0:
            return "Empty {cls}".format(cls=self.__class__.__name__)

        header_str = "{cls} with {keylen} tables:".format(
            cls=self.__class__.__name__, keylen=ntables
        )
        body_str = "\n".join(
            [
                "\t'{t_number}:{t_name}' with {ncol} column(s) "
                "and {nrow} row(s) ".format(
                    t_number=t_number,
                    t_name=t_name,
                    nrow=len(self[t_number]),
                    ncol=len(self[t_number].colnames),
                )
                for t_number, t_name in enumerate(self.keys())
            ]
        )

        return "\n".join([header_str, body_str])

    def print_table_list(self):
        """Print Table List.

        calls ``format_table_list``

        """
        print(self.format_table_list())

    def pprint(self, **kwargs):
        """Helper function to make API more similar to astropy.Tables."""
        if kwargs != {}:
            warnings.warn(
                "TableList is a container of astropy.Tables.", InputWarning
            )

        self.print_table_list()

    # -----------------
    # I/O

    def _save_table_iter(self, format, **table_kw):
        for i, name in enumerate(self.keys()):  # order-preserving

            # get kwargs for table writer
            # first get all general keys (by filtering)
            # then update with table-specific dictionary (if present)
            kw = {
                k: v for k, v in table_kw.items() if not k.startswith("table_")
            }
            kw.update(table_kw.get("table_" + name, {}))

            if isinstance(format, str):
                fmt = format
            else:
                fmt = format[i]

            yield name, fmt, kw

    def write_asdf(
        self,
        drct: str,
        split: bool = False,
        serialize_method: T.Union[None, str, T.Mapping] = None,
        **table_kw,
    ):
        """Write to ASDF.

        Parameters
        ----------
        drct : str
            The drct path.
        split : bool
            Whether to save the tables as individual drct
            with `drct` coordinating by reference.

        serialize_method : str, dict, optional
            Serialization method specifier for columns.

        **table_kw
            kwargs into each table.
            1. dictionary with table name as key
            2. General keys

        """
        # -----------
        # Path checks

        path = pathlib.Path(drct)

        if path.suffix == "":  # no suffix
            path = path.with_suffix(".asdf")
        if path.suffix != ".asdf":  # ensure only asdf
            raise ValueError("file type must be `.asdf`.")

        drct = path.parent  # directory in which to save

        # -----------
        # Table Types

        if self._types is None:  # TablesList
            table_type = tuple(
                [
                    tp.__class__.__module__ + "." + tp.__class__.__name__
                    for tp in self.values()
                ]
            )
        else:  # Standard Type
            table_type = self._types.__module__ + "." + self._types.__name__

        # -----------
        # Saving

        tl = asdf.AsdfFile()
        tl.tree["meta"] = tuple(self.meta.items())
        tl.tree["table_names"] = tuple(self.keys())  # in order
        tl.tree["save_format"] = "asdf"
        tl.tree["table_type"] = table_type

        if not split:  # save as single file
            for name in self.keys():  # add to tree
                tl.tree[name] = self[name]

        else:  # save as individual files
            for name, fmt, kw in self._save_table_iter("asdf", **table_kw):

                # name of table
                table_path = drct.joinpath(name).with_suffix(".asdf")

                # internal save
                self[name].write(
                    table_path,
                    format=fmt,
                    data_key=name,
                    serialize_method=serialize_method,
                    **kw,
                )

                # save by relative reference
                with asdf.open(table_path) as f:
                    tl.tree[name] = f.make_reference(path=[name])

        # /if
        tl.write_to(str(path))  # actually save

    def write(
        self,
        drct: str,
        format="asdf",
        split=True,
        serialize_method=None,
        **table_kw,
    ):
        """Write to ASDF.

        Parameters
        ----------
        drct : str
            The main directory path.
        format : str or list, optional
            save format. default "asdf"
            can be list of same length as TableList
        split : bool, optional
            *Applies to asdf `format` only*

            Whether to save the tables as individual file
            with `file` coordinating by reference.

        serialize_method : str, dict, optional
            Serialization method specifier for columns.

        **table_kw
            kwargs into each table.
            1. dictionary with table name as key
            2. General keys

        """
        if format == "asdf":
            self.write_asdf(
                drct=drct,
                split=split,
                serialize_method=serialize_method,
                **table_kw,
            )

        # -----------
        # Path checks

        path = pathlib.Path(drct)

        if path.suffix == "":  # no suffix
            path = path.with_suffix(".asdf")

        if path.suffix != ".asdf":  # ensure only asdf
            raise ValueError("file type must be `.asdf`.")

        drct = path.parent  # directory in which to save

        # -----------
        # TableType

        if self._types is None:
            table_type = [
                tp.__class__.__module__ + "." + tp.__class__.__name__
                for tp in self.values()
            ]
        else:
            table_type = self._types.__module__ + "." + self._types.__name__

        # -----------
        # Saving

        tl = asdf.AsdfFile()
        tl.tree["meta"] = tuple(self.meta.items())
        tl.tree["table_names"] = tuple(self.keys())  # in order
        tl.tree["save_format"] = format
        tl.tree["table_type"] = table_type

        for name, fmt, kw in self._save_table_iter(format, **table_kw):

            # name of table
            table_path = drct.joinpath(name)
            if table_path.suffix == "":  # TODO currently always. CLEANUP
                table_path = table_path.with_suffix("." + fmt.split(".")[-1])

            # internal save
            self[name].write(
                table_path, format=fmt, serialize_method=serialize_method, **kw
            )

            # save by relative reference
            tl.tree[name] = str(table_path.relative_to(drct))

        # /if
        tl.write_to(str(path))  # save directory

    @classmethod
    def _read_table_iter(cls, f, format, **table_kw):
        names = f.tree["table_names"]
        # table type, for casting
        # so that QTableList can open a saved TableList correctly
        # TablesList specifies no type, so must rely on saved info
        if cls._types is None:
            table_type = f.tree["table_type"]
            if not isinstance(table_type, cabc.Sequence):
                table_type = [table_type] * len(names)
            ttypes = [resolve_name(t) for t in table_type]
        else:
            ttypes = [cls._types] * len(names)

        for i, name in enumerate(names):  # order-preserving

            if isinstance(format, str):
                fmt = format
            else:
                fmt = format[i]

            # get kwargs for table writer
            # first get all general keys (by filtering)
            # then update with table-specific dictionary (if present)
            kw = {
                k: v for k, v in table_kw.items() if not k.startswith("table_")
            }
            kw.update(table_kw.get("table_" + name, {}))

            yield name, ttypes[i], fmt, kw

    @classmethod
    def read_asdf(cls, drct: str, **table_kw):
        """Read from ASDF.

        Parameters
        ----------
        drct : str
            The drct path to the coordinating object.
        strict : bool
            TODO, whether to only permit object of `_type`

        Returns
        -------
        cls : cls-type
            with tables in "table_names"

        Notes
        -----
        .. todo::

            single-table read permits assigning units and descriptions.

        """

        return cls.read(drct, format="asdf", **table_kw)

    @classmethod
    def read(
        cls,
        drct: str,
        format: T.Union[str, T.Sequence] = None,
        suffix: T.Optional[str] = None,
        **table_kw,
    ):
        """Write to ASDF.

        Parameters
        ----------
        drct : str
            The main directory path.
        format : str or list, optional
            read format. default "asdf"
            can be list of same length as TableList
        suffix : str, optional
            suffix to apply to table names.
            will be superceded by an "fnames" argument, when added

        **table_kw
            kwargs into each table.
            1. dictionary with table name as key
            2. General keys

        """
        # -----------
        # Path checks

        path = pathlib.Path(drct)

        if path.suffix == "":  # no suffix
            path = path.with_suffix(".asdf")

        if path.suffix != ".asdf":  # ensure only asdf
            raise ValueError("file type must be `.asdf`.")

        drct = path.parent  # directory

        # -----------
        # Read

        TL = cls()
        with asdf.open(path) as f:

            # load in the metadata
            TL.meta = OrderedDict(f.tree["meta"])

            if format is None:
                format = f.tree["save_format"]

            if format == "asdf":  # special case
                f.resolve_references()  # for split=True

                # iterate through tables
                for name, ttype, fmt, kw in cls._read_table_iter(
                    f, "asdf", **table_kw
                ):
                    TL[name] = ttype(f.tree[name], **kw)

            else:

                # iterate through tables
                for name, reader, fmt, kw in cls._read_table_iter(
                    f, format, **table_kw
                ):

                    # name of table
                    if suffix is None:
                        suffix = fmt.split(".")[-1]
                    if not suffix.startswith("."):
                        suffix = "." + suffix
                    table_path = drct.joinpath(name).with_suffix(suffix)

                    TL[name] = reader.read(str(table_path), format=fmt, **kw)

        return TL

    def copy(self):
        """Shallow copy."""
        out = self.__class__(self)
        out.meta = self.meta

        return out


# /class


# -------------------------------------------------------------------


class TableList(TablesList):
    """Homogeneous TablesList."""

    _types = Table


# /class


# -------------------------------------------------------------------


class QTableList(TablesList):
    """Astroquery-style QTable list.

    Attributes
    ----------
    meta : :class:`~astropy.utils.metadata.MetaData`

    """

    _types = QTable


# -------------------------------------------------------------------
