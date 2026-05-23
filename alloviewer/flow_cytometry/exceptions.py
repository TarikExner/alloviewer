import warnings


class NotCompensatedError(Exception):
    """Raised when compensated events are requested before compensation.

    Attributes
    ----------
    message : str
        Error message describing the missing compensation state.
    """

    def __init__(self):
        self.message = "File is not compensated. Please compensate first."
        super().__init__(self.message)


class EventsNotFoundError(Exception):
    """Raised when requested events are not stored in an FCS file object.

    Parameters
    ----------
    events : str
        Name of the missing event source.
    """

    def __init__(self, events: str):
        self.message = f"Events {events} are not contained within this FCSFile."
        super().__init__(self.message)


class TruncationWarning(Warning):
    """Warning emitted when event values exceed FCS channel ranges.

    Parameters
    ----------
    exceeded_channels : iterable
        Channel names with at least one value above the PnR range.
    number_exceeded_cells : iterable
        Counts of out-of-range values for the corresponding channels.
    """

    def __init__(
        self,
        exceeded_channels,
        number_exceeded_cells,
    ) -> None:
        self.message = (
            "Some data points exceed the PnR value. "
            "The data points are truncated. To avoid "
            "truncation, set the PnR value manually or "
            "pass `truncate_max_range = False`. The "
            "following counts were outside the channel range: "
        )

        channel_count_mapping = [
            f"{ch}: {count}"
            for ch, count in zip(exceeded_channels, number_exceeded_cells)
            if count != 0
        ]

        self.message += f"{', '.join(channel_count_mapping)}"
        warnings.warn(self.message, UserWarning)

    def __str__(self):
        """Return the warning message.

        Returns
        -------
        str
            Warning message representation.
        """
        return repr(self.message)


class InfRemovalWarning(Warning):
    """Warning emitted when rows with infinite values are removed.

    Parameters
    ----------
    message : str
        Warning message.
    """

    def __init__(
        self,
        message,
    ) -> None:
        self.message = message
        warnings.warn(message, UserWarning)

    def __str__(self):
        """Return the warning message.

        Returns
        -------
        str
            Warning message representation.
        """
        return repr(self.message)


class NaNRemovalWarning(Warning):
    """Warning emitted when rows with NaN values are removed.

    Parameters
    ----------
    message : str
        Warning message.
    """

    def __init__(
        self,
        message,
    ) -> None:
        self.message = message
        warnings.warn(message, UserWarning)

    def __str__(self):
        """Return the warning message.

        Returns
        -------
        str
            Warning message representation.
        """
        return repr(self.message)
