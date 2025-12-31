
import warnings

class NotCompensatedError(Exception):

    def __init__(self):
        self.message = "File is not compensated. Please compensate first."
        super().__init__(self.message)

class TruncationWarning(Warning):
    def __init__(self,
                 exceeded_channels,
                 number_exceeded_cells) -> None:
        self.message = "Some data points exceed the PnR value. " + \
                       "The data points are truncated. To avoid " + \
                       "truncation, set the PnR value manually or " + \
                       "pass `truncate_max_range = False`. The " + \
                       "following counts were outside the channel range: "
        channel_count_mapping = [f"{ch}: {count}"
                                 for ch, count in
                                 zip(exceeded_channels, number_exceeded_cells)
                                 if count != 0]
        self.message += f"{', '.join(channel_count_mapping)}"
        warnings.warn(self.message, UserWarning)

    def __str__(self):
        return repr(self.message)

class InfRemovalWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)

    def __str__(self):
        return repr(self.message)

class NaNRemovalWarning(Warning):
    def __init__(self,
                 message) -> None:
        self.message = message
        warnings.warn(message, UserWarning)
    
    def __str__(self):
        return repr(self.message)



