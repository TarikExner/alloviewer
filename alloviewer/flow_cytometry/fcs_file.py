from __future__ import annotations

import os
import warnings
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from flowio import FlowData
from flowio.exceptions import FCSParsingError
from flowutils.compensate import get_spill

from .exceptions import (
    EventsNotFoundError,
    InfRemovalWarning,
    NaNRemovalWarning,
    NotCompensatedError,
    TruncationWarning,
)
from .matrix import Matrix


class FCSFile:
    """Intermediate representation of an FCS sample file.

    The class loads an FCS file from disk, parses metadata and channel
    information, processes raw events, extracts or creates a compensation
    matrix, and stores compensated events.

    Parameters
    ----------
    file_dir : str
        Path to the FCS file.
    subsample : int or None, optional
        Number of events to randomly sample. If ``None``, all events are kept.
        If the requested number is larger than the available event count, all
        events are kept.
    truncate_max_range : bool, optional
        Whether to clip event values that exceed the channel range defined in
        the FCS metadata. The default is ``True``.

    Attributes
    ----------
    original_filename : str
        Basename of the input file.
    transform_status : str
        Current transformation status.
    gating_status : str
        Current gating status.
    version : str or None
        FCS version parsed from the file header.
    fcs_metadata : dict
        FCS text metadata.
    channels : pandas.DataFrame
        Channel metadata indexed by PnN channel names.
    original_events : numpy.ndarray
        Processed uncompensated events.
    compensated_events : numpy.ndarray
        Compensation-corrected events.
    event_count : int
        Number of events after optional subsampling and cleanup.
    matrix : Matrix
        Compensation matrix.
    compensation_status : str
        Current compensation status.
    """

    def __init__(
        self,
        file_dir: str,
        subsample: Optional[int] = None,
        truncate_max_range: bool = True,
    ) -> None:
        self.original_filename = os.path.basename(file_dir)

        raw_data = self._load_fcs_file_from_disk(
            file_dir,
            ignore_offset_error=False,
        )

        self.transform_status = "untransformed"
        self.gating_status = "ungated"

        self._fcs_event_count = self._parse_event_count(raw_data)
        self.version = self._parse_fcs_version(raw_data)
        self.fcs_metadata = self._parse_fcs_metadata(raw_data)
        self.channels = self._parse_channel_information(raw_data)
        self.original_events = self._parse_and_process_original_events(
            raw_data,
            subsample,
            truncate_max_range,
        )
        self.event_count = self.original_events.shape[0]
        self.matrix = self._parse_compensation_matrix_from_fcs()

        self.compensated_events = self.matrix.apply(self)
        self.compensation_status = "compensated"

    def __repr__(self) -> str:
        """Return a compact string representation of the FCS file.

        Returns
        -------
        str
            Representation including version, filename, channel count, event
            count, gating status, compensation status, and transform status.
        """
        return (
            f"{self.__class__.__name__}("
            f"v{self.version}, "
            f"{self.original_filename}, "
            f"{self.channels.shape[0]} channels, "
            f"{self.event_count} events, "
            f"gating status: {self.gating_status}, "
            f"compensation status: {self.compensation_status}, "
            f"transform status: {self.transform_status})"
        )

    def get_events(self, source: Literal["raw", "comp"]) -> np.ndarray:
        """Return events from the requested data source.

        Parameters
        ----------
        source : {'raw', 'comp'}
            Event source. ``"raw"`` returns uncompensated events and ``"comp"``
            returns compensated events.

        Returns
        -------
        numpy.ndarray
            Event matrix with shape ``(n_events, n_channels)``.

        Raises
        ------
        EventsNotFoundError
            If raw events are unavailable.
        NotCompensatedError
            If compensated events are requested before compensation is present.
        NotImplementedError
            If ``source`` is not ``"raw"`` or ``"comp"``.
        """
        if source == "raw":
            return self._get_original_events()
        elif source == "comp":
            return self._get_compensated_events()
        else:
            raise NotImplementedError(
                "Only Raw ('raw') and compensated events ('comp') can be fetched."
            )

    def _get_original_events(self) -> np.ndarray:
        """Return uncompensated events.

        Returns
        -------
        numpy.ndarray
            Processed uncompensated event matrix.

        Raises
        ------
        EventsNotFoundError
            If ``original_events`` is missing.
        """
        if self.original_events is None:
            raise EventsNotFoundError("original")

        return self.original_events

    def _get_compensated_events(self) -> np.ndarray:
        """Return compensated events.

        Returns
        -------
        numpy.ndarray
            Compensation-corrected event matrix.

        Raises
        ------
        NotCompensatedError
            If the file is not marked as compensated.
        """
        if self.compensation_status != "compensated":
            raise NotCompensatedError()

        return self.compensated_events

    def get_channel_index(self, channel_label: str) -> int:
        """Return the zero-based column index for a channel label.

        Parameters
        ----------
        channel_label : str
            PnN channel label.

        Returns
        -------
        int
            Zero-based event matrix column index.

        Raises
        ------
        IndexError
            If the channel label is not present.
        """
        return (
            self.channels.loc[
                self.channels.index == channel_label,
                "channel_numbers",
            ].iloc[0]
            - 1
        )

    def to_df(
        self,
        source: Literal["raw", "comp"],
        colnames: Literal["pnn", "pns"] = "pnn",
    ) -> pd.DataFrame:
        """Return events as a pandas DataFrame.

        Parameters
        ----------
        source : {'raw', 'comp'}
            Event source passed to :meth:`get_events`.
        colnames : {'pnn', 'pns'}, optional
            Requested channel naming mode. The current implementation uses PnN
            labels from ``self.channels.index``. The default is ``"pnn"``.

        Returns
        -------
        pandas.DataFrame
            Event matrix with channels as columns.
        """
        return pd.DataFrame(
            data=self.get_events(source),
            columns=self.channels.index,
        )

    def _parse_compensation_matrix_from_fcs(self) -> Matrix:
        """Parse or create a compensation matrix.

        Returns
        -------
        Matrix
            Compensation matrix from FCS spillover metadata. If no spillover
            metadata is present, an identity matrix is created for detector
            channels.

        Raises
        ------
        AssertionError
            If spillover matrix dimensions do not match parsed detectors or
            fluorochrome labels.
        """
        mtx_kwds = ["spill", "spillover"]

        if not any(k in self.fcs_metadata for k in mtx_kwds):
            detectors = [
                channel
                for channel in self.channels.index
                if any(k not in channel.lower() for k in ["fsc", "ssc", "time"])
            ]
            detector_n = len(detectors)
            fluorochromes = self.channels.loc[
                self.channels.index.isin(detectors),
                "pns",
            ].tolist()

            return Matrix(
                matrix_id="empty_matrix",
                detectors=detectors,
                fluorochromes=fluorochromes,
                spill_data_or_file=np.eye(N=detector_n, M=detector_n),
            )

        kwd_present = next((k for k in mtx_kwds if k in self.fcs_metadata), None)
        matrix, detectors = get_spill(self.fcs_metadata[kwd_present])
        fluorochromes = self.channels.loc[
            self.channels.index.isin(detectors),
            "pns",
        ].tolist()

        assert matrix.shape[0] == len(detectors)
        assert matrix.shape[0] == len(fluorochromes)

        return Matrix(
            matrix_id="acquisition_defined",
            detectors=detectors,
            fluorochromes=fluorochromes,
            spill_data_or_file=matrix,
        )

    def _parse_event_count(self, fcs_data: FlowData) -> int:
        """Return the event count stored in the FCS file.

        Parameters
        ----------
        fcs_data : FlowData
            Parsed FCS data object.

        Returns
        -------
        int
            Number of events reported by the file.
        """
        return fcs_data.event_count

    def _subsample_events(self, events: np.ndarray, size: int) -> np.ndarray:
        """Randomly subsample events.

        Parameters
        ----------
        events : numpy.ndarray
            Event matrix with shape ``(n_events, n_channels)``.
        size : int
            Number of events to sample.

        Returns
        -------
        numpy.ndarray
            Subsampled event matrix, or the original matrix if ``size`` is at
            least the number of available events.

        Notes
        -----
        Sampling is performed with replacement.
        """
        if size >= events.shape[0]:
            return events

        return events[
            np.random.randint(
                events.shape[0],
                size=size,
            ),
            :,
        ]

    def _parse_and_process_original_events(
        self,
        fcs_data: FlowData,
        subsample: Optional[int],
        truncate_max_range: bool,
    ) -> np.ndarray:
        """Parse, optionally subsample, and process raw events.

        Parameters
        ----------
        fcs_data : FlowData
            Parsed FCS data object.
        subsample : int or None
            Optional number of events to sample.
        truncate_max_range : bool
            Whether to clip values above channel ranges.

        Returns
        -------
        numpy.ndarray
            Processed uncompensated event matrix.
        """
        tmp_orig_events = self._parse_original_events(fcs_data)

        if subsample is not None:
            tmp_orig_events = self._subsample_events(
                tmp_orig_events,
                subsample,
            )

        tmp_orig_events = self._process_original_events(
            tmp_orig_events,
            truncate_max_range,
        )

        return tmp_orig_events

    def _process_original_events(
        self,
        tmp_orig_events: np.ndarray,
        truncate_max_range: bool,
    ) -> np.ndarray:
        """Apply raw event cleanup and metadata-based corrections.

        Parameters
        ----------
        tmp_orig_events : numpy.ndarray
            Raw event matrix.
        truncate_max_range : bool
            Whether to clip values above channel ranges.

        Returns
        -------
        numpy.ndarray
            Processed event matrix.

        Notes
        -----
        Processing includes optional range clipping, removal of NaN and
        infinite values, time-step adjustment, logarithmic decade adjustment,
        and channel gain correction.
        """
        if truncate_max_range:
            tmp_orig_events = self._adjust_range(tmp_orig_events)

        tmp_orig_events = self._remove_nans_from_events(tmp_orig_events)
        tmp_orig_events = self._adjust_time_channel(tmp_orig_events)
        tmp_orig_events = self._adjust_decades(tmp_orig_events)
        tmp_orig_events = self._adjust_channel_gain(tmp_orig_events)

        return tmp_orig_events

    def _adjust_range(self, arr: np.ndarray) -> np.ndarray:
        """Clip event values to their FCS channel ranges.

        Parameters
        ----------
        arr : numpy.ndarray
            Event matrix.

        Returns
        -------
        numpy.ndarray
            Range-clipped event matrix when any value exceeds its channel range;
            otherwise the original matrix.
        """
        channel_ranges = self.channels["pnr"].to_numpy(dtype=arr.dtype)
        range_exceeded_cells = arr > channel_ranges
        range_exceeded_channels = range_exceeded_cells.any(axis=0)

        if any(range_exceeded_channels):
            exceeded_channels = self.channels[range_exceeded_channels].index.tolist()
            number_of_exceeded_cells = range_exceeded_cells.sum(axis=0)

            TruncationWarning(exceeded_channels, number_of_exceeded_cells)

            array_mins = np.min(arr, axis=0).astype(arr.dtype)

            return np.clip(
                arr,
                array_mins,
                channel_ranges,
                dtype=np.float32,
            )

        return arr

    def _remove_nans_from_events(self, arr: np.ndarray) -> np.ndarray:
        """Remove rows containing NaN or infinite values.

        Parameters
        ----------
        arr : numpy.ndarray
            Event matrix.

        Returns
        -------
        numpy.ndarray
            Event matrix with invalid rows removed.
        """
        if np.isinf(arr).any():
            idxs = np.argwhere(np.isinf(arr))[:, 0]
            arr = arr[~np.in1d(np.arange(arr.shape[0]), idxs)]
            warning_message = (
                f"{idxs.shape[0]} cells were removed from "
                f"{self.original_filename} due to the presence of Infinity values"
            )
            InfRemovalWarning(warning_message)

        if np.isnan(arr).any():
            idxs = np.argwhere(np.isnan(arr))[:, 0]
            arr = arr[~np.in1d(np.arange(arr.shape[0]), idxs)]
            warning_message = (
                f"{idxs.shape[0]} cells were removed from "
                f"{self.original_filename} due to the presence of NaN values"
            )
            NaNRemovalWarning(warning_message)

        return arr

    def _adjust_channel_gain(self, events: np.ndarray) -> np.ndarray:
        """Divide event values by channel gain.

        Parameters
        ----------
        events : numpy.ndarray
            Event matrix.

        Returns
        -------
        numpy.ndarray
            Gain-adjusted event matrix.
        """
        channel_gains = (
            self.channels.sort_values("channel_numbers")["png"].to_numpy()
        )

        return np.divide(events, channel_gains)

    def _adjust_decades(self, events: np.ndarray) -> np.ndarray:
        """Apply logarithmic decade scaling from FCS channel metadata.

        Parameters
        ----------
        events : numpy.ndarray
            Event matrix.

        Returns
        -------
        numpy.ndarray
            Event matrix with logarithmic channels adjusted.
        """
        for (decades, log0), channel_number, channel_range in zip(
            self.channels["pne"],
            self.channels["channel_numbers"],
            self.channels["pnr"],
        ):
            if decades > 0:
                events[:, channel_number - 1] = (
                    10 ** (decades * events[:, channel_number - 1] / channel_range)
                ) * log0

        return events

    def _adjust_time_channel(self, events: np.ndarray) -> np.ndarray:
        """Apply the FCS time-step multiplier to the time channel.

        Parameters
        ----------
        events : numpy.ndarray
            Event matrix.

        Returns
        -------
        numpy.ndarray
            Event matrix with the time channel adjusted when present.
        """
        if self._time_channel_exists:
            time_index, time_step = self._find_time_channel()
            events[:, time_index] = events[:, time_index] * time_step

        return events

    def _find_time_channel(self) -> tuple[int, float]:
        """Return the time channel index and time step.

        Returns
        -------
        tuple of int and float
            Zero-based time channel index and time-step multiplier.

        Raises
        ------
        IndexError
            If no time channel exists.
        """
        time_step = (
            float(self.fcs_metadata["timestep"])
            if "timestep" in self.fcs_metadata
            else 1.0
        )
        time_index = (
            int(
                self.channels.loc[
                    self.channels.index.isin(["Time", "time"]),
                    "channel_numbers",
                ].iloc[0]
            )
            - 1
        )

        return time_index, time_step

    def _time_channel_exists(self) -> bool:
        """Return whether a time channel is present.

        Returns
        -------
        bool
            ``True`` if a ``"Time"`` or ``"time"`` channel exists.
        """
        return any(
            time_symbol in self.channels.index
            for time_symbol in ["Time", "time"]
        )

    def _parse_original_events(self, fcs_data: FlowData) -> np.ndarray:
        """Parse original events from a FlowData object.

        Parameters
        ----------
        fcs_data : FlowData
            Parsed FCS data object.

        Returns
        -------
        numpy.ndarray
            Event matrix with dtype ``float32`` and C-contiguous layout.
        """
        return np.array(
            fcs_data.events,
            dtype=np.float32,
            order="C",
        ).reshape(-1, fcs_data.channel_count)

    def _remove_disallowed_characters_from_string(
        self,
        input_string: str,
    ) -> str:
        """Replace disallowed label characters with underscores.

        Parameters
        ----------
        input_string : str
            Input channel label.

        Returns
        -------
        str
            Cleaned channel label.
        """
        for char in [" ", "/", "-"]:
            if char in input_string:
                input_string = input_string.replace(char, "_")

        return input_string

    def _parse_channel_information(self, fcs_data: FlowData) -> pd.DataFrame:
        """Parse channel metadata into a DataFrame.

        Parameters
        ----------
        fcs_data : FlowData
            Parsed FCS data object.

        Returns
        -------
        pandas.DataFrame
            Channel metadata indexed by PnN labels and sorted by FCS channel
            number. Columns include ``"pns"``, ``"png"``, ``"pne"``, ``"pnr"``,
            and ``"channel_numbers"``.
        """
        channels: dict = fcs_data.channels

        pnn_labels = [
            self._parse_pnn_label(channels, channel_number)
            for channel_number in channels
        ]
        pns_labels = [
            self._parse_pns_label(channels, channel_number)
            for channel_number in channels
        ]
        channel_gains = [
            self._parse_channel_gain(channel_number)
            for channel_number in channels
        ]
        channel_lin_log = [
            self._parse_channel_lin_log(channel_number)
            for channel_number in channels
        ]
        channel_ranges = [
            self._parse_channel_range(channel_number)
            for channel_number in channels
        ]

        channel_numbers = [int(k) for k in channels]

        channel_frame = pd.DataFrame(
            data={
                "pns": pns_labels,
                "png": channel_gains,
                "pne": channel_lin_log,
                "pnr": channel_ranges,
                "channel_numbers": channel_numbers,
            },
            index=pnn_labels,
        )

        return channel_frame.sort_values("channel_numbers")

    def _parse_pnn_label(
        self,
        channels: dict,
        channel_number: str,
    ) -> str:
        """Parse the PnN channel label.

        Parameters
        ----------
        channels : dict
            Channel metadata from FlowData.
        channel_number : str
            FCS channel number as a string.

        Returns
        -------
        str
            PnN label.
        """
        return channels[channel_number]["PnN"]

    def _parse_pns_label(
        self,
        channels: dict,
        channel_number: str,
    ) -> str:
        """Parse the PnS channel label.

        Parameters
        ----------
        channels : dict
            Channel metadata from FlowData.
        channel_number : str
            FCS channel number as a string.

        Returns
        -------
        str
            Cleaned PnS label, or an empty string if no PnS value is present.
        """
        try:
            return self._remove_disallowed_characters_from_string(
                channels[channel_number]["PnS"]
            )
        except KeyError:
            return ""

    def _parse_channel_range(
        self,
        channel_number: str,
    ) -> Union[int, float]:
        """Parse the PnR channel range.

        Parameters
        ----------
        channel_number : str
            FCS channel number as a string.

        Returns
        -------
        int or float
            Parsed channel range. Returns ``numpy.inf`` for malformed range
            values that cannot be parsed as integers.

        Raises
        ------
        ValueError
            If parsing fails for a reason other than malformed integer text.
        """
        try:
            return int(self.fcs_metadata[f"p{channel_number}r"])
        except ValueError as e:
            if "invalid literal for int() with base 10" in str(e):
                return np.inf

            raise ValueError from e

    def _parse_channel_lin_log(
        self,
        channel_number: str,
    ) -> tuple[float, float]:
        """Parse the PnE linear/logarithmic scaling metadata.

        Parameters
        ----------
        channel_number : str
            FCS channel number as a string.

        Returns
        -------
        tuple of float
            ``(decades, log0)`` values. Returns ``(0.0, 0.0)`` when PnE is
            missing.
        """
        try:
            decades, log0 = [
                float(x)
                for x in self.fcs_metadata[f"p{channel_number}e"].split(",")
            ]

            if log0 == 0.0 and decades != 0:
                log0 = 1.0

            return decades, log0
        except KeyError:
            return 0.0, 0.0

    def _parse_channel_gain(
        self,
        channel_number: str,
    ) -> float:
        """Parse the PnG channel gain.

        Parameters
        ----------
        channel_number : str
            FCS channel number as a string.

        Returns
        -------
        float
            Channel gain. Time channels and missing PnG values return ``1.0``.
        """
        if self.fcs_metadata[f"p{channel_number}n"] in ["Time", "time"]:
            return 1.0

        try:
            return float(self.fcs_metadata[f"p{channel_number}g"])
        except KeyError:
            return 1.0

    def _parse_fcs_metadata(self, fcs_data: FlowData) -> dict:
        """Return FCS text metadata.

        Parameters
        ----------
        fcs_data : FlowData
            Parsed FCS data object.

        Returns
        -------
        dict
            FCS text segment metadata.
        """
        return fcs_data.text

    def _parse_fcs_version(self, fcs_data: FlowData) -> Optional[str]:
        """Return the FCS version from the file header.

        Parameters
        ----------
        fcs_data : FlowData
            Parsed FCS data object.

        Returns
        -------
        str or None
            FCS version string, or ``None`` if unavailable.
        """
        try:
            return str(fcs_data.header["version"])
        except KeyError:
            return None

    def _load_fcs_file_from_disk(
        self,
        file_dir: str,
        ignore_offset_error: bool,
    ) -> FlowData:
        """Load an FCS file from disk.

        Parameters
        ----------
        file_dir : str
            Path to the FCS file.
        ignore_offset_error : bool
            Whether FlowIO should ignore FCS offset errors.

        Returns
        -------
        FlowData
            Parsed FCS data object.

        Warns
        -----
        UserWarning
            If parsing fails with the requested offset setting and the file is
            retried with ``ignore_offset_error=True``.
        """
        try:
            return FlowData(file_dir, ignore_offset_error)
        except FCSParsingError:
            warnings.warn(
                "FACSPy IO: FCS file could not be read with "
                f"ignore_offset_error set to {ignore_offset_error}. "
                "Parameter is set to True."
            )
            return FlowData(file_dir, ignore_offset_error=True)
