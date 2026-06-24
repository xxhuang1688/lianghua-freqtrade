"""Auditable candidates adapted from public strategy sources.

CommunityMACDCCI and CommunityADXMomentum are derived from
freqtrade/freqtrade-strategies (GPL-3.0).  Risk settings and timeframes were
changed for this workspace.  TimeSeriesMomentum follows the long-only form of
Moskowitz, Ooi, and Pedersen's time-series momentum concept.
"""

from pandas import DataFrame

import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from technical import qtpylib


class CommunityMACDCCI(IStrategy):
    """MACD crossover confirmed by CCI, slowed from 5m to 1h."""

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "1h"
    startup_candle_count = 60
    stoploss = -0.05
    minimal_roi = {"48": 0.0, "24": 0.01, "0": 0.025}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["macd"], dataframe["macdsignal"])
                & (dataframe["cci"] <= -50)
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "macd_cci")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])
                    | (dataframe["cci"] >= 100)
                )
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "macd_or_cci_exit")
        return dataframe


class CommunityADXMomentum(IStrategy):
    """ADX, directional movement and momentum on hourly candles."""

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "1h"
    startup_candle_count = 80
    stoploss = -0.06
    minimal_roi = {"72": 0.0, "24": 0.01, "0": 0.03}

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=25)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=25)
        dataframe["mom"] = ta.MOM(dataframe, timeperiod=14)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["adx"] > 25)
                & (dataframe["mom"] > 0)
                & (dataframe["plus_di"] > 25)
                & (dataframe["plus_di"] > dataframe["minus_di"])
                & (dataframe["close"] > dataframe["ema_200"])
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "adx_momentum")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["mom"] < 0)
                    | (dataframe["plus_di"] < dataframe["minus_di"])
                    | (dataframe["close"] < dataframe["ema_200"])
                )
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "momentum_reversal")
        return dataframe


class TimeSeriesMomentum(IStrategy):
    """Low-frequency Donchian breakout with time-series momentum confirmation."""

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "4h"
    startup_candle_count = 400
    stoploss = -0.08
    minimal_roi = {"1440": 0.0, "720": 0.04, "0": 0.12}
    trailing_stop = True
    trailing_stop_positive = 0.04
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["return_30d"] = dataframe["close"].pct_change(180)
        dataframe["return_60d"] = dataframe["close"].pct_change(360)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["entry_high"] = dataframe["high"].rolling(120).max().shift(1)
        dataframe["exit_low"] = dataframe["low"].rolling(60).min().shift(1)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe["close"], dataframe["entry_high"])
                & (dataframe["ema_100"] > dataframe["ema_200"])
                & (dataframe["return_30d"] > 0)
                & (dataframe["return_60d"] > 0)
                & (dataframe["atr_pct"] < 0.05)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "tsmom_positive")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    qtpylib.crossed_below(dataframe["close"], dataframe["exit_low"])
                    | qtpylib.crossed_below(dataframe["ema_100"], dataframe["ema_200"])
                    | (dataframe["return_30d"] < 0)
                )
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "tsmom_negative")
        return dataframe


class TrendFilteredBinCluc(IStrategy):
    """CombinedBinHAndCluc mean reversion with a one-hour trend regime."""

    INTERFACE_VERSION = 3
    can_short = False
    timeframe = "5m"
    startup_candle_count = 2400
    stoploss = -0.04
    minimal_roi = {"180": 0.0, "60": 0.01, "0": 0.03}
    exit_profit_only = True

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        mean_40 = dataframe["close"].rolling(40).mean()
        std_40 = dataframe["close"].rolling(40).std()
        dataframe["lower_40"] = mean_40 - 2 * std_40
        dataframe["bbdelta"] = (mean_40 - dataframe["lower_40"]).abs()
        dataframe["closedelta"] = dataframe["close"].diff().abs()
        dataframe["tail"] = (dataframe["close"] - dataframe["low"]).abs()

        bands = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )
        dataframe["bb_lowerband"] = bands["lower"]
        dataframe["bb_middleband"] = bands["mid"]
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_mean_30"] = dataframe["volume"].rolling(30).mean()
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        binhv45 = (
            dataframe["lower_40"].shift(1).gt(0)
            & dataframe["bbdelta"].gt(dataframe["close"] * 0.008)
            & dataframe["closedelta"].gt(dataframe["close"] * 0.0175)
            & dataframe["tail"].lt(dataframe["bbdelta"] * 0.25)
            & dataframe["close"].lt(dataframe["lower_40"].shift(1))
            & dataframe["close"].le(dataframe["close"].shift(1))
        )
        cluc = (
            (dataframe["close"] < dataframe["ema_50"])
            & (dataframe["close"] < 0.985 * dataframe["bb_lowerband"])
            & (dataframe["volume"] < dataframe["volume_mean_30"].shift(1) * 20)
            & (dataframe["rsi"] < 35)
        )
        safe_regime = (
            (dataframe["close_1h"] > dataframe["ema_200_1h"])
            & (dataframe["rsi_1h"] > 35)
        )
        dataframe.loc[
            (binhv45 | cluc) & safe_regime & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_filtered_bincluc")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe["close"] > dataframe["bb_middleband"])
                    | (dataframe["rsi"] > 60)
                )
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "mean_reverted")
        return dataframe
