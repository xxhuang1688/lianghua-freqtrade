"""Exit-risk grid for the validated TrendFilteredBinCluc entry signal."""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame

from ResearchedCommunityStrategies import TrendFilteredBinCluc


class BinClucFixedExitBase(TrendFilteredBinCluc):
    """Long-only USDT perpetual strategy using conservative fixed leverage."""

    fixed_leverage = 2.0
    use_exit_signal = False
    exit_profit_only = False
    minimal_roi = {"0": 0.015}
    stoploss = -0.015

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(self.fixed_leverage, max_leverage)


class BinClucTP10SL10(BinClucFixedExitBase):
    minimal_roi = {"0": 0.010}
    stoploss = -0.010


class BinClucTP10SL20(BinClucFixedExitBase):
    minimal_roi = {"0": 0.010}
    stoploss = -0.020


class BinClucTP15SL10(BinClucFixedExitBase):
    minimal_roi = {"0": 0.015}
    stoploss = -0.010


class BinClucTP15SL20(BinClucFixedExitBase):
    minimal_roi = {"0": 0.015}
    stoploss = -0.020


class BinClucTP20SL10(BinClucFixedExitBase):
    minimal_roi = {"0": 0.020}
    stoploss = -0.010


class BinClucTP20SL20(BinClucFixedExitBase):
    minimal_roi = {"0": 0.020}
    stoploss = -0.020


class BinClucTP25SL20(BinClucFixedExitBase):
    minimal_roi = {"0": 0.025}
    stoploss = -0.020


class BinClucTP30SL20(BinClucFixedExitBase):
    minimal_roi = {"0": 0.030}
    stoploss = -0.020


class BinClucTP30SL30(BinClucFixedExitBase):
    minimal_roi = {"0": 0.030}
    stoploss = -0.030


class BinClucTP40SL20(BinClucFixedExitBase):
    minimal_roi = {"0": 0.040}
    stoploss = -0.020


class BinClucTP40SL30(BinClucFixedExitBase):
    minimal_roi = {"0": 0.040}
    stoploss = -0.030


class BinClucFutures10x(BinClucFixedExitBase):
    """10x leverage; approximately +2% / -1% underlying price movement."""

    fixed_leverage = 10.0
    minimal_roi = {"0": 0.20}
    stoploss = -0.10


class BinClucFutures20x(BinClucFixedExitBase):
    """20x leverage; approximately +2% / -1% underlying price movement."""

    fixed_leverage = 20.0
    minimal_roi = {"0": 0.40}
    stoploss = -0.20


class BinClucFutures30x(BinClucFixedExitBase):
    """30x leverage; approximately +2% / -1% underlying price movement."""

    fixed_leverage = 30.0
    minimal_roi = {"0": 0.60}
    stoploss = -0.30


class BinClucFuturesATR(TrendFilteredBinCluc):
    """BinCluc entry signal + ATR-based dynamic position sizing, stop-loss, and take-profit.

    Risks a fixed percentage of the wallet per trade (``risk_per_trade``).
    Stop distance = ``atr_multiplier_sl`` × ATR (as % of price).
    Take-profit = stop distance × ``reward_risk_ratio``.
    Position size = (wallet × risk_per_trade) / (stop_distance × leverage).
    """

    # ── risk controls (overrides parent's static values) ──────────────
    stoploss = -0.15              # hard floor: never lose >15% of margin
    minimal_roi = {"0": 0.50}     # high ceiling so ATR TP takes priority
    use_custom_stoploss = True
    exit_profit_only = False      # allow custom_exit regardless of profit sign

    # ── ATR parameters ────────────────────────────────────────────────
    atr_period = 14
    atr_multiplier_sl = 3.0       # stop = multiplier × ATR  (wider for 5m noise)
    reward_risk_ratio = 2.5       # TP:SL ratio
    fixed_leverage = 10.0
    risk_per_trade = 0.02         # 2 % of wallet at risk per trade
    min_stop_price_pct = 0.015    # minimum stop distance: 1.5% of price

    # ── volatility filter ─────────────────────────────────────────────
    max_atr_pct = 0.08            # skip entry when ATR/close > 8 %
    min_atr_pct = 0.001           # skip entry when ATR/close < 0.1 %

    # ── indicators ────────────────────────────────────────────────────
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    # ── entry signals (parent + volatility filter) ────────────────────
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_entry_trend(dataframe, metadata)
        dataframe.loc[
            (dataframe["atr_pct"] > self.max_atr_pct)
            | (dataframe["atr_pct"] < self.min_atr_pct),
            ["enter_long", "enter_tag"],
        ] = (0, None)
        return dataframe

    # ── leverage ──────────────────────────────────────────────────────
    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(self.fixed_leverage, max_leverage)

    # ── ATR position sizing ───────────────────────────────────────────
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """Risk only ``risk_per_trade`` of wallet, sized by ATR stop distance."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]
        atr = last.get("atr")
        close = last.get("close")
        if atr is None or close is None or close == 0 or atr <= 0:
            return proposed_stake

        atr_pct = atr / close
        stop_dist = max(
            self.atr_multiplier_sl * atr_pct,
            self.min_stop_price_pct,
        )
        margin_risk = stop_dist * leverage                  # margin % lost if stopped
        if margin_risk <= 0:
            return proposed_stake

        wallet = self.wallets.get_total_stake_amount() or max_stake
        risk_dollars = wallet * self.risk_per_trade
        stake = risk_dollars / margin_risk

        stake = max(stake, min_stake or 0)
        stake = min(stake, max_stake)
        return stake

    # ── ATR dynamic stop-loss ─────────────────────────────────────────
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        """Stop set at ``atr_multiplier_sl`` × ATR below current price."""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss

        last = dataframe.iloc[-1]
        atr = last.get("atr")
        close = last.get("close")
        if atr is None or close is None or close == 0:
            return self.stoploss

        atr_pct = atr / close
        stop_distance = max(
            self.atr_multiplier_sl * atr_pct,
            self.min_stop_price_pct,
        )
        # custom_stoploss returns margin-relative value (framework divides by leverage).
        return -(stop_distance * self.fixed_leverage)

    # ── ATR take-profit ───────────────────────────────────────────────
    def custom_exit(
        self,
        pair: str,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        """Exit when profit reaches ATR stop-distance × reward:risk ratio."""
        if current_profit <= 0:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        atr = last.get("atr")
        close = last.get("close")
        if atr is None or close is None or close == 0:
            return None

        atr_pct = atr / close
        stop_dist = max(
            self.atr_multiplier_sl * atr_pct,
            self.min_stop_price_pct,
        )
        # current_profit is margin-relative, so TP must be margin-relative too.
        tp_target = stop_dist * self.fixed_leverage * self.reward_risk_ratio
        if current_profit >= tp_target:
            return "atr_tp_hit"
        return None
