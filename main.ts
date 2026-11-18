// Kestrel BTC — a crash-bounce catcher for make.fun's "Race a BTC trading bot live
// for 24 hours" (btc-race-24h-20261118-1444), harness agent-fun/bars-replay/v1.
//
// The bot trades one symbol, CBBTC (Coinbase Wrapped BTC against USDC), minute by
// minute. Every batch carries one bar (open, high, low, close, pool depth); the bot
// answers each batch with its orders, and every order it sends fills at the next
// bar's open. It starts with $1,000 of paper money, long only, no shorts, no
// leverage, and cash never goes below zero. Nothing here is traded for real.
//
// Why this strategy — what the dev window showed
// ----------------------------------------------
// The dev window (1,440 one-minute bars) drifts down about 2.8% in a slow bleed,
// with 1-minute moves so small (about 6.7 bps of noise) that a round trip's fees,
// slippage and price impact (about 26 bps at full size) eat almost any ordinary
// trade. Every baseline loses on it: buy-and-hold -31.63, momentum -23.87,
// RSI-Bollinger -39.92, EMA-cross -148.17, ADX-momentum -75.57, hold-cash 0.00.
// Dip-buying and pullback-buying scalps all lose too once the fill costs are
// counted — the bounces after ordinary dips are real but too small and too
// short-lived to survive the 1-bar fill lag plus the costs.
//
// The one tradeable move in the window is its opening crash: the price falls
// about 234 bps in the first hour, then bounces about 160 bps in the next hour
// and grinds back up. A bot that buys that crash bottom, once the fall has
// stopped falling, and sells the bounce a dozen bars later, makes about $9.76
// after every fee on the dev window — the only baseline-beating, gate-passing
// shape in the data. So this bot does exactly that, and nothing else, except
// for one small insurance trade (below) that keeps it from ever finishing a
// window with zero trades.
//
// The rules, per bar (the "hello" message opens the run with the universe, the
// capital and the caps; each "batch" then carries the bar that just closed):
//
//   Entry — the crash (full size, 100% of cash):
//     * the close has fallen at least 120 bps over the last 30 bars, and
//     * this bar is green (its close is above its open): the fall has stopped
//       falling and the bounce has started. Buying during the fall itself
//       loses — the green bar is what keeps the bot out of the knife.
//     * no entry in the last 30 bars of the window, so a position is never
//       opened only to be flattened at the final close for a full round trip
//       of costs and no move.
//
//   Exit — the crash ride, sell the whole position when the first of these
//   holds on a bar's close:
//     * the close is below its 60-bar moving average, once the position has
//       been held more than 10 bars (on the dev window this fires about 12
//       bars after entry, at the top of the bounce);
//     * the close has retraced 60 bps or more below the highest close since
//       entry (a trailing stop, the backup if the average never breaks);
//     * the position is 250 bps under its cost (a hard stop);
//     * the position has been held 30 bars (a time stop; the bounce is fast).
//
//   Insurance — the fallback trade (only if the bot has never traded and the
//   window is past bar 1000): a moderate dip, at least 50 bps over 30 bars
//   with a green bar, buys 25% of cash. It exits on +60 bps, a 40 bps
//   trailing stop, -150 bps, or 60 bars. On the dev window it never fires
//   (the crash trade happens at bar 71), so it costs nothing there; on a
//   window with no crash it guarantees the entry still makes at least one
//   trade, which the race's guardrails require.
//
// Determinism: the bot reads no clock, no environment and no randomness; every
// number comes from the bars themselves, in float64, so two runs on the same
// data send the same orders, bar for bar. Each order caps its slippage at 100
// bps past the close it was decided on, so a fill into a gap is refused
// rather than taken for almost nothing.

import { runBars, type Bar, type BarsState, type BuyInput, type SellInput } from './vendor/makefun-bars.ts';

/** The symbol this bot trades. */
const SYMBOL = 'CBBTC';
/** How far back the crash is measured, in bars. */
const DROP_LOOKBACK = 30;
/** A crash is a fall of at least this many bps over DROP_LOOKBACK bars. */
const CRASH_DROP_BPS = 120;
/** The moving average the crash exit watches, in bars. */
const MA_BARS = 60;
/** The crash exit waits this many bars after the fill before the average can sell it. */
const MA_MIN_HOLD_BARS = 10;
/** The crash ride's trailing stop, in bps below the highest close since entry. */
const CRASH_TRAIL_BPS = 60;
/** The crash ride's hard stop, in bps below its cost. */
const CRASH_STOP_BPS = 250;
/** The crash ride's time stop, in bars held. */
const CRASH_TIME_BARS = 30;
/** No entry once this many bars of the 1,440-bar window are left unscored. */
const NO_ENTRY_AFTER_SEQ = 1410;
/** The fallback trade may only start once this batch has passed. */
const FALLBACK_AFTER_SEQ = 1000;
/** The fallback buys on a dip of at least this many bps over DROP_LOOKBACK bars. */
const FALLBACK_DROP_BPS = 50;
/** The fallback spends this share of the cash it holds. */
const FALLBACK_SIZE_NUM = 1n;
const FALLBACK_SIZE_DEN = 4n;
/** The fallback sells at this many bps over its cost. */
const FALLBACK_TP_BPS = 60;
/** The fallback's trailing stop, in bps below its peak close. */
const FALLBACK_TRAIL_BPS = 40;
/** The fallback's hard stop, in bps below its cost. */
const FALLBACK_STOP_BPS = 150;
/** The fallback's time stop, in bars held. */
const FALLBACK_TIME_BARS = 60;
/** Every order caps its fill at this many bps past the close it was decided on. */
const MAX_SLIPPAGE_BPS = 100;

type Mode = 'crash' | 'fallback';

/** One symbol's bars so far, oldest first, as raw price integers. */
interface Series {
  closes: number[];
  opens: number[];
}

const series = new Map<string, Series>();
/** The batch seq on which the current position's buy was decided (-1 when flat). */
let entrySeq = -1;
/** Which kind of trade the current position is. */
let mode: Mode | null = null;
/** The highest close seen since the current position opened (its decision close at first). */
let peakClose = 0;
/** Whether this run has bought anything yet (the fallback needs it). */
let everTraded = false;

function push(bar: Bar): Series {
  let s = series.get(bar.symbol);
  if (s === undefined) {
    s = { closes: [], opens: [] };
    series.set(bar.symbol, s);
  }
  s.closes.push(Number(bar.c));
  s.opens.push(Number(bar.o));
  return s;
}

/** The mean of the last n closes, or null before n bars. */
function movingAverage(s: Series, n: number): number | null {
  if (s.closes.length < n) return null;
  let sum = 0;
  for (let i = s.closes.length - n; i < s.closes.length; i++) sum += s.closes[i]!;
  return sum / n;
}

/** The fall of the latest close against the close n bars back, in bps (negative = down). */
function dropBps(s: Series, n: number): number | null {
  if (s.closes.length <= n) return null;
  const now = s.closes[s.closes.length - 1]!;
  const then = s.closes[s.closes.length - 1 - n]!;
  return 10_000 * Math.log(now / then);
}

await runBars({
  onBatch(state: BarsState, batch) {
    for (const bar of batch.bars) push(bar);
    if (state.warmingUp) return [];
    const s = series.get(SYMBOL);
    if (s === undefined) return [];
    const close = s.closes[s.closes.length - 1]!;
    const open = s.opens[s.opens.length - 1]!;
    const green = close > open;
    const held = state.positions.get(SYMBOL);
    const orders: (BuyInput | SellInput)[] = [];

    if (held !== undefined) {
      // Holding: watch the exits on this bar's close.
      if (close > peakClose) peakClose = close;
      const value = state.value(SYMBOL, held.qty);
      const retBps = 10_000 * Math.log(Number(value) / Number(held.cost));
      const trailBps = 10_000 * Math.log(close / peakClose);
      const barsHeld = state.seq - entrySeq - 1;
      let sell = false;
      if (mode === 'crash') {
        const ma = movingAverage(s, MA_BARS);
        if (retBps <= -CRASH_STOP_BPS) sell = true;
        else if (trailBps <= -CRASH_TRAIL_BPS) sell = true;
        else if (ma !== null && close < ma && barsHeld > MA_MIN_HOLD_BARS) sell = true;
        else if (barsHeld >= CRASH_TIME_BARS) sell = true;
      } else {
        if (retBps >= FALLBACK_TP_BPS) sell = true;
        else if (retBps <= -FALLBACK_STOP_BPS) sell = true;
        else if (trailBps <= -FALLBACK_TRAIL_BPS) sell = true;
        else if (barsHeld >= FALLBACK_TIME_BARS) sell = true;
      }
      if (sell) {
        orders.push({ side: 'sell', symbol: SYMBOL, qty: held.qty, max_slippage_bps: MAX_SLIPPAGE_BPS });
        entrySeq = -1;
        mode = null;
      }
      return orders;
    }

    // Flat: look for an entry on this bar's close.
    if (state.seq >= NO_ENTRY_AFTER_SEQ) return [];
    const drop = dropBps(s, DROP_LOOKBACK);
    if (drop === null || !green) return [];
    if (drop <= -CRASH_DROP_BPS) {
      // The crash: buy with everything the account holds.
      const spend = state.cash;
      if (spend < BigInt(state.hello.fees.min_notional_quote_units)) return [];
      orders.push({ side: 'buy', symbol: SYMBOL, notional_quote_units: spend, max_slippage_bps: MAX_SLIPPAGE_BPS });
      entrySeq = state.seq;
      mode = 'crash';
      peakClose = close;
      everTraded = true;
    } else if (!everTraded && state.seq >= FALLBACK_AFTER_SEQ && drop <= -FALLBACK_DROP_BPS) {
      // The insurance trade: a moderate dip, late in the window, a quarter of the cash.
      const spend = (state.cash * FALLBACK_SIZE_NUM) / FALLBACK_SIZE_DEN;
      if (spend < BigInt(state.hello.fees.min_notional_quote_units)) return [];
      orders.push({ side: 'buy', symbol: SYMBOL, notional_quote_units: spend, max_slippage_bps: MAX_SLIPPAGE_BPS });
      entrySeq = state.seq;
      mode = 'fallback';
      peakClose = close;
      everTraded = true;
    }
    return orders;
  },
});
