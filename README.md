# MLB Daily High-Probability Outcome Finder

## Setup
```bash
pip install requests pybaseball pandas
```
(`pybaseball`/`pandas` are only needed for the Statcast similarity feature —
the script still runs without them, just falls back to hand-split matching.)

## Run
```bash
python mlb_daily_analysis.py                      # today's games (Statcast OFF by default -- see below)
python mlb_daily_analysis.py --date 2026-08-22     # specific date
python mlb_daily_analysis.py --min-bvp-ab 10       # require 10+ ABs before trusting a sample
python mlb_daily_analysis.py --use-statcast        # opt in to Statcast (slower, backtested as slightly worse)
python mlb_daily_analysis.py --similarity-threshold 0.9   # stricter pitcher matching
python mlb_daily_analysis.py --team Dodgers                # scope to one game, fast test run
```

## What it does
1. Pulls today's schedule + probable starting pitchers (MLB Stats API).
2. For each team, gets the actual posted lineup if available, otherwise
   estimates it from the 9 most frequent starters over the last 20 days.
3. For every batter, pulls a layered set of matchup data, used in this
   priority order (falls through to the next when the sample is too thin):
   1. **Direct BvP** — career AB/H vs today's exact opposing pitcher.
   2. **Statcast pitcher-similarity** — builds today's opposing pitcher's
      arsenal profile (pitch-type mix, velocity, movement) from his own
      Statcast pitch log, then scans every pitcher this batter has
      historically faced for a similar arsenal (cosine similarity ≥
      threshold, default 0.85) and aggregates his real outcomes against
      that group.
   3. **Vs-pitcher-hand split** — career line vs all L/R pitchers, as a
      last-resort proxy.
   4. **Recent form** — last 15 days of hitting, used both as a blend
      component and as the sole basis if nothing else qualifies.
4. Combines these into Hit / Run / RBI probability scores (0-100, weighted
   toward BvP/Statcast-similar per your preference, with lineup slot
   nudging Run vs RBI scores) AND real expected-count projections:
   - `expected_hits` / `expected_runs` / `expected_rbi` / `expected_combined`
     — a projected Hits+Runs+RBIs count for tonight, matching how
     sportsbooks price that prop (a literal sum, so a solo homer alone
     projects as high as 3).
   - `expected_total_bases` — projected total bases (1×1B + 2×2B + 3×3B +
     4×HR), its own standalone prop, built from a slugging-percentage blend
     using the same BvP > Statcast-similar > hand-split > recent-form
     priority chain as everything else.
5. Prints leaderboards (Top 10 Projected H+R+RBI, Top 10 Projected Total
   Bases, and the three individual Hit/Run/RBI score leaderboards) and
   saves a full CSV with every field, including which data source each
   batter's score leaned on.

## Statcast similarity status: OFF by default, opt-in only
A head-to-head backtest (2026-08-18 to 2026-08-24, n=1,158 matched player-dates
where the Statcast-similar tier actually fired) found it performed slightly
WORSE than the simpler hand-split fallback on both metrics:
- Hits: r=0.374 (Statcast) vs r=0.394 (hand-split alone)
- Total Bases: r=0.296 (Statcast) vs r=0.350 (hand-split alone)

Likely cause: the "similar pitcher" pool averaged 142 pitchers (up to 290) --
far too broad to represent genuine repertoire similarity, closer to "the
league-average pitcher" than a targeted match. The code is kept, not deleted,
in case a much stricter `--similarity-threshold` (try 0.95+) is worth testing
later, but it's opt-in (`--use-statcast`) rather than the default path. It's
also just plain slow -- `pybaseball` scrapes Baseball Savant, capped at 4
concurrent calls regardless of `--workers`.

## How the Statcast similarity works (v2)
- Pulls the *target* pitcher's own pitch log for the lookback window
  (default ~395 days) and buckets pitches into Fastball / Breaking /
  Offspeed, computing usage %, avg velocity, and avg movement per bucket.
- Pulls the *batter's* full pitch log for the same window — this single
  call also tells us every pitcher he's faced and lets us build a rough
  arsenal profile for each of them from the pitches actually thrown to him.
- Compares profiles via cosine similarity on a 9-dimension feature vector.
- Aggregates the batter's real AB/H against every pitcher above the
  similarity threshold (excluding the target pitcher, already covered by
  direct BvP).
- **Caveat:** a faced-pitcher's profile is built only from the subset of
  pitches he threw to *this specific batter*, not his full repertoire —
  so for batters who've seen a given pitcher only a handful of times, that
  profile can be noisy. This is a reasonable v1 tradeoff (one Statcast call
  per batter instead of one per historical opponent) but worth knowing.
- Statcast calls are only made when the direct BvP sample is too small
  (`--min-bvp-ab`), so a game where most regulars already have solid BvP
  history won't trigger many Statcast pulls.

## Known limitations / things to tune once you're running it live
- **Lineups aren't posted until 1-3 hrs before first pitch.** Early in the
  day you'll mostly get the "estimated" fallback lineup. Re-run closer to
  game time for real lineups.
- **The MLB Stats API is undocumented and can shift.** If a call fails,
  the script prints the exact URL/params — that's usually enough to spot
  a renamed field. Common trouble spots: `stats=vsPlayer` sometimes
  needs `season=0` explicitly for true career totals; `sitCodes` values
  for hand splits are `vl`/`vr`.
- **Rate limiting**: there's a small delay between MLB Stats API calls
  (`--delay`, default 0.15s). A full day's slate (~15 games x ~18 batters)
  is roughly 500-800 API calls, so a run takes a few minutes even without
  Statcast.
- **Statcast is slow the first time.** `pybaseball` scrapes Baseball
  Savant; a batter's ~400-day pitch log can be tens of thousands of rows.
  `pyb.cache.enable()` is already on in the script, so repeat runs for the
  same batter/date range are much faster. Statcast is OFF by default now
  anyway (see status note above) -- use `--use-statcast` deliberately, not
  as your everyday default.
- Pitch-type classification (`FF` vs `FT` vs `SI`, etc.) comes straight
  from Statcast's auto-classifier, which isn't perfect — the 3-bucket
  grouping (Fastball/Breaking/Offspeed) is deliberately coarse to reduce
  sensitivity to that noise.

## Ideas for v3
- Build target-pitcher-style profiles from each faced-pitcher's *full*
  repertoire (one Statcast call per unique historical opponent) instead
  of just the pitches thrown to this batter — more accurate, more calls.
- Weight recent-form window by sample size (10 vs 30 days) automatically.
- Park factor + weather (wind out at Wrigley, etc.) as a multiplier.
- Bullpen matchup for late-game runs/RBI, not just the starter.

## Backtesting (backtest.py)
```bash
python backtest.py --start-date 2026-08-01 --end-date 2026-08-14
python backtest.py --start-date 2026-08-01 --end-date 2026-08-14                 # Statcast OFF by default
python backtest.py --start-date 2026-08-01 --end-date 2026-08-14 --team Dodgers # scope to one team, fast test
```

Runs the same engine against a range of past dates (only games with `Final`
status), pulls each batter's real result for that game, and reports:
- **Top-decile hit rate**: of the batters the model ranked in its top 10%
  for hit/run/RBI score each day, what fraction actually delivered? Shown
  against the baseline rate across all batters analyzed, plus the % lift.
- **Correlation**: Pearson correlation between each 0-100 score and whether
  the outcome actually happened, across every batter-game in the window.
- **Combined projection accuracy**: correlation AND mean absolute error
  (MAE) between `expected_combined` (the projected Hits+Runs+RBIs count)
  and the batter's actual combined count that game — this is the metric
  that matters most for validating the prop-style projection, since it's a
  continuous count rather than a hit/miss score.
- A full CSV (`backtest_<start>_<end>.csv`) with every prediction next to
  the actual H/R/RBI (and actual_combined), so you can slice it further
  yourself.

**Leakage control:** recent-form and Statcast pulls are naturally bounded to
data before the date being tested. BvP and hand-split are NOT bounded by
default (they pull "as of right now" totals) — for the backtest, they're
restricted to the two seasons strictly before the year being tested, which
is airtight against leakage but means backtested BvP samples are thinner
than what a live run would show (no in-season-to-date at-bats counted).
That's a deliberate correctness-over-sample-size tradeoff — read as: the
backtest numbers are a conservative lower bound on how the live model
(with full current-season BvP) would likely perform.

Recommended first run: a 2-3 day window (Statcast is OFF by default, so this
is already fast) to confirm the plumbing and actual-results join work.

## Speed (--workers)
Both scripts analyze batters concurrently now instead of one at a time --
this is I/O-bound work (waiting on the MLB Stats API / Statcast), so
threading gives a large speedup with no infra changes needed. Control it
with `--workers` (default 8):
```bash
python backtest.py --start-date 2026-08-01 --end-date 2026-08-14 --workers 12
python mlb_daily_analysis.py --workers 12
```
Statcast (pybaseball/Baseball Savant) calls are capped at `min(workers, 4)`
internally regardless of what you set `--workers` to, since Baseball Savant
is more likely to throttle or error under heavy parallel load than the MLB
Stats API is. If you hit errors or rate-limit responses, lower `--workers`;
if it's running smoothly, you can push it higher. The old `--delay` flag no
longer does anything (kept for backward compatibility) -- concurrency is
now the throttle.

## Fitting blend weights (fit_weights.py)
```bash
python fit_weights.py --csv backtest_2026-07-15_2026-08-21.csv
python fit_weights.py --csv backtest_....csv --holdout-frac 0.25
python fit_weights.py --csv backtest_....csv --split-date 2026-08-10
```

The blend weights in `score_batter()` (e.g. 65% BvP + 35% recent form) were
originally hand-picked. This script replaces the guess with weights fit
directly from backtest data, per tier (BvP-trusted vs hand-split-fallback),
for each of Hits/Total Bases/Runs/RBI -- and validates the fit on a holdout
slice of dates it never saw, so a fit isn't just memorizing one window.

Requires a backtest CSV run with the current schema (raw per-tier component
columns like `hand_split_avg`, `bvp_slg`, `recent_runs_rate`, etc. -- these
were added after the original wide backtest, so **you'll need to delete old
backtest CSVs and re-run `backtest.py` once** to get a CSV with the new
columns before this will work).

Output for each tier/metric: the fitted weights, and a head-to-head
correlation comparison between the fitted model and the CURRENT hardcoded
formula, both evaluated on the same holdout dates. Only adopt a fit into
`score_batter()` if it (a) beats the current weights on holdout, not just
train, and (b) has a reasonable holdout sample size -- the script flags
tiers with fewer than 30 holdout rows as too thin to trust.

Statcast-similar isn't fit (no data since it's OFF by default now -- and per
the status note above, it backtested as slightly worse than the fallback it
would replace, so there's little reason to fit it as-is).

## Probability calibration (shared by odds_value_finder.py and squad_builder.py)
Backtesting `squad_builder.py` itself (not just the underlying projections)
found a serious problem: simulating its actual top-4-per-day squad selection
against real historical outcomes showed **74.6-76.9% average predicted joint
probability vs only 36.6% actual rate all 4 picks cleared** -- roughly double
the real number. Root cause: both squad-building and value-bet detection
specifically select from the highest-probability tail each day (top-N by
probability, or "model beats market"), and the raw Poisson-derived
probability is well-calibrated in the middle of its range but increasingly
OVERCONFIDENT at the top decile (+14.6% for Combined, +19.1% for Total Bases,
measured by binning ~9,800 candidates by predicted probability and comparing
against actual clear rate per bin) -- a classic selection-effect / "winner's
curse" pattern.

Fixed with an empirical calibration curve (`calibrate_probability()` in
`odds_value_finder.py`, used by both tools) that corrects any raw
Combined/Total-Bases probability against the real observed rate instead of
trusting the raw Poisson output. Re-validated by re-simulating the exact
same historical squads with calibration applied: predicted probability
dropped to 20.0-20.8%, actual stayed at 17.1% -- gap closed from ~38-40
points down to 3.7. Only Combined and Total Bases are calibrated (that's
the data we have); individual Hits/Runs/RBI markets pass through unchanged.
Re-derive this curve periodically from a fresh backtest -- it isn't permanent,
and it was measured specifically at a 1.5 line (applied to other lines as a
reasonable approximation, not separately validated).

## Odds & Profitability (odds_value_finder.py)
```bash
export ODDS_API_KEY=your_key_here    # free key: https://the-odds-api.com/
python odds_value_finder.py --date 2026-08-26
python odds_value_finder.py --date 2026-08-26 --team "Dodgers,Yankees"   # cheaper on quota
python odds_value_finder.py --csv mlb_report_2026-08-26.csv              # reuse an existing run
```

Matches the model's projections against LIVE sportsbook player prop odds
and surfaces bets where the model's implied probability meaningfully beats
what the market is pricing.

**Markets used** (via The Odds API): `batter_hits`, `batter_total_bases`,
`batter_rbis`, `batter_runs_scored`, and `batter_hits_runs_rbis` -- the last
one is the exact combined prop this model already projects as
`expected_combined`.

**How a point projection becomes a probability**: sportsbook lines are
over/under a number, but the model outputs a single expected value (e.g.
"1.8 total bases"), not a probability. This script converts:
- Hits: Binomial(n=at-bats-per-game, p=hit-probability-per-AB) -- both are
  already available from the model, and hits are naturally capped by AB.
- Total Bases / Runs / RBI / Combined: Poisson(lambda=projection) -- a
  standard approximation for count-like stats. Not exact, but reasonable
  without a fitted distribution.

**De-vigging**: when both Over and Under prices are available, they're
normalized to sum to 1 for a fairer "true" market probability (raw implied
probability includes the book's built-in margin).

**Cost**: [markets requested] x [regions] credits per game on Odds API's
free 500-credit/month tier. All 5 markets x 1 region = 5 credits/game --
~75 for a full 15-game slate, so roughly 6-7 full-slate days/month free.
Use `--team` to scope to specific games and stretch the quota much further.

**Important limitation**: historical player prop odds require a PAID Odds
API plan. This script only works with current/live odds -- it can't
retroactively backtest profitability against real historical lines. The
free path to validating profitability is prospective: run this daily going
forward and track actual results against the odds offered that day, rather
than backtesting.

## Squad Builder (squad_builder.py)
```bash
python squad_builder.py --date 2026-08-26                          # mixed pool, both stats, default 1.5 lines
python squad_builder.py --date 2026-08-26 --combined-line 2.5 --total-bases-line 1.5
python squad_builder.py --date 2026-08-26 --metric total_bases      # restrict to just one stat
python squad_builder.py --csv mlb_report_2026-08-26.csv
python squad_builder.py --date 2026-08-26 --squad-size 6 --max-per-game 1
```

By default, builds squads from a MIXED pool of both Hits+Runs+RBIs picks and
Total Bases picks for every batter (each is a separate "pick" -- a slip
is really 4 picks, not 4 players) -- so the same player can legitimately
fill two slots if both his picks rank highly (e.g. Ohtani's Combined pick
AND his Total Bases pick both in one squad). That is intended, not a bug:
the tool actively flags it in the squad output as a distinct, stronger
correlation warning than the ordinary same-game one, since two picks on
the literal same player share the exact same at-bats.

**Correlation-adjusted joint probability**: the tool now shows TWO numbers per
squad -- naive independence and a correlation-adjusted version -- using
correlations measured directly from backtest data (2026-07-16 to 2026-08-25):
same-player-both-stats correlation is ~0.71 (their outcomes share the same
at-bats), same-game-different-player correlation is much smaller (~0.01-0.04),
and different-game pairs measured near 0 (confirming independence is
reasonable across games). Re-measure periodically from a fresh backtest --
these are empirical, not permanent constants. Use `--metric
combined` or `--metric total_bases` to restrict to a single stat if you
want single-metric squads instead.

**Game-total prioritization**: set `ODDS_API_KEY` (same key as `odds_value_finder.py`)
to enable a squad built by GAME first (highest Over/Under total-runs line =
most expected offense), then best player(s) within each -- rather than
ranking players first and diversifying as an afterthought. Costs just 1
credit for the whole day's slate (uses the cheap bulk totals endpoint, not
the per-event player-props one). Optional -- everything else still works
fine without an API key, you just won't get this specific squad option.

For "pick N players, all must clear the line" formats (Underdog/PrizePicks-style
pick'ems) rather than single best picks or odds-based value bets. Converts
every batter's projection into P(actual > line) using the same Poisson math
validated in `odds_value_finder.py`, ranks everyone, and for a fixed squad
size the best joint probability is just the top-N by individual probability
(mathematically provable, and verified against brute force in testing).

Also builds a `--max-per-game`-diversified squad (default caps at 2 players
per game) and a few non-overlapping alternative squads, since two players
from the same game are NOT independent in real life.

**Read the caveat in the file docstring before trusting the joint probability
number** — it's a product of individual probabilities assuming independence,
which understates same-game correlation (both the extra risk and the extra
upside). Use it to compare candidate squads against each other, not as a
precise number on its own.

## Pitcher strikeouts integrated into Squad Builder
`squad_builder.py --metric all` now mixes Combined, Total Bases, AND Pitcher
Strikeouts picks in one pool (`--metric strikeouts` for strikeouts alone,
`--metric both` keeps the old batter-only default). Strikeouts draws from a
different underlying model (`pitcher_strikeouts.py`) than the batter props --
`load_rows()` only fetches each data source when actually needed. The
strikeouts probability gets the same empirical calibration correction as
Combined/Total Bases (see the `CALIBRATION_CURVE["strikeouts"]` entry in
`odds_value_finder.py`) -- built and applied BEFORE integration, not after,
learning from the batter-model sequence where an uncalibrated probability
briefly made it into live squad-building before the problem was caught.

```bash
python squad_builder.py --date 2026-08-26 --metric all
python squad_builder.py --date 2026-08-26 --metric strikeouts --strikeouts-line 5.5
```
