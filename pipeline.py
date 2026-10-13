"""Reproducible Cricsheet IPL ingestion. Run from any working directory."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
URL = 'https://cricsheet.org/downloads/ipl_json.zip'
AS_OF = '2026-10-12'
EXTRAS = ['wides', 'noballs', 'byes', 'legbyes', 'penalty']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def obtain(refresh=False):
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / 'ipl_json.zip'
    metadata = raw / 'download_manifest.json'
    if refresh or not archive.exists():
        temp = archive.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=120) as response:
            temp.write_bytes(response.read())
            headers = dict(response.headers)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError('Corrupt archive')
        temp.replace(archive)
        metadata.write_text(json.dumps({'url': URL, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
            'http_headers': headers, 'sha256': digest(archive)}, indent=2))
    if not metadata.exists():
        metadata.write_text(json.dumps({'url': URL, 'retrieved_date': AS_OF,
            'retrieval_method': 'Initial curl download; exact HTTP headers not captured',
            'sha256': digest(archive)}, indent=2))
    if json.loads(metadata.read_text())['sha256'] != digest(archive):
        raise ValueError('Raw archive differs from recorded checksum')
    return archive

def flatten(match_id, obj):
    info = obj['info']
    if info['gender'] != 'male' or info['match_type'] != 'T20' or info['event']['name'] != 'Indian Premier League':
        raise ValueError(f'Unexpected competition: {match_id}')
    if obj['meta']['data_version'] not in ('1.2.0', '1.3.0'):
        raise ValueError('Review new schema before ingestion')
    registry = info['registry']['people']
    date = min(info['dates'])
    if max(info['dates']) > AS_OF:
        raise ValueError('Source contains future-dated match')
    match = dict(match_id=match_id, date=date, end_date=max(info['dates']), season=int(date[:4]),
        season_source=str(info['season']), venue=info['venue'], city=info.get('city'),
        team1=info['teams'][0], team2=info['teams'][1], scheduled_overs=info['overs'],
        balls_per_over=info['balls_per_over'], event_stage=info['event'].get('stage'),
        event_match_number=info['event'].get('match_number'), toss_winner=info['toss']['winner'],
        toss_decision=info['toss']['decision'], result=info['outcome'].get('result'),
        winner=info['outcome'].get('winner'), outcome_method=info['outcome'].get('method'),
        data_version=obj['meta']['data_version'], source_revision=obj['meta']['revision'])
    rows, innings_rows, wickets, rosters = [], [], [], []
    for team, names in info['players'].items():
        for name in names:
            rosters.append(dict(match_id=match_id, date=date, season=match['season'], team=team,
                player_id=registry[name], player_name=name))
    for ix, inn in enumerate(obj.get('innings', []), 1):
        super_over = inn.get('super_over', False)
        target = inn.get('target', {})
        unusual = bool(info.get('missing') or inn.get('miscounted_overs'))
        standard = (info['overs'] == 20 and not info['outcome'].get('method')
            and target.get('overs', 20) == 20 and not unusual)
        score = inn.get('penalty_runs', {}).get('pre', 0)
        wickets_before = legal_before = sequence = 0
        batting_positions = {}
        innings_rows.append(dict(match_id=match_id, innings=ix, batting_team=inn['team'],
            is_super_over=super_over, target_runs=target.get('runs'), target_overs=target.get('overs'),
            powerplays_json=json.dumps(inn.get('powerplays', [])),
            miscounted_overs_json=json.dumps(inn.get('miscounted_overs', {})),
            penalty_runs_json=json.dumps(inn.get('penalty_runs', {})),
            absent_hurt_json=json.dumps(inn.get('absent_hurt', [])), standard_conditions=standard))
        for over in inn['overs']:
            for ball_ix, ball in enumerate(over['deliveries'], 1):
                sequence += 1
                for name in (ball['batter'], ball['non_striker']):
                    if name not in batting_positions:
                        batting_positions[name] = len(batting_positions) + 1
                extra = {k: ball.get('extras', {}).get(k, 0) for k in EXTRAS}
                if set(ball.get('extras', {})) - set(EXTRAS):
                    raise ValueError('Unknown extra type')
                runs = ball['runs']
                legal = extra['wides'] == 0 and extra['noballs'] == 0
                boundary = runs['batter'] in (4, 6) and not runs.get('non_boundary', False)
                # Explicit source ordering, never float ball numbers or runs.total == 4.
                key = f'{match_id}:{ix}:{sequence}'
                row = dict(delivery_id=key, match_id=match_id, date=date, season=match['season'],
                    innings=ix, is_super_over=super_over, over=over['over'], event_in_over=ball_ix,
                    delivery_sequence=sequence, actual_delivery=ball.get('actual_delivery'),
                    batter_id=registry[ball['batter']], batter_name=ball['batter'],
                    non_striker_id=registry[ball['non_striker']], bowler_id=registry[ball['bowler']],
                    bowler_name=ball['bowler'], batting_team=inn['team'],
                    bowling_team=next(t for t in info['teams'] if t != inn['team']),
                    venue=info['venue'], phase='powerplay' if over['over'] < 6 else ('middle' if over['over'] < 15 else 'death'),
                    batting_entry_order=batting_positions[ball['batter']],
                    batter_runs=runs['batter'], extras_runs=runs['extras'], total_runs=runs['total'],
                    non_boundary=runs.get('non_boundary', False), is_boundary=boundary,
                    is_legal=legal, is_nonwide=extra['wides'] == 0,
                    primary_eligible=legal and not super_over,
                    standard_conditions=standard, score_before=score, wickets_before=wickets_before,
                    legal_balls_before=legal_before, replacements_json=json.dumps(ball.get('replacements', {})), **extra)
                rows.append(row)
                for wi, wicket in enumerate(ball.get('wickets', []), 1):
                    wickets.append(dict(delivery_id=key, wicket_index=wi, match_id=match_id, innings=ix,
                        player_out_id=registry[wicket['player_out']], player_out_name=wicket['player_out'],
                        kind=wicket['kind'], fielders_json=json.dumps(wicket.get('fielders', []))))
                    wickets_before += int(wicket['kind'] != 'retired hurt')
                score += runs['total']
                legal_before += int(legal)
    return match, rows, innings_rows, wickets, rosters

def prior_day_features(innings):
    """History available before match day, excluding all same-day outcomes."""
    daily = innings.groupby(['batter_id', 'date'], as_index=False).agg(N=('N', 'sum'), B=('B', 'sum'))
    daily = daily.sort_values(['batter_id', 'date'])
    for col in ['N', 'B']:
        daily['prior_' + col] = daily.groupby('batter_id')[col].cumsum() - daily[col]
    daily['prior_boundary_rate'] = daily.prior_B / daily.prior_N.replace(0, float('nan'))
    return daily[['batter_id', 'date', 'prior_N', 'prior_B', 'prior_boundary_rate']]

def build():
    archive = obtain()
    collections = [[] for _ in range(5)]
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.json'):
                continue
            parts = flatten(Path(name).stem, json.loads(z.read(name)))
            collections[0].append(parts[0])
            for coll, part in zip(collections[1:], parts[1:]):
                coll.extend(part)
        for name in z.namelist():
            if Path(name).name.upper().startswith('README'):
                (ROOT / 'data/raw' / Path(name).name).write_bytes(z.read(name))
    matches, deliveries, innings_meta, wickets, rosters = map(pd.DataFrame, collections)
    deliveries = deliveries.sort_values(['date', 'match_id', 'innings', 'delivery_sequence']).reset_index(drop=True)
    if deliveries.delivery_id.duplicated().any() or matches.match_id.duplicated().any():
        raise ValueError('Duplicate keys')
    assert (deliveries.batter_runs + deliveries.extras_runs == deliveries.total_runs).all()
    assert (deliveries[EXTRAS].sum(axis=1) == deliveries.extras_runs).all()
    assert deliveries[['batter_id', 'bowler_id', 'non_striker_id']].notna().all().all()
    assert (deliveries[EXTRAS + ['batter_runs', 'total_runs']] >= 0).all().all()
    assert deliveries.match_id.isin(matches.match_id).all()
    assert wickets.delivery_id.isin(deliveries.delivery_id).all()
    model = deliveries.loc[deliveries.primary_eligible].copy()
    keys = ['match_id', 'date', 'season', 'innings', 'batter_id', 'batting_team', 'bowling_team', 'venue']
    batting = model.groupby(keys, as_index=False, dropna=False).agg(N=('is_boundary', 'size'), B=('is_boundary', 'sum'),
        runs_on_eligible_balls=('batter_runs', 'sum'), batting_entry_order=('batting_entry_order', 'first'),
        standard_conditions=('standard_conditions', 'all'))
    batting['boundary_rate'] = batting.B / batting.N
    batting = batting.merge(prior_day_features(batting), on=['batter_id', 'date'], validate='many_to_one')
    batting = batting.sort_values(['date', 'match_id', 'innings', 'batter_id'])
    seasons = batting.groupby(['season', 'batter_id'], as_index=False).agg(N=('N', 'sum'), B=('B', 'sum'), innings=('N', 'size'))
    seasons['boundary_rate'] = seasons.B / seasons.N
    # One global calendar cut per season; never a separate midpoint for each player.
    splits = []
    for season, group in matches.groupby('season'):
        dates = pd.to_datetime(group.date)
        midpoint = dates.min() + (dates.max() - dates.min()) / 2
        cutoff = midpoint.date().isoformat()
        for match_id, date in zip(group.match_id, group.date):
            splits.append(dict(match_id=match_id, season=season, cutoff_date=cutoff,
                half='early' if date <= cutoff else 'late',
                experiment_role='development' if season <= 2023 else ('validation' if season == 2024 else 'locked_test')))
    splits = pd.DataFrame(splits)
    half = model.merge(splits[['match_id', 'half']], on='match_id', validate='many_to_one').groupby(
        ['season', 'half', 'batter_id'], as_index=False).agg(N=('is_boundary', 'size'), B=('is_boundary', 'sum'))
    aliases = rosters[['player_id', 'player_name']].drop_duplicates().sort_values(['player_id', 'player_name'])
    coverage = matches.groupby('season', as_index=False).agg(matches=('match_id', 'size'), first_date=('date', 'min'), last_date=('end_date', 'max'))
    counts = deliveries.groupby('season', as_index=False).agg(raw_deliveries=('delivery_id', 'size'),
        primary_deliveries=('primary_eligible', 'sum'), super_over_deliveries=('is_super_over', 'sum'))
    coverage = coverage.merge(counts, on='season', validate='one_to_one')
    assert int(batting.N.sum()) == len(model) == int(seasons.N.sum()) == int(half.N.sum())
    assert int(batting.B.sum()) == int(model.is_boundary.sum()) == int(half.B.sum())
    assert (batting.B <= batting.N).all()
    tables = dict(matches=matches, deliveries=deliveries, innings_metadata=innings_meta, wickets=wickets,
        match_players=rosters, player_aliases=aliases, model_deliveries=model, batter_innings=batting,
        batter_seasons=seasons, season_halves=half, match_splits=splits, coverage=coverage)
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f'{name}.parquet', index=False)
        if name not in ('deliveries', 'model_deliveries'):
            table.to_csv(out / f'{name}.csv', index=False)
    dictionary = [dict(table=name, column=col, dtype=str(table[col].dtype),
        missing=int(table[col].isna().sum())) for name, table in tables.items() for col in table.columns]
    reports = ROOT / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dictionary).to_csv(reports / 'data_dictionary.csv', index=False)
    category_rows = [dict(field=col, source_value=value, matches=count)
        for col in ['venue', 'team1', 'team2'] for value, count in matches[col].value_counts().items()]
    pd.DataFrame(category_rows).to_csv(reports / 'category_audit.csv', index=False)
    report = dict(as_of=AS_OF, source_url=URL, archive_sha256=digest(archive),
        built_at_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), pandas=pd.__version__,
        counts={name: len(table) for name, table in tables.items()},
        first_date=matches.date.min(), last_date=matches.end_date.max(),
        boundaries=int(model.is_boundary.sum()), raw_deliveries=len(deliveries),
        primary_deliveries=len(model), nonwide_regular_deliveries=int((deliveries.is_nonwide & ~deliveries.is_super_over).sum()),
        exclusions_disjoint={'super_over_all': int(deliveries.is_super_over.sum()),
            'regular_wides': int((~deliveries.is_super_over & (deliveries.wides > 0)).sum()),
            'regular_noballs_not_wide': int((~deliveries.is_super_over & (deliveries.wides == 0) & (deliveries.noballs > 0)).sum())},
        extra_audit={'nonboundary_fours_sixes': int((deliveries.batter_runs.isin([4, 6]) & deliveries.non_boundary).sum()),
            'primary_nonstandard_conditions': int((model.standard_conditions == False).sum()),
            'matches_with_outcome_method': int(matches.outcome_method.notna().sum()),
            'missing_actual_delivery': int(deliveries.actual_delivery.isna().sum())},
        checks='Unique keys; registry IDs; run/extras accounting; nonnegative runs; foreign keys; aggregate reconciliation passed',
        caveats=['Coverage is what Cricsheet supplies, not independently reconciled to every official scorecard.',
            'Final target metadata is retained separately and must not be used as a historical delivery feature in revised chases.',
            'Season halves use retrospective calendar midpoints; pre-register a schedule cutoff for prospective experiments.',
            'Phase is a nominal 1-6/7-15/16-20 over band; consult recorded powerplays for shortened matches.',
            'This audit describes ingestion only; research.py fits the frozen forecasting experiments.'])
    (ROOT / 'reports/data_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Explicitly replace raw snapshot with current archive')
    args = parser.parse_args()
    if args.refresh:
        obtain(refresh=True)
    build()
