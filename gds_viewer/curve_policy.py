"""Curve types are allowed globally; per-record CSV links remain provenance only."""

GLOBAL_POLICY = 'global_legacy_curve_types_v1'


def curve_types(metadata):
    return sorted({r['Type'].strip().upper() for r in metadata if r['Type'].strip()})


def plottable(curve, allowed):
    return (curve['role'] == 'curve'
            and curve['mnemonic'].strip().upper() in allowed
            and int(curve['valid_count']) > 0)
