"""Find source PDFs using the Reference values already present in the CSV."""
import hashlib
import re


def split_references(value):
    return [part.strip() for part in (value or '').split(' | ') if part.strip()]


def reference_key(reference):
    return ' '.join(reference.split()).casefold()


def reference_id(reference):
    return hashlib.sha256(reference_key(reference).encode('utf-8')).hexdigest()


def pdf_key(reference):
    return 'publications/' + reference_id(reference) + '.pdf'


def pdf_filename(reference):
    name = re.sub(r'[\\/\x00-\x1f<>:"|?*]', '_', reference).strip('. ')[:180]
    return (name or 'publication') + '.pdf'


def existing_references(records, metadata):
    result = {}
    for row in [*metadata, *records]:
        for reference in split_references(row.get('Reference', '')):
            result.setdefault(reference_key(reference), reference)
    return result
