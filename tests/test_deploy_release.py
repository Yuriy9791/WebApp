from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from unittest.mock import patch
from io import BytesIO
import json
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = spec_from_file_location('deploy_release', TOOLS / 'deploy_release.py')
deploy = module_from_spec(spec)
spec.loader.exec_module(deploy)


def test_deploy_requires_updated_web_app_before_aws_writes():
    with patch.object(deploy, 'urlopen', return_value=BytesIO(b'{"status":"ready"}')):
        with pytest.raises(ValueError, match='Deploy the updated'):
            deploy.web_status('https://example.com')
    value = {'supported_curve_policy': deploy.GLOBAL_POLICY}
    with patch.object(deploy, 'urlopen', return_value=BytesIO(json.dumps(value).encode())):
        assert deploy.web_status('https://example.com/') == value
    with pytest.raises(ValueError, match='HTTPS'):
        deploy.web_status('http://example.com')


def test_v002_upload_activation_and_application_read_path(tmp_path):
    import os
    root = os.environ.get('GDS_EXPANDED_TEST_ROOT')
    if not root:
        pytest.skip('Set GDS_EXPANDED_TEST_ROOT to the rebuilt v002 payload')
    assert deploy.validate_payload(Path(root), 'v002') == (164, 332)
    from moto import mock_aws
    from gds_viewer.data import Repository, S3Store
    with mock_aws():
        store = deploy.Store('gds-global-types-integration', 'eu-north-1')
        store.provision(tmp_path / 'state')
        store.upload(Path(root), 'v002')
        assert store.get('current.json')[0] is None
        store.activate(Path(root), 'v002')
        repo = Repository(S3Store(store.bucket, store.region))
        snap = repo.active()
        assert snap.release == 'v002' and len(snap.curves) == 332
        rid = next(k for k, r in snap.records.items() if r['las_filename'] == 'Kurdistan-1.las')
        assert {r['Curve'] for r in repo.curve_rows(snap, [rid])} == {'GR', 'DT'}
        assert repo.table(snap, rid).column_names == ['depth', 'c002', 'c003']


def test_deploy_does_not_require_ui_or_parquet_packages():
    import os
    import subprocess
    code = '''
import importlib.abc, sys
from pathlib import Path
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'pyarrow', 'dash', 'plotly', 'lasio', 'gunicorn'}:
            raise AssertionError('Unnecessary upload dependency: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
sys.path.insert(0, sys.argv[1])
import deploy_release
if len(sys.argv) > 2:
    assert deploy_release.validate_payload(Path(sys.argv[2]), 'v002') == (164, 332)
'''
    command = [sys.executable, '-c', code, str(TOOLS)]
    if os.environ.get('GDS_EXPANDED_TEST_ROOT'):
        command.append(os.environ['GDS_EXPANDED_TEST_ROOT'])
    subprocess.run(command, check=True)
