import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from botocore.exceptions import ClientError
from gds_viewer.cloudshell_check import resolve_connection, bucket_region


def test_saved_config_beats_wrong_shell_region(tmp_path):
    settings=tmp_path/'GDS_CloudShell_Kit/state/connection.json'
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({'bucket':'gds-viewer-data-real-eu-north-1','region':'eu-north-1'}))
    session=MagicMock()
    bucket,region,_=resolve_connection({'AWS_REGION':'us-east-2'},tmp_path/'GDS_Viewer_Iteration_2',tmp_path,session)
    assert bucket=='gds-viewer-data-real-eu-north-1' and region=='eu-north-1'
    session.client.assert_not_called()


def test_discovery_uses_actual_bucket_not_shell_region(tmp_path):
    session=MagicMock(); client=session.client.return_value
    client.get_paginator.return_value.paginate.return_value=[{'Buckets':[{'Name':'for-metadata'}]},{'Buckets':[{'Name':'gds-viewer-data-real-eu-north-1'}]}]
    client.head_bucket.return_value={'BucketRegion':'eu-north-1'}
    bucket,region,_=resolve_connection({'AWS_REGION':'us-east-2'},tmp_path/'app',tmp_path,session)
    assert bucket=='gds-viewer-data-real-eu-north-1' and region=='eu-north-1'


def test_multiple_buckets_do_not_choose_first(tmp_path):
    session=MagicMock(); session.client.return_value.get_paginator.return_value.paginate.return_value=[{'Buckets':[{'Name':'gds-viewer-data-a'},{'Name':'gds-viewer-data-b'}]}]
    with pytest.raises(ValueError,match='exactly one'):
        resolve_connection({},tmp_path/'app',tmp_path,session)


def test_explicit_bucket_skips_list_and_gets_region(tmp_path):
    session=MagicMock();client=session.client.return_value
    client.head_bucket.return_value={'ResponseMetadata':{'HTTPHeaders':{'x-amz-bucket-region':'eu-west-1'}}}
    bucket,region,_=resolve_connection({'GDS_S3_BUCKET':'custom-gds','AWS_REGION':'us-east-2'},tmp_path/'app',tmp_path,session)
    assert bucket=='custom-gds' and region=='eu-west-1'
    client.get_paginator.assert_not_called()


def test_denied_head_with_region_does_not_require_listbucket():
    client=MagicMock()
    client.head_bucket.side_effect=ClientError({'Error':{'Code':'403'},'ResponseMetadata':{'HTTPHeaders':{'x-amz-bucket-region':'eu-north-1'}}},'HeadBucket')
    assert bucket_region(client,'a-real-bucket')=='eu-north-1'


def test_missing_config_is_reported(tmp_path):
    with pytest.raises(ValueError,match='not found'):
        resolve_connection({'GDS_STORAGE_CONFIG':str(tmp_path/'missing.json')},tmp_path/'app',tmp_path,MagicMock())


def test_moto_discovery_in_another_region(tmp_path):
    from moto import mock_aws
    import boto3
    with mock_aws():
        session=boto3.Session(region_name='us-east-2')
        s3=session.client('s3',region_name='eu-north-1')
        s3.create_bucket(Bucket='gds-viewer-data-existing-stockholm',CreateBucketConfiguration={'LocationConstraint':'eu-north-1'})
        bucket,region,_=resolve_connection({'AWS_REGION':'us-east-2'},tmp_path/'app',tmp_path,session)
        assert bucket=='gds-viewer-data-existing-stockholm' and region=='eu-north-1'
